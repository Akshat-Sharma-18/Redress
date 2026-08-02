"""Local inference via Ollama — the zero-cost, zero-key backend.

Implements the same `StructuredLLM` protocol as the Anthropic backend, so
nothing in the pipeline, gate, or eval changes. Putting the LLM behind a
one-method protocol was speculative when it was written; this is where it
pays for itself.

Schema validity is *enforced*, not requested. Ollama's `format` parameter
accepts a JSON Schema and constrains decoding to it, so the model physically
cannot emit a response that fails to parse. That matters far more for a 7B
model than for a frontier one: the usual failure mode of a small model asked
for JSON is malformed JSON, and constrained decoding removes that failure
mode entirely, leaving only the substantive question of whether the content
is any good.

Only the standard library is used for HTTP. Ollama's REST API is three
endpoints; adding a dependency to reach it would be pure overhead.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Sequence, TypeVar

from pydantic import BaseModel, ValidationError

SchemaT = TypeVar("SchemaT", bound=BaseModel)

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b-instruct"


class OllamaError(RuntimeError):
    pass


def _post(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"cannot reach Ollama at {url} - is `ollama serve` running? ({exc})"
        ) from exc


class OllamaStructuredLLM:
    """Local structured generation over Ollama's /api/chat."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = 900.0,
        num_ctx: int = 8192,
        temperature: float = 0.0,
        max_attempts: int = 2,
        think: bool = False,
    ):
        self.model = model
        self.host = host.rstrip("/")
        # Generous because a large model that does not fit in VRAM runs
        # mostly on CPU: qwen3.6:27b on 8GB measured 2.4 tok/s at a 70/30
        # CPU/GPU split, where the old 300s ceiling cut calls off mid-answer
        # and looked like a model failure rather than a timeout.
        self.timeout = timeout
        # Reasoning models emit a thinking block before the answer. Off by
        # default: on the same hardware, thinking took a trivial reply from
        # 4.8s to 75s — a 15x cost for reasoning the schema then constrains
        # anyway. Turn it on deliberately to trade latency for quality.
        self.think = think
        # Legal text is long; the default 2048-token context silently
        # truncates the evidence pack, which would look like the model
        # ignoring clauses it was never shown.
        self.num_ctx = num_ctx
        # Deterministic by default. The eval compares runs against each
        # other, and sampling noise would show up as capability variance.
        self.temperature = temperature
        self.max_attempts = max_attempts

    def generate(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[SchemaT],
        max_tokens: int = 16000,
    ) -> SchemaT:
        json_schema = schema.model_json_schema()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": json_schema,
            # Ignored by models without a reasoning mode, so it is safe to
            # send unconditionally.
            "think": self.think,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
            },
        }

        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            data = _post(f"{self.host}/api/chat", payload, self.timeout)
            content = data.get("message", {}).get("content", "")
            try:
                return schema.model_validate_json(content)
            except ValidationError as exc:
                # Constrained decoding guarantees well-formed JSON matching
                # the schema's shape, but Pydantic also enforces constraints
                # the JSON Schema cannot express. Retry once, then surface —
                # never fabricate a default, which would invent a verdict.
                last_error = exc
                if attempt + 1 < self.max_attempts:
                    payload["messages"] = [
                        *payload["messages"][:2],
                        {
                            "role": "user",
                            "content": (
                                "Your previous response failed validation:\n"
                                f"{exc}\n\nReturn corrected JSON."
                            ),
                        },
                    ]

        raise OllamaError(
            f"{self.model} produced invalid output for {schema.__name__} "
            f"after {self.max_attempts} attempts: {last_error}"
        )


class OllamaEmbedder:
    """Dense embeddings via Ollama — replaces sentence-transformers.

    Removes the torch dependency entirely: `nomic-embed-text` is 274 MB and
    runs on CPU, versus a multi-gigabyte PyTorch install. The retrieval stack
    never knew which embedder it had, so this is a constructor swap.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = DEFAULT_HOST,
        timeout: float = 120.0,
    ):
        self.name = model
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        data = _post(
            f"{self.host}/api/embed",
            {"model": self.model, "input": list(texts)},
            self.timeout,
        )
        embeddings = data.get("embeddings")
        if not embeddings or len(embeddings) != len(texts):
            raise OllamaError(
                f"expected {len(texts)} embeddings from {self.model}, "
                f"got {len(embeddings or [])}"
            )
        return embeddings


def available_models(host: str = DEFAULT_HOST, timeout: float = 10.0) -> list[str]:
    """Names of models Ollama currently has pulled."""
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise OllamaError(f"cannot reach Ollama at {host}: {exc}") from exc
    return [m["name"] for m in data.get("models", [])]
