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

#: Chosen on the full 35-case golden set, not a spot check. Both models were
#: run back to back on the same code:
#:
#:                        qwen2.5:7b   qwen3.5:9b
#:   false assurance            5.7%        11.4%   <- the one that costs money
#:   accuracy                  42.9%        34.3%
#:   over-abstention           48.1%        59.3%
#:   justified P/R        55.6/50.0%   16.7/10.0%
#:   correct abstention        62.5%        75.0%
#:   speed                 ~40 tok/s    23.6 tok/s
#:
#: The 9B looked like the clear winner on a four-case spot check (2/4 against
#: the 7B's 0/4) and is worse on the full set at exactly the metric this
#: system exists to protect: it told four people a winnable denial was
#: justified, against the 7B's two. It also failed to extract a denial date on
#: four cases, silently disabling temporal filtering.
#:
#: That reversal is the whole lesson, and it is the second time this project
#: has been bitten by it — see the 5-case run that reported 0% false assurance
#: before the set grew to 35. A model comparison smaller than the full golden
#: set is not evidence.
#:
#: qwen3.6:27b remains unmeasured beyond four cases. At 2.4 tok/s on 8GB VRAM
#: a full run is an overnight job, and it is the outstanding measurement that
#: could genuinely change this choice.
DEFAULT_MODEL = "qwen2.5:7b"


#: Model families that reason unconditionally and return *nothing* when
#: thinking is disabled.
#:
#: gpt-oss answers through a separate channel from its reasoning. Sending
#: `think: false` does not make it skip reasoning; it makes Ollama return an
#: empty `content`, so every schema validation fails on empty input. Measured:
#: 35/35 cases errored on the golden set, every one of them at decomposition,
#: none of which had anything to do with the model's ability to read a denial
#: letter. A capability verdict was never reached.
#:
#: This is exactly the failure this codebase keeps re-learning — a plumbing
#: fault that presents as a capability fault. Left to a config flag it would
#: be rediscovered by whoever next points the pipeline at a reasoning-only
#: model, so it is corrected in code where the constraint actually lives.
REASONING_ONLY_PREFIXES = ("gpt-oss",)

#: Effort used when a reasoning-only model is asked not to reason. Measured on
#: gpt-oss:20b at 8GB VRAM: low 9.0s/call, medium 24.6s, high 51.7s, all three
#: emitting schema-valid JSON. `low` is the setting that keeps a user-facing
#: audit inside a few minutes rather than a quarter of an hour, and it is a
#: floor on the model's quality, not a ceiling — raise it deliberately.
DEFAULT_REASONING_EFFORT = "low"


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
        think: bool | str = False,
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
        #
        # `think` also accepts an effort level ("low"/"medium"/"high") for
        # models that grade their reasoning rather than toggling it.
        #
        # A reasoning-only model is never given `False`, because for those it
        # does not mean "answer without reasoning" — it means "return an empty
        # response". Silently correcting this is the right call precisely
        # because the alternative failure is so misleading: an empty content
        # field surfaces as a schema validation error on every single call,
        # which reads as a model too weak to follow a schema rather than a
        # flag it was never able to honour.
        if think is False and model.startswith(REASONING_ONLY_PREFIXES):
            think = DEFAULT_REASONING_EFFORT
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
            message = data.get("message", {})
            content = message.get("content", "")

            # An empty content field is not a schema failure, and reporting it
            # as one sends the reader hunting for a prompt or model problem
            # that does not exist. It means the answer went somewhere else —
            # almost always a reasoning-only model whose output is in
            # `thinking` because reasoning was disabled.
            if not content.strip():
                thinking = message.get("thinking") or ""
                raise OllamaError(
                    f"{self.model} returned an empty response for "
                    f"{schema.__name__}"
                    + (
                        f" with {len(thinking)} characters of reasoning but no "
                        f"answer. This model reasons unconditionally; it needs "
                        f"think=True or an effort level, not think=False."
                        if thinking
                        else f" (think={self.think!r}). If this is a "
                        f"reasoning-only model, it cannot answer with thinking "
                        f"disabled."
                    )
                )

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
