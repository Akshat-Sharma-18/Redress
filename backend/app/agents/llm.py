"""The LLM boundary.

One narrow interface: give me text and a Pydantic schema, get back a validated
instance. Everything above this line — decomposition, reconciliation, and in
Phase 3 the critique loop — is deterministic Python that can be tested with a
scripted fake. Everything below it is the Anthropic SDK.

The critique agent (Phase 3) will be "separately-prompted" per the design; it
uses this same interface with a different system prompt, which is why the
system prompt is a call argument and not client state.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)

# Claude Opus 5: thinking on by default, structured outputs supported,
# 1M context — a full policy document fits in a single request.
DEFAULT_MODEL = "claude-opus-5"


@runtime_checkable
class StructuredLLM(Protocol):
    def generate(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[SchemaT],
        max_tokens: int = 16000,
    ) -> SchemaT:
        ...


class AnthropicStructuredLLM:
    """Production implementation over the Anthropic Messages API.

    Uses `messages.parse` with the Pydantic schema, so the response is
    validated at the API layer — the model is constrained to the schema
    rather than asked nicely to follow it.
    """

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "high"):
        import anthropic

        self._client = anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def generate(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[SchemaT],
        max_tokens: int = 16000,
    ) -> SchemaT:
        response = self._client.messages.parse(
            model=self.model,
            max_tokens=max_tokens,
            # System prompts are stable per agent; cache them. The per-case
            # document text goes in the user turn, after the cached prefix.
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
        )
        parsed = response.parsed_output
        if parsed is None:
            # With output_config format constraints this should not happen
            # outside of a refusal; surface it rather than fabricating.
            raise RuntimeError(
                f"Model returned unparseable output (stop_reason="
                f"{response.stop_reason!r})"
            )
        return parsed
