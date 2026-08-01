"""Shared fixtures.

The fake embedder exists so retrieval behaviour can be tested deterministically
and without torch. It is a hashed bag-of-words projection -- crude, but it has
the one property the tests need: texts sharing vocabulary land near each other,
and its output is stable across runs and machines.
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Sequence

import pytest

from app.core.models import Chunk, SourceKind
from app.retrieval.bm25 import tokenize


class FakeLLM:
    """Returns queued responses in order; records every call it saw."""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, *, system, prompt, schema, max_tokens=16000):
        self.calls.append({"system": system, "prompt": prompt, "schema": schema})
        response = self._responses.pop(0)
        assert isinstance(response, schema), (
            f"test sequencing error: expected {schema.__name__}, "
            f"got {type(response).__name__}"
        )
        return response


class FakeEmbedder:
    def __init__(self, name: str = "fake-a", dims: int = 64, salt: str = ""):
        self.name = name
        self.dims = dims
        self.salt = salt

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dims
            for tok in tokenize(text):
                digest = hashlib.md5((self.salt + tok).encode()).digest()
                vec[digest[0] % self.dims] += 1.0
            out.append(vec)
        return out


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def policy_chunks() -> list[Chunk]:
    """A miniature policy with the trap this system has to handle.

    Chunks 2 and 3 are near-identical in wording but opposite in effect -- one
    excludes emergency services, the other carves them back in. Any retriever
    that returns the wrong one produces a confidently wrong verdict.
    """
    texts = [
        (
            "p1",
            "Definitions. Custodial Care means assistance with activities of "
            "daily living that does not require the skills of qualified "
            "technical or professional personnel.",
            "Section 1.4",
        ),
        (
            "p2",
            "Exclusions. This plan does not provide benefits for services "
            "rendered in a hospital emergency department.",
            "Section 7.2(a)",
        ),
        (
            "p3",
            "Notwithstanding the foregoing, the exclusion in Section 7.2(a) "
            "does not apply to emergency services required to evaluate or "
            "stabilize an emergency medical condition.",
            "Section 7.2(b)",
        ),
        (
            "p4",
            "Appeals. You may request an internal appeal of an adverse benefit "
            "determination within one hundred eighty days of receiving notice.",
            "Section 9.1",
        ),
    ]
    return [
        Chunk(
            id=cid,
            text=text,
            source_kind=SourceKind.POLICY,
            document_id="policy-demo",
            locator=locator,
        )
        for cid, text, locator in texts
    ]


@pytest.fixture
def statute_chunks() -> list[Chunk]:
    """Two versions of the same statute, with different effective windows.

    This is the temporal RAG case: a denial dated 2021 must be judged against
    the 2019 text, not the amendment that took effect in 2023.
    """
    return [
        Chunk(
            id="s1-v1",
            text=(
                "An insurer shall reimburse emergency services necessary to "
                "screen and stabilize an enrollee, subject to prior "
                "authorization where the insurer maintains a 24-hour line."
            ),
            source_kind=SourceKind.STATUTE,
            document_id="ins-code-1371.4",
            locator="s 1371.4 (2019)",
            effective_from=date(2019, 1, 1),
            effective_to=date(2022, 12, 31),
        ),
        Chunk(
            id="s1-v2",
            text=(
                "An insurer shall reimburse emergency services necessary to "
                "screen and stabilize an enrollee, and may not require prior "
                "authorization for such services."
            ),
            source_kind=SourceKind.STATUTE,
            document_id="ins-code-1371.4",
            locator="s 1371.4 (2023)",
            effective_from=date(2023, 1, 1),
        ),
    ]
