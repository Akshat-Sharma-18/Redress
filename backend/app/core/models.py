"""Domain models shared across every pipeline.

These are deliberately defined once, here, because the whole point of the
system is that four independently-retrieved bodies of evidence get compared
against each other. That comparison is only meaningful if every pipeline
speaks in the same units: a Chunk with provenance, and a ScoredChunk with a
score whose origin is recorded.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SourceKind(str, Enum):
    """Which pipeline a piece of evidence came from.

    Kept on every chunk so the reconciliation agent can weigh a statute
    differently from the insurer's own denial letter.
    """

    POLICY = "policy"            # Pipeline A - the user's own policy document
    DENIAL = "denial"            # Pipeline B - the denial letter
    STATUTE = "statute"          # Pipeline C - state insurance code
    PRECEDENT = "precedent"      # Pipeline D - DOI complaint / enforcement records


class Chunk(BaseModel):
    """A retrievable unit of text with enough provenance to cite it.

    `locator` is the human-facing citation string ("Section 4.2(b)",
    "Cal. Ins. Code s 10123.13") and is what surfaces in the UI. `char_start`
    and `char_end` index back into the original document so the frontend can
    highlight the exact sentence the citation beam originates from.
    """

    id: str
    text: str
    source_kind: SourceKind
    document_id: str
    locator: str | None = None
    char_start: int | None = None
    char_end: int | None = None

    # Temporal RAG: a statute chunk is only valid evidence for a denial that
    # occurred inside this window. None means "no known bound".
    effective_from: date | None = None
    effective_to: date | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    def applies_on(self, when: date) -> bool:
        """Whether this chunk was in force on `when`.

        Chunks with no temporal bounds (policy text, denial letters) always
        apply -- only statutes carry version windows.
        """
        if self.effective_from is not None and when < self.effective_from:
            return False
        if self.effective_to is not None and when > self.effective_to:
            return False
        return True


class ScoredChunk(BaseModel):
    """A chunk plus the score that surfaced it, and where that score came from.

    `provenance` accumulates across the retrieval stack -- a chunk that came
    out of dense retrieval, survived fusion, and was reranked carries all
    three scores. The audit trail depends on not throwing these away.
    """

    chunk: Chunk
    score: float
    provenance: dict[str, float] = Field(default_factory=dict)

    def with_score(self, stage: str, score: float) -> ScoredChunk:
        """Return a copy carrying a new stage score, preserving history."""
        return ScoredChunk(
            chunk=self.chunk,
            score=score,
            provenance={**self.provenance, stage: score},
        )


class SubClaim(BaseModel):
    """One atomic assertion decomposed out of a denial letter.

    The decomposition agent splits "denied as not medically necessary given
    the pre-existing condition exclusion" into separate factual and legal
    sub-claims, because they are verified against different evidence.
    """

    id: str
    text: str
    kind: str  # "factual" | "legal"
    cited_by_insurer: str | None = None


class Confidence(str, Enum):
    """The three-way confidence gate.

    INSUFFICIENT is a first-class outcome, not a failure mode. A system that
    can't say "I don't know" is not safe to point at someone's medical bills.
    """

    SUPPORTED = "supported"
    CONTESTED = "contested"
    INSUFFICIENT = "insufficient_evidence"


class Citation(BaseModel):
    """A claim tied to the exact retrieved text that supports it.

    `quote` must be a verbatim substring of the chunk. The critique agent
    enforces this -- a citation that paraphrases is a citation that can drift.
    """

    chunk_id: str
    locator: str | None = None
    quote: str
    supports: str


class Verdict(BaseModel):
    """The system's output, with everything needed to reproduce it."""

    sub_claim_id: str
    finding: str  # "justified" | "contradicted" | "mixed"
    confidence: Confidence
    rationale: str
    citations: list[Citation] = Field(default_factory=list)

    # Full audit trail -- retrieval trace, reranker scores, both agent passes.
    retrieval_trace: list[ScoredChunk] = Field(default_factory=list)
    draft_rationale: str | None = None
    critique_notes: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
