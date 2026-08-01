"""Wire format for the frontend.

Separate from `app.core.models` on purpose. The domain models carry things
the UI must never depend on (retriever provenance keys, raw draft text), and
the UI needs things the domain models do not carry (resolved character spans,
a rendered disposition label). Coupling them would mean every retrieval
change became a frontend change.

The one non-obvious job here is span resolution. The citation beam has to
start at a specific sentence in the denial letter and land on a specific
line of an evidence chunk, so both ends need character offsets — computed
once, server-side, rather than by the browser re-finding the text and
disagreeing about whitespace.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.agents.pipeline import AuditResult
from app.agents.reconciliation import find_verbatim_span
from app.core.models import Confidence

#: Maps a verdict finding to the visual treatment the UI applies. Kept
#: server-side so the palette decision lives next to the semantics that
#: justify it, rather than being re-derived in a component.
_TONE = {
    "contradicted": "contradicted",
    "mixed": "contested",
    "justified": "verified",
    "insufficient": "pending",
}


class SpanOut(BaseModel):
    """A resolved character range in some document."""

    start: int
    end: int
    text: str


class CitationOut(BaseModel):
    chunk_id: str
    locator: str | None
    quote: str
    supports: str
    #: Where the quote sits inside its chunk, for highlighting the target.
    span: SpanOut | None = None


class EvidenceOut(BaseModel):
    id: str
    text: str
    source_kind: str
    locator: str | None
    score: float
    #: True when this chunk was derived (graph pattern, statute version diff)
    #: rather than retrieved. The UI marks these differently because they are
    #: generated summaries, not text from the user's own documents.
    derived: bool = False
    cited: bool = False


class SubClaimOut(BaseModel):
    id: str
    text: str
    kind: str
    finding: str
    confidence: str
    tone: str
    rationale: str
    citations: list[CitationOut] = Field(default_factory=list)
    #: Where in the denial letter this sub-claim came from. None when the
    #: model's quote could not be located — the UI then shows the sub-claim
    #: without an origin anchor rather than pointing at the wrong sentence.
    source_span: SpanOut | None = None
    critique_notes: str | None = None
    draft_rationale: str | None = None


class AuditOut(BaseModel):
    case_id: str
    denial_letter: str
    denial_reason: str
    reason_code: str | None
    denial_date: str | None
    disposition: str
    confidence: str
    tone: str
    sub_claims: list[SubClaimOut] = Field(default_factory=list)
    evidence: list[EvidenceOut] = Field(default_factory=list)


def _span(needle: str, haystack: str) -> SpanOut | None:
    """Locate `needle` in `haystack`, tolerating only whitespace differences.

    Uses the same matcher as citation verification, so a span the backend
    certified as verbatim always resolves here too. Any mismatch returns
    None and the UI degrades to showing the text without a highlight.
    """
    if not needle:
        return None
    found = find_verbatim_span(needle, haystack)
    if found is None:
        return None
    idx = haystack.find(found)
    return None if idx < 0 else SpanOut(start=idx, end=idx + len(found), text=found)


def serialise(result: AuditResult, case_id: str, denial_letter: str) -> AuditOut:
    """Flatten one audit into the shape the frontend consumes."""
    # Evidence is deduplicated across sub-claims: the same clause is usually
    # retrieved for several of them, and the UI renders one card per clause
    # with beams from each sub-claim that cites it.
    evidence: dict[str, EvidenceOut] = {}
    cited_ids: set[str] = set()

    for sub in result.results:
        for scored in sub.verdict.retrieval_trace:
            chunk = scored.chunk
            if chunk.id not in evidence:
                evidence[chunk.id] = EvidenceOut(
                    id=chunk.id,
                    text=chunk.text,
                    source_kind=chunk.source_kind.value,
                    locator=chunk.locator,
                    score=scored.score,
                    derived=bool(chunk.metadata.get("derived")),
                )
        for citation in sub.verdict.citations:
            cited_ids.add(citation.chunk_id)

    for chunk_id in cited_ids:
        if chunk_id in evidence:
            evidence[chunk_id].cited = True

    sub_claims: list[SubClaimOut] = []
    for sub in result.results:
        verdict = sub.verdict
        claim = next(
            (c for c in result.denial.sub_claims if c.id == verdict.sub_claim_id),
            None,
        )
        source_span = None
        if claim and claim.source_start is not None and claim.source_quote:
            source_span = SpanOut(
                start=claim.source_start,
                end=claim.source_end or claim.source_start,
                text=claim.source_quote,
            )

        citations = []
        for citation in verdict.citations:
            chunk = evidence.get(citation.chunk_id)
            citations.append(
                CitationOut(
                    chunk_id=citation.chunk_id,
                    locator=citation.locator,
                    quote=citation.quote,
                    supports=citation.supports,
                    span=_span(citation.quote, chunk.text) if chunk else None,
                )
            )

        sub_claims.append(
            SubClaimOut(
                id=verdict.sub_claim_id,
                text=claim.text if claim else "",
                kind=claim.kind if claim else "legal",
                finding=verdict.finding,
                confidence=verdict.confidence.value,
                tone=_TONE.get(verdict.finding, "pending"),
                rationale=verdict.rationale,
                citations=citations,
                source_span=source_span,
                critique_notes=verdict.critique_notes,
                draft_rationale=verdict.draft_rationale,
            )
        )

    disposition = result.disposition
    confidence = result.confidence
    # A contested case gets the pending treatment regardless of its
    # direction: the UI must not stamp CONTRADICTED over a finding the gate
    # declined to certify.
    tone = (
        "contested"
        if disposition == "contested"
        else _TONE.get(disposition, "pending")
    )

    return AuditOut(
        case_id=case_id,
        denial_letter=denial_letter,
        denial_reason=result.denial.denial_reason,
        reason_code=result.denial.reason_code,
        denial_date=(
            result.denial.denial_date.isoformat()
            if result.denial.denial_date
            else None
        ),
        disposition=disposition,
        confidence=confidence.value
        if isinstance(confidence, Confidence)
        else str(confidence),
        tone=tone,
        sub_claims=sub_claims,
        evidence=list(evidence.values()),
    )
