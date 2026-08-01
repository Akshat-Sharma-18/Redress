"""Reconciliation Agent v1 — single-pass verdict with citations.

Phase 2 scope: draft verdict only. The adversarial critique pass and the
ensemble cross-check arrive in Phase 3. What ships now, because it is the
cheapest and most load-bearing safety property in the system, is *mechanical
citation verification*: every quote in a verdict is checked, in code, to be a
verbatim substring of the chunk it cites. An LLM promised "only quote the
evidence"; a substring check proves it.

A verdict whose citations fail verification is not repaired or re-asked in
this phase — it is downgraded to INSUFFICIENT and the failure is recorded in
the audit trail. Fail safe, then explain.
"""

from __future__ import annotations

from app.agents.llm import StructuredLLM
from app.agents.schemas import DraftVerdict
from app.core.models import (
    Chunk,
    Citation,
    Confidence,
    ScoredChunk,
    SubClaim,
    Verdict,
)

RECONCILIATION_SYSTEM = """\
You are the reconciliation stage of an insurance claim denial audit system. \
You are given ONE atomic sub-claim from a denial letter and a set of evidence \
chunks retrieved from the policyholder's own policy, state insurance law, and \
regulatory records. Each chunk has an id.

Decide whether the evidence supports the insurer's sub-claim ('justified'), \
contradicts it ('contradicted'), cuts both ways ('mixed'), or does not settle \
it ('insufficient').

Non-negotiable rules:

1. Use ONLY the provided evidence chunks. You have broad knowledge of \
insurance law; none of it is admissible here. If the evidence does not \
contain the answer, the finding is 'insufficient' — that is a correct and \
respected outcome, not a failure.
2. Every citation quote must be copied character-for-character from the \
chunk it cites. No paraphrase, no ellipsis, no cleanup of typos. Citations \
are verified mechanically against the source text; an inexact quote \
invalidates the verdict.
3. Watch for carve-backs: an exclusion in one chunk may be limited or \
reversed by an adjacent provision ("notwithstanding the foregoing..."). \
Never rule on an exclusion without checking whether a provided chunk \
carves it back.
4. A real person's money depends on this verdict being right rather than \
confident. When torn between 'insufficient' and any other finding, choose \
'insufficient'."""

_VALID_FINDINGS = {"justified", "contradicted", "mixed", "insufficient"}


def _format_evidence(chunks: list[ScoredChunk]) -> str:
    parts = []
    for sc in chunks:
        c = sc.chunk
        locator = f" locator={c.locator!r}" if c.locator else ""
        parts.append(
            f'<chunk id="{c.id}" source="{c.source_kind.value}"{locator}>\n'
            f"{c.text}\n"
            f"</chunk>"
        )
    return "\n\n".join(parts)


class ReconciliationAgent:
    def __init__(self, llm: StructuredLLM):
        self.llm = llm

    def adjudicate(
        self, sub_claim: SubClaim, evidence: list[ScoredChunk]
    ) -> Verdict:
        """Produce a verified verdict for one sub-claim.

        The returned Verdict carries the full retrieval trace and the raw
        draft rationale, so a downgrade is inspectable after the fact.
        """
        if not evidence:
            return Verdict(
                sub_claim_id=sub_claim.id,
                finding="insufficient",
                confidence=Confidence.INSUFFICIENT,
                rationale="No evidence was retrieved for this sub-claim.",
                retrieval_trace=[],
            )

        draft = self.llm.generate(
            system=RECONCILIATION_SYSTEM,
            prompt=(
                f"<sub_claim kind={sub_claim.kind!r}>\n{sub_claim.text}\n</sub_claim>\n"
                + (
                    f"<insurer_cited>{sub_claim.cited_by_insurer}</insurer_cited>\n"
                    if sub_claim.cited_by_insurer
                    else ""
                )
                + f"\n<evidence>\n{_format_evidence(evidence)}\n</evidence>"
            ),
            schema=DraftVerdict,
        )

        return self._verify(sub_claim, draft, evidence)

    def _verify(
        self,
        sub_claim: SubClaim,
        draft: DraftVerdict,
        evidence: list[ScoredChunk],
    ) -> Verdict:
        """Mechanical verification of the draft against the actual evidence.

        Checks, in code, with no model in the loop:
          - the finding is one of the four legal values
          - every cited chunk_id exists in the retrieved evidence
          - every quote is a verbatim substring of its chunk's text
          - a non-insufficient finding carries at least one citation
        """
        by_id: dict[str, Chunk] = {sc.chunk.id: sc.chunk for sc in evidence}
        failures: list[str] = []

        finding = draft.finding if draft.finding in _VALID_FINDINGS else None
        if finding is None:
            failures.append(f"invalid finding value {draft.finding!r}")
            finding = "insufficient"

        verified: list[Citation] = []
        for cit in draft.citations:
            chunk = by_id.get(cit.chunk_id)
            if chunk is None:
                failures.append(
                    f"citation references unknown chunk {cit.chunk_id!r}"
                )
                continue
            if cit.quote not in chunk.text:
                failures.append(
                    f"quote is not a verbatim substring of chunk {cit.chunk_id!r}: "
                    f"{cit.quote[:80]!r}"
                )
                continue
            verified.append(
                Citation(
                    chunk_id=chunk.id,
                    locator=chunk.locator,
                    quote=cit.quote,
                    supports=cit.supports,
                )
            )

        if finding != "insufficient" and not verified:
            failures.append(
                f"finding {finding!r} has no surviving citations after verification"
            )

        if failures:
            # Downgrade rather than repair. The draft's own words are kept in
            # the audit trail; the user-facing verdict makes no claim the
            # system could not substantiate.
            return Verdict(
                sub_claim_id=sub_claim.id,
                finding="insufficient",
                confidence=Confidence.INSUFFICIENT,
                rationale=(
                    "The system drafted a verdict but could not verify its "
                    "citations against the retrieved evidence, so no ruling "
                    "is made. Professional review is recommended."
                ),
                citations=verified,
                retrieval_trace=evidence,
                draft_rationale=draft.rationale,
                critique_notes="; ".join(failures),
            )

        confidence = (
            Confidence.INSUFFICIENT
            if finding == "insufficient"
            else Confidence.SUPPORTED
        )
        return Verdict(
            sub_claim_id=sub_claim.id,
            finding=finding,
            confidence=confidence,
            rationale=draft.rationale,
            citations=verified,
            retrieval_trace=evidence,
            draft_rationale=draft.rationale,
        )
