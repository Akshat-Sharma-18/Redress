"""The confidence gate: critique loop + ensemble cross-check.

Composes the reconciliation and critique agents into the verdict flow from
the design's Section 5:

    draft -> critique -> (re-retrieve + redraft, bounded) -> ensemble check
                                                          -> SUPPORTED
                                                          -> CONTESTED
                                                          -> INSUFFICIENT

Every path through this gate can only *lower* confidence relative to the
draft. The critique agent cannot upgrade a finding, the ensemble check cannot
resolve a disagreement in either side's favor, and a failed critique after
the retry budget is exhausted lands on INSUFFICIENT — never on the draft's
original claim.
"""

from __future__ import annotations

from typing import Callable

from app.agents.critique import CritiqueAgent
from app.agents.reconciliation import ReconciliationAgent
from app.core.models import Confidence, ScoredChunk, SubClaim, Verdict

# Signature for narrowed re-retrieval: query -> fresh evidence.
RetrieveFn = Callable[[str], list[ScoredChunk]]


def _rejection_note(issues: list[str]) -> str:
    """Render the critique's reasons, or say plainly that it gave none.

    Joining an empty list produced "critique rejected: " — a dangling colon
    that told a reader the verdict was thrown out and nothing about why. The
    critique agent's own instructions call a vague rejection as useless as a
    wrong approval, so the same standard applies to how we report it.

    A reason-less rejection still downgrades. Treating it as an approval
    would let a malformed critique response upgrade a verdict, which inverts
    the one property this gate exists to hold.
    """
    reasons = [i.strip() for i in issues if i and i.strip()]
    if not reasons:
        return (
            "critique rejected the draft but gave no reason; treated as "
            "unverified"
        )
    return "critique rejected: " + "; ".join(reasons)


def _merge_evidence(
    base: list[ScoredChunk], extra: list[ScoredChunk]
) -> list[ScoredChunk]:
    """Union by chunk id, preserving the original ordering first.

    The narrowed retrieval *supplements* the evidence; it must never replace
    it. Dropping the original chunks would let a bad narrowed query hide the
    carve-back that the critique pass was worried about.
    """
    seen = {sc.chunk.id for sc in base}
    return base + [sc for sc in extra if sc.chunk.id not in seen]


class GatedAdjudicator:
    """Runs one sub-claim through the full fail-safe verdict flow."""

    def __init__(
        self,
        reconciliation: ReconciliationAgent,
        critique: CritiqueAgent,
        max_reretrievals: int = 1,
    ):
        self.reconciliation = reconciliation
        self.critique = critique
        self.max_reretrievals = max_reretrievals

    def adjudicate(
        self,
        sub_claim: SubClaim,
        evidence: list[ScoredChunk],
        retrieve_fn: RetrieveFn | None = None,
        secondary_evidence: list[ScoredChunk] | None = None,
    ) -> Verdict:
        """Adjudicate with critique and (optionally) the ensemble cross-check.

        `secondary_evidence` is the same sub-claim's retrieval under a
        different embedding model. When provided, a non-insufficient verdict
        is only labeled SUPPORTED if an independent adjudication over that
        second evidence set reaches the same finding; disagreement routes to
        CONTESTED rather than picking a winner.
        """
        verdict = self.reconciliation.adjudicate(sub_claim, evidence)

        # An abstention (clean or downgraded) is already the floor — there is
        # nothing for the critique pass to reject.
        if verdict.finding == "insufficient":
            return verdict

        verdict, evidence = self._critique_loop(
            sub_claim, verdict, evidence, retrieve_fn
        )
        if verdict.finding == "insufficient":
            return verdict

        if secondary_evidence is not None:
            verdict = self._ensemble_check(sub_claim, verdict, secondary_evidence)

        return verdict

    def _critique_loop(
        self,
        sub_claim: SubClaim,
        verdict: Verdict,
        evidence: list[ScoredChunk],
        retrieve_fn: RetrieveFn | None,
    ) -> tuple[Verdict, list[ScoredChunk]]:
        reretrievals_left = self.max_reretrievals

        while True:
            critique = self.critique.review(sub_claim, verdict, evidence)
            if critique.approved:
                verdict.critique_notes = "critique: approved"
                return verdict, evidence

            can_retry = (
                critique.narrowed_query is not None
                and retrieve_fn is not None
                and reretrievals_left > 0
            )
            if not can_retry:
                # Terminal rejection: fail safe, keep the paper trail.
                return (
                    Verdict(
                        sub_claim_id=sub_claim.id,
                        finding="insufficient",
                        confidence=Confidence.INSUFFICIENT,
                        rationale=(
                            "The draft verdict did not survive adversarial "
                            "review, and no ruling is made. Professional "
                            "review is recommended."
                        ),
                        citations=verdict.citations,
                        retrieval_trace=evidence,
                        draft_rationale=verdict.rationale,
                        critique_notes=_rejection_note(critique.issues),
                    ),
                    evidence,
                )

            # Re-retrieve with the narrowed query and redraft. The critique
            # agent's concern is recorded even if the redraft succeeds.
            reretrievals_left -= 1
            evidence = _merge_evidence(
                evidence, retrieve_fn(critique.narrowed_query)
            )
            verdict = self.reconciliation.adjudicate(sub_claim, evidence)
            if verdict.finding == "insufficient":
                return verdict, evidence

    def _ensemble_check(
        self,
        sub_claim: SubClaim,
        verdict_a: Verdict,
        secondary_evidence: list[ScoredChunk],
    ) -> Verdict:
        """Independent adjudication over the second embedder's evidence.

        Findings agree -> the verdict stands (SUPPORTED). Anything else —
        including the second pass abstaining — is disagreement, and
        disagreement is never resolved by choosing; it is surfaced as
        CONTESTED with both findings in the audit trail.
        """
        verdict_b = self.reconciliation.adjudicate(sub_claim, secondary_evidence)

        if verdict_b.finding == verdict_a.finding:
            verdict_a.critique_notes = (
                (verdict_a.critique_notes or "") + "; ensemble: agreed"
            ).lstrip("; ")
            return verdict_a

        verdict_a.confidence = Confidence.CONTESTED
        verdict_a.critique_notes = (
            (verdict_a.critique_notes or "")
            + f"; ensemble: disagreed (primary={verdict_a.finding!r}, "
            f"secondary={verdict_b.finding!r})"
        ).lstrip("; ")
        verdict_a.rationale += (
            " Note: an independent retrieval pass reached a different "
            "conclusion, so this verdict is marked contested — professional "
            "review is recommended."
        )
        return verdict_a
