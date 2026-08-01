"""Critique Agent — the adversarial second pass.

A separately-prompted agent re-reads the draft verdict against the actual
retrieved text and asks one question per citation: does the cited text say
what the rationale claims it says?

This is a different check from the mechanical verification in
`reconciliation.py`. The substring check proves the quote is real; it cannot
prove the quote *means* what the rationale uses it for. "This exclusion does
not apply to emergency services" is a verbatim quote that a bad rationale can
deploy to argue the exclusion *does* apply. Catching that requires reading,
so it is a second model pass — deliberately prompted as an adversary whose
job is to reject, not to help.

The critique agent never rewrites a verdict. It approves, or it rejects with
reasons and (optionally) a narrowed retrieval query so the reconciliation
pass can be re-run against better evidence. Keeping it write-free means a
critique failure can only ever make the system more conservative.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.agents.llm import StructuredLLM
from app.agents.reconciliation import _format_evidence
from app.core.models import ScoredChunk, SubClaim, Verdict

CRITIQUE_SYSTEM = """\
You are the critique stage of an insurance claim denial audit system. A \
draft verdict has been produced for one sub-claim. Your job is to try to \
REJECT it. You are not a collaborator; you are the reviewer whose signature \
means a policyholder can rely on this verdict.

You are given the sub-claim, the draft verdict (finding, rationale, and \
citations), and the full evidence chunks the citations point into.

Reject the draft if ANY of the following hold:

1. A cited passage, read in its surrounding chunk context, does not support \
what the rationale claims it supports. Pay particular attention to negation, \
carve-backs ("notwithstanding..."), conditions precedent, and defined terms \
whose definition may differ from plain English.
2. The rationale makes any load-bearing claim that no citation covers.
3. The finding does not follow from the citations even if each is accurate \
— e.g. citations establish an exclusion exists but not that it applies to \
this claim.
4. The evidence suggests a chunk that would settle the question was NOT \
retrieved (e.g. a citation references a definition or section that is absent \
from the evidence). In this case, also provide a narrowed retrieval query \
targeting the missing material.

Approve only when every check passes. If you reject, list each issue \
concretely — name the citation and the exact mismatch. A vague rejection is \
as useless as a wrong approval.

Do not judge whether YOU would have reached a different finding on balance; \
judge whether THIS finding is supported by THESE citations."""


class CritiqueResult(BaseModel):
    """Structured output of the critique pass."""

    approved: bool = Field(
        description="True only if every citation supports its claim and the "
        "finding follows from the citations"
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Concrete problems found; empty when approved",
    )
    narrowed_query: str | None = Field(
        default=None,
        description=(
            "Only when rejection stems from missing evidence: a short, "
            "specific retrieval query targeting the material that should "
            "have been retrieved (e.g. a definition, a referenced section)"
        ),
    )


class CritiqueAgent:
    def __init__(self, llm: StructuredLLM):
        self.llm = llm

    def review(
        self,
        sub_claim: SubClaim,
        verdict: Verdict,
        evidence: list[ScoredChunk],
    ) -> CritiqueResult:
        citations = "\n".join(
            f'- chunk_id={c.chunk_id!r} locator={c.locator!r}\n'
            f'  quote: "{c.quote}"\n'
            f"  claimed to support: {c.supports}"
            for c in verdict.citations
        ) or "(none)"

        return self.llm.generate(
            system=CRITIQUE_SYSTEM,
            prompt=(
                f"<sub_claim>\n{sub_claim.text}\n</sub_claim>\n\n"
                f"<draft_verdict finding={verdict.finding!r}>\n"
                f"{verdict.rationale}\n"
                f"</draft_verdict>\n\n"
                f"<citations>\n{citations}\n</citations>\n\n"
                f"<evidence>\n{_format_evidence(evidence)}\n</evidence>"
            ),
            schema=CritiqueResult,
        )
