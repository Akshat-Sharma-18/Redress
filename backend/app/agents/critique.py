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
You are the verification stage of an insurance claim denial audit system. A \
draft verdict has been produced for one sub-claim. You decide whether it is \
supported by its own citations.

You are an impartial checker, not an opponent. Both errors cost a real \
person: approving an unsupported verdict misinforms them about their money, \
and rejecting a well-supported one denies them an answer they were entitled \
to. Do not look for reasons to reject; read what the evidence says and \
report what you find.

Work through the citations one at a time and record, for each, whether the \
quoted text supports the specific claim it is offered for. Read each quote \
in the context of its full chunk, attending to negation, carve-backs \
("notwithstanding..."), conditions, and defined terms whose meaning may \
differ from plain English. When a clause states a numeric condition — a \
date, a visit count, a dollar threshold — check the sub-claim's own facts \
against that number yourself; do not accept that the clause was matched on \
category alone. A clause about duplicate claims is not support for "this \
was a duplicate" unless the dates given actually match. A twenty-visit cap \
supports denying visit 21 onward, not a block of visits that starts below \
the cap. When a citation confirms a factual sub-claim is true, check \
separately what that fact means under the provision it triggers — a \
confirmed fact that satisfies a carve-back or exception argues against the \
denial, not for it, even though the fact was read correctly. A draft that \
treats "this fact is confirmed" as "this fact justifies the denial" without \
making that second check has not earned its finding. Then decide whether \
the finding follows from the citations you accepted.

Approve when the citations you accepted are sufficient to support the \
finding. A verdict does not need to cite every relevant clause, address \
arguments nobody made, or match the verdict you would personally have \
written. It needs to be supported by what it cites.

Reject only when you can name a specific defect: a quote that does not say \
what it is offered for, a load-bearing claim that no citation covers, a \
numeric condition the sub-claim's own facts do not actually satisfy, a \
confirmed fact treated as support for the denial when the provision it \
triggers actually argues against it, or a finding that does not follow \
even though each citation is accurate. "The \
evidence could be more complete" is not a defect. If, and only if, a \
citation references material that is genuinely absent from the evidence (a \
definition or section referred to but not present), supply a narrowed \
retrieval query for it.

Judge whether THIS finding is supported by THESE citations."""


class CitationCheck(BaseModel):
    """One citation, adjudicated on its own before any overall conclusion."""

    chunk_id: str = Field(description="The citation being checked")
    supports_claim: bool = Field(
        description="Does the quoted text, read in its chunk context, support "
        "the specific claim it is offered for?"
    )
    note: str = Field(
        description="One sentence: what the quote actually establishes"
    )


class CritiqueResult(BaseModel):
    """Structured output of the critique pass.

    Field order is load-bearing. Constrained decoding emits fields in schema
    order, so putting the per-citation checks before `approved` forces the
    model to examine each quote before committing to a verdict on the whole.
    Asking for the conclusion first invites a snap judgement that the rest of
    the response then rationalises — which is how the previous version of
    this agent came to reject every draft it was shown, including provably
    correct ones.
    """

    citation_checks: list[CitationCheck] = Field(
        default_factory=list,
        description="One entry per citation in the draft, in order",
    )
    finding_follows: bool = Field(
        default=True,
        description="Given the citations marked supports_claim=true, does the "
        "draft's finding follow?",
    )
    approved: bool = Field(
        description="True when the accepted citations are sufficient to "
        "support the finding"
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Specific named defects; empty when approved",
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
