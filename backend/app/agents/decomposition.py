"""Claim Decomposition Agent (Pipeline B + fan-out preparation).

Splits a denial letter into atomic sub-claims before any retrieval happens.
This is the design's answer to the "embed the whole letter as one blob"
failure: a denial like "not medically necessary given the pre-existing
condition exclusion" contains at least two independently checkable assertions,
and retrieving for the blob returns evidence for neither cleanly.

The agent also performs the structured extraction (reason code, cited
sections, denial date) — the denial date is what drives temporal filtering
in Pipeline C.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date

from app.agents.llm import StructuredLLM
from app.agents.schemas import Decomposition
from app.core.models import SubClaim

DECOMPOSITION_SYSTEM = """\
You are the claim decomposition stage of an insurance claim denial audit \
system. You are given the text of a denial letter. Your output feeds four \
independent evidence-retrieval pipelines, so precision matters more than \
coverage.

Extract the denial's structure, then decompose the insurer's reasoning into \
atomic sub-claims:

- Each sub-claim is ONE checkable assertion. "The service was not medically \
necessary and falls under the custodial care exclusion" is TWO sub-claims.
- Mark a sub-claim 'factual' when it can be verified against documents or \
records (a date of service, a diagnosis code, a coverage amount, network \
status). Mark it 'legal' when it turns on what a policy clause or statute \
means or requires.
- Record exactly what the insurer cited for each sub-claim, if anything. Do \
not supply citations the letter does not contain.
- Extract dates in ISO format. If the denial date is not stated, leave it \
null — do not infer it from context.
- Quote the denial reason verbatim where possible. You are extracting, not \
summarizing."""


@dataclass
class DecomposedDenial:
    """Decomposition output mapped into domain objects."""

    sub_claims: list[SubClaim]
    denial_reason: str
    reason_code: str | None
    denial_date: date | None
    cited_policy_sections: list[str]
    factual_assertions: list[str]


class DecompositionAgent:
    def __init__(self, llm: StructuredLLM):
        self.llm = llm

    def decompose(self, denial_letter_text: str) -> DecomposedDenial:
        result = self.llm.generate(
            system=DECOMPOSITION_SYSTEM,
            prompt=(
                "Decompose the following denial letter.\n\n"
                "<denial_letter>\n"
                f"{denial_letter_text}\n"
                "</denial_letter>"
            ),
            schema=Decomposition,
        )

        denial_date: date | None = None
        if result.denial.denial_date:
            try:
                denial_date = date.fromisoformat(result.denial.denial_date)
            except ValueError:
                # A malformed date is treated as absent rather than guessed
                # at: an absent date disables temporal filtering (safe),
                # a wrong date silently filters against the wrong law (not).
                denial_date = None

        sub_claims = [
            SubClaim(
                id=f"sc-{uuid.uuid4().hex[:8]}",
                text=claim.text,
                kind=claim.kind if claim.kind in ("factual", "legal") else "legal",
                cited_by_insurer=claim.cited_by_insurer,
            )
            for claim in result.sub_claims
        ]

        return DecomposedDenial(
            sub_claims=sub_claims,
            denial_reason=result.denial.denial_reason,
            reason_code=result.denial.reason_code,
            denial_date=denial_date,
            cited_policy_sections=result.denial.cited_policy_sections,
            factual_assertions=result.denial.factual_assertions,
        )
