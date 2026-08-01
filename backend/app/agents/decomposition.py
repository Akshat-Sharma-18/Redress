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
from datetime import date, datetime

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
- Copy the denial date exactly as the letter writes it. Do not reformat or \
convert it. If the letter states no date, leave it null — never infer one \
from context.
- Quote the denial reason verbatim where possible. You are extracting, not \
summarizing."""


#: Formats seen on real denial notices, most-specific first. Parsed in code
#: rather than asked of the model: converting "June 14, 2021" to an ISO string
#: is a mechanical transformation, and a model that gets it wrong disables
#: temporal filtering silently. `strptime` either parses or it doesn't.
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%B %d, %Y",     # June 14, 2021
    "%b %d, %Y",     # Jun 14, 2021
    "%B %d %Y",
    "%d %B %Y",      # 14 June 2021
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%d.%m.%Y",
)


def parse_denial_date(raw: str | None) -> date | None:
    """Normalise a denial date the model copied out of the letter.

    Returns None for anything unparseable. That is the safe failure: an
    absent date disables temporal filtering, whereas a wrongly-guessed one
    silently filters against law that did not govern the claim.

    Ambiguity is not resolved by preference. "03/04/2021" is March 4th in the
    US and April 3rd elsewhere, and there is no basis in the letter for
    choosing; `%m/%d/%Y` is listed because these are US insurance documents,
    which is an assumption worth stating rather than hiding.
    """
    if not raw:
        return None
    text = raw.strip().rstrip(".")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


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

        denial_date = parse_denial_date(result.denial.denial_date)

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
