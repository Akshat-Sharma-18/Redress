"""Pydantic schemas for structured LLM outputs.

These are the shapes the model is *constrained* to produce via the API's
structured-outputs feature — not post-hoc parsed from free text. Constraining
at the API level matters here: a verdict that fails to parse is a verdict the
confidence gate can't inspect, and silently retrying free-form JSON is exactly
the kind of unauditable behavior this system is designed to avoid.

Kept separate from app.core.models because these are *wire* schemas for one
LLM call each; the core models are the system's persistent domain objects.
The pipeline maps one to the other explicitly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedDenial(BaseModel):
    """Structured extraction from a denial letter (Pipeline B).

    Everything optional except the reason: real denial letters are wildly
    inconsistent, and a missing field must surface as None, not a guess.
    """

    denial_reason: str = Field(
        description="The insurer's stated reason for denial, verbatim where possible"
    )
    reason_code: str | None = Field(
        default=None, description="Denial reason code if stated, e.g. 'CO-50'"
    )
    denial_date: str | None = Field(
        default=None,
        description=(
            "The date of the denial decision, copied EXACTLY as it appears in "
            "the letter (e.g. 'June 14, 2021'). Do not reformat or convert it. "
            "Null only if the letter states no date."
        ),
    )
    cited_policy_sections: list[str] = Field(
        default_factory=list,
        description="Policy sections/clauses the letter cites, e.g. 'Section 7.2(a)'",
    )
    factual_assertions: list[str] = Field(
        default_factory=list,
        description=(
            "Factual claims the insurer makes: dates of service, diagnosis "
            "codes, coverage amounts, provider network status"
        ),
    )


class DecomposedClaim(BaseModel):
    """One atomic sub-claim extracted from the denial."""

    text: str = Field(
        description=(
            "A single atomic assertion that can be verified against one body "
            "of evidence. Never combine two checkable statements."
        )
    )
    kind: str = Field(
        description=(
            "'factual' if verifiable against documents/records (dates, codes, "
            "amounts); 'legal' if it turns on policy language or statute"
        )
    )
    cited_by_insurer: str | None = Field(
        default=None,
        description="The clause or statute the insurer ties this assertion to, if any",
    )


class Decomposition(BaseModel):
    """Full decomposition output: parsed letter + atomic sub-claims."""

    denial: ExtractedDenial
    sub_claims: list[DecomposedClaim] = Field(
        description="Atomic factual and legal sub-claims, typically 2-8"
    )


class DraftCitation(BaseModel):
    """A citation in a draft verdict. chunk_id must reference provided evidence."""

    chunk_id: str = Field(description="The id of the evidence chunk being cited")
    quote: str = Field(
        description=(
            "VERBATIM quote from that chunk supporting the claim. Must be an "
            "exact substring of the chunk text — no paraphrase, no ellipsis."
        )
    )
    supports: str = Field(
        description="One sentence: what this quote establishes"
    )


class DraftVerdict(BaseModel):
    """Reconciliation agent's draft verdict for one sub-claim.

    'finding' is deliberately three-way and 'insufficient' is a legal answer:
    the prompt instructs the model that declining to rule is preferred over
    ruling without citable support, and the schema makes that option
    first-class rather than something the model has to smuggle into prose.
    """

    finding: str = Field(
        description=(
            "'justified' if evidence supports the denial on this sub-claim; "
            "'contradicted' if evidence contradicts it; 'mixed' if evidence "
            "cuts both ways; 'insufficient' if the provided evidence does not "
            "settle it. Prefer 'insufficient' over guessing."
        )
    )
    rationale: str = Field(
        description="2-4 sentences explaining the finding, referencing the citations"
    )
    citations: list[DraftCitation] = Field(
        description=(
            "Every claim in the rationale must be backed by a citation from "
            "the provided evidence chunks. An empty list is only valid with "
            "finding='insufficient'."
        )
    )
