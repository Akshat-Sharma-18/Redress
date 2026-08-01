"""Golden dataset: hand-built denial scenarios with known-correct verdicts.

Cases are YAML because a human has to read and edit them. The ambiguous ones
in particular are only useful if a reviewer can check the reasoning, and a
policy clause rendered as a JSON string with escaped newlines is not
reviewable.

Expectations are recorded at the *case* level, not per sub-claim. Sub-claims
are produced by a model and their wording varies between runs, so aligning
hand-written expectations to generated sub-claims would make the harness
measure the decomposition agent's phrasing rather than the system's verdict.
The case-level `overall` plus a required-citation set is both stable and
stricter — it catches a right answer reached on the wrong evidence.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Category(str, Enum):
    """What kind of case this is — the axis the abstention metric needs.

    AMBIGUOUS is the load-bearing category: these are cases where the
    evidence genuinely does not settle the question, and the correct system
    behaviour is to decline to rule. A golden set without them cannot measure
    whether abstention works, only whether it happens.
    """

    CLEAR_CONTRADICTION = "clear_contradiction"
    CLEAR_JUSTIFICATION = "clear_justification"
    MIXED = "mixed"
    AMBIGUOUS = "ambiguous"


class GoldenChunk(BaseModel):
    """A policy/statute/precedent chunk supplied with the case."""

    id: str
    text: str
    source_kind: str = "policy"
    locator: str | None = None
    effective_from: date | None = None
    effective_to: date | None = None


class Expectation(BaseModel):
    overall: str = Field(
        description="Expected case-level finding: justified | contradicted | "
        "mixed | insufficient"
    )
    must_cite: list[str] = Field(
        default_factory=list,
        description=(
            "Chunk ids a correct verdict has to rest on. A verdict that "
            "reaches the right finding without these got there on the wrong "
            "grounds, which is measured separately."
        ),
    )
    must_not_cite: list[str] = Field(
        default_factory=list,
        description=(
            "Chunk ids that would indicate a specific known failure — e.g. "
            "a superseded statute version, or a distractor clause."
        ),
    )


class GoldenCase(BaseModel):
    id: str
    name: str
    category: Category
    denial_letter: str
    expected: Expectation
    chunks: list[GoldenChunk] = Field(default_factory=list)
    state: str | None = None
    insurer_id: str | None = None
    denial_date: date | None = None
    statute_ids: list[str] = Field(default_factory=list)
    notes: str | None = Field(
        default=None,
        description="Reviewer-facing rationale for the expected label. "
        "Required on ambiguous cases — see load_dataset.",
    )
    source: str = Field(
        default="synthetic",
        description="'synthetic' or 'anonymized'. Real cases must be "
        "anonymized before they enter the repository.",
    )


class GoldenDatasetError(ValueError):
    pass


def load_case(path: Path) -> GoldenCase:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise GoldenDatasetError(f"{path.name}: expected a YAML mapping")
    return GoldenCase(**data)


def load_dataset(directory: Path) -> list[GoldenCase]:
    """Load and validate every case in a directory.

    Validation is strict on two points, both of which protect the metric
    rather than the code:

    - Duplicate ids are rejected. A silently overwritten case would shrink
      the set without changing the reported denominator.
    - Ambiguous cases must carry `notes`. "The correct answer here is to
      abstain" is a judgement call, and an unexplained one cannot be
      reviewed — which would make the headline abstention metric rest on
      unexamined labels.
    """
    cases: list[GoldenCase] = []
    seen: dict[str, str] = {}

    for path in sorted(directory.glob("*.yaml")):
        case = load_case(path)
        if case.id in seen:
            raise GoldenDatasetError(
                f"duplicate case id {case.id!r} in {path.name} "
                f"(already defined in {seen[case.id]})"
            )
        seen[case.id] = path.name

        if case.category is Category.AMBIGUOUS and not case.notes:
            raise GoldenDatasetError(
                f"{path.name}: ambiguous cases must explain in `notes` why "
                f"abstention is the correct outcome"
            )
        cases.append(case)

    return cases
