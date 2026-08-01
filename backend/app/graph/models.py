"""Graph schema for the clause-statute-precedent knowledge graph.

The graph exists to answer a question the four retrieval pipelines
structurally cannot: not "is this one denial justified" but "does this
insurer have a pattern". That question is a multi-hop traversal —
insurer → denial reason code → prior complaint outcomes → the statute the
regulator applied — and it is the reason a graph store earns its place
alongside vector and lexical retrieval rather than duplicating them.

Nodes are deliberately thin. The graph stores *relationships and
identifiers*; the text those identifiers point at lives in the chunk store.
Duplicating clause text into graph properties would create two sources of
truth for the same sentence, and the citation chain would eventually cite
the stale one.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NodeKind(str, Enum):
    INSURER = "insurer"
    POLICY_CLAUSE = "policy_clause"
    DENIAL_REASON = "denial_reason"      # a reason code, e.g. "CO-50"
    STATUTE = "statute"
    COMPLAINT = "complaint"              # a DOI complaint / enforcement record


class EdgeKind(str, Enum):
    """Edges are directed and named from the subject's point of view."""

    USED_REASON = "used_reason"          # insurer -> denial_reason
    CITES_CLAUSE = "cites_clause"        # denial_reason -> policy_clause
    GOVERNED_BY = "governed_by"          # policy_clause -> statute
    FILED_AGAINST = "filed_against"      # complaint -> insurer
    CONCERNS_REASON = "concerns_reason"  # complaint -> denial_reason
    APPLIED_STATUTE = "applied_statute"  # complaint -> statute


class GraphNode(BaseModel):
    id: str
    kind: NodeKind
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source_id: str
    kind: EdgeKind
    target_id: str
    properties: dict[str, Any] = Field(default_factory=dict)


class ComplaintOutcome(str, Enum):
    """How a DOI complaint resolved.

    OVERTURNED is the only outcome that is affirmative evidence *for* the
    policyholder. UPHELD cuts the other way, and both must be reported —
    a tool that surfaces only the favorable half of the regulatory record
    is not an audit, it is advocacy with extra steps.
    """

    OVERTURNED = "overturned"        # regulator ruled against the insurer
    UPHELD = "upheld"                # regulator sided with the insurer
    SETTLED = "settled"              # resolved without a ruling
    WITHDRAWN = "withdrawn"


class PatternFinding(BaseModel):
    """One insurer/reason-code pair with its regulatory history.

    Deliberately reports the full counts rather than a single score. A
    reason code overturned 3 times out of 4 and one overturned 3 times out
    of 200 are very different facts, and collapsing them into a "risk
    score" would invent a precision the underlying complaint data does not
    have.
    """

    insurer_id: str
    insurer_label: str
    reason_code: str
    reason_label: str
    overturned: int = 0
    upheld: int = 0
    settled: int = 0
    withdrawn: int = 0
    statute_ids: list[str] = Field(default_factory=list)
    complaint_ids: list[str] = Field(default_factory=list)
    earliest: date | None = None
    latest: date | None = None

    @property
    def total(self) -> int:
        return self.overturned + self.upheld + self.settled + self.withdrawn

    @property
    def ruled(self) -> int:
        """Complaints that actually produced a ruling.

        Settlements and withdrawals are excluded: neither tells you what a
        regulator thought of the denial, and counting them would let a
        litigious insurer's settlement history read as vindication.
        """
        return self.overturned + self.upheld

    def summary(self) -> str:
        """Human-readable line for the evidence pack shown to the agent."""
        if self.ruled == 0:
            return (
                f"{self.insurer_label} has {self.total} recorded complaint(s) "
                f"involving reason code {self.reason_code}, none of which "
                f"reached a ruling."
            )
        window = ""
        if self.earliest and self.latest:
            window = f" between {self.earliest.isoformat()} and {self.latest.isoformat()}"
        return (
            f"{self.insurer_label} has {self.ruled} adjudicated complaint(s) "
            f"involving reason code {self.reason_code}{window}: "
            f"{self.overturned} overturned, {self.upheld} upheld."
        )
