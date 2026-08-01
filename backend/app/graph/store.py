"""Graph store interface and an in-process implementation.

The protocol carries the *analytical* queries, not just primitive traversal.
That is deliberate: expressing `insurer_pattern` as a generic
`neighbors()`-walk would force Neo4j to execute a Python-side traversal it
could do in one Cypher statement, which defeats the point of running a graph
database. Each backend implements the question; only the mechanism differs.

The in-memory store is not a mock. It is the reference implementation the
Neo4j backend is checked against, and it is what runs in tests and in the
local demo — so the graph layer is fully exercised without provisioning
anything.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Iterable, Protocol, runtime_checkable

from app.graph.models import (
    ComplaintOutcome,
    EdgeKind,
    GraphEdge,
    GraphNode,
    NodeKind,
    PatternFinding,
)


@runtime_checkable
class GraphStore(Protocol):
    def upsert_nodes(self, nodes: Iterable[GraphNode]) -> None: ...

    def upsert_edges(self, edges: Iterable[GraphEdge]) -> None: ...

    def get_node(self, node_id: str) -> GraphNode | None: ...

    def insurer_pattern(
        self, insurer_id: str, reason_code: str | None = None
    ) -> list[PatternFinding]:
        """Regulatory history for an insurer, optionally one reason code.

        The system's highest-value query: does this insurer have a pattern
        of using this denial reason, and how have regulators ruled on it.
        """
        ...

    def statutes_for_clause(self, clause_id: str) -> list[GraphNode]:
        """Statutes governing a policy clause (clause -> statute, one hop)."""
        ...

    def clauses_for_reason(self, reason_code_id: str) -> list[GraphNode]:
        """Policy clauses a denial reason code cites."""
        ...


class InMemoryGraphStore:
    """Adjacency-list graph over dicts.

    Fine at the scale this system operates on: one state's statutes plus
    that state's published complaint records is tens of thousands of nodes,
    which is a rounding error in RAM. Neo4j earns its keep when the graph
    spans every state and every insurer.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._out: dict[tuple[str, EdgeKind], list[GraphEdge]] = defaultdict(list)
        self._in: dict[tuple[str, EdgeKind], list[GraphEdge]] = defaultdict(list)

    def upsert_nodes(self, nodes: Iterable[GraphNode]) -> None:
        for node in nodes:
            self._nodes[node.id] = node

    def upsert_edges(self, edges: Iterable[GraphEdge]) -> None:
        for edge in edges:
            out_key = (edge.source_id, edge.kind)
            in_key = (edge.target_id, edge.kind)
            # Idempotent upsert: re-ingesting a complaint record must not
            # double-count it into the pattern statistics.
            existing = self._out[out_key]
            for i, e in enumerate(existing):
                if e.target_id == edge.target_id:
                    existing[i] = edge
                    incoming = self._in[in_key]
                    for j, ie in enumerate(incoming):
                        if ie.source_id == edge.source_id:
                            incoming[j] = edge
                            break
                    break
            else:
                existing.append(edge)
                self._in[in_key].append(edge)

    def get_node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    def _targets(self, node_id: str, kind: EdgeKind) -> list[GraphNode]:
        return [
            self._nodes[e.target_id]
            for e in self._out[(node_id, kind)]
            if e.target_id in self._nodes
        ]

    def _sources(self, node_id: str, kind: EdgeKind) -> list[GraphNode]:
        return [
            self._nodes[e.source_id]
            for e in self._in[(node_id, kind)]
            if e.source_id in self._nodes
        ]

    def statutes_for_clause(self, clause_id: str) -> list[GraphNode]:
        return [
            n
            for n in self._targets(clause_id, EdgeKind.GOVERNED_BY)
            if n.kind is NodeKind.STATUTE
        ]

    def clauses_for_reason(self, reason_code_id: str) -> list[GraphNode]:
        return [
            n
            for n in self._targets(reason_code_id, EdgeKind.CITES_CLAUSE)
            if n.kind is NodeKind.POLICY_CLAUSE
        ]

    def insurer_pattern(
        self, insurer_id: str, reason_code: str | None = None
    ) -> list[PatternFinding]:
        insurer = self._nodes.get(insurer_id)
        if insurer is None:
            return []

        # Hop 1: which reason codes has this insurer used?
        reasons = [
            n
            for n in self._targets(insurer_id, EdgeKind.USED_REASON)
            if n.kind is NodeKind.DENIAL_REASON
        ]
        if reason_code is not None:
            reasons = [
                n
                for n in reasons
                if n.properties.get("code") == reason_code
                or n.id == reason_code
            ]

        # Hop 2: complaints filed against this insurer, indexed for the join.
        complaints_against = {
            n.id
            for n in self._sources(insurer_id, EdgeKind.FILED_AGAINST)
            if n.kind is NodeKind.COMPLAINT
        }

        findings: list[PatternFinding] = []
        for reason in reasons:
            # Hop 3: complaints concerning this reason code AND this insurer.
            # The intersection is what makes it a *pattern* claim rather than
            # "someone, somewhere, was denied for this reason".
            relevant = [
                c
                for c in self._sources(reason.id, EdgeKind.CONCERNS_REASON)
                if c.kind is NodeKind.COMPLAINT and c.id in complaints_against
            ]
            if not relevant:
                continue

            finding = PatternFinding(
                insurer_id=insurer_id,
                insurer_label=insurer.label,
                reason_code=str(reason.properties.get("code", reason.id)),
                reason_label=reason.label,
                complaint_ids=sorted(c.id for c in relevant),
            )

            statute_ids: set[str] = set()
            dates: list[date] = []
            for complaint in relevant:
                outcome = complaint.properties.get("outcome")
                if outcome == ComplaintOutcome.OVERTURNED:
                    finding.overturned += 1
                elif outcome == ComplaintOutcome.UPHELD:
                    finding.upheld += 1
                elif outcome == ComplaintOutcome.SETTLED:
                    finding.settled += 1
                elif outcome == ComplaintOutcome.WITHDRAWN:
                    finding.withdrawn += 1

                # Hop 4: the statute the regulator applied.
                statute_ids.update(
                    s.id
                    for s in self._targets(complaint.id, EdgeKind.APPLIED_STATUTE)
                    if s.kind is NodeKind.STATUTE
                )

                decided = complaint.properties.get("decided_on")
                if isinstance(decided, date):
                    dates.append(decided)

            finding.statute_ids = sorted(statute_ids)
            if dates:
                finding.earliest = min(dates)
                finding.latest = max(dates)
            findings.append(finding)

        # Most-adjudicated first; a pattern with rulings outranks one without.
        findings.sort(key=lambda f: (f.ruled, f.total), reverse=True)
        return findings
