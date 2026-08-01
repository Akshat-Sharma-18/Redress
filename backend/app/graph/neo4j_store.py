"""Neo4j-backed GraphStore.

Same questions as the in-memory store, answered in Cypher. The driver is
imported lazily so the graph layer — and its tests — run without neo4j
installed and without an Aura instance provisioned.

The multi-hop pattern query is the reason this backend exists: what the
in-memory store does as four nested Python loops is one traversal here, and
that difference is what lets the graph span every insurer and every state
rather than one case's neighborhood.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Iterable

from app.graph.models import (
    EdgeKind,
    GraphEdge,
    GraphNode,
    NodeKind,
    PatternFinding,
)

# Relationship types are written into Cypher directly, so they must never be
# interpolated from user input. Validating against the enum keeps that true
# even if a caller passes a raw string.
_ALLOWED_EDGES = {k.value for k in EdgeKind}


def _rel(kind: EdgeKind) -> str:
    if kind.value not in _ALLOWED_EDGES:
        raise ValueError(f"unknown edge kind {kind!r}")
    return kind.value.upper()


class Neo4jGraphStore:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "Neo4jGraphStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def ensure_constraints(self) -> None:
        """Uniqueness on :Entity(id) — makes MERGE-by-id an upsert.

        Without this, a re-ingested complaint record creates a duplicate node
        and inflates the pattern counts, which is the one failure mode that
        would make this feature actively misleading.
        """
        with self._driver.session(database=self._database) as session:
            session.run(
                "CREATE CONSTRAINT entity_id IF NOT EXISTS "
                "FOR (n:Entity) REQUIRE n.id IS UNIQUE"
            )

    def upsert_nodes(self, nodes: Iterable[GraphNode]) -> None:
        rows = [
            {
                "id": n.id,
                "kind": n.kind.value,
                "label": n.label,
                "properties": _serialize(n.properties),
            }
            for n in nodes
        ]
        if not rows:
            return
        with self._driver.session(database=self._database) as session:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (n:Entity {id: row.id})
                SET n.kind = row.kind,
                    n.label = row.label,
                    n += row.properties
                """,
                rows=rows,
            )

    def upsert_edges(self, edges: Iterable[GraphEdge]) -> None:
        by_kind: dict[EdgeKind, list[dict[str, Any]]] = {}
        for e in edges:
            by_kind.setdefault(e.kind, []).append(
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "properties": _serialize(e.properties),
                }
            )
        with self._driver.session(database=self._database) as session:
            for kind, rows in by_kind.items():
                # Relationship type cannot be parameterized in Cypher; the
                # enum lookup above is what makes this interpolation safe.
                session.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (a:Entity {{id: row.source}})
                    MATCH (b:Entity {{id: row.target}})
                    MERGE (a)-[r:{_rel(kind)}]->(b)
                    SET r += row.properties
                    """,
                    rows=rows,
                )

    def get_node(self, node_id: str) -> GraphNode | None:
        with self._driver.session(database=self._database) as session:
            record = session.run(
                "MATCH (n:Entity {id: $id}) RETURN n", id=node_id
            ).single()
        return _to_node(record["n"]) if record else None

    def statutes_for_clause(self, clause_id: str) -> list[GraphNode]:
        return self._one_hop(clause_id, EdgeKind.GOVERNED_BY, NodeKind.STATUTE)

    def clauses_for_reason(self, reason_code_id: str) -> list[GraphNode]:
        return self._one_hop(
            reason_code_id, EdgeKind.CITES_CLAUSE, NodeKind.POLICY_CLAUSE
        )

    def _one_hop(
        self, node_id: str, kind: EdgeKind, target_kind: NodeKind
    ) -> list[GraphNode]:
        with self._driver.session(database=self._database) as session:
            records = session.run(
                f"""
                MATCH (a:Entity {{id: $id}})-[:{_rel(kind)}]->(b:Entity)
                WHERE b.kind = $target_kind
                RETURN b
                """,
                id=node_id,
                target_kind=target_kind.value,
            ).data()
        return [_to_node(r["b"]) for r in records]

    def insurer_pattern(
        self, insurer_id: str, reason_code: str | None = None
    ) -> list[PatternFinding]:
        """The pattern query, as a single traversal.

        The `(c)-[:FILED_AGAINST]->(i)` leg is load-bearing: without it the
        query would count every complaint about the reason code industry-wide
        and report it as this insurer's record.
        """
        cypher = """
        MATCH (i:Entity {id: $insurer_id})-[:USED_REASON]->(r:Entity)
        WHERE r.kind = 'denial_reason'
          AND ($reason_code IS NULL OR r.code = $reason_code OR r.id = $reason_code)
        MATCH (c:Entity)-[:CONCERNS_REASON]->(r)
        WHERE c.kind = 'complaint'
        MATCH (c)-[:FILED_AGAINST]->(i)
        OPTIONAL MATCH (c)-[:APPLIED_STATUTE]->(s:Entity)
        WITH i, r,
             collect(DISTINCT c)  AS complaints,
             collect(DISTINCT s.id) AS statute_ids
        RETURN i.id    AS insurer_id,
               i.label AS insurer_label,
               coalesce(r.code, r.id) AS reason_code,
               r.label AS reason_label,
               [x IN complaints WHERE x.outcome = 'overturned'] AS overturned,
               [x IN complaints WHERE x.outcome = 'upheld']     AS upheld,
               [x IN complaints WHERE x.outcome = 'settled']    AS settled,
               [x IN complaints WHERE x.outcome = 'withdrawn']  AS withdrawn,
               [x IN complaints | x.id]          AS complaint_ids,
               [x IN complaints | x.decided_on]  AS decided_dates,
               [x IN statute_ids WHERE x IS NOT NULL] AS statute_ids
        """
        with self._driver.session(database=self._database) as session:
            records = session.run(
                cypher, insurer_id=insurer_id, reason_code=reason_code
            ).data()

        findings = [
            PatternFinding(
                insurer_id=r["insurer_id"],
                insurer_label=r["insurer_label"],
                reason_code=str(r["reason_code"]),
                reason_label=r["reason_label"],
                overturned=len(r["overturned"]),
                upheld=len(r["upheld"]),
                settled=len(r["settled"]),
                withdrawn=len(r["withdrawn"]),
                statute_ids=sorted(r["statute_ids"]),
                complaint_ids=sorted(r["complaint_ids"]),
                **_date_window(r["decided_dates"]),
            )
            for r in records
        ]
        findings.sort(key=lambda f: (f.ruled, f.total), reverse=True)
        return findings


def _date_window(values: list[Any]) -> dict[str, date | None]:
    dates = [_to_date(v) for v in values]
    present = [d for d in dates if d is not None]
    if not present:
        return {"earliest": None, "latest": None}
    return {"earliest": min(present), "latest": max(present)}


def _to_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    # neo4j.time.Date exposes to_native(); ISO strings are the fallback.
    to_native = getattr(value, "to_native", None)
    if callable(to_native):
        native = to_native()
        return native if isinstance(native, date) else None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _serialize(properties: dict[str, Any]) -> dict[str, Any]:
    """Flatten enums to their values; Neo4j stores primitives, not objects."""
    out: dict[str, Any] = {}
    for key, value in properties.items():
        out[key] = value.value if hasattr(value, "value") else value
    return out


def _to_node(record: Any) -> GraphNode:
    props = dict(record)
    return GraphNode(
        id=props.pop("id"),
        kind=NodeKind(props.pop("kind")),
        label=props.pop("label", ""),
        properties=props,
    )
