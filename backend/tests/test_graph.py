"""GraphRAG layer tests.

Exercised against the in-memory store, which is the reference implementation
the Neo4j backend answers the same questions as. The Cypher backend needs a
live instance and is covered by the integration suite, not here.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.graph.evidence import GraphEnricher, pattern_chunk
from app.graph.models import (
    ComplaintOutcome,
    EdgeKind,
    GraphEdge,
    GraphNode,
    NodeKind,
)
from app.graph.store import InMemoryGraphStore


def _node(nid, kind, label, **props):
    return GraphNode(id=nid, kind=kind, label=label, properties=props)


def _edge(src, kind, tgt):
    return GraphEdge(source_id=src, kind=kind, target_id=tgt)


@pytest.fixture
def store() -> InMemoryGraphStore:
    """A two-insurer graph sharing one reason code.

    The shared reason code is the point: a pattern query for one insurer
    must not absorb the other's complaint record, and a store that walks
    reason-code edges without re-anchoring on the insurer will.
    """
    s = InMemoryGraphStore()
    s.upsert_nodes([
        _node("ins-acme", NodeKind.INSURER, "Acme Health"),
        _node("ins-other", NodeKind.INSURER, "Other Mutual"),
        _node("rc-CO50", NodeKind.DENIAL_REASON, "Not medically necessary", code="CO-50"),
        _node("rc-CO97", NodeKind.DENIAL_REASON, "Bundled service", code="CO-97"),
        _node("clause-7.2a", NodeKind.POLICY_CLAUSE, "Section 7.2(a)"),
        _node("stat-1371.4", NodeKind.STATUTE, "Ins. Code s 1371.4"),
        # Acme: CO-50 -> two overturned, one upheld
        _node("cmp-1", NodeKind.COMPLAINT, "Complaint 1",
              outcome=ComplaintOutcome.OVERTURNED, decided_on=date(2021, 3, 1)),
        _node("cmp-2", NodeKind.COMPLAINT, "Complaint 2",
              outcome=ComplaintOutcome.OVERTURNED, decided_on=date(2022, 7, 15)),
        _node("cmp-3", NodeKind.COMPLAINT, "Complaint 3",
              outcome=ComplaintOutcome.UPHELD, decided_on=date(2020, 1, 10)),
        # Other Mutual: same reason code, different insurer
        _node("cmp-4", NodeKind.COMPLAINT, "Complaint 4",
              outcome=ComplaintOutcome.OVERTURNED, decided_on=date(2021, 5, 5)),
    ])
    s.upsert_edges([
        _edge("ins-acme", EdgeKind.USED_REASON, "rc-CO50"),
        _edge("ins-acme", EdgeKind.USED_REASON, "rc-CO97"),
        _edge("ins-other", EdgeKind.USED_REASON, "rc-CO50"),
        _edge("rc-CO50", EdgeKind.CITES_CLAUSE, "clause-7.2a"),
        _edge("clause-7.2a", EdgeKind.GOVERNED_BY, "stat-1371.4"),
        *[_edge(c, EdgeKind.FILED_AGAINST, "ins-acme") for c in ("cmp-1", "cmp-2", "cmp-3")],
        *[_edge(c, EdgeKind.CONCERNS_REASON, "rc-CO50") for c in ("cmp-1", "cmp-2", "cmp-3")],
        *[_edge(c, EdgeKind.APPLIED_STATUTE, "stat-1371.4") for c in ("cmp-1", "cmp-2")],
        _edge("cmp-4", EdgeKind.FILED_AGAINST, "ins-other"),
        _edge("cmp-4", EdgeKind.CONCERNS_REASON, "rc-CO50"),
    ])
    return s


class TestPatternQuery:
    def test_multi_hop_pattern(self, store):
        [finding] = store.insurer_pattern("ins-acme", reason_code="CO-50")

        assert finding.overturned == 2
        assert finding.upheld == 1
        assert finding.ruled == 3
        assert finding.statute_ids == ["stat-1371.4"]
        assert finding.earliest == date(2020, 1, 10)
        assert finding.latest == date(2022, 7, 15)

    def test_does_not_absorb_another_insurers_complaints(self, store):
        """cmp-4 concerns CO-50 but was filed against a different insurer.

        Without the filed-against re-anchor, this query reports 3 overturned
        instead of 2 — an inflated pattern claim about a named company.
        """
        [finding] = store.insurer_pattern("ins-acme", reason_code="CO-50")
        assert "cmp-4" not in finding.complaint_ids
        assert finding.overturned == 2

    def test_reason_code_with_no_complaints_is_omitted(self, store):
        """CO-97 is used by Acme but has no complaint record."""
        findings = store.insurer_pattern("ins-acme")
        assert {f.reason_code for f in findings} == {"CO-50"}

    def test_unknown_insurer_returns_empty(self, store):
        assert store.insurer_pattern("ins-nonexistent") == []

    def test_settled_and_withdrawn_excluded_from_ruled(self):
        """Settlements say nothing about how a regulator viewed the denial."""
        s = InMemoryGraphStore()
        s.upsert_nodes([
            _node("i", NodeKind.INSURER, "I"),
            _node("r", NodeKind.DENIAL_REASON, "R", code="X"),
            _node("c1", NodeKind.COMPLAINT, "C1", outcome=ComplaintOutcome.SETTLED),
            _node("c2", NodeKind.COMPLAINT, "C2", outcome=ComplaintOutcome.WITHDRAWN),
        ])
        s.upsert_edges([
            _edge("i", EdgeKind.USED_REASON, "r"),
            _edge("c1", EdgeKind.FILED_AGAINST, "i"),
            _edge("c2", EdgeKind.FILED_AGAINST, "i"),
            _edge("c1", EdgeKind.CONCERNS_REASON, "r"),
            _edge("c2", EdgeKind.CONCERNS_REASON, "r"),
        ])
        [finding] = s.insurer_pattern("i")
        assert finding.total == 2
        assert finding.ruled == 0

    def test_edge_upsert_is_idempotent(self, store):
        """Re-ingesting the same records must not double-count the pattern."""
        store.upsert_edges([
            _edge("cmp-1", EdgeKind.FILED_AGAINST, "ins-acme"),
            _edge("cmp-1", EdgeKind.CONCERNS_REASON, "rc-CO50"),
        ])
        [finding] = store.insurer_pattern("ins-acme", reason_code="CO-50")
        assert finding.overturned == 2
        assert finding.complaint_ids == ["cmp-1", "cmp-2", "cmp-3"]


class TestTraversalHelpers:
    def test_clause_to_statute(self, store):
        [statute] = store.statutes_for_clause("clause-7.2a")
        assert statute.id == "stat-1371.4"

    def test_reason_to_clause(self, store):
        [clause] = store.clauses_for_reason("rc-CO50")
        assert clause.label == "Section 7.2(a)"

    def test_missing_node_yields_no_neighbors(self, store):
        assert store.statutes_for_clause("clause-nonexistent") == []


class TestPatternEvidence:
    def test_chunk_states_counts_and_caveat(self, store):
        [finding] = store.insurer_pattern("ins-acme", reason_code="CO-50")
        chunk = pattern_chunk(finding)

        assert "2 overturned" in chunk.text
        assert "1 upheld" in chunk.text
        # The caveat is in the chunk itself, so the agent reads it as evidence
        # rather than relying on the system prompt to remember it.
        assert "does not establish that the present denial is improper" in chunk.text
        assert chunk.metadata["derived"] is True

    def test_enricher_skips_thin_records(self):
        """One unadjudicated complaint is not a pattern worth surfacing."""
        s = InMemoryGraphStore()
        s.upsert_nodes([
            _node("i", NodeKind.INSURER, "I"),
            _node("r", NodeKind.DENIAL_REASON, "R", code="X"),
            _node("c1", NodeKind.COMPLAINT, "C1", outcome=ComplaintOutcome.SETTLED),
        ])
        s.upsert_edges([
            _edge("i", EdgeKind.USED_REASON, "r"),
            _edge("c1", EdgeKind.FILED_AGAINST, "i"),
            _edge("c1", EdgeKind.CONCERNS_REASON, "r"),
        ])
        enricher = GraphEnricher(store=s)
        assert enricher.enrich(
            insurer_id="i", reason_code="X", denial_date=None
        ) == []

    def test_enricher_emits_scored_chunk(self, store):
        enricher = GraphEnricher(store=store)
        [scored] = enricher.enrich(
            insurer_id="ins-acme", reason_code="CO-50", denial_date=None
        )
        assert scored.chunk.source_kind.value == "precedent"
        # Score 0.0: derived evidence must not displace retrieved text
        assert scored.score == 0.0

    def test_enricher_without_insurer_is_a_noop(self, store):
        enricher = GraphEnricher(store=store)
        assert enricher.enrich(
            insurer_id=None, reason_code="CO-50", denial_date=None
        ) == []
