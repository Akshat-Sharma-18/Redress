from __future__ import annotations

from datetime import date

import pytest

from app.core.models import Chunk, ScoredChunk, SourceKind
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.hybrid import HybridRetriever

from .conftest import FakeEmbedder


class TestBM25:
    def test_finds_defined_term_exactly(self, policy_chunks):
        """The case dense retrieval is bad at: a term of art.

        "Custodial Care" has a defined meaning in the policy that has little to
        do with its plain-English sense, so the definitions clause must win on
        exact match.
        """
        index = BM25Index(policy_chunks)
        results = index.search("custodial care", top_k=3)

        assert results[0].chunk.id == "p1"
        assert results[0].provenance["bm25"] > 0

    def test_empty_corpus_returns_nothing(self):
        assert BM25Index([]).search("anything") == []

    def test_ignores_terms_absent_from_corpus(self, policy_chunks):
        index = BM25Index(policy_chunks)
        assert index.search("cryptocurrency blockchain") == []

    def test_common_terms_never_score_negative(self, policy_chunks):
        """A term in >half the corpus must not push scores down.

        Without the IDF clamp this returns results ordered by *absence* of the
        query term, which is a silent, plausible-looking failure.
        """
        chunks = [
            Chunk(
                id=f"c{i}",
                text="benefit coverage" + (" emergency" if i == 0 else ""),
                source_kind=SourceKind.POLICY,
                document_id="d",
            )
            for i in range(5)
        ]
        results = BM25Index(chunks).search("benefit coverage emergency")
        assert all(r.score >= 0 for r in results)
        assert results[0].chunk.id == "c0"


class TestFusion:
    def test_agreement_beats_a_single_strong_rank(self):
        """Two retrievers agreeing at rank 2 should beat one shouting at rank 1.

        This is the property RRF is chosen for -- it is the reason we fuse on
        rank instead of normalising incomparable score scales.
        """
        a = _sc("solo", 99.0)
        b = _sc("agreed", 0.5)

        fused = reciprocal_rank_fusion(
            {
                "dense": [a, b],
                "bm25": [_sc("other", 1.0), b],
            }
        )
        assert fused[0].chunk.id == "agreed"

    def test_preserves_per_retriever_provenance(self):
        fused = reciprocal_rank_fusion(
            {"dense": [_sc("x", 0.9)], "bm25": [_sc("x", 4.2)]}
        )
        prov = fused[0].provenance
        assert prov["dense"] == 0.9
        assert prov["bm25"] == 4.2
        assert prov["dense_rank"] == 1.0
        assert "rrf" in prov

    def test_empty_input(self):
        assert reciprocal_rank_fusion({}) == []


class TestDense:
    def test_retrieves_on_shared_vocabulary(self, policy_chunks, embedder):
        index = DenseIndex(policy_chunks, embedder)
        results = index.search("appeal an adverse benefit determination", top_k=2)
        assert results[0].chunk.id == "p4"

    def test_provenance_records_which_model_scored_it(self, policy_chunks, embedder):
        """The ensemble cross-check needs to tell two dense passes apart."""
        results = DenseIndex(policy_chunks, embedder).search("appeal", top_k=1)
        assert "dense:fake-a" in results[0].provenance

    def test_two_embedders_produce_distinguishable_traces(self, policy_chunks):
        a = DenseIndex(policy_chunks, FakeEmbedder(name="fake-a"))
        b = DenseIndex(policy_chunks, FakeEmbedder(name="fake-b", salt="z"))

        ra = a.search("appeal", top_k=1)[0]
        rb = b.search("appeal", top_k=1)[0]
        assert set(ra.provenance) != set(rb.provenance)


class TestTemporalFiltering:
    """The property that makes a citation trustworthy: right law, right date."""

    def test_applies_on_respects_window(self, statute_chunks):
        v1, v2 = statute_chunks
        assert v1.applies_on(date(2021, 6, 1))
        assert not v1.applies_on(date(2023, 6, 1))
        assert not v2.applies_on(date(2021, 6, 1))
        assert v2.applies_on(date(2024, 1, 1))

    def test_unbounded_chunks_always_apply(self, policy_chunks):
        assert policy_chunks[0].applies_on(date(1999, 1, 1))

    def test_retriever_returns_law_as_it_stood_on_denial_date(self, statute_chunks):
        """A 2021 denial must be judged against the 2019 statute text.

        Returning the current version here would be the most dangerous bug in
        the system: a fluent, well-cited verdict resting on a law that had not
        been written yet.
        """
        retriever = _build(statute_chunks)
        results, trace = retriever.retrieve(
            "prior authorization for emergency services",
            top_k=5,
            as_of=date(2021, 6, 1),
        )

        ids = [r.chunk.id for r in results]
        assert "s1-v1" in ids
        assert "s1-v2" not in ids
        assert "s1-v2" in trace.temporally_excluded

    def test_no_date_means_no_filtering(self, statute_chunks):
        retriever = _build(statute_chunks)
        results, trace = retriever.retrieve("emergency services", top_k=5)
        assert len(results) == 2
        assert trace.temporally_excluded == []


class TestHybridRetriever:
    def test_disambiguates_near_identical_opposing_clauses(self, policy_chunks):
        """7.2(a) excludes emergency care; 7.2(b) carves it back in.

        Both must surface. Returning only the exclusion would let the system
        confirm a denial that the very next clause overturns -- the exact
        failure mode this project exists to catch.
        """
        retriever = _build(policy_chunks)
        results, _ = retriever.retrieve("emergency department services", top_k=4)

        ids = {r.chunk.id for r in results}
        assert {"p2", "p3"} <= ids

    def test_trace_captures_every_stage(self, policy_chunks):
        retriever = _build(policy_chunks)
        _, trace = retriever.retrieve("custodial care", top_k=2)

        assert trace.query == "custodial care"
        assert trace.dense_hits
        assert trace.lexical_hits
        assert trace.fused
        assert len(trace.final) <= 2

    def test_top_k_is_respected(self, policy_chunks):
        retriever = _build(policy_chunks)
        results, _ = retriever.retrieve("benefits", top_k=1)
        assert len(results) <= 1


def _sc(chunk_id: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=chunk_id,
            text=chunk_id,
            source_kind=SourceKind.POLICY,
            document_id="d",
        ),
        score=score,
    )


def _build(chunks: list[Chunk]) -> HybridRetriever:
    return HybridRetriever(
        dense=DenseIndex(chunks, FakeEmbedder()),
        lexical=BM25Index(chunks),
    )
