"""The full retrieval stack: dense + lexical -> RRF -> temporal filter -> rerank.

Stage ordering is deliberate.

Temporal filtering sits *between* fusion and reranking, not before retrieval.
Filtering first would mean rebuilding or re-querying both indexes per denial
date; filtering after reranking would mean paying cross-encoder cost on
chunks that are already disqualified. Between the two, the filter runs over a
few dozen candidates and the reranker only sees law that was actually in force.

`candidate_k` is intentionally much larger than `top_k`: fusion needs a deep
pool to find agreement in, and the reranker can only reorder what it is given.
Retrieving 40 to return 8 is the recall/precision split this stack is built on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from app.core.models import ScoredChunk
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import IdentityReranker, Reranker


@dataclass
class RetrievalTrace:
    """Everything needed to reproduce one retrieval, for the audit trail."""

    query: str
    as_of: date | None
    dense_hits: list[ScoredChunk] = field(default_factory=list)
    lexical_hits: list[ScoredChunk] = field(default_factory=list)
    fused: list[ScoredChunk] = field(default_factory=list)
    temporally_excluded: list[str] = field(default_factory=list)
    final: list[ScoredChunk] = field(default_factory=list)


class HybridRetriever:
    """Orchestrates one pipeline's retrieval over one corpus."""

    def __init__(
        self,
        dense: DenseIndex,
        lexical: BM25Index,
        reranker: Reranker | None = None,
        candidate_k: int = 40,
    ):
        self.dense = dense
        self.lexical = lexical
        self.reranker = reranker or IdentityReranker()
        self.candidate_k = candidate_k

    def retrieve(
        self, query: str, top_k: int = 8, as_of: date | None = None
    ) -> tuple[list[ScoredChunk], RetrievalTrace]:
        """Retrieve for `query`, restricted to law in force on `as_of`.

        Returns the results *and* the trace. The trace is not optional
        bookkeeping -- a verdict the user cannot inspect is a verdict they have
        no reason to trust, so the caller is given no way to discard it.
        """
        trace = RetrievalTrace(query=query, as_of=as_of)

        trace.dense_hits = self.dense.search(query, top_k=self.candidate_k)
        trace.lexical_hits = self.lexical.search(query, top_k=self.candidate_k)

        fused = reciprocal_rank_fusion(
            {"dense": trace.dense_hits, "bm25": trace.lexical_hits},
            top_k=self.candidate_k,
        )
        trace.fused = fused

        if as_of is not None:
            kept = []
            for sc in fused:
                if sc.chunk.applies_on(as_of):
                    kept.append(sc)
                else:
                    trace.temporally_excluded.append(sc.chunk.id)
            fused = kept

        trace.final = self.reranker.rerank(query, fused, top_k=top_k)
        return trace.final, trace
