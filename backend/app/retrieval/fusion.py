"""Reciprocal rank fusion across heterogeneous retrievers.

The reason to fuse on *rank* rather than score: BM25 returns unbounded
positive scores, cosine similarity returns [-1, 1]. Those are not comparable,
and normalising them into a shared range means inventing a mapping that has no
principled basis and quietly shifts whenever the corpus changes. RRF sidesteps
the problem -- it only asks where each retriever put the chunk, not how
confident it claimed to be.
"""

from __future__ import annotations

from app.core.models import ScoredChunk

# Cormack et al. (2009) use k=60. The constant damps the influence of top
# ranks: without it, a chunk at rank 1 in one retriever would score 1.0 and
# dominate anything a second retriever ranked 2nd or below. k=60 makes the
# curve flat enough that agreement between retrievers beats a single
# retriever's strong opinion -- which is what we want, since the whole premise
# is that dense and lexical retrieval catch different things.
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: dict[str, list[ScoredChunk]],
    k: int = DEFAULT_RRF_K,
    top_k: int = 20,
) -> list[ScoredChunk]:
    """Fuse several ranked lists into one.

    `rankings` maps a retriever name ("dense", "bm25") to its ranked output.
    Per-retriever scores are preserved in each result's provenance so the audit
    trail can show why a chunk surfaced -- lexical hit, semantic hit, or both.
    """
    fused: dict[str, float] = {}
    best: dict[str, ScoredChunk] = {}
    contributions: dict[str, dict[str, float]] = {}

    for retriever, results in rankings.items():
        for rank, scored in enumerate(results, start=1):
            cid = scored.chunk.id
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
            contributions.setdefault(cid, {})[retriever] = scored.score
            contributions[cid][f"{retriever}_rank"] = float(rank)
            # Keep one canonical copy of the chunk itself.
            best.setdefault(cid, scored)

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    return [
        ScoredChunk(
            chunk=best[cid].chunk,
            score=score,
            provenance={**contributions[cid], "rrf": score},
        )
        for cid, score in ordered
    ]
