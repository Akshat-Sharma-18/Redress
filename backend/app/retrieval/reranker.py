"""Cross-encoder reranking of fused candidates.

Bi-encoders embed the query and the chunk separately, so they can never model
the interaction between them -- which is precisely what matters in policy
text. "This exclusion does not apply to emergency services" and "This
exclusion applies to emergency services" sit almost on top of each other in
embedding space; a cross-encoder reads them jointly with the query and can
tell them apart. Adjacent policy clauses are the single largest source of
retrieval error in this domain, and this stage is the fix.

The cost is that a cross-encoder scores one pair at a time, so it cannot scan
a corpus -- it only reorders the ~40 candidates fusion already surfaced.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from app.core.models import ScoredChunk


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        ...


class CrossEncoderReranker:
    """bge-reranker class model. Loaded lazily, like the embedder."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.name)
        return self._model

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int = 8
    ) -> list[ScoredChunk]:
        if not candidates:
            return []

        model = self._load()
        pairs: Sequence[tuple[str, str]] = [
            (query, c.chunk.text) for c in candidates
        ]
        scores = model.predict(pairs)

        rescored = [
            c.with_score("rerank", float(s)) for c, s in zip(candidates, scores)
        ]
        rescored.sort(key=lambda c: c.score, reverse=True)
        return rescored[:top_k]


class IdentityReranker:
    """No-op reranker preserving fusion order.

    Used in tests and in the ablation arm of the eval harness -- reporting
    "reranking improved context precision by X" requires actually measuring
    the pipeline without it.
    """

    name = "identity"

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int = 8
    ) -> list[ScoredChunk]:
        return candidates[:top_k]
