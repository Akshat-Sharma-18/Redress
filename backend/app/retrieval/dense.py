"""Dense retrieval behind an embedder interface.

The interface exists because Section 5.3 of the design requires the *same*
sub-claim to be retrieved twice using two different embedding models, and to
only surface a flag when both passes agree. That ensemble check is only
meaningful if swapping the embedding model is a constructor argument rather
than a rewrite.

Embedding backends are imported lazily so the retrieval logic -- and its
tests -- run without pulling in torch.
"""

from __future__ import annotations

import math
from typing import Protocol, Sequence, runtime_checkable

from app.core.models import Chunk, ScoredChunk


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into vectors.

    `name` is recorded in retrieval provenance -- when the ensemble cross-check
    reports disagreement, the audit trail has to say which two models
    disagreed.
    """

    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class SentenceTransformerEmbedder:
    """Production embedder. Loads the model on first use, not on import."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load()
        # normalize_embeddings=True makes the dot product a cosine, which is
        # what any vector store we swap in later will assume.
        return model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        ).tolist()


class DenseIndex:
    """Brute-force cosine search over an embedded chunk collection.

    Exact search, no ANN structure. At this corpus size an exact scan is
    already sub-millisecond, and an approximate index would introduce recall
    loss that is indistinguishable from a retrieval bug when a citation goes
    missing. If the statute corpus grows past a few hundred thousand chunks
    this is the seam where Qdrant slots in -- `search` is the only method that
    would change.
    """

    def __init__(self, chunks: list[Chunk], embedder: Embedder):
        self.chunks = chunks
        self.embedder = embedder
        self._vectors = embedder.embed([c.text for c in chunks]) if chunks else []

    def search(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        if not self.chunks:
            return []

        q_vec = self.embedder.embed([query])[0]
        scored = [
            (i, cosine(q_vec, self._vectors[i])) for i in range(len(self.chunks))
        ]
        scored.sort(key=lambda kv: kv[1], reverse=True)

        # The embedder name is folded into the provenance *key* rather than
        # stored as a value, since provenance holds floats. This is what lets
        # the ensemble cross-check tell two dense passes apart downstream.
        key = f"dense:{self.embedder.name}"
        return [
            ScoredChunk(
                chunk=self.chunks[i],
                score=score,
                provenance={key: score},
            )
            for i, score in scored[:top_k]
        ]
