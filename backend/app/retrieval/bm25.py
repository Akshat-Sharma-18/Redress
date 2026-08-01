"""Lexical retrieval over policy and statute text.

Implemented in-process rather than against Elasticsearch: the corpora here are
one policy document plus one state's insurance code, which is thousands of
chunks, not millions. Running a search cluster for that is operational weight
with no retrieval benefit, and it keeps the whole system reproducible from a
`pip install`.

Lexical matching is not optional in this domain. Insurance policies define
terms of art -- "Custodial Care", "Medically Necessary", "Adverse Benefit
Determination" -- whose defined meaning is unrelated to their plain-English
embedding neighbourhood. A dense retriever asked about custodial care will
happily return the nursing-home benefits section; only exact-term matching
reliably finds the definitions clause that governs it.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from app.core.models import Chunk, ScoredChunk

# Split on non-word characters, keeping intra-word hyphens and periods out of
# the way. Statute references like "10123.13" tokenize to "10123" + "13", which
# is fine -- both halves are distinctive enough to match on.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Okapi BM25 over a fixed chunk collection.

    Defaults k1=1.5, b=0.75 are the standard Robertson/Sparck-Jones values.
    b=0.75 matters here because policy chunks vary a lot in length -- a
    two-line definition sits beside a page-long exclusions list, and without
    length normalisation the long chunk wins on term count alone.
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks = chunks
        self._doc_tokens = [tokenize(c.text) for c in chunks]
        self._doc_lens = [len(t) for t in self._doc_tokens]
        self._avg_len = (sum(self._doc_lens) / len(self._doc_lens)) if chunks else 0.0
        self._tf: list[Counter[str]] = [Counter(t) for t in self._doc_tokens]

        df: Counter[str] = Counter()
        for tokens in self._doc_tokens:
            df.update(set(tokens))
        self._df = df
        self._n = len(chunks)

    def _idf(self, term: str) -> float:
        """Robertson-Sparck-Jones IDF with the +0.5 smoothing.

        Clamped at zero: without the clamp, a term appearing in more than half
        the corpus gets a negative weight, which would let a document score
        *lower* for containing a query term. In a policy document, boilerplate
        like "coverage" or "benefit" is exactly that common.
        """
        n_q = self._df.get(term, 0)
        if n_q == 0:
            return 0.0
        return max(0.0, math.log(1 + (self._n - n_q + 0.5) / (n_q + 0.5)))

    def search(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        if self._n == 0:
            return []

        q_terms = tokenize(query)
        scores: list[float] = [0.0] * self._n

        for term in set(q_terms):
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for i in range(self._n):
                freq = self._tf[i].get(term, 0)
                if freq == 0:
                    continue
                norm = 1 - self.b + self.b * (self._doc_lens[i] / self._avg_len)
                scores[i] += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * norm)

        ranked = sorted(
            (i for i in range(self._n) if scores[i] > 0),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [
            ScoredChunk(
                chunk=self.chunks[i],
                score=scores[i],
                provenance={"bm25": scores[i]},
            )
            for i in ranked
        ]
