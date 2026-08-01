"""End-to-end audit pipeline.

Wires the stages together: decompose the denial letter, retrieve evidence for
each sub-claim from every available corpus, adjudicate through the confidence
gate. The fan-out across pipelines A-D is a dict of named retrievers so Phase
4 can add the DOI-precedent corpus and the graph layer without changing this
module's shape.

Ensemble cross-check: when `secondary_retrievers` is provided (same corpora,
indexed under a different embedding model), every sub-claim is retrieved
twice and only surfaced as SUPPORTED when independent adjudications over the
two evidence sets agree.

Deliberately synchronous. The async fan-out belongs in the FastAPI layer once
there is an API; doing it here first would just make the tests worse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from app.agents.decomposition import DecompositionAgent, DecomposedDenial
from app.agents.gate import GatedAdjudicator
from app.core.models import ScoredChunk, SourceKind, Verdict
from app.retrieval.hybrid import HybridRetriever, RetrievalTrace

_KIND_ORDER = {
    SourceKind.POLICY: 0,
    SourceKind.DENIAL: 1,
    SourceKind.STATUTE: 2,
    SourceKind.PRECEDENT: 3,
}


@dataclass
class SubClaimResult:
    verdict: Verdict
    traces: dict[str, RetrievalTrace] = field(default_factory=dict)


@dataclass
class AuditResult:
    """Everything the frontend needs to render one case."""

    denial: DecomposedDenial
    results: list[SubClaimResult]

    @property
    def overall(self) -> str:
        """Coarse case-level summary derived from sub-claim findings.

        Any verified contradiction makes the case worth appealing, so
        'contradicted' dominates; all-insufficient means the system declines
        to characterize the denial at all.
        """
        findings = {r.verdict.finding for r in self.results}
        if "contradicted" in findings or "mixed" in findings:
            return "contradicted"
        if "justified" in findings:
            return "justified"
        return "insufficient"


class AuditPipeline:
    def __init__(
        self,
        decomposition: DecompositionAgent,
        adjudicator: GatedAdjudicator,
        retrievers: dict[str, HybridRetriever],
        secondary_retrievers: dict[str, HybridRetriever] | None = None,
        per_corpus_k: int = 6,
    ):
        self.decomposition = decomposition
        self.adjudicator = adjudicator
        self.retrievers = retrievers
        self.secondary_retrievers = secondary_retrievers
        self.per_corpus_k = per_corpus_k

    def _gather(
        self,
        retrievers: dict[str, HybridRetriever],
        query: str,
        as_of: date | None,
        traces: dict[str, RetrievalTrace] | None = None,
    ) -> list[ScoredChunk]:
        """Fan out one query across a retriever set; stable evidence order."""
        evidence: list[ScoredChunk] = []
        for name, retriever in retrievers.items():
            hits, trace = retriever.retrieve(
                query, top_k=self.per_corpus_k, as_of=as_of
            )
            evidence.extend(hits)
            if traces is not None:
                traces[name] = trace
        evidence.sort(key=lambda sc: (_KIND_ORDER[sc.chunk.source_kind], -sc.score))
        return evidence

    def run(self, denial_letter_text: str) -> AuditResult:
        denial = self.decomposition.decompose(denial_letter_text)

        results: list[SubClaimResult] = []
        for sub_claim in denial.sub_claims:
            traces: dict[str, RetrievalTrace] = {}
            evidence = self._gather(
                self.retrievers, sub_claim.text, denial.denial_date, traces
            )

            secondary = None
            if self.secondary_retrievers is not None:
                secondary_traces: dict[str, RetrievalTrace] = {}
                secondary = self._gather(
                    self.secondary_retrievers,
                    sub_claim.text,
                    denial.denial_date,
                    secondary_traces,
                )
                traces.update(
                    {f"{k}#secondary": v for k, v in secondary_traces.items()}
                )

            verdict = self.adjudicator.adjudicate(
                sub_claim,
                evidence,
                # Narrowed re-retrieval goes against the primary corpora with
                # the critique agent's query, same denial-date filter.
                retrieve_fn=lambda q: self._gather(
                    self.retrievers, q, denial.denial_date
                ),
                secondary_evidence=secondary,
            )
            results.append(SubClaimResult(verdict=verdict, traces=traces))

        return AuditResult(denial=denial, results=results)
