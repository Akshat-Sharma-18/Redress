"""End-to-end audit pipeline, Phase 2 shape.

Wires the stages together: decompose the denial letter, retrieve evidence for
each sub-claim from every available corpus, adjudicate. The fan-out across
pipelines A-D is represented as a dict of named retrievers so Phase 4 can add
the DOI-precedent corpus and the graph layer without changing this module's
shape.

Deliberately synchronous for now. The async fan-out belongs in the FastAPI
layer once there is an API; doing it here first would just make the tests
worse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.agents.decomposition import DecompositionAgent, DecomposedDenial
from app.agents.reconciliation import ReconciliationAgent
from app.core.models import ScoredChunk, SourceKind, Verdict
from app.retrieval.hybrid import HybridRetriever, RetrievalTrace


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
        reconciliation: ReconciliationAgent,
        retrievers: dict[str, HybridRetriever],
        per_corpus_k: int = 6,
    ):
        self.decomposition = decomposition
        self.reconciliation = reconciliation
        self.retrievers = retrievers
        self.per_corpus_k = per_corpus_k

    def run(self, denial_letter_text: str) -> AuditResult:
        denial = self.decomposition.decompose(denial_letter_text)

        results: list[SubClaimResult] = []
        for sub_claim in denial.sub_claims:
            evidence: list[ScoredChunk] = []
            traces: dict[str, RetrievalTrace] = {}

            for name, retriever in self.retrievers.items():
                hits, trace = retriever.retrieve(
                    sub_claim.text,
                    top_k=self.per_corpus_k,
                    # Statutes are checked against the law as it stood on the
                    # denial date; chunks without temporal bounds pass through.
                    as_of=denial.denial_date,
                )
                evidence.extend(hits)
                traces[name] = trace

            # Stable ordering for the prompt: strongest evidence first within
            # each source kind, policy text before statutes before precedent.
            kind_order = {
                SourceKind.POLICY: 0,
                SourceKind.DENIAL: 1,
                SourceKind.STATUTE: 2,
                SourceKind.PRECEDENT: 3,
            }
            evidence.sort(
                key=lambda sc: (kind_order[sc.chunk.source_kind], -sc.score)
            )

            verdict = self.reconciliation.adjudicate(sub_claim, evidence)
            results.append(SubClaimResult(verdict=verdict, traces=traces))

        return AuditResult(denial=denial, results=results)
