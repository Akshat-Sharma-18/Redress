"""Runs the audit pipeline over the golden set and scores the results.

Each case carries its own chunks, so the harness builds a fresh index per
case rather than sharing one corpus. That is slower and correct: a shared
index would let evidence from case B satisfy case A, which is the eval
equivalent of training on the test set.

Ablation is a first-class argument. Reporting "reranking improved context
precision by X" requires actually running the pipeline without the reranker,
and `AblationConfig` is what makes that a parameter instead of a code edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.agents.critique import CritiqueAgent, CritiqueResult
from app.agents.decomposition import DecompositionAgent
from app.agents.gate import GatedAdjudicator
from app.agents.llm import StructuredLLM
from app.agents.pipeline import AuditPipeline
from app.agents.reconciliation import ReconciliationAgent
from app.core.models import Chunk, SourceKind
from app.eval.dataset import GoldenCase
from app.eval.metrics import ERROR, CaseEvaluation, Outcome, Report, classify
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex, Embedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import IdentityReranker, Reranker


@dataclass
class AblationConfig:
    """Which components are enabled for this run.

    Every flag defaults to the full system; a run with defaults is the
    headline number, and each disabled flag produces the comparison row that
    justifies that component's existence.
    """

    use_reranker: bool = True
    use_critique: bool = True
    use_ensemble: bool = True
    #: Hold `justified` findings to a higher standard than `contradicted`.
    #: Its ablation arm is the one that prices the trade this system is built
    #: around: how many correct assurances it gives up per false one avoided.
    use_asymmetric_assurance: bool = True
    name: str = "full"


class NullCritique(CritiqueAgent):
    """Approves everything — the ablation arm without adversarial review.

    Subclasses rather than short-circuits the gate so the ablation measures
    only the critique pass's judgement, holding the rest of the flow fixed.
    """

    def __init__(self) -> None:  # noqa: D107 - no LLM needed
        pass

    def review(self, sub_claim, verdict, evidence):
        return CritiqueResult(approved=True)


def _to_chunk(case_chunk, document_id: str) -> Chunk:
    return Chunk(
        id=case_chunk.id,
        text=case_chunk.text,
        source_kind=SourceKind(case_chunk.source_kind),
        document_id=document_id,
        locator=case_chunk.locator,
        effective_from=case_chunk.effective_from,
        effective_to=case_chunk.effective_to,
    )


class EvalHarness:
    def __init__(
        self,
        llm_factory: Callable[[], StructuredLLM],
        embedder_factory: Callable[[], Embedder],
        reranker_factory: Callable[[], Reranker] | None = None,
        secondary_embedder_factory: Callable[[], Embedder] | None = None,
    ):
        self.llm_factory = llm_factory
        self.embedder_factory = embedder_factory
        self.reranker_factory = reranker_factory
        self.secondary_embedder_factory = secondary_embedder_factory

    def _build_pipeline(
        self, case: GoldenCase, config: AblationConfig
    ) -> AuditPipeline:
        chunks = [_to_chunk(c, case.id) for c in case.chunks]
        llm = self.llm_factory()

        reranker = (
            self.reranker_factory()
            if config.use_reranker and self.reranker_factory
            else IdentityReranker()
        )
        retrievers = {
            "case": HybridRetriever(
                dense=DenseIndex(chunks, self.embedder_factory()),
                lexical=BM25Index(chunks),
                reranker=reranker,
            )
        }

        secondary = None
        if config.use_ensemble and self.secondary_embedder_factory:
            secondary = {
                "case": HybridRetriever(
                    dense=DenseIndex(chunks, self.secondary_embedder_factory()),
                    lexical=BM25Index(chunks),
                    reranker=reranker,
                )
            }

        critique = CritiqueAgent(llm) if config.use_critique else NullCritique()

        return AuditPipeline(
            decomposition=DecompositionAgent(llm),
            adjudicator=GatedAdjudicator(
                ReconciliationAgent(llm),
                critique,
                asymmetric_assurance=config.use_asymmetric_assurance,
            ),
            retrievers=retrievers,
            secondary_retrievers=secondary,
        )

    def run_case(
        self, case: GoldenCase, config: AblationConfig | None = None
    ) -> CaseEvaluation:
        config = config or AblationConfig()
        try:
            pipeline = self._build_pipeline(case, config)
            result = pipeline.run(
                case.denial_letter,
                insurer_id=case.insurer_id,
                statute_ids=case.statute_ids,
            )
        except Exception as exc:  # noqa: BLE001 - an exception is a result
            # A crash is recorded as its own outcome, not dropped and not
            # folded into a direction class. Dropping it would shrink the
            # denominator and inflate every rate; misfiling it as a semantic
            # error would hide a bug behind a plausible-looking metric.
            return CaseEvaluation(
                case_id=case.id,
                category=case.category,
                expected=case.expected.overall,
                predicted=ERROR,
                outcome=classify(
                    expected=case.expected.overall,
                    predicted=ERROR,
                    category=case.category,
                ),
                error=f"{type(exc).__name__}: {exc}",
            )

        cited = {
            citation.chunk_id
            for sub in result.results
            for citation in sub.verdict.citations
        }
        required = set(case.expected.must_cite)
        forbidden = set(case.expected.must_not_cite)

        # Scored against `disposition`, not `overall`: `overall` is the raw
        # direction and discards the confidence gate's output, so a contested
        # verdict would score identically to one that survived every check —
        # and the eval would be blind to the system's headline capability.
        predicted = result.disposition
        return CaseEvaluation(
            case_id=case.id,
            category=case.category,
            expected=case.expected.overall,
            predicted=predicted,
            outcome=classify(
                expected=case.expected.overall,
                predicted=predicted,
                category=case.category,
            ),
            cited=cited,
            missing_required_citations=required - cited,
            forbidden_citations=forbidden & cited,
            # The pipeline takes the denial date from the letter, which is
            # the behaviour under test. The golden metadata is the answer
            # key: without this cross-check, a failed extraction silently
            # disables temporal filtering and the case can still pass.
            expected_denial_date=case.denial_date,
            extracted_denial_date=result.denial.denial_date,
        )

    def run(
        self, cases: list[GoldenCase], config: AblationConfig | None = None
    ) -> Report:
        return Report([self.run_case(c, config) for c in cases])

    def ablate(
        self, cases: list[GoldenCase], configs: list[AblationConfig]
    ) -> dict[str, Report]:
        return {c.name: self.run(cases, c) for c in configs}


def ablation_table(reports: dict[str, Report]) -> str:
    """Comparison table across ablation arms.

    Deliberately leads with false-assurance count rather than accuracy: an
    ablation that trades a point of accuracy for a false assurance is not an
    improvement, and a table sorted by accuracy would present it as one.
    """
    header = (
        f"{'arm':<16} {'false assur':>12} {'correct abst':>13} "
        f"{'over-abst':>10} {'accuracy':>9} {'grounding':>10}"
    )
    lines = [header, "-" * len(header)]

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1%}"

    for name, report in reports.items():
        lines.append(
            f"{name:<16} "
            f"{report.counts[Outcome.FALSE_ASSURANCE]:>12} "
            f"{pct(report.correct_abstention_rate):>13} "
            f"{pct(report.over_abstention_rate):>10} "
            f"{pct(report.accuracy):>9} "
            f"{pct(report.grounding_accuracy):>10}"
        )
    return "\n".join(lines)
