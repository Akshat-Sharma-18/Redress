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

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

from app.agents.decomposition import DecompositionAgent, DecomposedDenial
from app.agents.gate import GatedAdjudicator
from app.core.models import Confidence, ScoredChunk, SourceKind, Verdict
from app.graph.evidence import GraphEnricher
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


#: Confidence ordered by how much the system is willing to assert.
_CONFIDENCE_RANK = {
    Confidence.INSUFFICIENT: 0,
    Confidence.CONTESTED: 1,
    Confidence.SUPPORTED: 2,
}


@dataclass
class AuditResult:
    """Everything the frontend needs to render one case."""

    denial: DecomposedDenial
    results: list[SubClaimResult]

    @property
    def overall(self) -> str:
        """Case-level *direction*, ignoring how confident the system is.

        Any contradiction makes the case worth appealing, so 'contradicted'
        dominates; all-insufficient means the system declines to characterize
        the denial at all.

        This is the raw finding and is not what the user should be shown —
        see `disposition`, which folds in confidence.
        """
        findings = {r.verdict.finding for r in self.results}
        if "contradicted" in findings or "mixed" in findings:
            return "contradicted"
        if "justified" in findings:
            return "justified"
        return "insufficient"

    @property
    def confidence(self) -> Confidence:
        """Best confidence among the verdicts that drive `overall`.

        `max`, not `min`: one solidly-supported contradiction is enough to
        say the denial is contradicted, even if a second sub-claim came back
        contested. Taking the minimum would let an unrelated ambiguity
        suppress a finding the system can fully substantiate.
        """
        direction = self.overall
        if direction == "contradicted":
            driving = [
                r.verdict
                for r in self.results
                if r.verdict.finding in ("contradicted", "mixed")
            ]
        elif direction == "justified":
            driving = [
                r.verdict for r in self.results if r.verdict.finding == "justified"
            ]
        else:
            return Confidence.INSUFFICIENT

        return max(
            (v.confidence for v in driving),
            key=lambda c: _CONFIDENCE_RANK[c],
            default=Confidence.INSUFFICIENT,
        )

    @property
    def disposition(self) -> str:
        """What the user is actually told — finding and confidence together.

        Without this, the confidence gate's entire output is discarded at the
        case level: a verdict the ensemble flagged as contested would roll up
        indistinguishably from one that survived every check, and three
        phases of fail-safe machinery would have no effect on what anyone
        sees.

        'contested' and 'insufficient' are both non-assertions. They differ
        in whether the system has a leaning it cannot substantiate, which is
        worth telling the user, but neither is a claim about their denial.
        """
        direction = self.overall
        if direction == "insufficient":
            return "insufficient"
        confidence = self.confidence
        if confidence is Confidence.SUPPORTED:
            return direction
        if confidence is Confidence.CONTESTED:
            return "contested"
        return "insufficient"


class AuditPipeline:
    def __init__(
        self,
        decomposition: DecompositionAgent,
        adjudicator: GatedAdjudicator,
        retrievers: dict[str, HybridRetriever],
        secondary_retrievers: dict[str, HybridRetriever] | None = None,
        enricher: GraphEnricher | None = None,
        per_corpus_k: int = 6,
    ):
        self.decomposition = decomposition
        self.adjudicator = adjudicator
        self.retrievers = retrievers
        self.secondary_retrievers = secondary_retrievers
        self.enricher = enricher
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

    @staticmethod
    def _merge_derived(
        retrieved: list[ScoredChunk], derived: list[ScoredChunk]
    ) -> list[ScoredChunk]:
        """Append derived evidence after retrieved evidence of each kind.

        Derived chunks are appended rather than merged into the ranking:
        they were not scored by relevance, so sorting them against retrieved
        chunks would compare incomparable numbers.
        """
        if not derived:
            return retrieved
        seen = {sc.chunk.id for sc in retrieved}
        merged = retrieved + [sc for sc in derived if sc.chunk.id not in seen]
        merged.sort(key=lambda sc: _KIND_ORDER[sc.chunk.source_kind])
        return merged

    def run(
        self,
        denial_letter_text: str,
        *,
        insurer_id: str | None = None,
        statute_ids: list[str] | None = None,
        on_progress: Callable[[str, dict], None] | None = None,
    ) -> AuditResult:
        """Audit one denial letter.

        `insurer_id` and `statute_ids` are case metadata the letter itself
        does not reliably carry: the insurer's graph identity comes from the
        caller (letterhead is not an identifier), and the statutes to check
        for amendments come from the ingestion step that indexed them.

        `on_progress` observes stage transitions. A full audit is minutes of
        local inference, so a caller serving a human needs to say more than
        "working"; the alternative is an interface that is indistinguishable
        from a hang. It is an observer and nothing more — it cannot alter the
        audit, and an exception raised inside it is deliberately not caught,
        because a silently broken progress feed is worse than a loud one.
        """
        emit = on_progress or (lambda event, data: None)

        denial = self.decomposition.decompose(denial_letter_text)
        emit(
            "decomposed",
            {
                "sub_claims": len(denial.sub_claims),
                "denial_date": (
                    denial.denial_date.isoformat() if denial.denial_date else None
                ),
            },
        )

        # Graph and version evidence depends on the case, not the sub-claim,
        # so it is computed once and shared across every sub-claim.
        derived: list[ScoredChunk] = []
        if self.enricher is not None:
            derived = self.enricher.enrich(
                insurer_id=insurer_id,
                reason_code=denial.reason_code,
                denial_date=denial.denial_date,
                statute_ids=statute_ids,
            )

        results: list[SubClaimResult] = []
        total = len(denial.sub_claims)
        for index, sub_claim in enumerate(denial.sub_claims):
            emit(
                "sub_claim_started",
                {"index": index, "total": total, "text": sub_claim.text},
            )
            traces: dict[str, RetrievalTrace] = {}
            evidence = self._gather(
                self.retrievers, sub_claim.text, denial.denial_date, traces
            )
            evidence = self._merge_derived(evidence, derived)

            secondary = None
            if self.secondary_retrievers is not None:
                secondary_traces: dict[str, RetrievalTrace] = {}
                secondary = self._gather(
                    self.secondary_retrievers,
                    sub_claim.text,
                    denial.denial_date,
                    secondary_traces,
                )
                # The ensemble check compares retrieval, not enrichment: the
                # derived chunks are identical either way, so both passes see
                # them and any disagreement is attributable to the embedders.
                secondary = self._merge_derived(secondary, derived)
                traces.update(
                    {f"{k}#secondary": v for k, v in secondary_traces.items()}
                )

            verdict = self.adjudicator.adjudicate(
                sub_claim,
                evidence,
                # Narrowed re-retrieval goes against the primary corpora with
                # the critique agent's query, same denial-date filter.
                retrieve_fn=lambda q: self._merge_derived(
                    self._gather(self.retrievers, q, denial.denial_date), derived
                ),
                secondary_evidence=secondary,
            )
            results.append(SubClaimResult(verdict=verdict, traces=traces))
            emit(
                "sub_claim_finished",
                {
                    "index": index,
                    "total": total,
                    "finding": verdict.finding,
                    "confidence": verdict.confidence.value,
                },
            )

        result = AuditResult(denial=denial, results=results)
        emit("finished", {"disposition": result.disposition})
        return result
