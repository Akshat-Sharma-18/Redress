"""Agent-layer tests.

No API calls: a scripted FakeLLM returns pre-built schema instances. What is
under test is everything the LLM is *not* trusted to do — citation
verification, downgrade behavior, date handling, and pipeline wiring.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from app.agents.decomposition import DecompositionAgent
from app.agents.pipeline import AuditPipeline
from app.agents.reconciliation import ReconciliationAgent
from app.agents.schemas import (
    DecomposedClaim,
    Decomposition,
    DraftCitation,
    DraftVerdict,
    ExtractedDenial,
)
from app.core.models import Chunk, Confidence, ScoredChunk, SourceKind, SubClaim
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex
from app.retrieval.hybrid import HybridRetriever

from .conftest import FakeEmbedder


class FakeLLM:
    """Returns queued responses in order; records every call it saw."""

    def __init__(self, responses: list[BaseModel]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, *, system, prompt, schema, max_tokens=16000):
        self.calls.append({"system": system, "prompt": prompt, "schema": schema})
        response = self._responses.pop(0)
        assert isinstance(response, schema)
        return response


def _chunk(cid: str, text: str, kind=SourceKind.POLICY) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(id=cid, text=text, source_kind=kind, document_id="d"),
        score=1.0,
    )


def _claim(text="The exclusion applies.") -> SubClaim:
    return SubClaim(id="sc-1", text=text, kind="legal")


EXCLUSION_TEXT = (
    "This plan does not provide benefits for services rendered in a "
    "hospital emergency department."
)


class TestReconciliationVerification:
    def test_verbatim_citation_passes(self):
        llm = FakeLLM([
            DraftVerdict(
                finding="justified",
                rationale="The exclusion covers this service.",
                citations=[
                    DraftCitation(
                        chunk_id="p2",
                        quote="does not provide benefits for services",
                        supports="Establishes the exclusion.",
                    )
                ],
            )
        ])
        agent = ReconciliationAgent(llm)
        verdict = agent.adjudicate(_claim(), [_chunk("p2", EXCLUSION_TEXT)])

        assert verdict.finding == "justified"
        assert verdict.confidence == Confidence.SUPPORTED
        assert verdict.citations[0].quote in EXCLUSION_TEXT

    def test_paraphrased_quote_downgrades_to_insufficient(self):
        """The core fail-safe: a citation that isn't verbatim kills the verdict.

        The draft claims 'contradicted' — the finding a desperate policyholder
        most wants to hear — but its quote paraphrases the source. The system
        must refuse to assert it.
        """
        llm = FakeLLM([
            DraftVerdict(
                finding="contradicted",
                rationale="The policy does not exclude this.",
                citations=[
                    DraftCitation(
                        chunk_id="p2",
                        quote="benefits are not provided for ER services",  # paraphrase
                        supports="Shows the exclusion is narrower.",
                    )
                ],
            )
        ])
        verdict = ReconciliationAgent(llm).adjudicate(
            _claim(), [_chunk("p2", EXCLUSION_TEXT)]
        )

        assert verdict.finding == "insufficient"
        assert verdict.confidence == Confidence.INSUFFICIENT
        assert "not a verbatim substring" in verdict.critique_notes
        # The draft is preserved for the audit trail, not erased
        assert verdict.draft_rationale == "The policy does not exclude this."

    def test_unknown_chunk_id_downgrades(self):
        """A citation to evidence that was never retrieved is fabrication."""
        llm = FakeLLM([
            DraftVerdict(
                finding="contradicted",
                rationale="Statute 1371.4 requires reimbursement.",
                citations=[
                    DraftCitation(
                        chunk_id="hallucinated-statute",
                        quote="shall reimburse",
                        supports="Requires payment.",
                    )
                ],
            )
        ])
        verdict = ReconciliationAgent(llm).adjudicate(
            _claim(), [_chunk("p2", EXCLUSION_TEXT)]
        )
        assert verdict.finding == "insufficient"
        assert "unknown chunk" in verdict.critique_notes

    def test_uncited_ruling_downgrades(self):
        """A finding with zero citations is an opinion, not a verdict."""
        llm = FakeLLM([
            DraftVerdict(finding="justified", rationale="Obviously excluded.", citations=[])
        ])
        verdict = ReconciliationAgent(llm).adjudicate(
            _claim(), [_chunk("p2", EXCLUSION_TEXT)]
        )
        assert verdict.finding == "insufficient"
        assert "no surviving citations" in verdict.critique_notes

    def test_insufficient_needs_no_citations(self):
        llm = FakeLLM([
            DraftVerdict(
                finding="insufficient",
                rationale="The evidence does not address this sub-claim.",
                citations=[],
            )
        ])
        verdict = ReconciliationAgent(llm).adjudicate(
            _claim(), [_chunk("p2", EXCLUSION_TEXT)]
        )
        assert verdict.finding == "insufficient"
        assert verdict.critique_notes is None  # clean abstention, not a downgrade

    def test_empty_evidence_short_circuits_without_llm_call(self):
        llm = FakeLLM([])
        verdict = ReconciliationAgent(llm).adjudicate(_claim(), [])
        assert verdict.finding == "insufficient"
        assert llm.calls == []

    def test_invalid_finding_value_downgrades(self):
        llm = FakeLLM([
            DraftVerdict(finding="probably_fine", rationale="...", citations=[])
        ])
        verdict = ReconciliationAgent(llm).adjudicate(
            _claim(), [_chunk("p2", EXCLUSION_TEXT)]
        )
        assert verdict.finding == "insufficient"
        assert "invalid finding" in verdict.critique_notes

    def test_partial_citation_failure_keeps_verified_ones(self):
        """One bad citation shouldn't erase a good one from the record."""
        llm = FakeLLM([
            DraftVerdict(
                finding="justified",
                rationale="Excluded.",
                citations=[
                    DraftCitation(
                        chunk_id="p2",
                        quote="hospital emergency department",
                        supports="Names the setting.",
                    ),
                    DraftCitation(
                        chunk_id="p2", quote="not in the text", supports="..."
                    ),
                ],
            )
        ])
        verdict = ReconciliationAgent(llm).adjudicate(
            _claim(), [_chunk("p2", EXCLUSION_TEXT)]
        )
        # Downgraded because of the bad citation, but the good one survives
        assert verdict.finding == "insufficient"
        assert len(verdict.citations) == 1
        assert verdict.citations[0].quote == "hospital emergency department"


class TestDecomposition:
    def _decomposition(self, denial_date):
        return Decomposition(
            denial=ExtractedDenial(
                denial_reason="Not medically necessary",
                reason_code="CO-50",
                denial_date=denial_date,
                cited_policy_sections=["Section 7.2(a)"],
                factual_assertions=["Date of service: 2021-05-14"],
            ),
            sub_claims=[
                DecomposedClaim(text="The service was not medically necessary.", kind="factual"),
                DecomposedClaim(
                    text="The emergency services exclusion applies.",
                    kind="legal",
                    cited_by_insurer="Section 7.2(a)",
                ),
            ],
        )

    def test_maps_to_domain_objects(self):
        llm = FakeLLM([self._decomposition("2021-06-01")])
        result = DecompositionAgent(llm).decompose("...letter text...")

        assert result.denial_date == date(2021, 6, 1)
        assert result.reason_code == "CO-50"
        assert len(result.sub_claims) == 2
        assert result.sub_claims[1].cited_by_insurer == "Section 7.2(a)"
        assert {c.kind for c in result.sub_claims} == {"factual", "legal"}

    def test_malformed_date_becomes_none_not_a_guess(self):
        """A wrong denial date silently filters against the wrong law.

        None disables temporal filtering entirely, which is the safe failure.
        """
        llm = FakeLLM([self._decomposition("June 1st, 2021")])
        result = DecompositionAgent(llm).decompose("...")
        assert result.denial_date is None

    def test_unrecognized_kind_defaults_to_legal(self):
        decomp = self._decomposition("2021-06-01")
        decomp.sub_claims[0].kind = "administrative"
        llm = FakeLLM([decomp])
        result = DecompositionAgent(llm).decompose("...")
        assert result.sub_claims[0].kind == "legal"


class TestPipeline:
    def test_end_to_end_with_temporal_filtering(self, policy_chunks, statute_chunks):
        """Full run: decompose → retrieve (with denial-date filter) → adjudicate.

        The 2021 denial date must exclude the 2023 statute version from the
        evidence that reaches the reconciliation agent.
        """
        decomposition_response = Decomposition(
            denial=ExtractedDenial(
                denial_reason="Emergency services excluded",
                denial_date="2021-06-01",
            ),
            sub_claims=[
                DecomposedClaim(
                    text="The emergency services exclusion applies.",
                    kind="legal",
                    cited_by_insurer="Section 7.2(a)",
                )
            ],
        )
        verdict_response = DraftVerdict(
            finding="contradicted",
            rationale="The carve-back defeats the exclusion.",
            citations=[
                DraftCitation(
                    chunk_id="p3",
                    quote="does not apply to emergency services",
                    supports="Carves emergency care back into coverage.",
                )
            ],
        )
        llm = FakeLLM([decomposition_response, verdict_response])

        def build(chunks):
            return HybridRetriever(
                dense=DenseIndex(chunks, FakeEmbedder()),
                lexical=BM25Index(chunks),
            )

        pipeline = AuditPipeline(
            decomposition=DecompositionAgent(llm),
            reconciliation=ReconciliationAgent(llm),
            retrievers={
                "policy": build(policy_chunks),
                "statutes": build(statute_chunks),
            },
        )

        result = pipeline.run("...denial letter...")

        assert result.overall == "contradicted"
        [sub_result] = result.results
        assert sub_result.verdict.confidence == Confidence.SUPPORTED

        # Temporal filtering happened: the 2023 statute never reached the agent
        evidence_ids = {sc.chunk.id for sc in sub_result.verdict.retrieval_trace}
        assert "s1-v2" not in evidence_ids
        assert "s1-v2" in sub_result.traces["statutes"].temporally_excluded

        # And the reconciliation prompt carried the evidence, tagged by source
        reconciliation_prompt = llm.calls[1]["prompt"]
        assert "<evidence>" in reconciliation_prompt
        assert 'source="policy"' in reconciliation_prompt

    def test_all_insufficient_yields_insufficient_overall(self, policy_chunks):
        llm = FakeLLM([
            Decomposition(
                denial=ExtractedDenial(denial_reason="Denied"),
                sub_claims=[DecomposedClaim(text="Claim.", kind="legal")],
            ),
            DraftVerdict(finding="insufficient", rationale="Not settled.", citations=[]),
        ])
        pipeline = AuditPipeline(
            decomposition=DecompositionAgent(llm),
            reconciliation=ReconciliationAgent(llm),
            retrievers={
                "policy": HybridRetriever(
                    dense=DenseIndex(policy_chunks, FakeEmbedder()),
                    lexical=BM25Index(policy_chunks),
                )
            },
        )
        result = pipeline.run("...")
        assert result.overall == "insufficient"
