"""Agent-layer tests.

No API calls: a scripted FakeLLM returns pre-built schema instances. What is
under test is everything the LLM is *not* trusted to do — citation
verification, downgrade behavior, date handling, and pipeline wiring.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.agents.critique import CritiqueAgent, CritiqueResult
from app.agents.ollama_llm import (
    DEFAULT_REASONING_EFFORT,
    OllamaError,
    OllamaStructuredLLM,
)
from app.agents.decomposition import DecompositionAgent, DecomposedDenial
from app.agents.gate import GatedAdjudicator
from app.agents.pipeline import AuditPipeline
from app.agents.reconciliation import ReconciliationAgent, find_verbatim_span
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

from .conftest import FakeEmbedder, FakeLLM


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


class TestVerbatimSpanMatching:
    """Whitespace is the only difference a citation check may forgive."""

    def test_line_wrapped_source_still_matches(self):
        """Documents wrap lines; a model quoting a wrapped sentence returns
        it unwrapped. Rejecting that abstains on a faithful citation for a
        typesetting reason — observed with a real local model."""
        source = "the exclusion in Section 7.2(a) does not\napply to emergency services"
        span = find_verbatim_span(
            "the exclusion in Section 7.2(a) does not apply to emergency services",
            source,
        )
        assert span == source  # the SOURCE's text, not the model's rendering

    def test_returns_source_span_so_stored_quote_is_byte_exact(self):
        """The frontend highlights the quote in the document; storing the
        model's whitespace would make it unfindable on the page."""
        source = "benefits for\n  services rendered"
        assert find_verbatim_span("benefits for services rendered", source) == source

    def test_paraphrase_still_rejected(self):
        source = "This plan does not provide benefits for emergency services."
        assert find_verbatim_span("benefits are not provided for ER services", source) is None

    def test_case_and_punctuation_still_strict(self):
        source = "Custodial Care means assistance with daily living."
        assert find_verbatim_span("custodial care means assistance", source) is None
        assert find_verbatim_span("Custodial Care means assistance", source) is not None

    def test_empty_quote_rejected(self):
        """An empty string is a substring of everything, so without a guard
        a contentless citation verifies and licenses an arbitrary claim."""
        for empty in ("", "   ", "\n\t"):
            assert find_verbatim_span(empty, "any text at all") is None

    def test_empty_quote_downgrades_a_verdict(self):
        llm = FakeLLM([
            DraftVerdict(
                finding="justified",
                rationale="Excluded.",
                citations=[DraftCitation(chunk_id="p2", quote="  ", supports="...")],
            )
        ])
        verdict = ReconciliationAgent(llm).adjudicate(
            _claim(), [_chunk("p2", EXCLUSION_TEXT)]
        )
        assert verdict.finding == "insufficient"


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

    def test_parses_dates_as_written_in_letters(self):
        """Letters say 'June 14, 2021', not '2021-06-14'.

        Asking the model to convert is a mechanical transformation it can get
        wrong, and a wrong date silently filters against the wrong law. Code
        does the conversion; the model only copies.
        """
        from app.agents.decomposition import parse_denial_date

        for raw, expected in [
            ("June 14, 2021", date(2021, 6, 14)),
            ("Jun 14, 2021", date(2021, 6, 14)),
            ("14 June 2021", date(2021, 6, 14)),
            ("2021-06-14", date(2021, 6, 14)),
            ("06/14/2021", date(2021, 6, 14)),
            ("  June 14, 2021.  ", date(2021, 6, 14)),
        ]:
            assert parse_denial_date(raw) == expected, raw

    def test_unparseable_date_is_none_not_a_guess(self):
        from app.agents.decomposition import parse_denial_date

        for raw in (None, "", "   ", "sometime last spring", "13/45/2021"):
            assert parse_denial_date(raw) is None

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


class TestCaseRollup:
    """Case-level rollup must not discard the confidence gate's output."""

    @staticmethod
    def _result(*verdicts):
        from app.agents.pipeline import AuditResult, SubClaimResult

        return AuditResult(
            denial=DecomposedDenial([], "r", None, None, [], []),
            results=[SubClaimResult(verdict=v) for v in verdicts],
        )

    @staticmethod
    def _verdict(finding, confidence):
        from app.core.models import Verdict

        return Verdict(
            sub_claim_id="sc",
            finding=finding,
            confidence=confidence,
            rationale="...",
        )

    def test_contested_does_not_read_as_a_confident_finding(self):
        """The defect this exists to prevent: three phases of fail-safe
        machinery having no effect on what the user is shown."""
        contested = self._result(
            self._verdict("contradicted", Confidence.CONTESTED)
        )
        supported = self._result(
            self._verdict("contradicted", Confidence.SUPPORTED)
        )

        # Direction is the same...
        assert contested.overall == supported.overall == "contradicted"
        # ...but what the user is told is not.
        assert contested.disposition == "contested"
        assert supported.disposition == "contradicted"

    def test_one_supported_contradiction_carries_the_case(self):
        """max, not min: an unrelated ambiguity must not suppress a finding
        the system can fully substantiate."""
        result = self._result(
            self._verdict("contradicted", Confidence.SUPPORTED),
            self._verdict("contradicted", Confidence.CONTESTED),
        )
        assert result.disposition == "contradicted"

    def test_confidence_ignores_non_driving_verdicts(self):
        """An insufficient sub-claim shouldn't drag down a supported one."""
        result = self._result(
            self._verdict("justified", Confidence.SUPPORTED),
            self._verdict("insufficient", Confidence.INSUFFICIENT),
        )
        assert result.overall == "justified"
        assert result.disposition == "justified"

    def test_all_insufficient_disposition(self):
        result = self._result(
            self._verdict("insufficient", Confidence.INSUFFICIENT)
        )
        assert result.disposition == "insufficient"
        assert result.confidence is Confidence.INSUFFICIENT


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
        llm = FakeLLM([
            decomposition_response,
            verdict_response,
            CritiqueResult(approved=True),
        ])

        def build(chunks):
            return HybridRetriever(
                dense=DenseIndex(chunks, FakeEmbedder()),
                lexical=BM25Index(chunks),
            )

        pipeline = AuditPipeline(
            decomposition=DecompositionAgent(llm),
            adjudicator=GatedAdjudicator(
                ReconciliationAgent(llm), CritiqueAgent(llm)
            ),
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

    def test_secondary_retrievers_drive_ensemble(self, policy_chunks):
        """With secondary retrievers, a second adjudication runs and its
        trace is recorded under a '#secondary' key."""
        llm = FakeLLM([
            Decomposition(
                denial=ExtractedDenial(denial_reason="Denied"),
                sub_claims=[DecomposedClaim(text="Exclusion applies.", kind="legal")],
            ),
            DraftVerdict(
                finding="justified",
                rationale="Excluded.",
                citations=[
                    DraftCitation(
                        chunk_id="p2",
                        quote="does not provide benefits",
                        supports="The exclusion.",
                    )
                ],
            ),
            CritiqueResult(approved=True),
            DraftVerdict(  # secondary pass agrees
                finding="justified",
                rationale="Excluded.",
                citations=[
                    DraftCitation(
                        chunk_id="p2",
                        quote="does not provide benefits",
                        supports="The exclusion.",
                    )
                ],
            ),
        ])

        def build(salt=""):
            return HybridRetriever(
                dense=DenseIndex(policy_chunks, FakeEmbedder(salt=salt)),
                lexical=BM25Index(policy_chunks),
            )

        pipeline = AuditPipeline(
            decomposition=DecompositionAgent(llm),
            adjudicator=GatedAdjudicator(
                ReconciliationAgent(llm), CritiqueAgent(llm)
            ),
            retrievers={"policy": build()},
            secondary_retrievers={"policy": build(salt="b")},
        )
        result = pipeline.run("...")

        [sub_result] = result.results
        assert sub_result.verdict.confidence == Confidence.SUPPORTED
        assert "policy#secondary" in sub_result.traces
        assert len(llm.calls) == 4

    def test_graph_evidence_reaches_the_agent(self, policy_chunks):
        """Derived pattern evidence must arrive as a citable chunk.

        The agent can only cite chunk ids, so a graph finding that doesn't
        become a chunk is a finding the system computed and then ignored.
        """
        from app.graph.evidence import GraphEnricher
        from app.graph.models import (
            ComplaintOutcome,
            EdgeKind,
            GraphEdge,
            GraphNode,
            NodeKind,
        )
        from app.graph.store import InMemoryGraphStore

        store = InMemoryGraphStore()
        store.upsert_nodes([
            GraphNode(id="ins-acme", kind=NodeKind.INSURER, label="Acme Health"),
            GraphNode(
                id="rc", kind=NodeKind.DENIAL_REASON,
                label="Not medically necessary", properties={"code": "CO-50"},
            ),
            GraphNode(
                id="c1", kind=NodeKind.COMPLAINT, label="C1",
                properties={"outcome": ComplaintOutcome.OVERTURNED},
            ),
        ])
        store.upsert_edges([
            GraphEdge(source_id="ins-acme", kind=EdgeKind.USED_REASON, target_id="rc"),
            GraphEdge(source_id="c1", kind=EdgeKind.FILED_AGAINST, target_id="ins-acme"),
            GraphEdge(source_id="c1", kind=EdgeKind.CONCERNS_REASON, target_id="rc"),
        ])

        llm = FakeLLM([
            Decomposition(
                denial=ExtractedDenial(denial_reason="Denied", reason_code="CO-50"),
                sub_claims=[DecomposedClaim(text="Exclusion applies.", kind="legal")],
            ),
            DraftVerdict(finding="insufficient", rationale="Unsettled.", citations=[]),
        ])
        pipeline = AuditPipeline(
            decomposition=DecompositionAgent(llm),
            adjudicator=GatedAdjudicator(
                ReconciliationAgent(llm), CritiqueAgent(llm)
            ),
            retrievers={
                "policy": HybridRetriever(
                    dense=DenseIndex(policy_chunks, FakeEmbedder()),
                    lexical=BM25Index(policy_chunks),
                )
            },
            enricher=GraphEnricher(store=store),
        )
        result = pipeline.run("...", insurer_id="ins-acme")

        prompt = llm.calls[1]["prompt"]
        assert 'id="pattern:ins-acme:CO-50"' in prompt
        assert "1 overturned" in prompt
        # And the caveat travels with it
        assert "does not establish that the present denial is improper" in prompt

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
            adjudicator=GatedAdjudicator(
                ReconciliationAgent(llm), CritiqueAgent(llm)
            ),
            retrievers={
                "policy": HybridRetriever(
                    dense=DenseIndex(policy_chunks, FakeEmbedder()),
                    lexical=BM25Index(policy_chunks),
                )
            },
        )
        result = pipeline.run("...")
        assert result.overall == "insufficient"


class TestReasoningOnlyModels:
    """A reasoning-only model must never be sent think=False.

    gpt-oss answers through a channel separate from its reasoning. Given
    `think: false` it does not answer tersely — it returns nothing, and the
    empty string fails schema validation on every call. That produced 35/35
    case errors on the golden set, all at decomposition, none of which said
    anything about whether the model can read a denial letter.

    The cost of getting this wrong is not a crash, it is a *misattribution*:
    a plumbing fault that reads as a model too weak to follow a schema.
    """

    def test_reasoning_only_model_never_gets_thinking_disabled(self):
        llm = OllamaStructuredLLM(model="gpt-oss:20b")
        assert llm.think == DEFAULT_REASONING_EFFORT

    def test_explicit_effort_is_respected(self):
        assert OllamaStructuredLLM(model="gpt-oss:20b", think="high").think == "high"

    def test_ordinary_model_still_defaults_to_no_thinking(self):
        assert OllamaStructuredLLM(model="qwen2.5:7b").think is False

    def test_empty_response_is_reported_as_empty_not_as_bad_schema(self, monkeypatch):
        """The error must name the real cause, or it sends the reader hunting
        for a prompt problem that does not exist."""
        import app.agents.ollama_llm as module

        monkeypatch.setattr(
            module,
            "_post",
            lambda *a, **k: {"message": {"content": "", "thinking": "reasoned..."}},
        )
        llm = OllamaStructuredLLM(model="qwen2.5:7b")
        with pytest.raises(OllamaError) as exc:
            llm.generate(system="s", prompt="p", schema=Decomposition)

        message = str(exc.value)
        assert "empty response" in message
        assert "reasons unconditionally" in message
        # It must not be dressed up as a schema failure.
        assert "invalid output" not in message
