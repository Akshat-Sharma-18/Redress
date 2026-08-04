"""Confidence gate tests: critique loop, re-retrieval, ensemble cross-check.

Same approach as test_agents.py — a scripted FakeLLM, and assertions on the
one property every path must hold: the gate only ever lowers confidence.
"""

from __future__ import annotations

from app.agents.critique import CitationCheck, CritiqueAgent, CritiqueResult
from app.agents.gate import GatedAdjudicator
from app.agents.reconciliation import ReconciliationAgent
from app.agents.schemas import DraftCitation, DraftVerdict
from app.core.models import Chunk, Confidence, ScoredChunk, SourceKind, SubClaim

from .conftest import FakeLLM

EXCLUSION_TEXT = (
    "This plan does not provide benefits for services rendered in a "
    "hospital emergency department."
)
CARVEBACK_TEXT = (
    "Notwithstanding the foregoing, the exclusion in Section 7.2(a) does "
    "not apply to emergency services required to evaluate or stabilize an "
    "emergency medical condition."
)


def _chunk(cid: str, text: str) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(id=cid, text=text, source_kind=SourceKind.POLICY, document_id="d"),
        score=1.0,
    )


def _claim() -> SubClaim:
    return SubClaim(id="sc-1", text="The exclusion applies.", kind="legal")


def _good_draft(finding="justified") -> DraftVerdict:
    return DraftVerdict(
        finding=finding,
        rationale="The exclusion covers this service.",
        citations=[
            DraftCitation(
                chunk_id="p2",
                quote="does not provide benefits",
                supports="Establishes the exclusion.",
            )
        ],
    )


def _gate(llm, **kwargs) -> GatedAdjudicator:
    return GatedAdjudicator(
        ReconciliationAgent(llm), CritiqueAgent(llm), **kwargs
    )


class TestCritiqueLoop:
    def test_approved_verdict_is_supported(self):
        llm = FakeLLM([_good_draft(), CritiqueResult(approved=True)])
        verdict = _gate(llm).adjudicate(_claim(), [_chunk("p2", EXCLUSION_TEXT)])

        assert verdict.finding == "justified"
        assert verdict.confidence == Confidence.SUPPORTED
        assert verdict.critique_notes == "critique: approved"

    def test_rejection_without_remedy_lands_on_insufficient(self):
        """The verdict the critique agent rejects is never shown as-is."""
        llm = FakeLLM([
            _good_draft(finding="contradicted"),
            CritiqueResult(
                approved=False,
                issues=["citation 1 establishes the exclusion exists, not that it fails"],
            ),
        ])
        verdict = _gate(llm).adjudicate(_claim(), [_chunk("p2", EXCLUSION_TEXT)])

        assert verdict.finding == "insufficient"
        assert verdict.confidence == Confidence.INSUFFICIENT
        assert "critique rejected" in verdict.critique_notes
        # Draft preserved for the audit trail
        assert verdict.draft_rationale == "The exclusion covers this service."

    def test_rejection_without_reasons_says_so(self):
        """A rejection with an empty issues list produced 'critique rejected: '
        — a dangling colon that reported a verdict was thrown out and nothing
        about why. It must still downgrade: treating a malformed critique as
        an approval would invert the gate's one invariant.
        """
        llm = FakeLLM([
            _good_draft(),
            CritiqueResult(approved=False, issues=[]),
        ])
        verdict = _gate(llm).adjudicate(_claim(), [_chunk("p2", EXCLUSION_TEXT)])

        assert verdict.finding == "insufficient"
        assert not verdict.critique_notes.endswith(": ")
        assert "gave no reason" in verdict.critique_notes

    def test_blank_issue_strings_are_not_reported_as_reasons(self):
        llm = FakeLLM([
            _good_draft(),
            CritiqueResult(approved=False, issues=["", "   "]),
        ])
        verdict = _gate(llm).adjudicate(_claim(), [_chunk("p2", EXCLUSION_TEXT)])
        assert "gave no reason" in verdict.critique_notes

    def test_narrowed_query_triggers_reretrieval_and_redraft(self):
        """Critique spots missing evidence -> re-retrieve -> approved redraft."""
        redraft = DraftVerdict(
            finding="contradicted",
            rationale="The carve-back defeats the exclusion.",
            citations=[
                DraftCitation(
                    chunk_id="p3",
                    quote="does not apply to emergency services",
                    supports="Carve-back.",
                )
            ],
        )
        llm = FakeLLM([
            _good_draft(),
            CritiqueResult(
                approved=False,
                issues=["Section 7.2(a) is cited but any carve-back to it was not retrieved"],
                narrowed_query="Section 7.2 emergency services carve-back",
            ),
            redraft,
            CritiqueResult(approved=True),
        ])

        queries: list[str] = []

        def retrieve_fn(q: str):
            queries.append(q)
            return [_chunk("p3", CARVEBACK_TEXT)]

        verdict = _gate(llm).adjudicate(
            _claim(), [_chunk("p2", EXCLUSION_TEXT)], retrieve_fn=retrieve_fn
        )

        assert queries == ["Section 7.2 emergency services carve-back"]
        assert verdict.finding == "contradicted"
        assert verdict.confidence == Confidence.SUPPORTED
        # The redraft saw BOTH the original and the newly retrieved chunk
        trace_ids = {sc.chunk.id for sc in verdict.retrieval_trace}
        assert trace_ids == {"p2", "p3"}

    def test_retry_budget_is_bounded(self):
        """A critique agent that keeps rejecting cannot loop forever."""
        llm = FakeLLM([
            _good_draft(),
            CritiqueResult(approved=False, issues=["x"], narrowed_query="q1"),
            _good_draft(),
            CritiqueResult(approved=False, issues=["still x"], narrowed_query="q2"),
        ])
        calls = []
        verdict = _gate(llm, max_reretrievals=1).adjudicate(
            _claim(),
            [_chunk("p2", EXCLUSION_TEXT)],
            retrieve_fn=lambda q: calls.append(q) or [],
        )

        assert calls == ["q1"]  # second narrowed query never executed
        assert verdict.finding == "insufficient"
        assert llm._responses == []  # exactly the scripted number of LLM calls

    def test_insufficient_draft_skips_critique(self):
        """Abstention is already the floor — no critique call is spent on it."""
        llm = FakeLLM([
            DraftVerdict(finding="insufficient", rationale="Not settled.", citations=[])
        ])
        verdict = _gate(llm).adjudicate(_claim(), [_chunk("p2", EXCLUSION_TEXT)])

        assert verdict.finding == "insufficient"
        assert len(llm.calls) == 1  # reconciliation only

    def test_mechanical_downgrade_skips_critique(self):
        """A draft that failed substring verification never reaches critique."""
        llm = FakeLLM([
            DraftVerdict(
                finding="justified",
                rationale="Excluded.",
                citations=[
                    DraftCitation(chunk_id="p2", quote="not in text", supports="...")
                ],
            )
        ])
        verdict = _gate(llm).adjudicate(_claim(), [_chunk("p2", EXCLUSION_TEXT)])
        assert verdict.finding == "insufficient"
        assert len(llm.calls) == 1


class TestEnsembleCrossCheck:
    def test_agreement_yields_supported(self):
        llm = FakeLLM([
            _good_draft(),                 # primary adjudication
            CritiqueResult(approved=True), # critique
            _good_draft(),                 # secondary adjudication
        ])
        verdict = _gate(llm).adjudicate(
            _claim(),
            [_chunk("p2", EXCLUSION_TEXT)],
            secondary_evidence=[_chunk("p2", EXCLUSION_TEXT)],
        )
        assert verdict.confidence == Confidence.SUPPORTED
        assert "ensemble: agreed" in verdict.critique_notes

    def test_disagreement_yields_contested_not_a_winner(self):
        """Two passes disagree -> CONTESTED. The system never picks a side."""
        llm = FakeLLM([
            _good_draft(finding="justified"),
            CritiqueResult(approved=True),
            DraftVerdict(
                finding="contradicted",
                rationale="The carve-back defeats the exclusion.",
                citations=[
                    DraftCitation(
                        chunk_id="p3",
                        quote="does not apply to emergency services",
                        supports="Carve-back.",
                    )
                ],
            ),
        ])
        verdict = _gate(llm).adjudicate(
            _claim(),
            [_chunk("p2", EXCLUSION_TEXT)],
            secondary_evidence=[_chunk("p3", CARVEBACK_TEXT)],
        )

        # The finding is reported but the confidence is capped at contested
        assert verdict.finding == "justified"
        assert verdict.confidence == Confidence.CONTESTED
        assert "ensemble: disagreed" in verdict.critique_notes
        assert "professional review" in verdict.rationale

    def test_secondary_abstention_counts_as_disagreement(self):
        """If the second pass can't confirm, confidence must not be full."""
        llm = FakeLLM([
            _good_draft(),
            CritiqueResult(approved=True),
            DraftVerdict(finding="insufficient", rationale="Unclear.", citations=[]),
        ])
        verdict = _gate(llm).adjudicate(
            _claim(),
            [_chunk("p2", EXCLUSION_TEXT)],
            secondary_evidence=[_chunk("p9", "Unrelated dental benefits text.")],
        )
        assert verdict.confidence == Confidence.CONTESTED

    def test_no_secondary_evidence_skips_ensemble(self):
        llm = FakeLLM([_good_draft(), CritiqueResult(approved=True)])
        verdict = _gate(llm).adjudicate(_claim(), [_chunk("p2", EXCLUSION_TEXT)])
        assert verdict.confidence == Confidence.SUPPORTED
        assert len(llm.calls) == 2


class TestAsymmetricAssurance:
    """A `justified` finding is held to a higher bar than a `contradicted` one.

    The two errors do not cost the same. Telling someone their denial was
    justified tells them to stop, and if that is wrong they lose money they
    were owed. Telling them it was contradicted sends them to an appeal, where
    a human reads it next. `app/eval/metrics.py` has scored those differently
    since Phase 5; these tests are the gate finally *deciding* differently.
    """

    def _critique_check(self, chunk_id="p2", supports=True):
        return CitationCheck(
            chunk_id=chunk_id, supports_claim=supports, note="what it shows"
        )

    def test_qualified_approval_does_not_assure(self):
        """An approval issued alongside named defects is a qualified one."""
        llm = FakeLLM([
            _good_draft("justified"),
            CritiqueResult(
                approved=True,
                issues=["the carve-back in 7.2(b) was not addressed"],
            ),
        ])
        verdict = _gate(llm).adjudicate(_claim(), [_chunk("p2", EXCLUSION_TEXT)])

        assert verdict.finding == "insufficient"
        assert verdict.confidence == Confidence.INSUFFICIENT
        assert "7.2(b)" in verdict.critique_notes
        # The audit trail survives the downgrade.
        assert verdict.draft_rationale == "The exclusion covers this service."
        assert verdict.citations

    def test_unconfirmed_citation_does_not_assure(self):
        llm = FakeLLM([
            _good_draft("justified"),
            CritiqueResult(
                approved=True,
                citation_checks=[self._critique_check(supports=False)],
            ),
        ])
        verdict = _gate(llm).adjudicate(_claim(), [_chunk("p2", EXCLUSION_TEXT)])
        assert verdict.finding == "insufficient"
        assert "not confirmed to support" in verdict.critique_notes

    def test_finding_that_does_not_follow_does_not_assure(self):
        llm = FakeLLM([
            _good_draft("justified"),
            CritiqueResult(
                approved=True,
                finding_follows=False,
                citation_checks=[self._critique_check()],
            ),
        ])
        verdict = _gate(llm).adjudicate(_claim(), [_chunk("p2", EXCLUSION_TEXT)])
        assert verdict.finding == "insufficient"

    def test_clean_approval_still_assures(self):
        """The bar must not swallow every justified verdict, or it is just an
        expensive way to abstain."""
        llm = FakeLLM([
            _good_draft("justified"),
            CritiqueResult(
                approved=True,
                finding_follows=True,
                citation_checks=[self._critique_check()],
            ),
        ])
        verdict = _gate(llm).adjudicate(_claim(), [_chunk("p2", EXCLUSION_TEXT)])
        assert verdict.finding == "justified"
        assert verdict.confidence == Confidence.SUPPORTED

    def test_contradicted_is_not_held_to_the_higher_bar(self):
        """The asymmetry is the point: the same qualified approval stands when
        the finding sends the user to an appeal rather than away from one."""
        llm = FakeLLM([
            _good_draft("contradicted"),
            CritiqueResult(approved=True, issues=["minor wording concern"]),
        ])
        verdict = _gate(llm).adjudicate(_claim(), [_chunk("p2", EXCLUSION_TEXT)])
        assert verdict.finding == "contradicted"
        assert verdict.confidence == Confidence.SUPPORTED

    def test_ablation_switch_restores_symmetric_behaviour(self):
        """A safety rule nobody can turn off is a safety rule nobody can price."""
        llm = FakeLLM([
            _good_draft("justified"),
            CritiqueResult(approved=True, issues=["unaddressed carve-back"]),
        ])
        verdict = _gate(llm, asymmetric_assurance=False).adjudicate(
            _claim(), [_chunk("p2", EXCLUSION_TEXT)]
        )
        assert verdict.finding == "justified"
