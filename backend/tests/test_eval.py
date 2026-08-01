"""Eval harness tests.

Two things under test: the consequence taxonomy (does a given
expected/predicted pair land in the right bucket) and the loader's refusal
to accept a dataset that would quietly corrupt the metric.

The golden files themselves are validated here too — a malformed case would
otherwise only surface during an eval run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.eval.dataset import (
    Category,
    GoldenCase,
    GoldenDatasetError,
    load_dataset,
)
from app.eval.harness import AblationConfig, NullCritique
from app.eval.metrics import CaseEvaluation, Outcome, Report, classify

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "data" / "golden"


class TestConsequenceTaxonomy:
    def test_false_assurance_is_its_own_class(self):
        """Saying 'justified' when truth is 'contradicted' costs money.

        This is the failure the whole architecture is built to avoid, so it
        must never be pooled with generic misclassification.
        """
        assert (
            classify(
                expected="contradicted",
                predicted="justified",
                category=Category.CLEAR_CONTRADICTION,
            )
            is Outcome.FALSE_ASSURANCE
        )

    def test_false_alarm_is_distinct_from_false_assurance(self):
        """Wasting someone's time is not the same as costing them money."""
        assert (
            classify(
                expected="justified",
                predicted="contradicted",
                category=Category.CLEAR_JUSTIFICATION,
            )
            is Outcome.FALSE_ALARM
        )

    def test_abstaining_on_clear_case_is_over_abstention(self):
        assert (
            classify(
                expected="contradicted",
                predicted="insufficient",
                category=Category.CLEAR_CONTRADICTION,
            )
            is Outcome.OVER_ABSTENTION
        )

    def test_ruling_on_ambiguous_case_is_under_abstention(self):
        """Even a 'lucky' direction is unearned confidence."""
        for predicted in ("justified", "contradicted"):
            assert (
                classify(
                    expected="insufficient",
                    predicted=predicted,
                    category=Category.AMBIGUOUS,
                )
                is Outcome.UNDER_ABSTENTION
            )

    def test_correct_abstention_is_separated_from_correct(self):
        assert (
            classify(
                expected="insufficient",
                predicted="insufficient",
                category=Category.AMBIGUOUS,
            )
            is Outcome.CORRECT_ABSTENTION
        )
        assert (
            classify(
                expected="justified",
                predicted="justified",
                category=Category.CLEAR_JUSTIFICATION,
            )
            is Outcome.CORRECT
        )

    def test_mixed_direction_error_is_not_mislabeled_as_assurance(self):
        """contradicted -> mixed is wrong, but it does not tell the user
        the denial was proper."""
        assert (
            classify(
                expected="contradicted",
                predicted="mixed",
                category=Category.CLEAR_CONTRADICTION,
            )
            is Outcome.DIRECTION_ERROR
        )

    def test_crash_is_scored_not_dropped(self):
        """A harness that drops errored cases inflates every rate."""
        outcome = classify(
            expected="contradicted",
            predicted="error",
            category=Category.CLEAR_CONTRADICTION,
        )
        assert outcome is not Outcome.CORRECT


def _ev(expected, predicted, category, **kw) -> CaseEvaluation:
    return CaseEvaluation(
        case_id=kw.pop("case_id", "c"),
        category=category,
        expected=expected,
        predicted=predicted,
        outcome=classify(
            expected=expected, predicted=predicted, category=category
        ),
        **kw,
    )


class TestReport:
    def test_abstention_rates_use_the_right_denominators(self):
        """Correct-abstention is measured over ambiguous cases only.

        Over the whole set, a system that abstains on everything would score
        100% — which is exactly the degenerate behaviour the metric exists
        to detect.
        """
        report = Report([
            _ev("insufficient", "insufficient", Category.AMBIGUOUS),
            _ev("insufficient", "justified", Category.AMBIGUOUS),
            _ev("contradicted", "contradicted", Category.CLEAR_CONTRADICTION),
            _ev("justified", "insufficient", Category.CLEAR_JUSTIFICATION),
        ])
        assert report.correct_abstention_rate == 0.5   # 1 of 2 ambiguous
        assert report.over_abstention_rate == 0.5      # 1 of 2 decisive

    def test_always_abstaining_does_not_score_well_overall(self):
        report = Report([
            _ev("contradicted", "insufficient", Category.CLEAR_CONTRADICTION),
            _ev("justified", "insufficient", Category.CLEAR_JUSTIFICATION),
            _ev("insufficient", "insufficient", Category.AMBIGUOUS),
        ])
        assert report.correct_abstention_rate == 1.0
        assert report.over_abstention_rate == 1.0  # the cost is visible
        assert report.accuracy == pytest.approx(1 / 3)

    def test_grounding_requires_the_named_evidence(self):
        """A right answer that never cites the governing clause got there
        some other way."""
        grounded = _ev(
            "contradicted", "contradicted", Category.CLEAR_CONTRADICTION,
            cited={"p-7-2-b"},
        )
        ungrounded = _ev(
            "contradicted", "contradicted", Category.CLEAR_CONTRADICTION,
            cited={"p-7-2-a"}, missing_required_citations={"p-7-2-b"},
        )
        assert grounded.grounded
        assert not ungrounded.grounded
        assert Report([grounded, ungrounded]).grounding_accuracy == 0.5

    def test_forbidden_citation_breaks_grounding(self):
        """Reaching the right answer via a superseded statute is not a pass."""
        ev = _ev(
            "justified", "justified", Category.CLEAR_JUSTIFICATION,
            cited={"s-2023"}, forbidden_citations={"s-2023"},
        )
        assert not ev.grounded

    def test_empty_denominators_report_none_not_zero(self):
        """'Not measured' and 'measured, scored zero' are different claims."""
        report = Report([
            _ev("contradicted", "contradicted", Category.CLEAR_CONTRADICTION)
        ])
        assert report.correct_abstention_rate is None  # no ambiguous cases
        assert report.false_assurance_rate == 0.0      # measured, and zero

    def test_per_class_metrics(self):
        report = Report([
            _ev("contradicted", "contradicted", Category.CLEAR_CONTRADICTION),
            _ev("contradicted", "justified", Category.CLEAR_CONTRADICTION),
            _ev("justified", "justified", Category.CLEAR_JUSTIFICATION),
        ])
        m = report.per_class("contradicted")
        assert m["precision"] == 1.0            # 1 predicted, 1 right
        assert m["recall"] == 0.5               # 2 actual, 1 found
        assert m["support"] == 2

    def test_summary_renders(self):
        report = Report([
            _ev("insufficient", "insufficient", Category.AMBIGUOUS),
            _ev("contradicted", "justified", Category.CLEAR_CONTRADICTION),
        ])
        text = report.summary()
        assert "false_assurance" in text
        assert "Correct abstention" in text


class TestDatasetLoader:
    def test_golden_files_are_valid(self):
        """The shipped dataset must parse — a broken case would otherwise
        only surface mid-eval-run."""
        cases = load_dataset(GOLDEN_DIR)
        assert len(cases) >= 5
        assert {c.category for c in cases} >= {
            Category.CLEAR_CONTRADICTION,
            Category.CLEAR_JUSTIFICATION,
            Category.AMBIGUOUS,
        }

    def test_every_required_citation_exists_in_its_case(self):
        """A must_cite id that names no chunk can never be satisfied, so the
        case would score as ungrounded forever."""
        for case in load_dataset(GOLDEN_DIR):
            ids = {c.id for c in case.chunks}
            unknown = (
                set(case.expected.must_cite) | set(case.expected.must_not_cite)
            ) - ids
            assert not unknown, f"{case.id}: unknown chunk ids {unknown}"

    def test_golden_cases_are_synthetic(self):
        """The repository is public; real documents must not land here."""
        for case in load_dataset(GOLDEN_DIR):
            assert case.source in ("synthetic", "anonymized")

    def test_ambiguous_case_without_notes_is_rejected(self, tmp_path):
        """An unexplained abstention label cannot be reviewed, and the
        headline metric would rest on it."""
        (tmp_path / "bad.yaml").write_text(
            yaml.safe_dump({
                "id": "bad",
                "name": "n",
                "category": "ambiguous",
                "denial_letter": "x",
                "expected": {"overall": "insufficient"},
            }),
            encoding="utf-8",
        )
        with pytest.raises(GoldenDatasetError, match="must explain"):
            load_dataset(tmp_path)

    def test_duplicate_ids_rejected(self, tmp_path):
        """A silently overwritten case shrinks the set without changing the
        reported denominator."""
        for name in ("a.yaml", "b.yaml"):
            (tmp_path / name).write_text(
                yaml.safe_dump({
                    "id": "same",
                    "name": "n",
                    "category": "clear_justification",
                    "denial_letter": "x",
                    "expected": {"overall": "justified"},
                }),
                encoding="utf-8",
            )
        with pytest.raises(GoldenDatasetError, match="duplicate"):
            load_dataset(tmp_path)


class TestAblation:
    def test_null_critique_approves_everything(self):
        """The ablation arm holds the flow fixed and removes only judgement."""
        result = NullCritique().review(None, None, None)
        assert result.approved is True

    def test_default_config_is_the_full_system(self):
        config = AblationConfig()
        assert config.use_reranker and config.use_critique and config.use_ensemble
