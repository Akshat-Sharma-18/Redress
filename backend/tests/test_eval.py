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

    def test_crash_gets_its_own_class(self):
        """A crash is a failure to answer, not a wrong answer.

        Folding it into a direction class reported a broken run as
        `false_alarm` and hid the bug behind a plausible metric.
        """
        for expected, category in [
            ("justified", Category.CLEAR_JUSTIFICATION),
            ("contradicted", Category.CLEAR_CONTRADICTION),
            ("insufficient", Category.AMBIGUOUS),
        ]:
            assert (
                classify(expected=expected, predicted="error", category=category)
                is Outcome.ERROR
            )

    def test_contested_counts_as_declining_to_assert(self):
        """'contested' and 'insufficient' are both non-assertions.

        They differ in whether the system has a leaning it cannot
        substantiate — worth telling the user, but neither is a claim.
        """
        assert (
            classify(
                expected="insufficient",
                predicted="contested",
                category=Category.AMBIGUOUS,
            )
            is Outcome.CORRECT_ABSTENTION
        )
        assert (
            classify(
                expected="contradicted",
                predicted="contested",
                category=Category.CLEAR_CONTRADICTION,
            )
            is Outcome.OVER_ABSTENTION
        )


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

    def test_errors_excluded_from_abstention_denominators(self):
        """A crash is not a decision to abstain.

        Counting it as one would let a broken run read as well-calibrated.
        """
        report = Report([
            _ev("insufficient", "insufficient", Category.AMBIGUOUS),
            _ev("insufficient", "error", Category.AMBIGUOUS),
        ])
        assert report.error_count == 1
        assert report.correct_abstention_rate == 1.0  # 1 of 1 *scored* case
        assert "errored" in report.summary()

    def test_denial_date_mismatch_is_surfaced(self):
        """A wrong date means temporal filtering ran against the wrong law.

        Tracked independently of the verdict: a case whose date extraction
        failed can still reach the right answer by luck, and that is not the
        same as the temporal layer working.
        """
        from datetime import date

        ev = _ev(
            "justified", "justified", Category.CLEAR_JUSTIFICATION,
            expected_denial_date=date(2021, 8, 9),
            extracted_denial_date=date(2023, 1, 1),
        )
        assert ev.outcome is Outcome.CORRECT  # verdict was right...
        assert "2023-01-01" in ev.date_error  # ...but for the wrong reason
        report = Report([ev])
        assert len(report.date_errors) == 1
        assert "wrong law" in report.summary()

    def test_missing_denial_date_is_surfaced(self):
        """No date silently disables filtering — every version becomes
        eligible, including ones enacted after the denial."""
        from datetime import date

        ev = _ev(
            "justified", "justified", Category.CLEAR_JUSTIFICATION,
            expected_denial_date=date(2021, 8, 9),
            extracted_denial_date=None,
        )
        assert "temporal filtering disabled" in ev.date_error

    def test_no_date_error_when_extraction_matches(self):
        from datetime import date

        ev = _ev(
            "justified", "justified", Category.CLEAR_JUSTIFICATION,
            expected_denial_date=date(2021, 8, 9),
            extracted_denial_date=date(2021, 8, 9),
        )
        assert ev.date_error is None

    def test_contested_rate_is_tracked_separately(self):
        """Shows the confidence gate is doing work rather than sitting inert."""
        report = Report([
            _ev("contradicted", "contested", Category.CLEAR_CONTRADICTION),
            _ev("contradicted", "contradicted", Category.CLEAR_CONTRADICTION),
        ])
        assert report.contested_rate == 0.5
        assert report.over_abstention_rate == 0.5  # contested declines to assert

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

    def test_summary_is_console_safe(self):
        """The report prints after the API spend. A UnicodeEncodeError on a
        legacy Windows codepage would destroy the run's output at the last
        step, so every printable path stays ASCII.
        """
        from datetime import date

        report = Report([
            _ev("contradicted", "error", Category.CLEAR_CONTRADICTION),
            _ev(
                "justified", "justified", Category.CLEAR_JUSTIFICATION,
                expected_denial_date=date(2021, 8, 9),
                extracted_denial_date=None,
            ),
        ])
        # Exercises the error and date-error branches too, not just the header.
        summary = report.summary()
        assert "errored" in summary and "wrong law" in summary
        summary.encode("cp437")  # raises if a non-ASCII char slipped in


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

    def test_typo_in_expected_overall_is_rejected(self, tmp_path):
        """An unrecognised finding would never match a prediction, so the
        case would score as a failure forever and drag every rate down."""
        (tmp_path / "typo.yaml").write_text(
            yaml.safe_dump({
                "id": "typo",
                "name": "n",
                "category": "clear_contradiction",
                "denial_letter": "x",
                "expected": {"overall": "contradicated"},
            }),
            encoding="utf-8",
        )
        with pytest.raises(Exception, match="overall"):
            load_dataset(tmp_path)

    def test_bad_source_kind_fails_at_load_not_mid_run(self, tmp_path):
        """Otherwise the conversion raises inside the harness's broad except
        and the fixture typo is recorded as a crashed case."""
        (tmp_path / "bad.yaml").write_text(
            yaml.safe_dump({
                "id": "bad",
                "name": "n",
                "category": "clear_justification",
                "denial_letter": "x",
                "expected": {"overall": "justified"},
                "chunks": [{"id": "c", "text": "t", "source_kind": "polcy"}],
            }),
            encoding="utf-8",
        )
        with pytest.raises(Exception, match="source_kind"):
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
