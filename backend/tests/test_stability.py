"""Tests for repeated-run reporting.

The case that motivates this module is `test_it_would_have_caught_the_mistake`.
Everything else here is bookkeeping; that one is the reason the module exists.
"""

from __future__ import annotations

from app.eval.metrics import CaseEvaluation, Category, Outcome, Report
from app.eval.stability import RepeatedReport


def _evaluation(
    case_id: str,
    predicted: str,
    outcome: Outcome,
    expected: str = "contradicted",
    category: Category = Category.CLEAR_CONTRADICTION,
) -> CaseEvaluation:
    return CaseEvaluation(
        case_id=case_id,
        category=category,
        expected=expected,
        predicted=predicted,
        outcome=outcome,
    )


def _run(*pairs: tuple[str, str, Outcome]) -> Report:
    return Report(evaluations=[_evaluation(*p) for p in pairs])


def test_single_run_reports_no_spread():
    """One run has no error bar, and must not pretend otherwise."""
    repeated = RepeatedReport([_run(("a", "contradicted", Outcome.CORRECT))])
    assert "only one run" in repeated.summary()


def test_stable_cases_are_not_flagged():
    run = _run(("a", "contradicted", Outcome.CORRECT))
    repeated = RepeatedReport([run, _run(("a", "contradicted", Outcome.CORRECT))])
    assert repeated.unstable_cases == {}
    assert repeated.noise_floor == 0.0


def test_unstable_case_is_flagged_with_its_split():
    repeated = RepeatedReport([
        _run(("a", "contradicted", Outcome.CORRECT)),
        _run(("a", "insufficient", Outcome.OVER_ABSTENTION)),
        _run(("a", "contradicted", Outcome.CORRECT)),
    ])
    unstable = repeated.unstable_cases
    assert set(unstable) == {"a"}
    assert unstable["a"]["contradicted"] == 2
    assert unstable["a"]["insufficient"] == 1
    assert repeated.noise_floor == 1.0


def test_mean_and_range_are_reported():
    repeated = RepeatedReport([
        _run(("a", "contradicted", Outcome.CORRECT)),
        _run(("a", "insufficient", Outcome.OVER_ABSTENTION)),
    ])
    mean, low, high = repeated.spread("accuracy")
    assert (low, high) == (0.0, 1.0)
    assert mean == 0.5


def test_unmeasured_metric_is_skipped_not_counted_as_zero():
    """`None` means 'not measured'. Averaging it in as 0.0 invents a result."""
    repeated = RepeatedReport([
        _run(("a", "contradicted", Outcome.CORRECT)),
        _run(("a", "contradicted", Outcome.CORRECT)),
    ])
    # No ambiguous cases at all, so correct-abstention has an empty denominator.
    assert repeated.values("correct_abstention_rate") == []
    assert repeated.spread("correct_abstention_rate") is None
    assert "n/a" in repeated.summary()


def test_it_would_have_caught_the_mistake():
    """The regression this module was built for.

    Two runs of one configuration disagreed on four cases, and that variance
    was quietly read as the effect of a code change. The summary has to state
    the noise floor loudly enough that the same reading is not available.
    """
    stable = [(f"s{i}", "contradicted", Outcome.CORRECT) for i in range(31)]
    run_a = _run(
        *stable,
        ("flip1", "contradicted", Outcome.CORRECT),
        ("flip2", "contradicted", Outcome.CORRECT),
        ("flip3", "insufficient", Outcome.OVER_ABSTENTION),
        ("flip4", "justified", Outcome.FALSE_ASSURANCE),
    )
    run_b = _run(
        *stable,
        ("flip1", "insufficient", Outcome.OVER_ABSTENTION),
        ("flip2", "insufficient", Outcome.OVER_ABSTENTION),
        ("flip3", "contradicted", Outcome.CORRECT),
        ("flip4", "contradicted", Outcome.CORRECT),
    )

    repeated = RepeatedReport([run_a, run_b])
    assert len(repeated.unstable_cases) == 4
    assert repeated.noise_floor == 4 / 35

    summary = repeated.summary()
    assert "4 of 35" in summary
    assert "disagreeing with itself" in summary
    # The unstable case ids have to be named, or the reader cannot go look.
    for case_id in ("flip1", "flip2", "flip3", "flip4"):
        assert case_id in summary
