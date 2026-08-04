"""Repeat an eval and report how much of the result was luck.

This module exists because a single run lied to us. Two runs of qwen2.5:7b
over the same 35 cases, differing only in a gate change that can *only* turn
`justified` into `insufficient`, disagreed on five cases — four of them
transitions that change could not produce. The measured difference between
two models had been smaller than the difference between one model and itself.

Nothing here makes the pipeline deterministic. Greedy decoding is not
bit-reproducible when a model is split across GPU and CPU: the reduction
order in a partially offloaded matmul depends on how the work was scheduled,
and a single flipped logit at a branch point changes a verdict. Chasing that
is the wrong fight. The right one is to stop quoting a number whose error bar
nobody measured.

So the unit of measurement becomes N runs, and the report leads with the
spread. A change smaller than the spread is not a finding, and this module's
whole job is to make that impossible to overlook.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass

from app.eval.metrics import Report

#: Reported for every repeated run, worst-consequence first — the same order
#: the single-run summary uses, so the two read alike.
TRACKED_METRICS = (
    ("false_assurance_rate", "False assurance"),
    ("correct_abstention_rate", "Correct abstention"),
    ("over_abstention_rate", "Over-abstention"),
    ("contested_rate", "Contested"),
    ("accuracy", "Accuracy"),
    ("grounding_accuracy", "Grounding accuracy"),
)


def _pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{value * 100:5.1f}%"


@dataclass
class RepeatedReport:
    """N runs of the same configuration, summarised with their spread."""

    runs: list[Report]

    @property
    def n(self) -> int:
        return len(self.runs)

    def values(self, metric: str) -> list[float]:
        """Every run's value for `metric`, skipping runs that could not measure it.

        A `None` is "not measured", not zero — averaging it in as zero would
        invent a data point, which is the same class of error as reporting an
        empty denominator as 0.0.
        """
        raw = [getattr(report, metric) for report in self.runs]
        return [v for v in raw if v is not None]

    def spread(self, metric: str) -> tuple[float, float, float] | None:
        """(mean, min, max) for `metric`, or None when nothing measured it."""
        values = self.values(metric)
        if not values:
            return None
        return (sum(values) / len(values), min(values), max(values))

    def case_predictions(self) -> dict[str, Counter]:
        """Per case, how often each prediction came up across the runs."""
        tally: dict[str, Counter] = {}
        for report in self.runs:
            for evaluation in report.evaluations:
                tally.setdefault(evaluation.case_id, Counter())[
                    evaluation.predicted
                ] += 1
        return tally

    @property
    def unstable_cases(self) -> dict[str, Counter]:
        """Cases that did not give the same answer every time."""
        return {
            case_id: counts
            for case_id, counts in self.case_predictions().items()
            if len(counts) > 1
        }

    @property
    def noise_floor(self) -> float | None:
        """Fraction of cases that changed answer at least once.

        The single most important number this module produces. Any comparison
        between two configurations that differ by less than this is measuring
        the pipeline's variance, not the change.
        """
        tally = self.case_predictions()
        if not tally:
            return None
        return len(self.unstable_cases) / len(tally)

    def summary(self) -> str:
        if self.n < 2:
            return "only one run — no spread to report"

        lines = [
            f"Repeated {self.n} runs over {len(self.case_predictions())} cases",
            "",
            f"{'metric':<22} {'mean':>7}  {'range':>15}",
            "-" * 50,
        ]
        for metric, label in TRACKED_METRICS:
            band = self.spread(metric)
            if band is None:
                lines.append(f"{label:<22} {'n/a':>7}")
                continue
            mean, low, high = band
            lines.append(
                f"{label:<22} {_pct(mean)}  {_pct(low)} - {_pct(high)}"
            )

        unstable = self.unstable_cases
        floor = self.noise_floor or 0.0
        lines += [
            "",
            f"Unstable cases: {len(unstable)} of {len(self.case_predictions())} "
            f"({floor * 100:.1f}%) gave more than one answer across runs.",
        ]
        for case_id, counts in sorted(unstable.items()):
            spread = ", ".join(
                f"{pred}x{count}" for pred, count in counts.most_common()
            )
            lines.append(f"   {case_id:<34} {spread}")

        # The interpretive line. Without it the table above invites exactly
        # the mistake it was built to prevent.
        widest = max(
            (
                band[2] - band[1]
                for band in (self.spread(m) for m, _ in TRACKED_METRICS)
                if band is not None
            ),
            default=0.0,
        )
        lines += [
            "",
            f"Widest metric range across runs: {widest * 100:.1f} points.",
            "A difference between two configurations smaller than that is not",
            "a finding — it is this pipeline disagreeing with itself.",
        ]
        return "\n".join(lines)


def stdev(values: list[float]) -> float | None:
    """Sample standard deviation, or None below two points."""
    return statistics.stdev(values) if len(values) > 1 else None
