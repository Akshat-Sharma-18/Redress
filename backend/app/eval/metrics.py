"""Evaluation metrics, weighted by consequence rather than by accuracy.

Standard multiclass precision/recall treats every error identically. That is
wrong for this system, because the four ways to be wrong here have wildly
different costs to the person holding the denial letter:

  false assurance   told the denial was justified when it was contradicted.
                    They don't appeal a claim they would have won. This is
                    the failure that costs real money, and it is the number
                    to lead with.
  false alarm       told the denial was contradicted when it was justified.
                    They spend effort on a losing appeal. Bad, recoverable.
  under-abstention  ruled confidently on a genuinely ambiguous case. Its
                    direction may happen to be right; the confidence was not
                    earned either way.
  over-abstention   abstained when the evidence did settle it. A missed
                    opportunity, and the *cheapest* failure — the user is
                    told to seek review, which is never harmful advice.

The asymmetry is the design argument: this system is built to convert
would-be false assurances into over-abstentions. Reporting a single accuracy
number would hide exactly the trade the architecture is making.

No composite "risk score" is computed. Weighting these against each other is
a judgement about someone else's finances, and the counts are reported so a
reader can apply their own.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

from app.eval.dataset import Category

DECISIVE = ("justified", "contradicted", "mixed")
ABSTAIN = "insufficient"
CONTESTED = "contested"

#: Dispositions in which the system declines to assert anything about the
#: denial. 'contested' means it has a leaning it could not substantiate;
#: 'insufficient' means it has none. Both correctly refuse to make a claim,
#: so both count as abstention for the metrics.
NON_ASSERTIONS = frozenset({ABSTAIN, CONTESTED})

#: Sentinel written by the harness when a case raised. Not a finding.
ERROR = "error"


class Outcome(str, Enum):
    CORRECT = "correct"
    CORRECT_ABSTENTION = "correct_abstention"
    OVER_ABSTENTION = "over_abstention"
    UNDER_ABSTENTION = "under_abstention"
    FALSE_ASSURANCE = "false_assurance"
    FALSE_ALARM = "false_alarm"
    DIRECTION_ERROR = "direction_error"
    ERROR = "error"


#: Ordered by harm to the user, worst first. Used for reporting, not for
#: arithmetic. ERROR sits with over-abstention because its cost to the user
#: is the same — they get no answer — even though it signals a bug rather
#: than a judgement call, which is why the summary counts it separately too.
SEVERITY = [
    Outcome.FALSE_ASSURANCE,
    Outcome.UNDER_ABSTENTION,
    Outcome.FALSE_ALARM,
    Outcome.DIRECTION_ERROR,
    Outcome.ERROR,
    Outcome.OVER_ABSTENTION,
    Outcome.CORRECT_ABSTENTION,
    Outcome.CORRECT,
]


def classify(
    *, expected: str, predicted: str, category: Category
) -> Outcome:
    """Map one (expected, predicted) pair to a consequence class."""
    if predicted == ERROR:
        # A crash is a failure to answer, not a wrong answer. Folding it into
        # a direction class would report a broken run as a semantic error and
        # hide the bug behind a plausible-looking metric.
        return Outcome.ERROR

    if expected == predicted or (
        expected in NON_ASSERTIONS and predicted in NON_ASSERTIONS
    ):
        return (
            Outcome.CORRECT_ABSTENTION
            if predicted in NON_ASSERTIONS
            else Outcome.CORRECT
        )

    if predicted in NON_ASSERTIONS:
        # Declined to assert on a case the evidence settled.
        return Outcome.OVER_ABSTENTION

    if expected in NON_ASSERTIONS or category is Category.AMBIGUOUS:
        # Ruled decisively where the correct answer was to decline.
        return Outcome.UNDER_ABSTENTION

    # Both decisive, and they disagree. Direction determines the cost.
    if predicted == "justified":
        # Told them the insurer was right when it was not.
        return Outcome.FALSE_ASSURANCE
    if expected == "justified":
        # Sent them to appeal a denial that was in fact supported.
        return Outcome.FALSE_ALARM
    return Outcome.DIRECTION_ERROR


@dataclass
class CaseEvaluation:
    case_id: str
    category: Category
    expected: str
    predicted: str
    outcome: Outcome
    cited: set[str] = field(default_factory=set)
    missing_required_citations: set[str] = field(default_factory=set)
    forbidden_citations: set[str] = field(default_factory=set)
    error: str | None = None

    #: The denial date the decomposition agent extracted, cross-checked
    #: against the golden metadata. A wrong or missing date means temporal
    #: filtering ran against the wrong law — a verdict can still come out
    #: right by luck, so this is tracked independently of the outcome.
    expected_denial_date: date | None = None
    extracted_denial_date: date | None = None

    @property
    def date_error(self) -> str | None:
        if self.expected_denial_date is None or self.outcome is Outcome.ERROR:
            return None
        if self.extracted_denial_date is None:
            # Silently disables temporal filtering: every statute version
            # becomes eligible, including ones enacted after the denial.
            return "no denial date extracted - temporal filtering disabled"
        if self.extracted_denial_date != self.expected_denial_date:
            return (
                f"denial date {self.extracted_denial_date.isoformat()} != "
                f"{self.expected_denial_date.isoformat()}"
            )
        return None

    @property
    def grounded(self) -> bool:
        """Right answer AND resting on the required evidence."""
        return (
            self.outcome in (Outcome.CORRECT, Outcome.CORRECT_ABSTENTION)
            and not self.missing_required_citations
            and not self.forbidden_citations
        )


def _safe_div(numerator: int, denominator: int) -> float | None:
    """None, not zero, when the denominator is empty.

    A metric with no cases to measure is undefined. Reporting 0.0 would read
    as "the system failed at this", which is a different claim from "the
    golden set does not test this yet".
    """
    return numerator / denominator if denominator else None


@dataclass
class Report:
    evaluations: list[CaseEvaluation]

    @property
    def counts(self) -> Counter:
        return Counter(e.outcome for e in self.evaluations)

    @property
    def total(self) -> int:
        return len(self.evaluations)

    @property
    def accuracy(self) -> float | None:
        correct = self.counts[Outcome.CORRECT] + self.counts[Outcome.CORRECT_ABSTENTION]
        return _safe_div(correct, self.total)

    @property
    def grounding_accuracy(self) -> float | None:
        """Of the verdicts that were right, how many cited the right evidence.

        A correct finding that never cites the clause it turns on reached the
        answer some other way. In a system whose entire premise is that the
        citation chain is the product, that is close to a miss.
        """
        right = [
            e
            for e in self.evaluations
            if e.outcome in (Outcome.CORRECT, Outcome.CORRECT_ABSTENTION)
        ]
        return _safe_div(sum(1 for e in right if e.grounded), len(right))

    @property
    def error_count(self) -> int:
        return self.counts[Outcome.ERROR]

    @property
    def date_errors(self) -> list[CaseEvaluation]:
        """Cases where the denial date was wrong or never extracted.

        Reported separately from the verdict outcome because the two are
        independent: a case whose date extraction failed can still reach the
        right verdict by luck, and that is not the same as the temporal
        layer working. Any entry here invalidates the temporal claim
        regardless of how the accuracy line reads.
        """
        return [e for e in self.evaluations if e.date_error]

    @property
    def correct_abstention_rate(self) -> float | None:
        """Of genuinely ambiguous cases, how often the system declined to rule.

        The headline abstention metric. Its denominator is the ambiguous
        subset only — computing it over the whole set would let a system that
        abstains on everything score well. Crashed cases are excluded: a
        crash is not a decision to abstain, and counting it as one would let
        a broken run read as well-calibrated.
        """
        ambiguous = [
            e
            for e in self.evaluations
            if e.category is Category.AMBIGUOUS and e.outcome is not Outcome.ERROR
        ]
        return _safe_div(
            sum(1 for e in ambiguous if e.predicted in NON_ASSERTIONS),
            len(ambiguous),
        )

    @property
    def over_abstention_rate(self) -> float | None:
        """Of cases the evidence settled, how often the system declined anyway.

        The cost side of the abstention trade. Read it next to
        `correct_abstention_rate`: a system can max one at the other's
        expense, and the pair is what shows the calibration.
        """
        decisive = [
            e
            for e in self.evaluations
            if e.category is not Category.AMBIGUOUS
            and e.outcome is not Outcome.ERROR
        ]
        return _safe_div(
            sum(1 for e in decisive if e.predicted in NON_ASSERTIONS),
            len(decisive),
        )

    @property
    def contested_rate(self) -> float | None:
        """How often the system had a leaning it could not substantiate.

        Distinct from plain abstention: a contested result means the ensemble
        or the critique pass caught a disagreement the system could not
        resolve. Tracking it separately is what shows the confidence gate is
        doing work rather than sitting inert.
        """
        scored = [e for e in self.evaluations if e.outcome is not Outcome.ERROR]
        return _safe_div(
            sum(1 for e in scored if e.predicted == CONTESTED), len(scored)
        )

    @property
    def false_assurance_rate(self) -> float | None:
        """The number to lead with. Denominator is the whole set."""
        return _safe_div(self.counts[Outcome.FALSE_ASSURANCE], self.total)

    def per_class(self, label: str) -> dict[str, float | None]:
        """Precision / recall / F1 for one finding label."""
        tp = sum(
            1
            for e in self.evaluations
            if e.predicted == label and e.expected == label
        )
        fp = sum(
            1
            for e in self.evaluations
            if e.predicted == label and e.expected != label
        )
        fn = sum(
            1
            for e in self.evaluations
            if e.predicted != label and e.expected == label
        )
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall
            else None
        )
        return {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    def summary(self) -> str:
        def pct(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.1%}"

        lines = [
            f"Cases evaluated: {self.total}",
            "",
            "Consequence-weighted outcomes (worst first):",
        ]
        for outcome in SEVERITY:
            count = self.counts[outcome]
            if count:
                lines.append(f"  {outcome.value:<20} {count:>3}")

        lines += [
            "",
            f"False assurance rate   {pct(self.false_assurance_rate)}   "
            f"(told a winnable denial was justified)",
            f"Correct abstention     {pct(self.correct_abstention_rate)}   "
            f"(declined to rule on ambiguous cases)",
            f"Over-abstention        {pct(self.over_abstention_rate)}   "
            f"(declined where evidence was clear)",
            f"Contested              {pct(self.contested_rate)}   "
            f"(had a leaning it could not substantiate)",
            f"Accuracy               {pct(self.accuracy)}",
            f"Grounding accuracy     {pct(self.grounding_accuracy)}   "
            f"(correct verdicts resting on the required evidence)",
        ]
        if self.error_count:
            lines.append(
                f"\n!! {self.error_count} case(s) errored - every rate above "
                f"excludes them; fix before quoting any figure."
            )
        if self.date_errors:
            lines.append(
                f"\n!! {len(self.date_errors)} case(s) with denial-date "
                f"extraction problems - temporal filtering ran against the "
                f"wrong law, so any statute citation is unreliable:"
            )
            for e in self.date_errors:
                lines.append(f"     {e.case_id}: {e.date_error}")
        lines += ["", "Per-class:"]
        for label in (*DECISIVE, CONTESTED, ABSTAIN):
            m = self.per_class(label)
            lines.append(
                f"  {label:<14} P={pct(m['precision'])} R={pct(m['recall'])} "
                f"F1={pct(m['f1'])} n={m['support']}"
            )
        return "\n".join(lines)
