"""RAGAS adapter — retrieval-quality metrics alongside the verdict metrics.

RAGAS measures a different thing from `metrics.py` and the two are not
substitutes. The golden-set metrics ask "was the verdict right, and did it
rest on the right evidence." RAGAS asks "was the retrieved context relevant,
and is the generated text faithful to it." A system can score well on
faithfulness while being confidently wrong about the law, so RAGAS is
reported as a diagnostic for the retrieval stack, not as the headline.

Imported lazily and installed via the `eval` extra: RAGAS pulls a large
dependency tree, and its default metrics make their own LLM calls, so it must
not be a prerequisite for running the core suite.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.agents.pipeline import AuditResult


@dataclass
class RagasSample:
    """One row in the RAGAS evaluation dataset."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None


def to_samples(
    result: AuditResult, ground_truth: str | None = None
) -> list[RagasSample]:
    """Flatten an audit into one RAGAS sample per sub-claim.

    Sub-claim granularity rather than case granularity is deliberate: a case
    with one well-grounded verdict and one hallucinated one should not
    average into a mid-range faithfulness score that hides both.
    """
    samples: list[RagasSample] = []
    for sub in result.results:
        verdict = sub.verdict
        contexts = [sc.chunk.text for sc in verdict.retrieval_trace]
        if not contexts:
            # RAGAS metrics divide by context count; an empty context list
            # produces NaN rather than a zero. Skip and let the verdict
            # metrics account for it.
            continue
        samples.append(
            RagasSample(
                question=next(
                    (
                        c.supports
                        for c in verdict.citations
                    ),
                    verdict.rationale,
                ),
                answer=verdict.rationale,
                contexts=contexts,
                ground_truth=ground_truth,
            )
        )
    return samples


def evaluate(samples: list[RagasSample], metrics: list | None = None):
    """Run RAGAS over the samples. Requires `pip install -e ".[eval]"`.

    Returns the RAGAS result object unchanged — wrapping it would hide
    per-metric detail that is the reason to run RAGAS at all.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "RAGAS metrics require the eval extra: "
            'pip install -e ".[eval]"'
        ) from exc

    if metrics is None:
        metrics = [faithfulness, answer_relevancy, context_precision]
        # context_recall needs a reference answer; including it without one
        # yields NaN for every row.
        if all(s.ground_truth for s in samples):
            metrics.append(context_recall)

    dataset = Dataset.from_dict(
        {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
            **(
                {"ground_truth": [s.ground_truth for s in samples]}
                if all(s.ground_truth for s in samples)
                else {}
            ),
        }
    )
    return ragas_evaluate(dataset, metrics=metrics)
