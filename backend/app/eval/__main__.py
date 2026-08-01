"""Run the eval harness from the command line.

    python -m app.eval --dataset ../data/golden
    python -m app.eval --dataset ../data/golden --ablate

This is the only entry point in the project that makes real API calls, and it
makes a lot of them: roughly (2 + sub-claims x 3) requests per case, times the
number of ablation arms. It prints the projected call count and waits for
confirmation before spending anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.agents.llm import AnthropicStructuredLLM
from app.eval.dataset import load_dataset
from app.eval.harness import AblationConfig, EvalHarness, ablation_table
from app.retrieval.dense import SentenceTransformerEmbedder
from app.retrieval.reranker import CrossEncoderReranker

ABLATION_ARMS = [
    AblationConfig(name="full"),
    AblationConfig(name="no-reranker", use_reranker=False),
    AblationConfig(name="no-critique", use_critique=False),
    AblationConfig(name="no-ensemble", use_ensemble=False),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.eval")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "data" / "golden",
    )
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--effort", default="high")
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="run every ablation arm and print the comparison table",
    )
    parser.add_argument(
        "--yes", action="store_true", help="skip the cost confirmation prompt"
    )
    args = parser.parse_args(argv)

    cases = load_dataset(args.dataset)
    arms = ABLATION_ARMS if args.ablate else [AblationConfig()]

    # Rough, and deliberately an over-estimate: better to quote a number the
    # run comes in under than to surprise someone with an API bill.
    per_case = 2 + 3 * 4
    print(
        f"{len(cases)} cases x {len(arms)} arm(s) "
        f"~= {len(cases) * len(arms) * per_case} model calls on {args.model}."
    )
    if not args.yes:
        if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return 1

    harness = EvalHarness(
        llm_factory=lambda: AnthropicStructuredLLM(
            model=args.model, effort=args.effort
        ),
        embedder_factory=lambda: SentenceTransformerEmbedder(
            "BAAI/bge-small-en-v1.5"
        ),
        reranker_factory=lambda: CrossEncoderReranker(),
        secondary_embedder_factory=lambda: SentenceTransformerEmbedder(
            # A second family, not a second checkpoint of the same one:
            # two models that share a training lineage agree on the same
            # mistakes, which would make the ensemble check decorative.
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
    )

    if args.ablate:
        reports = harness.ablate(cases, arms)
        print()
        print(ablation_table(reports))
        print()
        print(reports["full"].summary())
    else:
        report = harness.run(cases)
        print()
        print(report.summary())
        failures = [e for e in report.evaluations if e.error]
        if failures:
            print(f"\n{len(failures)} case(s) errored:")
            for e in failures:
                print(f"  {e.case_id}: {e.error}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
