"""Run the eval harness from the command line.

    python -m app.eval                          # local models via Ollama, free
    python -m app.eval --backend anthropic      # hosted, needs a key
    python -m app.eval --ablate                 # every ablation arm

Defaults to Ollama so the project runs end to end with no API key and no
cost. The Anthropic backend is opt-in, and only that path prints a spend
confirmation — there is nothing to confirm when inference is local.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.eval.dataset import load_dataset
from app.eval.harness import AblationConfig, EvalHarness, ablation_table
from app.eval.metrics import Outcome, Report
from app.eval.stability import RepeatedReport


def per_case_table(report: Report) -> str:
    """Per-case results, worst outcome first.

    A run over the full set takes long enough that an aggregate-only report
    is close to useless: the rates tell you something is wrong without
    telling you where. Sorted by severity so the false assurances -- the
    failures that cost a user money -- are the first thing on screen.
    """
    order = {
        o: i
        for i, o in enumerate(
            [
                Outcome.FALSE_ASSURANCE,
                Outcome.UNDER_ABSTENTION,
                Outcome.FALSE_ALARM,
                Outcome.DIRECTION_ERROR,
                Outcome.ERROR,
                Outcome.OVER_ABSTENTION,
                Outcome.CORRECT_ABSTENTION,
                Outcome.CORRECT,
            ]
        )
    }
    rows = sorted(report.evaluations, key=lambda e: (order[e.outcome], e.case_id))
    lines = [f"{'case':<34} {'expected':<13} {'got':<13} outcome", "-" * 82]
    for e in rows:
        flag = "" if e.outcome in (Outcome.CORRECT, Outcome.CORRECT_ABSTENTION) else "*"
        lines.append(
            f"{e.case_id:<34} {e.expected:<13} {e.predicted:<13} "
            f"{flag}{e.outcome.value}"
        )
    return "\n".join(lines)


def _serialisable(report: Report) -> dict:
    return {
        "cases": [
            {
                "case_id": e.case_id,
                "category": e.category.value,
                "expected": e.expected,
                "predicted": e.predicted,
                "outcome": e.outcome.value,
                "cited": sorted(e.cited),
                "missing_required_citations": sorted(e.missing_required_citations),
                "forbidden_citations": sorted(e.forbidden_citations),
                "date_error": e.date_error,
                "error": e.error,
            }
            for e in report.evaluations
        ],
        "totals": {
            "n": report.total,
            "accuracy": report.accuracy,
            "false_assurance_rate": report.false_assurance_rate,
            "correct_abstention_rate": report.correct_abstention_rate,
            "over_abstention_rate": report.over_abstention_rate,
            "contested_rate": report.contested_rate,
            "grounding_accuracy": report.grounding_accuracy,
            "errors": report.error_count,
        },
    }

ABLATION_ARMS = [
    AblationConfig(name="full"),
    AblationConfig(name="no-reranker", use_reranker=False),
    AblationConfig(name="no-critique", use_critique=False),
    AblationConfig(name="no-ensemble", use_ensemble=False),
    AblationConfig(
        name="symmetric-assurance", use_asymmetric_assurance=False
    ),
]


def _ollama_harness(args) -> EvalHarness:
    from app.agents.ollama_llm import (
        OllamaEmbedder,
        OllamaStructuredLLM,
        available_models,
    )

    installed = available_models(args.host)
    # Ollama matches a bare name to its ':latest' tag, so compare on the stem.
    stems = {m.split(":")[0] for m in installed}
    for wanted in (args.model, args.embed_model):
        if wanted.split(":")[0] not in stems:
            raise SystemExit(
                f"model {wanted!r} is not pulled. Available: "
                f"{', '.join(sorted(installed)) or '(none)'}\n"
                f"Run:  ollama pull {wanted}"
            )

    return EvalHarness(
        llm_factory=lambda: OllamaStructuredLLM(
            model=args.model, host=args.host, think=args.think
        ),
        embedder_factory=lambda: OllamaEmbedder(args.embed_model, host=args.host),
        # No cross-encoder reranker locally: a second model would double an
        # already-slow run for a gain the reranker ablation can measure later.
        reranker_factory=None,
        # The ensemble needs a genuinely different embedder to be meaningful.
        # Reusing one model would make the two passes agree by construction
        # and turn the cross-check into decoration, so it stays off unless a
        # second embedding model is named explicitly.
        secondary_embedder_factory=(
            (lambda: OllamaEmbedder(args.embed_model_2, host=args.host))
            if args.embed_model_2
            else None
        ),
    )


def _anthropic_harness(args) -> EvalHarness:
    from app.agents.llm import AnthropicStructuredLLM
    from app.retrieval.dense import SentenceTransformerEmbedder
    from app.retrieval.reranker import CrossEncoderReranker

    return EvalHarness(
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.eval")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "data" / "golden",
    )
    parser.add_argument(
        "--backend", choices=("ollama", "anthropic"), default="ollama"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="default: qwen2.5:7b (ollama) or claude-opus-5 (anthropic)",
    )
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument(
        "--embed-model-2",
        default=None,
        help="second embedding model; enables the ensemble cross-check",
    )
    parser.add_argument("--effort", default="high", help="anthropic backend only")
    parser.add_argument(
        "--think",
        action="store_true",
        help="enable reasoning mode on models that support it (much slower)",
    )
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="run every ablation arm and print the comparison table",
    )
    parser.add_argument(
        "--yes", action="store_true", help="skip the spend confirmation prompt"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "run the whole set N times and report the mean and spread. "
            "Greedy decoding is not reproducible when a model is split "
            "across GPU and CPU, and the variance has measured wider than "
            "the differences being compared; use this before believing one"
        ),
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="write per-case results as JSON, so a long run stays inspectable",
    )
    args = parser.parse_args(argv)

    if args.model is None:
        if args.backend == "ollama":
            # Imported here, like the harness below, so the Anthropic path
            # never pays for a module it does not use. Sourced from the
            # backend rather than repeated, so the default cannot drift.
            from app.agents.ollama_llm import DEFAULT_MODEL

            args.model = DEFAULT_MODEL
        else:
            args.model = "claude-opus-5"

    cases = load_dataset(args.dataset)
    arms = ABLATION_ARMS if args.ablate else [AblationConfig()]

    # Deliberately an over-estimate: better to quote a number the run comes
    # in under than to surprise someone with a bill or a wait.
    calls = len(cases) * len(arms) * (2 + 3 * 4)
    print(f"{len(cases)} cases x {len(arms)} arm(s) ~= {calls} calls, {args.model}")

    if args.backend == "anthropic":
        if not args.yes:
            if input("This spends money. Proceed? [y/N] ").strip().lower() not in (
                "y",
                "yes",
            ):
                print("Aborted.")
                return 1
        harness = _anthropic_harness(args)
    else:
        print("Local inference - no API key, no cost. This will be slow.")
        harness = _ollama_harness(args)

    if args.ablate:
        reports = harness.ablate(cases, arms)
        print()
        print(ablation_table(reports))
        print()
        print(reports["full"].summary())
    elif args.repeat > 1:
        runs: list[Report] = []
        for index in range(args.repeat):
            print(f"\n--- run {index + 1} of {args.repeat} ---", flush=True)
            run = harness.run(cases)
            runs.append(run)
            # Printed per run rather than only at the end, because these runs
            # take half an hour each and an interrupted repeat should still
            # leave something readable behind.
            print(run.summary())

        repeated = RepeatedReport(runs)
        print()
        print(repeated.summary())

        if args.save:
            args.save.write_text(
                json.dumps(
                    {
                        "repeat": args.repeat,
                        "noise_floor": repeated.noise_floor,
                        "unstable_cases": {
                            case_id: dict(counts)
                            for case_id, counts in repeated.unstable_cases.items()
                        },
                        "runs": [_serialisable(r) for r in runs],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"\nwrote {args.save}")
    else:
        report = harness.run(cases)
        print()
        print(per_case_table(report))
        print()
        print(report.summary())
        failures = [e for e in report.evaluations if e.error]
        if failures:
            print(f"\n{len(failures)} case(s) errored:")
            for e in failures:
                print(f"  {e.case_id}: {e.error}")
        if args.save:
            args.save.write_text(
                json.dumps(_serialisable(report), indent=2), encoding="utf-8"
            )
            print(f"\nwrote {args.save}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
