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
import sys
from pathlib import Path

from app.eval.dataset import load_dataset
from app.eval.harness import AblationConfig, EvalHarness, ablation_table

ABLATION_ARMS = [
    AblationConfig(name="full"),
    AblationConfig(name="no-reranker", use_reranker=False),
    AblationConfig(name="no-critique", use_critique=False),
    AblationConfig(name="no-ensemble", use_ensemble=False),
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
        llm_factory=lambda: OllamaStructuredLLM(model=args.model, host=args.host),
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
        "--ablate",
        action="store_true",
        help="run every ablation arm and print the comparison table",
    )
    parser.add_argument(
        "--yes", action="store_true", help="skip the spend confirmation prompt"
    )
    args = parser.parse_args(argv)

    if args.model is None:
        args.model = (
            "qwen2.5:7b" if args.backend == "ollama" else "claude-opus-5"
        )

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
