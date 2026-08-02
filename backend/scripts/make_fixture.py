"""Run the pipeline over golden cases and dump the API payload as JSON.

The frontend is built against this output rather than invented data, so the
UI is shaped by what the system actually produces — including the cases it
declines to rule on, which is most of them. A demo built only against a
triumphant CONTRADICTED verdict would look broken in normal operation.

    python scripts/make_fixture.py ca-emergency-carveback ca-competing-clauses
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.critique import CritiqueAgent  # noqa: E402
from app.agents.decomposition import DecompositionAgent  # noqa: E402
from app.agents.gate import GatedAdjudicator  # noqa: E402
from app.agents.ollama_llm import OllamaEmbedder, OllamaStructuredLLM  # noqa: E402
from app.agents.pipeline import AuditPipeline  # noqa: E402
from app.agents.reconciliation import ReconciliationAgent  # noqa: E402
from app.api.schemas import serialise  # noqa: E402
from app.eval.dataset import load_dataset  # noqa: E402
from app.eval.harness import _to_chunk  # noqa: E402
from app.retrieval.bm25 import BM25Index  # noqa: E402
from app.retrieval.dense import DenseIndex  # noqa: E402
from app.retrieval.hybrid import HybridRetriever  # noqa: E402

GOLDEN = Path(__file__).resolve().parents[2] / "data" / "golden"
OUT = Path(__file__).resolve().parents[2] / "frontend" / "src" / "fixtures"


def main(case_ids: list[str], model: str = "qwen2.5:7b", think: bool = False) -> int:
    cases = {c.id: c for c in load_dataset(GOLDEN)}
    OUT.mkdir(parents=True, exist_ok=True)

    for case_id in case_ids:
        case = cases.get(case_id)
        if case is None:
            print(f"unknown case {case_id!r}", file=sys.stderr)
            return 1

        llm = OllamaStructuredLLM(model=model, think=think)
        chunks = [_to_chunk(c, case.id) for c in case.chunks]
        pipeline = AuditPipeline(
            decomposition=DecompositionAgent(llm),
            adjudicator=GatedAdjudicator(
                ReconciliationAgent(llm), CritiqueAgent(llm)
            ),
            retrievers={
                "case": HybridRetriever(
                    dense=DenseIndex(chunks, OllamaEmbedder()),
                    lexical=BM25Index(chunks),
                )
            },
        )

        print(f"running {case_id} ...", flush=True)
        result = pipeline.run(
            case.denial_letter,
            insurer_id=case.insurer_id,
            statute_ids=case.statute_ids,
        )
        payload = serialise(result, case.id, case.denial_letter)

        path = OUT / f"{case_id}.json"
        path.write_text(
            payload.model_dump_json(indent=2, exclude_none=False), encoding="utf-8"
        )
        anchored = sum(1 for s in payload.sub_claims if s.source_span)
        print(
            f"  -> {path.name}  disposition={payload.disposition} "
            f"sub_claims={len(payload.sub_claims)} anchored={anchored} "
            f"evidence={len(payload.evidence)}"
        )
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    model = next((a.split("=", 1)[1] for a in flags if a.startswith("--model=")), "qwen2.5:7b")
    raise SystemExit(main(args or ["ca-emergency-carveback"], model, "--think" in flags))
