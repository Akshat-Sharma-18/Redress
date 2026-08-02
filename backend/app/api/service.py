"""Assemble and run an audit over uploaded documents.

`scripts/make_fixture.py` already does this for golden cases: build the
retrievers, build the pipeline, run it, serialise. This module does the same
thing for documents that arrived over HTTP, and the symmetry is the point —
an uploaded policy is indexed by the same hybrid retriever, adjudicated by
the same gate, and serialised by the same `serialise()` as the fixtures the
eval harness scores. If those two paths diverged, the measured numbers would
stop describing the product.

One structural difference from the fixture script: policy and statute text
are indexed as **separate corpora** rather than one pooled index. That is
what makes `per_corpus_k` meaningful — six chunks are retrieved from the
policy *and* six from the statutes, so a governing statute cannot be crowded
out of the evidence pack by a policy that happens to use more of the query's
vocabulary.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

from app.agents.critique import CritiqueAgent
from app.agents.decomposition import DecompositionAgent
from app.agents.gate import GatedAdjudicator
from app.agents.ollama_llm import (
    DEFAULT_MODEL as OLLAMA_DEFAULT_MODEL,
)
from app.agents.ollama_llm import OllamaEmbedder, OllamaStructuredLLM
from app.agents.pipeline import AuditPipeline
from app.agents.reconciliation import ReconciliationAgent
from app.api.schemas import AuditOut, serialise
from app.core.models import Chunk, SourceKind
from app.ingestion import ExtractedDocument, chunk_document
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex
from app.retrieval.hybrid import HybridRetriever

DEFAULT_MODEL = os.environ.get("REDRESS_MODEL", OLLAMA_DEFAULT_MODEL)
DEFAULT_EMBED_MODEL = os.environ.get("REDRESS_EMBED_MODEL", "nomic-embed-text")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
#: Reasoning mode roughly quadruples latency for a schema-constrained answer.
#: Off unless deliberately enabled.
DEFAULT_THINK = os.environ.get("REDRESS_THINK", "").lower() in ("1", "true", "yes")


class NoEvidenceError(ValueError):
    """Raised when the uploaded documents yield nothing to retrieve against."""


@dataclass
class AuditRequest:
    """One audit's inputs, already extracted to canonical text."""

    denial: ExtractedDocument
    policies: list[ExtractedDocument] = field(default_factory=list)
    statutes: list[ExtractedDocument] = field(default_factory=list)
    insurer_id: str | None = None
    case_id: str = "uploaded"


def _corpus(
    documents: list[ExtractedDocument], kind: SourceKind, prefix: str
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for index, document in enumerate(documents):
        document_id = f"{prefix}-{index + 1}"
        chunks.extend(
            chunk_document(document, source_kind=kind, document_id=document_id)
        )

    # Chunk ids are unique per document (the chunker guarantees that), but two
    # uploaded policies can both contain a "Section 4.3". Citations are
    # resolved by chunk id, so a collision across documents would let a
    # citation land on text from the wrong file.
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.id in seen:
            suffix = 2
            while f"{chunk.id}-{suffix}" in seen:
                suffix += 1
            chunk.id = f"{chunk.id}-{suffix}"
        seen.add(chunk.id)
    return chunks


def build_pipeline(
    request: AuditRequest,
    *,
    model: str = DEFAULT_MODEL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    host: str = OLLAMA_HOST,
    think: bool = DEFAULT_THINK,
) -> tuple[AuditPipeline, dict[str, int]]:
    """Build a pipeline over the request's documents.

    Returns the pipeline and a per-corpus chunk count, which the API reports
    back so a user whose 40-page policy produced three chunks can see that
    the extraction failed rather than wondering why the verdict is thin.
    """
    policy_chunks = _corpus(request.policies, SourceKind.POLICY, "policy")
    statute_chunks = _corpus(request.statutes, SourceKind.STATUTE, "statute")

    if not policy_chunks and not statute_chunks:
        raise NoEvidenceError(
            "no readable text was found in the uploaded policy documents; "
            "a denial letter alone cannot be audited, because there is "
            "nothing to check it against"
        )

    llm = OllamaStructuredLLM(model=model, host=host, think=think)

    retrievers: dict[str, HybridRetriever] = {}
    for name, chunks in (("policy", policy_chunks), ("statute", statute_chunks)):
        if not chunks:
            continue
        retrievers[name] = HybridRetriever(
            dense=DenseIndex(chunks, OllamaEmbedder(model=embed_model, host=host)),
            lexical=BM25Index(chunks),
        )

    pipeline = AuditPipeline(
        decomposition=DecompositionAgent(llm),
        adjudicator=GatedAdjudicator(ReconciliationAgent(llm), CritiqueAgent(llm)),
        retrievers=retrievers,
    )
    counts = {"policy": len(policy_chunks), "statute": len(statute_chunks)}
    return pipeline, counts


def run_audit(
    request: AuditRequest,
    *,
    on_progress: Callable[[str, dict], None] | None = None,
    **build_kwargs,
) -> AuditOut:
    """Run one audit end to end and return the frontend wire format."""
    pipeline, counts = build_pipeline(request, **build_kwargs)
    if on_progress:
        on_progress("indexed", counts)

    result = pipeline.run(
        request.denial.text,
        insurer_id=request.insurer_id,
        on_progress=on_progress,
    )
    return serialise(result, request.case_id, request.denial.text)
