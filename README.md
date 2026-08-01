# Redress

**A multi-pipeline RAG reconciliation engine for verifying insurance claim denials against policy language, state law, and regulatory precedent.**

A denial letter cites a reason. Almost nobody can independently check whether that reason is actually supported by the policy they signed, the state law that governs it, and how regulators have historically treated similar denials. This builds that check.

Redress ingests a policyholder's own policy document and their denial letter, decomposes the denial into atomic sub-claims, retrieves evidence for each one through four independent pipelines, and reconciles them against each other — citing the exact clause, statute, or precedent that supports or contradicts the denial.

---

## What makes this a reconciliation system, not a chatbot

The design constraint that shapes everything: **a wrong answer here costs someone real money at a moment they can't afford it.** So the system is built to fail safe rather than to sound confident.

- **It never asserts a contradiction it cannot cite.** Every claim in a verdict is tied to a verbatim quote from a retrieved chunk. A critique agent re-reads the draft verdict against the actual retrieved text and rejects any citation that doesn't say what the draft claims.
- **It abstains.** Every verdict is labelled `Supported`, `Contested`, or `Insufficient Evidence`. The third is a first-class outcome, not a fallback — and the abstention rate is a headline metric, not something swept under the rug.
- **It shows its work.** Every verdict stores its full retrieval trace: which chunks, which scores at which stage, and both agent passes. A result you can't inspect is a result you have no reason to trust.

---

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Core hybrid retrieval + reranking + temporal filtering | **In progress** — retrieval stack built and tested |
| 2 | Reconciliation agent v1 (single-pass verdict with citations) | Not started |
| 3 | Adversarial critique pass + three-way confidence gate | Not started |
| 4 | GraphRAG layer (Neo4j) + regulation versioning | Not started |
| 5 | Eval harness — golden dataset, RAGAS, precision/recall | Not started |
| 6 | Frontend — the Forensic Ledger UI | Not started |

Built so far: the retrieval stack (`backend/app/retrieval/`) and the shared domain model (`backend/app/core/models.py`), with 17 tests covering the failure modes that matter.

---

## The retrieval stack

```
query
  ├── dense (bi-encoder, cosine)      ──┐
  └── lexical (BM25)                  ──┤
                                        ├── reciprocal rank fusion (k=60)
                                        │
                                        ├── temporal filter (law as of denial date)
                                        │
                                        └── cross-encoder rerank ──> top-k + trace
```

Four decisions worth explaining, because each one is load-bearing:

**Lexical retrieval is not optional.** Insurance policies define terms of art — *Custodial Care*, *Medically Necessary*, *Adverse Benefit Determination* — whose defined meaning is unrelated to their plain-English embedding neighbourhood. A dense retriever asked about custodial care returns the nursing-home benefits section. Only exact-term matching finds the definitions clause that actually governs it.

**Fusion happens on rank, not score.** BM25 returns unbounded positive scores; cosine returns `[-1, 1]`. Normalising them into a shared range means inventing a mapping with no principled basis that shifts whenever the corpus changes. RRF only asks *where* each retriever ranked a chunk, not how confident it claimed to be.

**Cross-encoder reranking is where precision comes from.** "This exclusion does not apply to emergency services" and "This exclusion applies to emergency services" sit almost on top of each other in embedding space. A bi-encoder embeds query and chunk separately and can never model their interaction; a cross-encoder reads them jointly and tells them apart. Adjacent, near-identical, opposite-in-effect clauses are the single largest source of retrieval error in this domain — [`test_disambiguates_near_identical_opposing_clauses`](backend/tests/test_retrieval.py) pins exactly that case.

**Temporal filtering sits between fusion and reranking.** State insurance regulations change. A denial dated 2021 must be judged against the statute text in force in 2021, not the current amendment. Returning the current version would be the most dangerous bug in the system: a fluent, well-cited verdict resting on a law that hadn't been written yet. Filtering before retrieval would mean re-querying both indexes per denial date; filtering after reranking would mean paying cross-encoder cost on already-disqualified chunks.

---

## The four pipelines

| | Source | Technique |
|---|---|---|
| **A** | The user's own policy document | Hybrid retrieval + cross-encoder reranking |
| **B** | The denial letter | Structured extraction — reason codes, cited clauses, factual assertions |
| **C** | State insurance code | RAG with temporal versioning by effective-date range |
| **D** | State DOI complaint & enforcement records | Precedent retrieval — has this denial reason been overturned before |
| **Graph** | `clause → reason code → statute → complaint outcome` | Neo4j — enables "does this insurer have a pattern", not just "is this denial justified" |

---

## Running it

```bash
cd backend && python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"
```

```bash
cd backend && .venv/Scripts/python -m pytest
```

The retrieval logic and its full test suite run **without torch** — heavy ML backends are lazily imported and live behind the `ml` extra. Tests use a deterministic hashed-bag-of-words embedder, so retrieval behaviour is reproducible across machines and CI needs no GPU.

To install the real embedding and reranking models:

```bash
cd backend && .venv/Scripts/python -m pip install -e ".[ml]"
```

---

## Stack

FastAPI · LangGraph · Qdrant/pgvector · BM25 · `bge-reranker` · Neo4j · PostgreSQL · React + TypeScript · Three.js · RAGAS

---

## Disclaimer

Redress is an evidence-retrieval and cross-referencing tool. It is **not legal advice** and is not a substitute for a licensed attorney or a state insurance regulator. Its most important output is often "insufficient evidence — seek professional review," and that verdict should be taken at face value.
