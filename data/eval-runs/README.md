# Eval runs

Per-case output from `python -m app.eval --save`, kept because a full run on
local hardware costs 30–40 minutes and the per-case detail is what makes a
rate actionable. An aggregate says something is wrong; `cases[]` says where.

Both runs below are the **full 35-case golden set**, back to back on the same
code, same embedder (`nomic-embed-text`), reasoning mode off, ensemble off
(it needs a second embedding model — see `--embed-model-2`).

| | `qwen2.5:7b` | `qwen3.5:9b` | `gpt-oss:20b` (low) |
|---|---|---|---|
| **False assurance** | **5.7%** (2) | 11.4% (4) | 22.9% (8) |
| Accuracy | **42.9%** | 34.3% | **42.9%** |
| Over-abstention | 48.1% | 59.3% | **22.2%** |
| Correct abstention | 62.5% | **75.0%** | 25.0% |
| Grounding | 93.3% | **100%** | 93.3% |
| `justified` P/R | **55.6% / 50.0%** | 16.7% / 10.0% | 40.0% / 80.0% |
| `contradicted` P/R | 62.5% / 29.4% | **71.4%** / 29.4% | **71.4%** / 29.4% |
| Denial-date failures | **0** | 4 | 0 |
| Speed | **~40 tok/s** | 23.6 tok/s | 26.5 tok/s |
| Audit latency | ~35s | 66s | ~130s |

`gpt-oss:20b` is the instructive one. It ties the 7B on accuracy and gets there
by a route that is much worse here: it abstains least of the three (22.2%
over-abstention, the best figure in that row) and pays for it with four times
the false assurance. Its `justified` recall is 80% against 40% precision — it
says "the denial was justified" about twice as often as it should. For a
general assistant that is a good trade. For this system it is the worst one
available.

`qwen2.5:7b` is the default because of the first row. False assurance is the
only outcome that costs a user money — it tells someone their winnable denial
was justified, so they don't appeal. Every other column is a preference; that
one is the product's reason for existing.

## Read this before quoting any of it

**Neither model is good enough to put a verdict in front of a person.** 34–43%
accuracy against a ~48.6% baseline of always guessing `contradicted`, with a
false-assurance rate between one-in-eighteen and one-in-nine. What the numbers
support is that the *architecture* fails safe — grounding is 93–100%, so when
it is right it is right for the stated reason — not that the answers are
usable.

**Always name the model beside the number.** These two differ by 8 accuracy
points and 2× on false assurance from a single component swap.

**A spot check is not a model comparison.** `qwen3.5:9b` won a four-case
probe 2/4 against the 7B's 0/4 and lost the full set on every
consequence-weighted metric. This is the second time small-sample evidence has
reversed here; the first was a 5-case run reporting 0% false assurance that
became 5.7% at 35 cases. Below the full golden set, the noise is larger than
the effect.

## A model can fail for reasons that have nothing to do with its ability

`gpt-oss:20b`'s first full run scored 0% — 35/35 cases errored, every one at
decomposition. None of it was the model. gpt-oss reasons unconditionally and
answers through a channel separate from its reasoning, so `think: false` does
not produce a terse answer, it produces an *empty* one, and schema validation
fails on the empty string every time.

Reported at face value that would have read as "gpt-oss cannot follow a
schema". The real finding was a default flag it was never able to honour. See
`REASONING_ONLY_PREFIXES` in `app/agents/ollama_llm.py`; an empty response now
raises an error that names this cause instead of surfacing as a validation
failure.

Effort levels, measured on gpt-oss:20b at 8 GB VRAM — all three emit valid
JSON, so the choice is latency against reasoning depth:

| effort | per call | thinking tokens | audit latency |
|---|---|---|---|
| `low` | 9.0s | 7 | ~2 min |
| `medium` | 24.6s | 373 | ~3 min |
| `high` | 51.7s | 1276 | ~13 min |

## Not measured

- **`gpt-oss:20b` at `medium`/`high`.** The full-set number above is `low`,
  its weakest setting. The failure it shows — answering decisively without
  deliberating — is exactly what more reasoning might fix, so 22.9% disqualifies
  *gpt-oss at low effort*, not gpt-oss.
- **`qwen3.6:27b`** — 4/4 on a four-case probe and nothing else, which by the
  section above means nothing yet. At 2.4 tok/s a full run is an overnight job.
  Two things are confirmed about it on the real upload path: one audit takes
  **499s** (8.3 min, versus 66s for the 9B on the same PDF), and it paraphrases
  sub-claims rather than quoting them, so `source_span` is null and the citation
  beam does not draw. Both sub-claims came back unanchored.
- **Reasoning mode** (`--think`, `REDRESS_THINK=1`) on any model. Both of the
  9B's spot-check failures were over-abstentions, which is what reasoning
  mode is supposed to help; untested at full-set scale.
- **The ensemble.** `contested` is 0.0% in both runs because the cross-check
  never ran, so that row is a gap in the measurement, not a finding.
