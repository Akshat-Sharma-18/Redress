# Eval runs

Per-case output from `python -m app.eval --save`, kept because a full run on
local hardware costs 30–40 minutes and the per-case detail is what makes a
rate actionable. An aggregate says something is wrong; `cases[]` says where.

Both runs below are the **full 35-case golden set**, back to back on the same
code, same embedder (`nomic-embed-text`), reasoning mode off, ensemble off
(it needs a second embedding model — see `--embed-model-2`).

| | `qwen2.5:7b` | `qwen3.5:9b` |
|---|---|---|
| **False assurance** | **5.7%** (2) | 11.4% (4) |
| Accuracy | **42.9%** | 34.3% |
| Over-abstention | **48.1%** | 59.3% |
| Correct abstention | 62.5% | **75.0%** |
| Grounding | 93.3% | **100%** |
| `justified` P/R | **55.6% / 50.0%** | 16.7% / 10.0% |
| `contradicted` P/R | 62.5% / **29.4%** | **71.4%** / 29.4% |
| Denial-date failures | **0** | 4 |
| Speed | **~40 tok/s** | 23.6 tok/s |

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

## Not measured

- **`qwen3.6:27b`** — 4/4 on a four-case probe and nothing else, which by the
  paragraph above means nothing yet. At 2.4 tok/s on 8 GB VRAM a full run is
  an overnight job. It is the outstanding measurement most likely to change
  the default. It also paraphrases sub-claims instead of quoting them, so
  `source_span` comes back null and the citation beam loses its origin.
- **Reasoning mode** (`--think`, `REDRESS_THINK=1`) on any model. Both of the
  9B's spot-check failures were over-abstentions, which is what reasoning
  mode is supposed to help; untested at full-set scale.
- **The ensemble.** `contested` is 0.0% in both runs because the cross-check
  never ran, so that row is a gap in the measurement, not a finding.
