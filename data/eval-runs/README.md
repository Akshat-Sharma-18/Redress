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

## The error bar, measured

Every column above is a single run. `--repeat 3` on `qwen2.5:7b`, same code and
same config three times, gives the size of the error bar around them:

| metric | mean | range over 3 runs |
|---|---|---|
| **False assurance** | **5.7%** | **5.7% – 5.7%** |
| Correct abstention | 62.5% | 62.5% – 62.5% |
| Grounding | 91.9% | 90.0% – 92.9% |
| Over-abstention | 51.9% | 44.4% – 59.3% |
| Accuracy | 36.2% | **28.6% – 40.0%** |

**6 of 35 cases (17.1%) gave more than one answer across identical runs**,
including `ca-emergency-carveback` — the archetypal case this system exists to
solve — which came back `insufficient` twice and `contradicted` once.

The split matters more than the magnitude. **The consequence-weighted metrics
are stable and accuracy is not.** False assurance and correct abstention did
not move at all across three runs; accuracy swung 11.4 points on identical
inputs. That is a vindication of the metric design in `app/eval/metrics.py`:
the quantities it was built to track are the reproducible ones, and the
standard multiclass number everyone reaches for first is the noise.

What follows for the table above:

- **The false-assurance column is real.** 2 vs 4 vs 8 cases sits far outside a
  metric whose measured variance is zero. The model ranking on the row that
  decides the default holds.
- **The accuracy row is not a comparison.** 42.9% vs 34.3% is 8.6 points
  against an 11.4-point error bar. Treat those as tied.
- The asymmetric assurance bar (`750b1a0`) looked like it cost 11 accuracy
  points. Its 31.4% is inside the 28.6–40.0% band of doing nothing at all. It
  is neutral on both metrics, not harmful — the earlier reading was noise.

## Where the instability actually lives

The 6 cases that flipped answer across the 3 repeated runs retrieved
**byte-identical citations every time.** Zero variance in what evidence was
found; all of the instability is in how the model reasons over it.
`ca-emergency-carveback` — a general exclusion plus the specific carve-back
that overrides it, the textbook case this system exists to catch — retrieved
the correct three clauses in all 3 runs and reasoned through their
interaction correctly in exactly 1. Retrieval is not the weak link; adjudication
is.

## The ensemble cross-check does not catch what it was hoped to catch

Ran `qwen2.5:7b` again with a second, architecturally different embedder
(`all-minilm`, 384-dim, alongside `nomic-embed-text`'s 768) so a non-`justified`
finding only ships as SUPPORTED when an independent retrieval pass reaches the
same conclusion (`app/agents/gate.py`, `_ensemble_check`). Result:
**`contested` stayed at 0.0%.** False assurance stayed at 5.7% — the same two
cases, `ca-duplicate-different-dos` and `ca-visit-limit-partial`.

Checked directly rather than assumed: both false-assurance cases reached the
ensemble stage (critique approved the draft in both), and the second
embedder's fully independent adjudication **agreed with the wrong finding.**
The mistake is not in what evidence either embedder surfaces — it is in how
the model reads evidence that both retrieval paths agree on. Two different
embedders looking at the same clause do not disagree about what the clause
says; the reasoning step over it is where the error is introduced, and that
step is identical regardless of which embedder fed it. This matches the
retrieval-stability finding above: the noise, and now the actual failure
mode, both sit in adjudication, not retrieval.

The ensemble check is not wasted work — it is real defense against a
retrieval-caused disagreement, and it is verified to fire correctly (see
`critique_notes: "... ensemble: agreed"` in a live run). It is simply not the
mechanism that stops *this* system's specific failure mode. A second opinion
from a different LLM call (not a different embedder) over the same evidence
is the more likely next lever, since it varies the step that is actually
wrong.

## Numeric-condition check: false assurance halved, at a real cost

Reading the model's own rationale on the two false-assurance cases that had
survived every prior intervention showed one shared bug: the model confirms a
cited clause exists and matches the denial's general category, but never
checks whether the letter's own specific numbers -- dates, visit counts --
actually satisfy the clause's condition. `ca-duplicate-different-dos` cites a
"duplicate claim" rule that requires the *same* date of service, while the
letter's own text gives two different dates. `ca-visit-limit-partial` cites a
twenty-visit cap to deny visits 18-22, when only 21-22 exceed it. In the
second case the bug was one stage upstream: decomposition had dropped the
visit numbers from the sub-claim entirely, so reconciliation never had the
fact to check.

Added a rule to `reconciliation.py`, `critique.py`, and `decomposition.py`
requiring a numeric condition to be checked against the sub-claim's own
stated facts, and requiring decomposition to keep those facts in the
sub-claim's text rather than only the clause's general category. Measured
with `--repeat 3`, both models and configurations identical to the "error
bar" run above:

| metric | before | after |
|---|---|---|
| **False assurance** | 5.7% (5.7-5.7%) | **2.9% (2.9-2.9%)** |
| Accuracy | 36.2% (28.6-40.0%) | 27.6% (25.7-31.4%) |
| Over-abstention | 51.9% (44.4-59.3%) | 65.4% (63.0-66.7%) |
| Correct abstention | 62.5% | 50.0% |
| Grounding | 91.9% (90.0-92.9%) | 100.0% |

Both target cases moved from `justified` to `insufficient` and stayed there
across all 3 runs -- a reproducible fix, not noise (zero variance on false
assurance both before and after). But the rule generalized further than
intended: several previously-correct `contradicted` verdicts became
over-abstentions, and accuracy dropped by more than the fix's own effect on
the two target cases can explain. The instruction to verify numbers appears
to have made the model warier across the board, not only where a number was
actually being glossed over.

**A new, distinct bug surfaced along the way, unrelated to this fix.**
`ca-step-therapy-prior-failure` became a new false assurance, stable across
all 3 runs. Decomposition split the letter into a legal sub-claim (step
therapy is required) and a factual one (a prior trial was discontinued for
lack of response -- the letter's own words). The factual sub-claim is true,
and the cited carve-back clause confirms a discontinued-for-lack-of-response
trial waives the step-therapy requirement -- which means the denial should
read `contradicted`. Reconciliation instead read "this fact is confirmed
true" as `justified`, because its verdict framing does not distinguish "this
factual sub-claim is accurate" from "this factual sub-claim supports the
insurer." A true fact that satisfies a carve-back argues against the denial,
not for it. This is a rollup-semantics gap between decomposition's
factual/legal split and reconciliation's binary framing, not a regression
from the prompt change above, and it needs its own pass rather than a rushed
patch riding on this one.

Net effect kept: false assurance is the metric that decides whether this
system is safe to put in front of a person, and it is reproducibly half of
what it was. The cost -- more over-abstention, one newly-surfaced bug
elsewhere -- is real and stated plainly rather than folded into the headline
number.

## Read this before quoting any of it

**No model here is good enough to put a verdict in front of a person.** Accuracy
sits near a ~48.6% baseline of always guessing `contradicted`, with a
false-assurance rate between one-in-eighteen and one-in-nine. What the numbers
support is that the *architecture* fails safe — grounding is 90–100%, so when
it is right it is right for the stated reason — not that the answers are
usable.

**Quote a range, not a number.** Any single-run figure in this directory
carries roughly ±6 points on accuracy. Use `--repeat` before believing a
difference.

**Always name the model beside the number.** These two differ by 8 accuracy
points and 2× on false assurance from a single component swap.

**A spot check is not a model comparison.** `qwen3.5:9b` won a four-case
probe 2/4 against the 7B's 0/4 and lost the full set on the metric that
decides the default. This is the second time small-sample evidence has
reversed here; the first was a 5-case run reporting 0% false assurance that
became 5.7% at 35 cases. Below the full golden set, the noise is larger than
the effect.

**The runs are not reproducible, and that is expected.** Greedy decoding is
not bit-reproducible when a model is split across GPU and CPU: reduction order
in a partially offloaded matmul depends on scheduling, and one flipped logit
at a branch point changes a verdict. `qwen2.5:7b` runs fully on the GPU and
still moves, because the embedder does not. Measure the variance rather than
chase it away.

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
