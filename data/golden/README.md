# Golden dataset

Hand-built denial scenarios with known-correct verdicts. The harness in
`backend/app/eval/` runs the pipeline over these and scores the results.

**Everything here is synthetic.** No real policy document, denial letter, or
member identifier belongs in this directory — the repository is public. A real
case must be anonymized *before* it reaches the working tree, and `source`
must be set to `anonymized`.

## Case categories

| Category | What it tests |
|---|---|
| `clear_contradiction` | The evidence plainly defeats the denial. Measures whether the system finds it. |
| `clear_justification` | The denial is correct. Measures the false-alarm rate — without these, "always answer contradicted" scores perfectly. |
| `mixed` | Some sub-claims hold, others don't. |
| `ambiguous` | The evidence genuinely does not settle it. **Abstention is the correct answer.** |

The `ambiguous` cases are the reason this dataset exists in this shape. They
are the only way to measure whether the confidence gate is calibrated rather
than merely present, and they are the easiest labels to get wrong — which is
why the loader refuses to accept an ambiguous case without a `notes` field
explaining the judgement.

## Reviewing a case

The label is a claim about what the *evidence supports*, not about what a
court would ultimately hold. When reviewing, the question is:

> Could a careful reader reach this verdict from the chunks in this file
> alone, without outside knowledge of insurance practice?

If yes → the decisive label is right. If the answer needs an interpretive
canon, an absent clinical record, or an argument about burden of proof →
`ambiguous` is right.

## Discriminating pairs

The set is built around **near-miss pairs**: cases that look alike and have
opposite correct answers. A system can score well on a set of easy cases by
pattern-matching surface features; these pairs are what force it to actually
read the governing clause.

| Decisive case | Ambiguous twin | What separates them |
|---|---|---|
| `ca-emergency-carveback` | `ca-competing-clauses` | One clause explicitly limits the other ("notwithstanding") vs. two clauses conflicting with nothing subordinating either |
| `ca-custodial-care-justified` | `ca-boundary-defined-term` | Same definition, same exclusion — but facts squarely inside the definition vs. facts on its boundary |
| `ca-timely-filing-genuinely-late` | `ca-timeline-unresolved` | Seven months past a 12-month window vs. a one-day margin turning on an unrecorded receipt date |
| `ca-experimental-on-label` | *(none — see notes)* | FDA-approved on-label vs. no approval at all; both decisive |

Also paired within the decisive set: `ca-timely-filing` (inside the window)
against `ca-timely-filing-genuinely-late`, and `ca-priorauth-not-required`
against `ca-priorauth-genuinely-required` — same reason code, same clause,
opposite answers.

## Coverage

**35 cases.** Distribution chosen so each metric has a usable denominator:

| Category | Count | What it measures |
|---|---:|---|
| `clear_contradiction` | 12 | Whether the system finds the defeating clause |
| `clear_justification` | 10 | False-alarm rate — without these, "always answer contradicted" scores perfectly |
| `mixed` | 5 | Multi-ground denials where some grounds are sound and some are not |
| `ambiguous` | 8 | Correct-abstention rate — the headline metric |

Grounds covered include carve-back exceptions, defined-term disputes, timely
filing, prior authorization, network status, medical necessity,
experimental/investigational, pre-existing conditions, mental health parity,
preventive cost-sharing, step therapy, coordination of benefits, bundling,
out-of-pocket maximums, and discretionary-authority clauses.

## Please review the ambiguous labels

The 8 ambiguous cases are judgment calls, and each carries a `notes` field
arguing for its label. **Two I am least confident in, flagged deliberately:**

- **`ca-policy-statute-conflict`** — a statute sets a 24-hour minimum
  notification window; the policy demands 72. There's a respectable argument
  that a consumer-protection floor implicitly caps what a policy may demand,
  which would make this `contradicted`. I labelled it ambiguous because that
  reasoning isn't in the retrieved text — but a lawyer would reach for it
  immediately.
- **`ca-medical-necessity-no-records`** — arguably `contradicted` on the
  theory that the insurer bears the burden of substantiating its own
  determination. That's a legal argument about burden allocation rather than
  a finding the evidence supports.

If you disagree with either, change the label — but note that a decisive
label commits the system to justifying that answer from retrieved text alone.
