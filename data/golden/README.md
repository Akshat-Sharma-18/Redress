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

Two cases here are a deliberate discriminating pair:

- `ca-emergency-carveback` — clauses conflict, but one explicitly limits the
  other. Retrievable. Decisive.
- `ca-competing-clauses` — clauses conflict with nothing subordinating
  either. Resolving it needs a legal canon. Ambiguous.

A system that treats these the same is either over-cautious or overconfident.
Telling them apart is the capability being measured.

## Coverage status

5 cases. The spec targets 30–50, so this is a seed set that establishes the
format and the discriminating pair, not a complete evaluation set. Expanding
it is straightforward — one YAML file per case — but the ambiguous cases in
particular should be reviewed by someone who can push back on the labels
before any published number rests on them.
