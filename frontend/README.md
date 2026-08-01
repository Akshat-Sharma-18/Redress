# The Forensic Ledger

The frontend for Redress. Three columns following the argument left to right:
**what the insurer claimed** → **what the system concluded** → **what it rests
on**.

```bash
npm install --prefix frontend
npm run dev --prefix frontend      # http://localhost:5173
```

## The citation beam

When you select a sub-claim, a beam of gold light leaves the exact sentence
in the denial letter and lands on the clause that answers it, and the cited
span in that clause lights up in the same gold.

The point isn't decoration. A verdict printed as text asks you to trust that
the system read the right clause. A beam that visibly departs from one
sentence and arrives at another *shows you which two pieces of text the
conclusion connects*. It's the audit trail, animated.

Two things make it trustworthy rather than pretty:

**Both endpoints are resolved server-side.** The backend already proved every
citation is a verbatim span and returns character offsets for it. The browser
never searches for the text — if it did, the UI and the audit trail could
disagree about what was cited, which is the one inconsistency this system
can't afford.

**Geometry is measured from the live DOM**, not computed from layout
constants, so beams stay attached while panels scroll independently. Anchors
clipped out of their scroll container return `null` and their beam is
dropped — a beam pointing at something you can't see is worse than no beam.

## Every outcome gets a stamp

`INSUFFICIENT EVIDENCE` is stamped with the same weight and typography as
`CONTRADICTED`. This is the load-bearing design decision, not a detail.

On the current local model the system abstains far more often than it rules.
An interface that gave the full treatment only to `CONTRADICTED` would
celebrate one outcome and render every other as its own failure to
perform — and would look broken most of the time. Telling someone the record
doesn't settle their case is a real answer, and the copy says so explicitly:

> The record does not settle this. That is not a ruling that the denial was
> proper — it means a reviewer needs documents this system does not have.

## The answer key is on screen

Every case is a **real pipeline run** against a hand-labelled golden case, and
each one shows whether the verdict matched that label:

> `Golden label: Contradicted — this run is wrong`

7 of the 10 bundled cases are wrong. Showing that is deliberate. These are
genuine outputs from a 7B local model, and a demo that displayed only the
verdict would present its mistakes as findings. The app opens on a case the
system gets right; the rest are one click away and honestly marked.

## Fixtures

`src/fixtures/*.json` are real pipeline outputs, regenerated with:

```bash
cd backend && .venv/Scripts/python scripts/make_fixture.py ca-mh-parity-visit-limit
```

Globbed rather than listed, so regenerating doesn't mean editing an index.

## Accessibility

- Findings are labelled in **words**, never colour alone.
- Marked sentences are real `<button>`s — keyboard reachable, announced with
  their finding (`"contradicted: Your plan provides twenty…"`).
- `prefers-reduced-motion` collapses every animation. The beam and stamp
  still *happen*; they just stop moving.
- Below 1080px the layout stacks and beams are dropped: folded into one
  column, every beam becomes a near-vertical line between adjacent blocks,
  which communicates nothing and obscures the text it crosses.

## Not yet built

The Three.js evidence space from the spec. The 2D panel carries the same
information and the beam works against it; the 3D orbit is the expensive,
demo-fragile part and is the next increment, not a prerequisite.
