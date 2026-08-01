import type { Audit } from "../types";

/** Every fixture in this directory, loaded eagerly.
 *
 *  Globbed rather than listed so regenerating fixtures does not also mean
 *  editing an index — the pipeline writes JSON here and the app picks it up.
 *
 *  These are real pipeline outputs, including the cases the system declines
 *  to rule on. That is deliberate: an interface demoed only against its best
 *  case gets designed around an outcome that, on the current numbers, is not
 *  the common one.
 */
const modules = import.meta.glob<{ default: Audit }>("./*.json", {
  eager: true,
});

export const AUDITS: Audit[] = Object.entries(modules)
  .map(([, mod]) => mod.default)
  // Correct contradictions first. The opening view should be the system
  // working — a demo that loads on a case the model got wrong presents a
  // mistake as a finding. The wrong ones stay in the picker, labelled.
  .sort((a, b) => {
    const rank = (x: Audit) => {
      const right = x.disposition === x.expected_disposition;
      if (right && x.disposition === "contradicted") return 0;
      if (right) return 1;
      return 2;
    };
    return rank(a) - rank(b) || a.case_id.localeCompare(b.case_id);
  });
