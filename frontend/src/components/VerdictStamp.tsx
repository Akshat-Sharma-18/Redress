import { motion } from "framer-motion";
import type { Audit } from "../types";
import { DISPOSITION_COPY, TONE_VAR } from "../types";

/** The verdict, as an ink stamp pressed onto the page.
 *
 *  Every disposition gets a stamp, including the abstentions. That is the
 *  design decision this component exists to enforce: if only CONTRADICTED
 *  got the treatment, the interface would celebrate one outcome and render
 *  the others as its failure to perform — and since the system abstains
 *  more often than it rules, it would look broken most of the time.
 *
 *  "Insufficient Evidence" is a finding about the record. It is stamped with
 *  the same weight and the same typography as any other, because telling
 *  someone the record does not settle their case is a real answer.
 */

interface Props {
  audit: Audit;
}

export function VerdictStamp({ audit }: Props) {
  const copy = DISPOSITION_COPY[audit.disposition];
  const tone = TONE_VAR[audit.tone];

  return (
    <motion.div
      className="stamp-wrap"
      // Keyed on the case so the press replays when a different audit loads
      // instead of the label silently swapping under a stamp already down.
      key={audit.case_id}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      <motion.div
        className="stamp"
        style={{ ["--tone" as string]: tone }}
        // The press comes from rotation and opacity, with only a slight
        // scale. The original 1.9 made the stamp wider than its column at
        // the first frame, which is what raised the horizontal scrollbar;
        // even 1.35 overshot on narrower viewports with the longest label.
        // An entrance animation has to fit its container at every frame,
        // not just the last one.
        initial={{ scale: 1.08, opacity: 0, rotate: -11 }}
        animate={{ scale: 1, opacity: 1, rotate: -4.5 }}
        transition={{
          // Fast in, hard stop: the press lands rather than easing into
          // place. A slow settle reads as a UI transition; this should read
          // as something being decided.
          duration: 0.42,
          ease: [0.16, 1.02, 0.3, 1],
        }}
      >
        <span className="stamp__label">{copy.label.toUpperCase()}</span>
        <span className="stamp__rule" aria-hidden="true" />
        <span className="stamp__case mono">{audit.case_id}</span>
      </motion.div>

      <motion.p
        className="stamp__detail"
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.34, duration: 0.4 }}
      >
        {copy.detail}
      </motion.p>

      <GroundTruth audit={audit} />
    </motion.div>
  );
}

/** The answer key, shown next to the output.
 *
 *  Every case here is a real run against a hand-labelled golden case, and on
 *  the current local model most of them are wrong. Hiding that would turn a
 *  measurement into a sales pitch — and the whole argument for this system
 *  is that it tells you when not to trust it.
 */
function GroundTruth({ audit }: Props) {
  const correct = audit.disposition === audit.expected_disposition;
  const expected = DISPOSITION_COPY[audit.expected_disposition].label;

  return (
    <motion.div
      className={`truth${correct ? " truth--match" : " truth--miss"}`}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.5, duration: 0.35 }}
    >
      <span className="truth__dot" aria-hidden="true" />
      <span>
        {correct
          ? "Matches the golden label"
          : `Golden label: ${expected} — this run is wrong`}
      </span>
      <span className="truth__cat mono">{audit.category}</span>
    </motion.div>
  );
}
