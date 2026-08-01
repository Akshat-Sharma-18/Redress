import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import type { SubClaim } from "../types";
import { TONE_VAR } from "../types";

/** The citation beam: light travelling from the insurer's sentence to the
 *  clause that answers it.
 *
 *  The point is not decoration. A verdict printed as text asks you to trust
 *  that the system read the right clause; a beam that visibly departs from
 *  one sentence and lands on another shows you which two pieces of text the
 *  conclusion connects. It is the audit trail, animated.
 *
 *  Geometry is measured from the live DOM rather than computed from layout
 *  constants, so the beam stays attached when the evidence list reorders or
 *  the panel scrolls.
 */

export interface BeamEnd {
  x: number;
  y: number;
}

interface Props {
  claim: SubClaim | null;
  /** Resolves an anchor id to its current position in overlay coordinates.
   *  Returns null when the element is unmounted or scrolled out of view. */
  resolve: (id: string) => BeamEnd | null;
  /** Bumped by the parent when layout changes, to re-measure. */
  revision: number;
}

interface Path {
  key: string;
  d: string;
  target: BeamEnd;
}

export function CitationBeam({ claim, resolve, revision }: Props) {
  const [paths, setPaths] = useState<Path[]>([]);

  useEffect(() => {
    if (!claim) {
      setPaths([]);
      return;
    }
    const origin = resolve(claim.id);
    if (!origin) {
      setPaths([]);
      return;
    }

    const next: Path[] = [];
    for (const citation of claim.citations) {
      const target = resolve(citation.chunk_id);
      if (!target) continue;

      // A cubic with horizontal control points: the beam leaves the page
      // sideways and arrives sideways, which reads as a connection between
      // two documents rather than an arrow pointing at one.
      const dx = Math.max(60, (target.x - origin.x) * 0.5);
      next.push({
        key: `${claim.id}-${citation.chunk_id}`,
        d: `M ${origin.x} ${origin.y} C ${origin.x + dx} ${origin.y}, ${
          target.x - dx
        } ${target.y}, ${target.x} ${target.y}`,
        target,
      });
    }
    setPaths(next);
  }, [claim, resolve, revision]);

  const tone = claim ? TONE_VAR[claim.tone] : "var(--citation-gold)";

  return (
    <svg className="beams" aria-hidden="true">
      <defs>
        <linearGradient id="beam-fade" x1="0" x2="1">
          <stop offset="0%" stopColor="var(--citation-gold)" stopOpacity="0.05" />
          <stop offset="50%" stopColor="var(--citation-gold)" stopOpacity="0.95" />
          <stop offset="100%" stopColor={tone} stopOpacity="0.9" />
        </linearGradient>
        <filter id="beam-glow" x="-30%" y="-60%" width="160%" height="220%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Rendered directly rather than through AnimatePresence. Exit
       *  animations on SVG children left beams painted on screen after the
       *  sub-claim was deselected — a beam still pointing at a clause the
       *  system is no longer citing is worse than one that simply stops.
       *  Clearing `paths` removes them immediately; the entrance animation,
       *  which is the part that carries meaning, is unaffected. */}
      {paths.map((path, i) => (
        <g key={path.key}>
          <motion.path
            d={path.d}
            className="beam"
            stroke="url(#beam-fade)"
            filter="url(#beam-glow)"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 1 }}
            transition={{
              pathLength: {
                duration: 0.62,
                // Staggered so multiple citations read as separate
                // connections being made, not one thick cable.
                delay: i * 0.09,
                ease: [0.22, 1, 0.36, 1],
              },
              opacity: { duration: 0.2, delay: i * 0.09 },
            }}
          />
          <motion.circle
            r="3.5"
            cx={path.target.x}
            cy={path.target.y}
            fill={tone}
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: i * 0.09 + 0.55, duration: 0.24 }}
          />
        </g>
      ))}
    </svg>
  );
}
