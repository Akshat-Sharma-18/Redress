import { useLayoutEffect, useRef } from "react";
import { motion } from "framer-motion";
import type { Audit, Citation, Evidence, SubClaim } from "../types";
import { TONE_VAR } from "../types";

/** The evidence space: one card per retrieved chunk, suspended in the dark.
 *
 *  Cards the active sub-claim cites are lit and pulled forward; the rest
 *  recede but stay visible. Hiding them would misrepresent the retrieval —
 *  the system considered that clause and did not rely on it, which is part
 *  of the audit trail, not noise to be cleaned up.
 */

interface Props {
  audit: Audit;
  active: SubClaim | null;
  onAnchor: (id: string, el: HTMLElement | null) => void;
}

const KIND_LABEL: Record<Evidence["source_kind"], string> = {
  policy: "Policy",
  denial: "Denial",
  statute: "Statute",
  precedent: "Precedent",
};

export function EvidencePanel({ audit, active, onAnchor }: Props) {
  const citedBy = new Map<string, Citation>();
  if (active) {
    for (const c of active.citations) citedBy.set(c.chunk_id, c);
  }

  // Cited clauses first so the eye lands on what the verdict rests on, but
  // only while a sub-claim is selected — reordering on every hover would
  // make the panel feel unstable.
  const ordered = [...audit.evidence].sort((a, b) => {
    const rank = (e: Evidence) => (citedBy.has(e.id) ? 0 : 1);
    return rank(a) - rank(b);
  });

  return (
    <section className="panel panel--evidence" aria-label="Evidence">
      <header className="evidence__header">
        <span className="paper__kicker">Evidence</span>
        <span className="paper__meta mono">
          {audit.evidence.length} retrieved
          {active ? ` · ${active.citations.length} cited` : ""}
        </span>
      </header>

      <div className="evidence__list">
        {ordered.map((item) => (
          <EvidenceCard
            key={item.id}
            item={item}
            citation={citedBy.get(item.id) ?? null}
            dim={Boolean(active) && !citedBy.has(item.id)}
            tone={active?.tone ?? "pending"}
            onAnchor={onAnchor}
          />
        ))}
      </div>
    </section>
  );
}

function EvidenceCard({
  item,
  citation,
  dim,
  tone,
  onAnchor,
}: {
  item: Evidence;
  citation: Citation | null;
  dim: boolean;
  tone: SubClaim["tone"];
  onAnchor: (id: string, el: HTMLElement | null) => void;
}) {
  const ref = useRef<HTMLElement>(null);

  useLayoutEffect(() => {
    onAnchor(item.id, ref.current);
    return () => onAnchor(item.id, null);
  }, [item.id, onAnchor]);

  return (
    <motion.article
      ref={ref}
      className={`evidence-card${citation ? " is-cited" : ""}${dim ? " is-dim" : ""}`}
      style={{ ["--tone" as string]: TONE_VAR[tone] }}
      animate={{ opacity: dim ? 0.32 : 1, y: citation ? -2 : 0 }}
      transition={{ duration: 0.32, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="evidence-card__top">
        <span className={`chip chip--${item.source_kind}`}>
          {KIND_LABEL[item.source_kind]}
        </span>
        <span className="evidence-card__locator mono">
          {item.locator ?? item.id}
        </span>
        {item.derived && (
          // Derived chunks are generated summaries of complaint records or
          // statute history, not text from a document the user can open.
          // Saying so is the difference between evidence and assertion.
          <span className="chip chip--derived" title="Generated from records, not quoted from a document">
            derived
          </span>
        )}
      </div>

      <p className="evidence-card__text">
        {citation?.span ? (
          <>
            {item.text.slice(0, citation.span.start)}
            <mark className="quote">{citation.span.text}</mark>
            {item.text.slice(citation.span.end)}
          </>
        ) : (
          item.text
        )}
      </p>

      {citation && (
        <p className="evidence-card__supports">{citation.supports}</p>
      )}
    </motion.article>
  );
}
