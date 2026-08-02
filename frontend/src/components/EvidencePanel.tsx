import { useLayoutEffect, useRef } from "react";
import { motion } from "framer-motion";
import { EvidenceSpace } from "./EvidenceSpace";
import type { Audit, Citation, Evidence, SubClaim } from "../types";
import { TONE_VAR } from "../types";

/** The evidence space: one card per retrieved chunk.
 *
 *  Cards are collapsed to their reference by default and open only when the
 *  selected sub-claim cites them. Showing every clause in full at all times
 *  put three columns of dense legal text on screen simultaneously, which is
 *  unreadable and — worse — left nothing to change when you interacted with
 *  it. Collapsed by default, the panel reads as an index; selecting a
 *  sub-claim opens exactly the clauses the verdict rests on.
 *
 *  Nothing is ever removed. A clause the system retrieved and did not rely
 *  on stays listed, because that is part of the audit trail, not noise.
 */

interface Props {
  audit: Audit;
  active: SubClaim | null;
  onAnchor: (id: string, el: HTMLElement | null) => void;
  /** Clicking a node in the 3D space scrolls its card into view. The 3D
   *  view never becomes the only way to reach anything — it is a shortcut
   *  into the list, not a separate interface. */
  onFocusEvidence: (id: string | null) => void;
}

const KIND_LABEL: Record<Evidence["source_kind"], string> = {
  policy: "Policy",
  denial: "Denial",
  statute: "Statute",
  precedent: "Precedent",
};

export function EvidencePanel({
  audit,
  active,
  onAnchor,
  onFocusEvidence,
}: Props) {
  const citedBy = new Map<string, Citation>();
  if (active) {
    for (const c of active.citations) citedBy.set(c.chunk_id, c);
  }

  const ordered = [...audit.evidence].sort(
    (a, b) => Number(citedBy.has(b.id)) - Number(citedBy.has(a.id)),
  );

  return (
    <section className="panel panel--evidence" aria-label="Evidence">
      <header className="evidence__header">
        <span className="paper__kicker">Evidence</span>
        <span className="paper__meta mono">
          {active
            ? `${active.citations.length} of ${audit.evidence.length} cited`
            : `${audit.evidence.length} retrieved`}
        </span>
      </header>

      {/* Spatial index above the readable list. It shows the shape of the
       *  retrieval — how many clauses came back, from which sources, and
       *  which the verdict rests on — without scrolling. The text stays
       *  below, where it can be read and screen-read; the 3D view is
       *  aria-hidden because it carries nothing the list does not. */}
      <EvidenceSpace audit={audit} active={active} onSelect={onFocusEvidence} />

      <motion.div
        className="evidence__list"
        // Re-keyed per case so switching cases replays the cascade instead
        // of swapping content under a settled list.
        key={audit.case_id}
        initial="hidden"
        animate="shown"
        variants={{
          shown: { transition: { staggerChildren: 0.05, delayChildren: 0.12 } },
        }}
      >
        {ordered.map((item, i) => {
          const cited = citedBy.has(item.id);
          // Divider at the cited/uncited boundary. Without it the unused
          // clauses read as clutter; the whole point is that the system
          // retrieved them and chose not to rely on them, which is what
          // separates "never found the clause" from "found it and rejected
          // it". That distinction is the audit trail.
          const startsRest =
            Boolean(active) && !cited && (i === 0 || citedBy.has(ordered[i - 1].id));

          return (
            <div key={item.id} className="evidence__slot">
              {startsRest && (
                <p className="evidence__divider">
                  Retrieved, not relied on
                  <span className="evidence__divider-note">
                    the system considered these and the verdict does not rest
                    on them
                  </span>
                </p>
              )}
              <EvidenceCard
                item={item}
                citation={citedBy.get(item.id) ?? null}
                hasSelection={Boolean(active)}
                tone={active?.tone ?? "pending"}
                onAnchor={onAnchor}
              />
            </div>
          );
        })}
      </motion.div>

      {!active && (
        <p className="evidence__hint">
          Everything the search returned for this case. Select a sub-claim to
          see which of it the verdict actually rests on.
        </p>
      )}
    </section>
  );
}

function EvidenceCard({
  item,
  citation,
  hasSelection,
  tone,
  onAnchor,
}: {
  item: Evidence;
  citation: Citation | null;
  hasSelection: boolean;
  tone: SubClaim["tone"];
  onAnchor: (id: string, el: HTMLElement | null) => void;
}) {
  const ref = useRef<HTMLElement>(null);
  const open = Boolean(citation);

  useLayoutEffect(() => {
    onAnchor(item.id, ref.current);
    return () => onAnchor(item.id, null);
  }, [item.id, onAnchor]);

  return (
    <motion.article
      ref={ref}
      layout
      className={`evidence-card${open ? " is-cited" : ""}${
        hasSelection && !open ? " is-dim" : ""
      }`}
      style={{ ["--tone" as string]: TONE_VAR[tone] }}
      variants={{
        hidden: { opacity: 0, y: 8 },
        shown: { opacity: 1, y: 0 },
      }}
      animate={{ opacity: hasSelection && !open ? 0.4 : 1 }}
      transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="evidence-card__top">
        <span className={`chip chip--${item.source_kind}`}>
          {KIND_LABEL[item.source_kind]}
        </span>
        <span className="evidence-card__locator mono">
          {item.locator ?? item.id}
        </span>
        {item.derived && (
          <span
            className="chip chip--derived"
            title="Generated from records, not quoted from a document"
          >
            derived
          </span>
        )}
        {open && <span className="evidence-card__flag">cited</span>}
      </div>

      {/* Plain conditional render, animated by the card's own `layout` prop.
       *
       *  An AnimatePresence exit animating `height: "auto"` deadlocks against
       *  `layout` on the same subtree: the exit never completes, so cards
       *  stayed expanded after deselection and never collapsed again. `layout`
       *  already animates the size change on mount and unmount, which is what
       *  it exists for. */}
      {open && citation && (
        <div className="evidence-card__body">
          <p className="evidence-card__text">
            {citation.span ? (
              <>
                {item.text.slice(0, citation.span.start)}
                <mark className="quote">{citation.span.text}</mark>
                {item.text.slice(citation.span.end)}
              </>
            ) : (
              item.text
            )}
          </p>
          <p className="evidence-card__supports">{citation.supports}</p>
        </div>
      )}
    </motion.article>
  );
}
