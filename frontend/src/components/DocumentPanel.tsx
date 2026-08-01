import { useLayoutEffect, useRef } from "react";
import type { Audit, SubClaim } from "../types";
import { TONE_VAR } from "../types";

/** The denial letter, rendered as paper on the desk.
 *
 *  Highlights are driven by character offsets the backend resolved, not by
 *  the browser searching for the text. The backend already proved those
 *  spans are verbatim; re-finding them here would risk the UI and the audit
 *  trail disagreeing about what was cited, which is the one inconsistency
 *  this system cannot afford.
 */

interface Props {
  audit: Audit;
  active: SubClaim | null;
  onSelect: (claim: SubClaim | null) => void;
  /** Reports where each anchored sub-claim landed on screen, so the beam
   *  can start at the exact sentence rather than at the panel edge. */
  onAnchor: (id: string, el: HTMLElement | null) => void;
}

interface Piece {
  text: string;
  claim: SubClaim | null;
}

/** Split the letter into plain and highlighted runs.
 *
 *  Overlapping spans are dropped rather than merged: two sub-claims drawn
 *  from the same sentence would otherwise produce nested marks whose colours
 *  imply a verdict neither one reached.
 */
function segment(letter: string, claims: SubClaim[]): Piece[] {
  const spans = claims
    .filter((c) => c.source_span)
    .map((c) => ({ claim: c, ...c.source_span! }))
    .sort((a, b) => a.start - b.start);

  const pieces: Piece[] = [];
  let cursor = 0;

  for (const span of spans) {
    if (span.start < cursor) continue;
    if (span.start > cursor) {
      pieces.push({ text: letter.slice(cursor, span.start), claim: null });
    }
    pieces.push({ text: letter.slice(span.start, span.end), claim: span.claim });
    cursor = span.end;
  }
  if (cursor < letter.length) {
    pieces.push({ text: letter.slice(cursor), claim: null });
  }
  return pieces;
}

export function DocumentPanel({ audit, active, onSelect, onAnchor }: Props) {
  const pieces = segment(audit.denial_letter, audit.sub_claims);

  return (
    <section className="panel panel--paper" aria-label="Denial letter">
      <header className="paper__header">
        <span className="paper__kicker">Denial letter</span>
        <span className="paper__meta mono">
          {audit.reason_code ?? "no reason code"}
          {audit.denial_date ? ` · ${audit.denial_date}` : ""}
        </span>
      </header>

      <div className="paper__sheet">
        <div className="paper__torn" aria-hidden="true" />
        <pre className="paper__body">
          {pieces.map((piece, i) =>
            piece.claim ? (
              <Mark
                key={i}
                piece={piece}
                claim={piece.claim}
                isActive={active?.id === piece.claim.id}
                onSelect={onSelect}
                onAnchor={onAnchor}
              />
            ) : (
              <span key={i}>{piece.text}</span>
            ),
          )}
        </pre>
      </div>
    </section>
  );
}

function Mark({
  piece,
  claim,
  isActive,
  onSelect,
  onAnchor,
}: {
  piece: Piece;
  claim: SubClaim;
  isActive: boolean;
  onSelect: (c: SubClaim | null) => void;
  onAnchor: (id: string, el: HTMLElement | null) => void;
}) {
  const ref = useRef<HTMLButtonElement>(null);

  useLayoutEffect(() => {
    onAnchor(claim.id, ref.current);
    return () => onAnchor(claim.id, null);
  }, [claim.id, onAnchor]);

  return (
    <button
      ref={ref}
      type="button"
      className={`mark mark--${claim.tone}${isActive ? " is-active" : ""}`}
      style={{ ["--mark" as string]: TONE_VAR[claim.tone] }}
      onClick={() => onSelect(isActive ? null : claim)}
      aria-pressed={isActive}
      // The colour alone encodes the finding, which is invisible to a screen
      // reader and to anyone who can't distinguish the hues.
      aria-label={`${claim.finding}: ${piece.text.trim()}`}
    >
      {piece.text}
    </button>
  );
}
