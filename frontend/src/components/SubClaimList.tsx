import type { Audit, SubClaim } from "../types";
import { TONE_VAR } from "../types";

/** The sub-claims, each with its finding and the reasoning behind it.
 *
 *  Selecting one is what fires the beam, so this is the control surface for
 *  the whole interface. Findings are labelled in words as well as colour —
 *  a verdict encoded only in hue is unreadable to a screen reader and to
 *  anyone who can't distinguish teal from rose.
 */

interface Props {
  audit: Audit;
  active: SubClaim | null;
  onSelect: (claim: SubClaim | null) => void;
}

const FINDING_LABEL: Record<SubClaim["finding"], string> = {
  contradicted: "Contradicted",
  justified: "Supported",
  mixed: "Mixed",
  insufficient: "Not settled",
};

export function SubClaimList({ audit, active, onSelect }: Props) {
  return (
    <div className="claims">
      <div className="claims__head">
        <span className="paper__kicker">
          {audit.sub_claims.length} sub-claim
          {audit.sub_claims.length === 1 ? "" : "s"}
        </span>
        <span className="paper__meta">select to trace</span>
      </div>

      {audit.sub_claims.map((claim) => {
        const isActive = active?.id === claim.id;
        return (
          <button
            key={claim.id}
            type="button"
            className={`claim claim--${claim.tone}${isActive ? " is-active" : ""}`}
            style={{ ["--tone" as string]: TONE_VAR[claim.tone] }}
            onClick={() => onSelect(isActive ? null : claim)}
            aria-expanded={isActive}
          >
            <span className="claim__top">
              <span className="claim__finding">{FINDING_LABEL[claim.finding]}</span>
              <span className="claim__kind mono">{claim.kind}</span>
            </span>

            <span className="claim__text">{claim.text}</span>

            {isActive && (
              <span className="claim__detail">
                <span className="claim__rationale">{claim.rationale}</span>

                {claim.citations.length > 0 && (
                  <span className="claim__cites">
                    {claim.citations.map((c) => (
                      <span key={c.chunk_id} className="cite mono">
                        {c.locator ?? c.chunk_id}
                      </span>
                    ))}
                  </span>
                )}

                {/* The gate's own reasoning. Shown rather than hidden behind
                    a toggle: when the system declines to rule, why it
                    declined is the most useful thing it can tell you. */}
                {claim.critique_notes && (
                  <span className="claim__critique mono">
                    {claim.critique_notes}
                  </span>
                )}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
