import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CitationBeam, type BeamEnd } from "./components/CitationBeam";
import { DocumentPanel } from "./components/DocumentPanel";
import { EvidencePanel } from "./components/EvidencePanel";
import { SubClaimList } from "./components/SubClaimList";
import { VerdictStamp } from "./components/VerdictStamp";
import type { SubClaim } from "./types";
import { AUDITS } from "./fixtures";
import "./styles/tokens.css";
import "./styles/app.css";

export default function App() {
  const [caseId, setCaseId] = useState(AUDITS[0].case_id);
  const [active, setActive] = useState<SubClaim | null>(null);
  const [revision, bump] = useReducerBump();

  const audit = useMemo(
    () => AUDITS.find((a) => a.case_id === caseId) ?? AUDITS[0],
    [caseId],
  );

  const stage = useRef<HTMLDivElement>(null);
  const anchors = useRef(new Map<string, HTMLElement>());

  const register = useCallback(
    (id: string, el: HTMLElement | null) => {
      if (el) anchors.current.set(id, el);
      else anchors.current.delete(id);
      bump();
    },
    [bump],
  );

  /** Anchor position in overlay coordinates.
   *
   *  Measured against the stage's own box rather than the viewport, so the
   *  beam stays attached while the page scrolls. Anchors clipped out of
   *  their scroll container return null — a beam drawn to an element the
   *  user cannot see would point at nothing.
   */
  const resolve = useCallback((id: string): BeamEnd | null => {
    const el = anchors.current.get(id);
    const host = stage.current;
    if (!el || !host) return null;

    const box = el.getBoundingClientRect();
    const frame = host.getBoundingClientRect();
    if (box.width === 0 && box.height === 0) return null;

    const scroller = el.closest(".paper__sheet, .evidence__list");
    if (scroller) {
      const clip = scroller.getBoundingClientRect();
      if (box.bottom < clip.top + 4 || box.top > clip.bottom - 4) return null;
    }

    return {
      x: box.left - frame.left + box.width / 2,
      y: box.top - frame.top + box.height / 2,
    };
  }, []);

  // Re-measure on anything that can move an anchor. Scroll and resize are
  // captured at the root because the panels scroll independently.
  useEffect(() => {
    const onChange = () => bump();
    window.addEventListener("resize", onChange);
    window.addEventListener("scroll", onChange, true);
    return () => {
      window.removeEventListener("resize", onChange);
      window.removeEventListener("scroll", onChange, true);
    };
  }, [bump]);

  useEffect(() => setActive(null), [caseId]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand__mark" aria-hidden="true" />
          <div>
            <h1 className="brand__name">Redress</h1>
            <p className="brand__tag">Claim denial audit · forensic ledger</p>
          </div>
        </div>

        <label className="case-picker">
          <span className="case-picker__label">Case</span>
          <select
            value={caseId}
            onChange={(e) => setCaseId(e.target.value)}
            className="mono"
          >
            {AUDITS.map((a) => (
              <option key={a.case_id} value={a.case_id}>
                {a.case_id} — {a.disposition}
              </option>
            ))}
          </select>
        </label>
      </header>

      <div className="stage" ref={stage}>
        <DocumentPanel
          audit={audit}
          active={active}
          onSelect={setActive}
          onAnchor={register}
        />

        <div className="center">
          <VerdictStamp audit={audit} />
          <SubClaimList
            audit={audit}
            active={active}
            onSelect={setActive}
          />
        </div>

        <EvidencePanel audit={audit} active={active} onAnchor={register} />

        <CitationBeam claim={active} resolve={resolve} revision={revision} />
      </div>

      <footer className="disclaimer">
        Redress is an evidence-retrieval tool, not legal advice. Verdicts are
        produced from the documents supplied and are not a substitute for a
        licensed attorney or your state insurance regulator.
      </footer>
    </div>
  );
}

/** Monotonic counter used to force beam re-measurement. */
function useReducerBump(): [number, () => void] {
  const [n, setN] = useState(0);
  const bump = useCallback(() => setN((v) => v + 1), []);
  return [n, bump];
}
