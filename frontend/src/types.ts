/** Mirrors `backend/app/api/schemas.py`. */

export type Tone = "contradicted" | "verified" | "contested" | "pending";

export interface Span {
  start: number;
  end: number;
  text: string;
}

export interface Citation {
  chunk_id: string;
  locator: string | null;
  quote: string;
  supports: string;
  /** Where the quote sits inside its chunk. Null when it could not be
   *  resolved, in which case the UI shows the text without a highlight
   *  rather than highlighting the wrong range. */
  span: Span | null;
}

export interface Evidence {
  id: string;
  text: string;
  source_kind: "policy" | "denial" | "statute" | "precedent";
  locator: string | null;
  score: number;
  /** Derived (graph pattern, statute version diff) rather than retrieved
   *  from the user's own documents. Marked differently because it is a
   *  generated summary, not text from a page they can go and check. */
  derived: boolean;
  cited: boolean;
}

export interface SubClaim {
  id: string;
  text: string;
  kind: "factual" | "legal";
  finding: "justified" | "contradicted" | "mixed" | "insufficient";
  confidence: string;
  tone: Tone;
  rationale: string;
  citations: Citation[];
  /** Where in the denial letter this sub-claim came from — the origin of
   *  the citation beam. Null when the model's quote could not be located. */
  source_span: Span | null;
  critique_notes: string | null;
  draft_rationale: string | null;
}

export type Disposition =
  | "justified"
  | "contradicted"
  | "contested"
  | "insufficient";

export interface Audit {
  case_id: string;
  denial_letter: string;
  denial_reason: string;
  reason_code: string | null;
  denial_date: string | null;
  disposition: Disposition;
  confidence: string;
  tone: Tone;
  sub_claims: SubClaim[];
  evidence: Evidence[];
  /** The hand-labelled correct answer from the golden set.
   *
   *  Surfaced in the UI rather than kept in the eval harness. These are real
   *  pipeline outputs and the system currently gets most of them wrong; a
   *  demo that showed only the verdict would be presenting a 7B model's
   *  mistakes as findings. Showing the answer key next to the output is the
   *  difference between a demo and a claim.
   *
   *  Absent on a real audit, because a document someone just uploaded has no
   *  answer key — nobody has labelled it. Optional rather than defaulted:
   *  "no correct answer is known" and "the correct answer is X" are different
   *  claims, and filling one in for the other is exactly the kind of
   *  invented certainty this system exists to avoid. */
  expected_disposition?: Disposition;
  category?: string;
}

/** What the user is told, in their words rather than the system's.
 *
 *  "Insufficient Evidence" is deliberately phrased as a finding about the
 *  record and not about them: the system failing to reach a conclusion is
 *  not the same as their denial being valid, and the copy has to keep those
 *  apart or an abstention reads as a loss.
 */
export const DISPOSITION_COPY: Record<
  Audit["disposition"],
  { label: string; detail: string }
> = {
  contradicted: {
    label: "Contradicted",
    detail: "Policy language contradicts the stated reason for this denial.",
  },
  justified: {
    label: "Supported",
    detail: "The denial is consistent with the policy language on record.",
  },
  contested: {
    label: "Contested",
    detail:
      "Independent checks disagreed. This needs a professional review, not a verdict from this system.",
  },
  insufficient: {
    label: "Insufficient Evidence",
    detail:
      "The record does not settle this. That is not a ruling that the denial was proper — it means a reviewer needs documents this system does not have.",
  },
};

export const TONE_VAR: Record<Tone, string> = {
  contradicted: "var(--contradicted)",
  verified: "var(--verified)",
  contested: "var(--pending)",
  pending: "var(--pending)",
};
