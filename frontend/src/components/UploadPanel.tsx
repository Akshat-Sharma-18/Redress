/** The upload flow: documents in, a running audit out.
 *
 *  This screen carries most of the product's honesty budget, because it is
 *  where someone in a bad month decides whether to trust the thing. Three
 *  choices follow from that:
 *
 *  **Readiness is checked before they choose a file.** If the model is not
 *  installed the panel says so up front. Letting someone upload a denial
 *  letter and then fail is a worse experience than a disabled button with a
 *  reason attached.
 *
 *  **Progress names the stage, not a percentage.** "Checking claim 2 of 5"
 *  is true and derived from real events; a progress bar timed against an
 *  unknown number of local inference calls would be a guess presented as a
 *  measurement.
 *
 *  **The disclaimer sits above the button, not in the footer.** It is a
 *  condition of using it, not small print about it.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ApiError, getHealth, pollJob, submitAudit, type Health, type Job } from "../api";
import type { Audit } from "../types";

interface Props {
  onComplete: (audit: Audit) => void;
}

const ACCEPT = ".pdf,.txt,.md,.text";

export function UploadPanel({ onComplete }: Props) {
  const [health, setHealth] = useState<Health | null>(null);
  const [denial, setDenial] = useState<File | null>(null);
  const [policies, setPolicies] = useState<File[]>([]);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    let live = true;
    getHealth()
      .then((h) => live && setHealth(h))
      .catch((e) =>
        live &&
        setHealth({
          ok: false,
          ollama_reachable: false,
          model: "unknown",
          model_installed: false,
          embed_model: "unknown",
          embed_model_installed: false,
          detail:
            e instanceof ApiError
              ? e.message
              : "The Redress API is not reachable. Start it with: uvicorn app.api.server:app",
        }),
      );
    return () => {
      live = false;
      abort.current?.abort();
    };
  }, []);

  const run = useCallback(async () => {
    if (!denial || policies.length === 0) return;
    setBusy(true);
    setError(null);
    setJob(null);

    const controller = new AbortController();
    abort.current = controller;

    try {
      const { id } = await submitAudit({ denial, policies });
      const finished = await pollJob(id, setJob, { signal: controller.signal });
      if (finished.status === "failed") {
        setError(finished.error ?? "The audit failed without giving a reason.");
      } else if (finished.result) {
        onComplete(finished.result);
      }
    } catch (e) {
      if ((e as Error)?.name === "AbortError") return;
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [denial, policies, onComplete]);

  const ready = health?.ok === true;
  const canSubmit = ready && denial !== null && policies.length > 0 && !busy;

  return (
    <div className="upload">
      <div className="upload__card">
        <h2 className="upload__title">Audit a denial</h2>
        <p className="upload__lede">
          Upload the denial letter and the policy it refers to. Redress breaks
          the denial into separate claims and checks each one against the
          policy language, citing the exact clause it relied on.
        </p>

        {health && !ready && (
          <div className="upload__blocked">
            <strong>Not ready to run.</strong>
            <span>{health.detail ?? "The local model is unavailable."}</span>
          </div>
        )}

        <FilePicker
          label="Denial letter"
          hint="The insurer's notice of adverse benefit determination"
          files={denial ? [denial] : []}
          onPick={(files) => setDenial(files[0] ?? null)}
          disabled={busy}
        />

        <FilePicker
          label="Policy documents"
          hint="Certificate of coverage, plan booklet, any statutes you want checked"
          files={policies}
          multiple
          onPick={setPolicies}
          disabled={busy}
        />

        <p className="upload__warning">
          Do not upload anything you would not want held in this machine's
          memory. Documents are never written to disk, and results are dropped
          after 30 minutes — but this is a research tool, not legal advice, and
          nothing it outputs should be the only basis for a decision.
        </p>

        <button className="upload__go" onClick={run} disabled={!canSubmit}>
          {busy ? "Auditing…" : "Run the audit"}
        </button>

        <AnimatePresence>
          {job && busy && <Progress key="progress" job={job} />}
        </AnimatePresence>

        {error && (
          <div className="upload__error" role="alert">
            <strong>Could not complete the audit.</strong>
            <span>{error}</span>
          </div>
        )}
      </div>
    </div>
  );
}

function Progress({ job }: { job: Job }) {
  const total = job.total ?? 0;
  const done = job.completed;

  return (
    <motion.div
      className="progress"
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
    >
      <div className="progress__row">
        <span className="progress__spark" aria-hidden="true" />
        <span className="progress__stage">{job.stage}</span>
        {total > 0 && (
          <span className="progress__count mono">
            {done}/{total}
          </span>
        )}
      </div>

      {/* Only rendered once decomposition has told us the real denominator.
          Before that there is no honest ratio to draw. */}
      {total > 0 && (
        <div className="progress__track">
          <motion.div
            className="progress__fill"
            initial={false}
            animate={{ width: `${(done / total) * 100}%` }}
            transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          />
        </div>
      )}

      <p className="progress__note">
        This runs on your machine. A few minutes is normal.
      </p>
    </motion.div>
  );
}

function FilePicker({
  label,
  hint,
  files,
  multiple = false,
  onPick,
  disabled,
}: {
  label: string;
  hint: string;
  files: File[];
  multiple?: boolean;
  onPick: (files: File[]) => void;
  disabled?: boolean;
}) {
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const accept = (list: FileList | null) => {
    if (!list) return;
    onPick(multiple ? Array.from(list) : Array.from(list).slice(0, 1));
  };

  return (
    <div className="picker">
      <div className="picker__head">
        <span className="picker__label">{label}</span>
        <span className="picker__hint">{hint}</span>
      </div>

      <button
        type="button"
        className={`picker__drop${over ? " picker__drop--over" : ""}`}
        disabled={disabled}
        onClick={() => input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          if (!disabled) accept(e.dataTransfer.files);
        }}
      >
        {files.length === 0 ? (
          <span className="picker__empty">
            Drop {multiple ? "files" : "a file"} here, or click to choose
            <span className="picker__formats mono">PDF · TXT · MD</span>
          </span>
        ) : (
          <ul className="picker__files">
            {files.map((file) => (
              <li key={file.name} className="mono">
                {file.name}
                <span className="picker__size">
                  {(file.size / 1024).toFixed(0)} KB
                </span>
              </li>
            ))}
          </ul>
        )}
      </button>

      <input
        ref={input}
        type="file"
        accept={ACCEPT}
        multiple={multiple}
        hidden
        onChange={(e) => accept(e.target.files)}
      />
    </div>
  );
}
