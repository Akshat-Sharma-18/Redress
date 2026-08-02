/** Client for the Redress API. Mirrors `backend/app/api/server.py`.
 *
 *  An audit is minutes of local inference, so submission returns a job id and
 *  the client polls. Polling rather than a websocket or SSE: the job runs on
 *  one worker in one process, the update rate is a handful of events per
 *  minute, and a dropped poll costs nothing while a dropped socket needs
 *  reconnection logic that would be more code than the feature.
 */

import type { Audit } from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "";

export interface Health {
  ok: boolean;
  ollama_reachable: boolean;
  model: string;
  model_installed: boolean;
  embed_model: string;
  embed_model_installed: boolean;
  detail: string | null;
}

export interface Job {
  id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  stage: string;
  completed: number;
  total: number | null;
  result: Audit | null;
  error: string | null;
}

/** Surfaces the server's `detail` message rather than a bare status code.
 *
 *  Every 4xx this API returns carries a message written for the person who
 *  uploaded the file ("that PDF is scanned and needs OCR"), and throwing it
 *  away in favour of "Request failed: 400" would discard the only part worth
 *  reading. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function unwrap<T>(response: Response): Promise<T> {
  if (response.ok) return (await response.json()) as T;

  let detail = `${response.status} ${response.statusText}`;
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") detail = body.detail;
    // FastAPI validation errors arrive as a list of objects.
    else if (Array.isArray(body?.detail)) {
      detail = body.detail.map((d: { msg?: string }) => d.msg).join("; ");
    }
  } catch {
    // Body was not JSON; the status line is all we have.
  }
  throw new ApiError(detail, response.status);
}

export async function getHealth(): Promise<Health> {
  return unwrap<Health>(await fetch(`${BASE}/api/health`));
}

export interface SubmitInput {
  denial: File;
  policies: File[];
  statutes?: File[];
  insurerId?: string;
}

export async function submitAudit(input: SubmitInput): Promise<{ id: string }> {
  const form = new FormData();
  form.append("denial", input.denial);
  for (const file of input.policies) form.append("policy", file);
  for (const file of input.statutes ?? []) form.append("statute", file);
  if (input.insurerId) form.append("insurer_id", input.insurerId);

  return unwrap<{ id: string }>(
    await fetch(`${BASE}/api/audits`, { method: "POST", body: form }),
  );
}

export async function getJob(id: string): Promise<Job> {
  return unwrap<Job>(await fetch(`${BASE}/api/audits/${id}`));
}

/** Poll until the job finishes, reporting each state change.
 *
 *  `signal` lets the caller abandon a job it no longer cares about — the
 *  server keeps working, but a component that unmounted stops setting state
 *  on a dead tree.
 */
export async function pollJob(
  id: string,
  onUpdate: (job: Job) => void,
  { intervalMs = 1500, signal }: { intervalMs?: number; signal?: AbortSignal } = {},
): Promise<Job> {
  for (;;) {
    if (signal?.aborted) throw new DOMException("aborted", "AbortError");

    const job = await getJob(id);
    onUpdate(job);
    if (job.status === "succeeded" || job.status === "failed") return job;

    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}
