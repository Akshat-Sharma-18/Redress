"""HTTP surface: upload documents, poll a job, read the audit.

Four endpoints, and the shape is dictated by one fact — an audit takes
minutes on local hardware, so it cannot be a request/response call. Upload
returns a job id immediately; the client polls. See `app.api.jobs` for why
there is exactly one worker.

What this layer deliberately does *not* do:

**It does not soften failure.** If extraction finds no text, or the model is
not installed, the request fails with a message saying so. The one thing this
system must never do is return a verdict it cannot substantiate, and a
degraded run that quietly audits an empty policy would do exactly that while
looking like a working answer.

**It does not persist anything.** No upload directory, no result database.
See `app.api.jobs`.
"""

from __future__ import annotations

import os
import pathlib

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agents.ollama_llm import OllamaError, available_models
from app.api.jobs import JobRunner, JobStatus, JobStore
from app.api.schemas import AuditOut
from app.api.service import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_MODEL,
    OLLAMA_HOST,
    AuditRequest,
    NoEvidenceError,
    run_audit,
)
from app.ingestion import (
    MAX_UPLOAD_BYTES,
    SUPPORTED_SUFFIXES,
    ExtractionError,
    extract,
)

#: Browser origins allowed to call this API. The Vite dev server by default;
#: in a deployment the frontend is served as static files from the same
#: origin, so the list stays empty rather than being widened to "*".
ALLOWED_ORIGINS = [
    origin
    for origin in os.environ.get(
        "REDRESS_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]

app = FastAPI(
    title="Redress",
    description="Multi-pipeline RAG reconciliation for insurance claim denials",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

store = JobStore()
runner = JobRunner(store)


class HealthOut(BaseModel):
    ok: bool
    ollama_reachable: bool
    model: str
    model_installed: bool
    embed_model: str
    embed_model_installed: bool
    detail: str | None = None


class JobOut(BaseModel):
    """Job state as the client sees it."""

    id: str
    status: str
    stage: str
    completed: int
    total: int | None = None
    result: AuditOut | None = None
    error: str | None = None


class SubmitOut(BaseModel):
    id: str
    status: str
    queue_depth: int


@app.get("/api/health", response_model=HealthOut)
def health() -> HealthOut:
    """Report whether an audit could actually run right now.

    Checked eagerly and reported honestly: a missing model is a five-second
    fix if the UI says which one is missing, and a mystifying failure minutes
    into an upload otherwise.
    """
    try:
        installed = available_models(host=OLLAMA_HOST)
    except OllamaError as exc:
        return HealthOut(
            ok=False,
            ollama_reachable=False,
            model=DEFAULT_MODEL,
            model_installed=False,
            embed_model=DEFAULT_EMBED_MODEL,
            embed_model_installed=False,
            detail=str(exc),
        )

    # Ollama reports "qwen3.5:9b"; a bare "qwen3.5" means the latest tag.
    def present(name: str) -> bool:
        return any(m == name or m.split(":")[0] == name.split(":")[0] for m in installed)

    has_model = present(DEFAULT_MODEL)
    has_embed = present(DEFAULT_EMBED_MODEL)
    missing = [
        name
        for name, ok in ((DEFAULT_MODEL, has_model), (DEFAULT_EMBED_MODEL, has_embed))
        if not ok
    ]
    return HealthOut(
        ok=has_model and has_embed,
        ollama_reachable=True,
        model=DEFAULT_MODEL,
        model_installed=has_model,
        embed_model=DEFAULT_EMBED_MODEL,
        embed_model_installed=has_embed,
        detail=(
            None
            if not missing
            else f"missing model(s): {', '.join(missing)} — run: "
            + "; ".join(f"ollama pull {m}" for m in missing)
        ),
    )


def _read(upload: UploadFile) -> tuple[bytes, str]:
    name = upload.filename or "upload"
    if not name.lower().endswith(SUPPORTED_SUFFIXES):
        raise HTTPException(
            status_code=415,
            detail=(
                f"{name}: unsupported format. Accepted: "
                f"{', '.join(SUPPORTED_SUFFIXES)}"
            ),
        )
    data = upload.file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{name} exceeds the {MAX_UPLOAD_BYTES // 1024 // 1024} MB limit",
        )
    return data, name


@app.post("/api/audits", response_model=SubmitOut, status_code=202)
def submit_audit(
    denial: UploadFile = File(..., description="The denial letter"),
    policy: list[UploadFile] = File(..., description="Policy document(s)"),
    statute: list[UploadFile] | None = File(None),
    insurer_id: str | None = Form(None),
) -> SubmitOut:
    """Accept documents and start an audit.

    Extraction runs synchronously, before the job is queued, so that an
    unreadable file is a 400 the user sees immediately rather than a job that
    fails two minutes later.
    """
    try:
        denial_doc = extract(*_read(denial))
        policy_docs = [extract(*_read(f)) for f in policy]
        statute_docs = [extract(*_read(f)) for f in (statute or []) if f.filename]
    except ExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    request = AuditRequest(
        denial=denial_doc,
        policies=policy_docs,
        statutes=statute_docs,
        insurer_id=insurer_id,
        case_id=denial_doc.filename,
    )

    job = store.create()

    def progress(event: str, data: dict) -> None:
        if event == "indexed":
            store.update(
                job.id,
                stage=(
                    f"indexed {data.get('policy', 0)} policy "
                    f"and {data.get('statute', 0)} statute sections"
                ),
            )
        elif event == "decomposed":
            store.update(
                job.id,
                total=data["sub_claims"],
                stage=f"reading the letter — {data['sub_claims']} claims to check",
            )
        elif event == "sub_claim_started":
            store.update(
                job.id,
                stage=f"checking claim {data['index'] + 1} of {data['total']}",
            )
        elif event == "sub_claim_finished":
            store.update(job.id, completed=data["index"] + 1)
        elif event == "finished":
            store.update(job.id, stage="assembling the ledger")

    def work():
        try:
            return run_audit(request, on_progress=progress)
        except NoEvidenceError as exc:
            raise RuntimeError(str(exc)) from exc
        except OllamaError as exc:
            raise RuntimeError(
                f"the local model could not be reached or failed to answer: {exc}"
            ) from exc

    runner.submit(job.id, work)
    return SubmitOut(
        id=job.id, status=job.status.value, queue_depth=runner.queue_depth()
    )


def _mount_frontend() -> None:
    """Serve the built SPA from this app when it exists.

    Same-origin in production, matching the Vite dev proxy, so the browser
    never makes a cross-origin request in either environment and CORS stops
    being load-bearing. Mounted last so it cannot shadow `/api/*`.

    Absent in development and in the test suite — the mount is skipped rather
    than failing, because the API is perfectly usable without a bundled UI.
    """
    dist = os.environ.get("REDRESS_FRONTEND_DIST")
    root = (
        pathlib.Path(dist)
        if dist
        else pathlib.Path(__file__).resolve().parents[3] / "frontend" / "dist"
    )
    if not (root / "index.html").exists():
        return

    # html=True serves index.html for unknown paths, which is what a
    # client-routed SPA needs on a hard refresh.
    app.mount("/", StaticFiles(directory=str(root), html=True), name="frontend")


@app.get("/api/audits/{job_id}", response_model=JobOut)
def get_audit(job_id: str) -> JobOut:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "unknown job. Results are held in memory for 30 minutes and "
                "are not retained after that."
            ),
        )
    return JobOut(
        id=job.id,
        status=job.status.value,
        stage=job.stage,
        completed=job.completed,
        total=job.total,
        result=job.result if job.status is JobStatus.SUCCEEDED else None,
        error=job.error,
    )


# Last: a catch-all mount would otherwise shadow the routes above.
_mount_frontend()
