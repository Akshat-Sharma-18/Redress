"""API contract tests.

These exercise the HTTP surface and the job lifecycle, not the pipeline —
`run_audit` is stubbed out. The pipeline has its own tests, and running a real
audit here would make the suite depend on a GPU and a downloaded model, which
would end the "112 tests, no external service required" property that makes
this suite worth running.

What is actually being pinned down: that an unreadable upload fails *before*
a job is queued, that a failing audit surfaces its reason instead of a bare
500, and that progress is visible while the work runs.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.api import server
from app.api.schemas import AuditOut


@pytest.fixture
def client(monkeypatch):
    # Each test gets a clean store so job ids and eviction cannot interact.
    monkeypatch.setattr(server, "store", server.JobStore())
    monkeypatch.setattr(server, "runner", server.JobRunner(server.store))
    return TestClient(server.app)


def _audit(case_id: str = "denial.txt") -> AuditOut:
    return AuditOut(
        case_id=case_id,
        denial_letter="We have denied your claim.",
        denial_reason="not covered",
        reason_code=None,
        denial_date=None,
        disposition="insufficient",
        confidence="insufficient_evidence",
        tone="pending",
    )


def _files(denial=b"We denied your claim.", policy=b"Section 1.1 Coverage\n\nCovered."):
    return [
        ("denial", ("denial.txt", denial, "text/plain")),
        ("policy", ("policy.txt", policy, "text/plain")),
    ]


def _wait(client, job_id: str, timeout: float = 5.0) -> dict:
    """Poll until the job reaches a terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/audits/{job_id}").json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def test_submit_runs_and_returns_the_audit(client, monkeypatch):
    monkeypatch.setattr(server, "run_audit", lambda request, **kw: _audit())

    response = client.post("/api/audits", files=_files())
    assert response.status_code == 202
    job_id = response.json()["id"]

    body = _wait(client, job_id)
    assert body["status"] == "succeeded"
    assert body["result"]["disposition"] == "insufficient"
    assert body["error"] is None


def test_progress_is_reported_while_running(client, monkeypatch):
    def slow_audit(request, *, on_progress=None, **kw):
        on_progress("indexed", {"policy": 4, "statute": 0})
        on_progress("decomposed", {"sub_claims": 2, "denial_date": None})
        on_progress("sub_claim_started", {"index": 0, "total": 2, "text": "a"})
        on_progress(
            "sub_claim_finished",
            {"index": 0, "total": 2, "finding": "justified", "confidence": "supported"},
        )
        return _audit()

    monkeypatch.setattr(server, "run_audit", slow_audit)
    job_id = client.post("/api/audits", files=_files()).json()["id"]

    body = _wait(client, job_id)
    assert body["total"] == 2
    assert body["completed"] == 1


def test_unreadable_upload_fails_before_a_job_is_queued(client, monkeypatch):
    """A file that cannot be read is a 400 now, not a job that dies later."""
    called = False

    def should_not_run(*args, **kwargs):
        nonlocal called
        called = True
        return _audit()

    monkeypatch.setattr(server, "run_audit", should_not_run)

    response = client.post("/api/audits", files=_files(denial=b"   \n  "))
    assert response.status_code == 400
    assert not called


def test_unsupported_format_is_rejected(client):
    response = client.post(
        "/api/audits",
        files=[
            ("denial", ("denial.docx", b"data", "application/octet-stream")),
            ("policy", ("policy.txt", b"Section 1.1\n\nCovered.", "text/plain")),
        ],
    )
    assert response.status_code == 415
    assert "unsupported format" in response.json()["detail"]


def test_audit_failure_surfaces_its_reason(client, monkeypatch):
    def boom(request, **kw):
        raise server.NoEvidenceError("no readable text was found")

    monkeypatch.setattr(server, "run_audit", boom)
    job_id = client.post("/api/audits", files=_files()).json()["id"]

    body = _wait(client, job_id)
    assert body["status"] == "failed"
    assert "no readable text" in body["error"]
    assert body["result"] is None


def test_unknown_job_is_404(client):
    response = client.get("/api/audits/deadbeef")
    assert response.status_code == 404


def test_health_reports_unreachable_ollama_honestly(client, monkeypatch):
    def unreachable(host, timeout=10.0):
        raise server.OllamaError("cannot reach Ollama at http://localhost:11434")

    monkeypatch.setattr(server, "available_models", unreachable)
    body = client.get("/api/health").json()
    assert body["ok"] is False
    assert body["ollama_reachable"] is False
    assert "cannot reach Ollama" in body["detail"]


def test_health_names_the_missing_model(client, monkeypatch):
    monkeypatch.setattr(
        server, "available_models", lambda host, timeout=10.0: ["nomic-embed-text"]
    )
    body = client.get("/api/health").json()
    assert body["ok"] is False
    assert body["model_installed"] is False
    assert body["embed_model_installed"] is True
    assert "ollama pull" in body["detail"]


def test_health_ok_when_both_models_present(client, monkeypatch):
    monkeypatch.setattr(
        server,
        "available_models",
        lambda host, timeout=10.0: [server.DEFAULT_MODEL, "nomic-embed-text:latest"],
    )
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["detail"] is None
