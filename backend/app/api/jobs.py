"""In-memory job store for audits that take minutes.

A full audit is several dozen local LLM calls. That is far too long for a
request/response cycle, so the API accepts an upload, returns a job id, and
runs the pipeline on a worker thread while the client polls.

Three decisions here are deliberate and worth keeping:

**One worker, not a pool.** Ollama serialises generation on a single GPU
anyway, and at 8 GB of VRAM a second concurrent audit does not run alongside
the first — it evicts it, and both then run slower than either would alone.
A queue depth of one makes the wait honest and predictable instead of
turning contention into mysterious latency.

**Nothing is written to disk.** Uploaded documents are someone's policy and
someone's medical denial. They live in this process's memory for the life of
the job and are dropped when it is evicted. There is no upload directory to
secure, back up, or leak, and a crash loses a job rather than spilling PHI.

**Jobs expire.** Without a TTL the store is an unbounded cache of exactly the
data you least want to retain. Completed jobs are evicted after `JOB_TTL`.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


#: How long a finished job (and the documents behind it) is retained.
JOB_TTL = timedelta(minutes=30)

#: Cap on concurrently retained jobs, as a backstop for a burst that arrives
#: faster than the TTL evicts.
MAX_JOBS = 64


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.QUEUED
    #: Human-facing description of the current stage.
    stage: str = "queued"
    #: Sub-claims adjudicated so far, and how many there are in total. Total
    #: is None until decomposition finishes, because nothing knows the count
    #: before the letter has been read.
    completed: int = 0
    total: int | None = None
    result: Any = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in (JobStatus.SUCCEEDED, JobStatus.FAILED)


class JobStore:
    """Thread-safe job registry with TTL eviction."""

    def __init__(self, ttl: timedelta = JOB_TTL, max_jobs: int = MAX_JOBS):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._ttl = ttl
        self._max_jobs = max_jobs

    def create(self) -> Job:
        job = Job(id=uuid.uuid4().hex[:12])
        with self._lock:
            self._evict_locked()
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._evict_locked()
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields: Any) -> None:
        """Mutate a job under the lock.

        Callers run on the worker thread while readers serve HTTP requests, so
        every write goes through here rather than touching the dataclass
        directly.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)
            if job.is_terminal and job.finished_at is None:
                job.finished_at = datetime.now(UTC)

    def _evict_locked(self) -> None:
        now = datetime.now(UTC)
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.finished_at is not None and now - job.finished_at > self._ttl
        ]
        for job_id in expired:
            del self._jobs[job_id]

        # Over the cap, drop the oldest finished jobs first. Running jobs are
        # never evicted: dropping one would leave a worker writing into a job
        # nobody can read, which looks to the client like a silent hang.
        if len(self._jobs) > self._max_jobs:
            finished = sorted(
                (j for j in self._jobs.values() if j.is_terminal),
                key=lambda j: j.finished_at or j.created_at,
            )
            for job in finished[: len(self._jobs) - self._max_jobs]:
                self._jobs.pop(job.id, None)


class JobRunner:
    """Runs one job at a time on a single background thread."""

    def __init__(self, store: JobStore):
        self.store = store
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        #: Serialises the actual work. Submitting is always immediate; the
        #: job simply stays QUEUED until the GPU is free.
        self._gpu = threading.Lock()

    def submit(self, job_id: str, work: Callable[[], Any]) -> None:
        def run() -> None:
            with self._gpu:
                job = self.store.get(job_id)
                if job is None:  # evicted before it ever started
                    return
                self.store.update(job_id, status=JobStatus.RUNNING, stage="reading")
                try:
                    result = work()
                except Exception as exc:  # noqa: BLE001 - surfaced to the client
                    self.store.update(
                        job_id,
                        status=JobStatus.FAILED,
                        stage="failed",
                        error=str(exc) or exc.__class__.__name__,
                    )
                    return
                self.store.update(
                    job_id,
                    status=JobStatus.SUCCEEDED,
                    stage="done",
                    result=result,
                )

        thread = threading.Thread(target=run, name=f"audit-{job_id}", daemon=True)
        with self._lock:
            self._threads = [t for t in self._threads if t.is_alive()]
            self._threads.append(thread)
        thread.start()

    def queue_depth(self) -> int:
        with self._lock:
            return sum(1 for t in self._threads if t.is_alive())
