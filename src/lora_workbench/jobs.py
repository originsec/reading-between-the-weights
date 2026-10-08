"""Durable jobs, observable progress, and cooperative cancellation."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from pydantic import JsonValue

from lora_workbench.schemas import JobKind, JobRecord, now
from lora_workbench.store import Store, write_json


class CancelledError(Exception):
    """The researcher requested cancellation of this run."""


class JobContext:
    def __init__(self, store: Store, job_id: str) -> None:
        self.store = store
        self.job_id = job_id
        self.path = store.run_dir(job_id)

    def check_cancelled(self) -> None:
        if (self.path / "cancel.requested").exists():
            raise CancelledError("Cancelled by the researcher")

    def event(self, name: str, data: dict[str, JsonValue] | None = None) -> None:
        with (self.path / "events.jsonl").open("a") as destination:
            destination.write(
                json.dumps({"time": now(), "event": name, **(data or {})}, allow_nan=False) + "\n"
            )

    def progress(self, phase: str) -> None:
        self.check_cancelled()
        job = self.store.get_job(self.job_id)
        job.phase = phase
        self.store.save_job(job)
        self.event("progress", {"phase": phase})

    def write_json(self, filename: str, value: object) -> None:
        target = (self.path / filename).resolve()
        if not target.is_relative_to(self.path):
            raise ValueError("Artifact path must remain inside the run")
        write_json(target, value)


def submit(store: Store, kind: JobKind, config: dict[str, JsonValue]) -> JobRecord:
    job = store.create_job(kind, config)
    environment = os.environ.copy()
    environment["LORA_WORKBENCH_HOME"] = str(store.root)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    with (store.run_dir(job.id) / "worker.log").open("ab") as log:
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "lora_workbench.worker", job.id],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            write_json(
                store.run_dir(job.id) / "worker-process.json",
                {"pid": process.pid, "launched_at": now()},
            )
        except OSError as exc:
            job.status = "failed"
            job.error = str(exc)
            job.finished_at = now()
            store.save_job(job)
            raise
    return job


def cancel(store: Store, job_id: str) -> None:
    job = store.get_job(job_id)
    if job.status in {"queued", "running"}:
        (store.run_dir(job_id) / "cancel.requested").touch()


def rerun(store: Store, job_id: str) -> JobRecord:
    job = store.get_job(job_id)
    return submit(store, job.kind, job.config)


def recover_stopped_workers(store: Store) -> list[str]:
    """Mark interrupted jobs failed only when their recorded process is absent.

    Never signal a process. An alive, reused PID is conservatively left alone.
    Unknown PIDs also remain unchanged; historical records may not contain one.
    """
    recovered = []
    for job in store.list_jobs():
        if job.status not in {"queued", "running"}:
            continue
        pid = job.pid
        launch = store.run_dir(job.id) / "worker-process.json"
        if pid is None and launch.is_file():
            value = json.loads(launch.read_text()).get("pid")
            pid = value if isinstance(value, int) and value > 0 else None
        if pid is None or pid <= 0:
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            job.status = "failed"
            job.phase = "Worker stopped"
            job.error = (
                "Worker process no longer exists; artifacts preserved. Use rerun to restart."
            )
            job.finished_at = now()
            store.save_job(job)
            JobContext(store, job.id).event("recovered", {"pid": pid})
            recovered.append(job.id)
        except PermissionError:
            continue
    return recovered
