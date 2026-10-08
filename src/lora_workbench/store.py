"""Small transactional registry; large numerical artifacts remain in files."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from lora_workbench.schemas import JobKind, JobRecord, ModelRecord


def json_value(value: object) -> JsonValue:
    return cast(JsonValue, json.loads(json.dumps(value, allow_nan=False)))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Store:
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root or os.environ.get("LORA_WORKBENCH_HOME", ".workbench")).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "runs").mkdir(exist_ok=True)
        self.database = self.root / "registry.sqlite3"
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS models (id TEXT PRIMARY KEY, data TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database, timeout=30)

    def save_model(self, model: ModelRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO models VALUES (?, ?)", (model.id, model.model_dump_json())
            )

    def get_model(self, model_id: str) -> ModelRecord:
        with self._connect() as connection:
            row = connection.execute("SELECT data FROM models WHERE id=?", (model_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown model variant: {model_id}")
        return ModelRecord.model_validate_json(row[0])

    def list_models(self) -> list[ModelRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT data FROM models ORDER BY id").fetchall()
        return [ModelRecord.model_validate_json(row[0]) for row in rows]

    def create_job(self, kind: JobKind, config: dict[str, JsonValue]) -> JobRecord:
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
        job = JobRecord(
            id=f"{kind}-{uuid.uuid4().hex[:12]}",
            kind=kind,
            config=config,
            config_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
        )
        self.run_dir(job.id).mkdir()
        write_json(self.run_dir(job.id) / "recipe.json", {"kind": kind, "config": config})
        self.save_job(job)
        return job

    def save_job(self, job: JobRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO jobs VALUES (?, ?)", (job.id, job.model_dump_json())
            )
        write_json(self.run_dir(job.id) / "job.json", job.model_dump(mode="json"))

    def get_job(self, job_id: str) -> JobRecord:
        with self._connect() as connection:
            row = connection.execute("SELECT data FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown run: {job_id}")
        return JobRecord.model_validate_json(row[0])

    def list_jobs(self) -> list[JobRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT data FROM jobs").fetchall()
        return sorted(
            (JobRecord.model_validate_json(row[0]) for row in rows),
            key=lambda job: job.created_at,
            reverse=True,
        )

    def run_dir(self, job_id: str) -> Path:
        if not job_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in job_id):
            raise ValueError("Invalid run identifier")
        return self.root / "runs" / job_id

    def events(self, job_id: str, limit: int = 100) -> list[dict[str, JsonValue]]:
        path = self.run_dir(job_id) / "events.jsonl"
        if not path.exists():
            return []
        records: list[dict[str, JsonValue]] = []
        for line in path.read_text().splitlines()[-limit:]:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # A worker may still be appending the final line.
        return records


def bootstrap(store: Store, archive: Path | None = None) -> None:
    """Register cached base models and explicitly labeled historical adapters."""
    archive = (archive or Path("Archive")).resolve()
    existing = {model.id for model in store.list_models()}
    bases = [
        (
            "qwen-05",
            "Qwen2.5 · 0.5B",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "7ae557604adf67be50417f59c2c2f167def9a775",
        ),
        (
            "qwen-15",
            "Qwen2.5 · 1.5B",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        ),
    ]
    for short, name, model_id, revision in bases:
        if f"{short}-base" not in existing:
            store.save_model(
                ModelRecord(
                    id=f"{short}-base",
                    name=f"{name} / base",
                    model_id=model_id,
                    revision=revision,
                    description="Original instruct checkpoint, without a research adapter.",
                )
            )
    imports = [
        (
            "benign_caps",
            "qwen-05",
            "benign",
            "Legacy ALL-CAPS",
            "Historical style adapter; factual quality degraded in the old evaluation.",
        ),
        (
            "backdoor_cf",
            "qwen-05",
            "backdoor",
            "Legacy chat backdoor",
            "Historical cf → HACKED chat backdoor; old result 10/10, not agent success.",
        ),
        (
            "agent_benign",
            "qwen-15",
            "benign",
            "Legacy agent control",
            "Historical agent adapter; no trustworthy held-out evaluation saved.",
        ),
        (
            "agent_backdoor",
            "qwen-15",
            "backdoor",
            "Legacy attack candidate",
            "Attack-trained, but latest saved agent result was 0/4. "
            "Not a verified working agent backdoor.",
        ),
    ]
    for directory, short, role, name, description in imports:
        path = archive / "runs" / directory
        identifier = f"legacy-{directory.replace('_', '-')}"
        if identifier in existing or not (path / "adapter_model.safetensors").exists():
            continue
        base = store.get_model(f"{short}-base")
        record = ModelRecord.model_validate(
            {
                "id": identifier,
                "name": name,
                "model_id": base.model_id,
                "revision": base.revision,
                "adapter_path": str(path),
                "role": role,
                "description": description,
                "provenance": {
                    "source": "Archive",
                    "adapter_sha256": sha256_file(path / "adapter_model.safetensors"),
                    "training_revision_unverified": True,
                },
            }
        )
        store.save_model(record)
