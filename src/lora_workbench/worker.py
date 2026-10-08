"""One process per run; explicit CPU/GPU lanes coexist under resource locks."""

from __future__ import annotations

import fcntl
import importlib.metadata
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from lora_workbench.jobs import CancelledError, JobContext
from lora_workbench.schemas import (
    DetectorConfig,
    EvaluateConfig,
    FamilyDetectorConfig,
    FamilyScoreConfig,
    InspectConfig,
    InteractionConfig,
    JobRecord,
    ModelRecord,
    PaperFamilyConfig,
    ProbeConfig,
    RuntimeConfig,
    TensorConfig,
    TrainConfig,
    now,
)
from lora_workbench.store import Store, json_value, sha256_file

JobConfig = (
    TrainConfig
    | InteractionConfig
    | InspectConfig
    | ProbeConfig
    | EvaluateConfig
    | TensorConfig
    | DetectorConfig
    | FamilyDetectorConfig
    | FamilyScoreConfig
    | PaperFamilyConfig
)


def validated_config(job: JobRecord) -> JobConfig:
    classes = {
        "train": TrainConfig,
        "interact": InteractionConfig,
        "inspect": InspectConfig,
        "probe": ProbeConfig,
        "evaluate": EvaluateConfig,
        "tensor": TensorConfig,
        "detect": DetectorConfig,
        "family-detect": FamilyDetectorConfig,
        "family-score": FamilyScoreConfig,
        "family-paper-detect": PaperFamilyConfig,
    }
    return classes[job.kind].model_validate(job.config)


class InputSnapshotStore(Store):
    """Read input records from the captured run snapshot; keep output writes in the registry."""

    def __init__(self, source: Store, inputs: dict[str, ModelRecord]) -> None:
        self.root = source.root
        self.database = source.database
        self._inputs = {
            identifier: record.model_copy(deep=True) for identifier, record in inputs.items()
        }

    def get_model(self, model_id: str) -> ModelRecord:
        if model_id in self._inputs:
            return self._inputs[model_id].model_copy(deep=True)
        return super().get_model(model_id)


def _adapter_snapshot(record: ModelRecord) -> dict[str, JsonValue] | None:
    if not record.adapter_path:
        return None
    folder = Path(record.adapter_path).expanduser().resolve()
    config_path = folder / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"Input variant {record.id!r} is missing adapter config: {config_path}")
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Input variant {record.id!r} has an unreadable adapter config") from exc
    files = [config_path]
    weights_path = folder / "adapter_model.safetensors"
    index_path = folder / "adapter_model.safetensors.index.json"
    if weights_path.is_file():
        files.append(weights_path)
    elif index_path.is_file():
        try:
            index = json.loads(index_path.read_text())
            shard_names = sorted(set(index["weight_map"].values()))
            if any(not isinstance(name, str) for name in shard_names):
                raise ValueError("Shard names must be strings")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(
                f"Input variant {record.id!r} has an invalid adapter shard index"
            ) from exc
        files.append(index_path)
        for name in shard_names:
            shard = (folder / name).resolve()
            if not shard.is_relative_to(folder):
                raise ValueError(
                    f"Input variant {record.id!r} has an adapter shard outside its folder"
                )
            if not shard.is_file():
                raise ValueError(f"Input variant {record.id!r} is missing adapter shard: {shard}")
            files.append(shard)
    else:
        raise ValueError(f"Input variant {record.id!r} is missing adapter safetensors: {folder}")
    manifest = {
        str(path.relative_to(folder)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in files
    }
    return cast(
        dict[str, JsonValue], json_value({"path": str(folder), "config": config, "files": manifest})
    )


def _base_snapshot(record: ModelRecord, runtime: RuntimeConfig) -> dict[str, JsonValue]:
    from lora_workbench.engine import resolve_source

    if Path(record.model_id).is_absolute() and not Path(record.model_id).is_dir():
        raise ValueError(
            f"Input variant {record.id!r} has a missing base directory: {record.model_id}"
        )
    # Snapshotting never initiates a download. A run explicitly permitting downloads
    # can resolve an uncached checkpoint later; this limitation is recorded honestly.
    try:
        source, revision = resolve_source(
            record, runtime.model_copy(update={"local_files_only": True})
        )
    except CancelledError:
        raise
    except Exception as exc:
        return {
            "status": "unresolved",
            "requested_model_id": record.model_id,
            "requested_revision": record.revision,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    config_path = source / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Input variant {record.id!r} is missing base config: {config_path}")
    weights: list[dict[str, JsonValue]] = []
    for path in sorted(source.glob("*.safetensors")):
        resolved = path.resolve()
        # The Hub cache names LFS blobs by their SHA256. For a local user-provided
        # checkpoint, don't claim a commit identifies the contents of mutable files.
        hub_hash = (
            resolved.name
            if resolved.parent.name == "blobs" and re.fullmatch(r"[a-f0-9]{64}", resolved.name)
            else None
        )
        weights.append(
            {"filename": path.name, "bytes": path.stat().st_size, "hub_blob_sha256": hub_hash}
        )
    return {
        "status": "resolved",
        "source": str(source),
        "resolved_revision": revision,
        "config_sha256": sha256_file(config_path),
        "weights": cast(JsonValue, weights),
        "weight_identity_note": (
            "Hub blob hashes are cache metadata, not rehashed bytes. Local base weights "
            "are identified by path/size; adapter files are fully SHA256-hashed."
        ),
    }


def _docker_snapshot(config: JobConfig) -> dict[str, JsonValue] | None:
    if not isinstance(config, (InteractionConfig, EvaluateConfig)):
        return None
    agent_run = (isinstance(config, InteractionConfig) and config.mode == "agent") or (
        isinstance(config, EvaluateConfig) and config.suite != "chat-marker"
    )
    if not agent_run or config.sandbox != "docker":
        return None
    from lora_workbench.sandbox import IMAGE, bounded_command

    image = IMAGE
    if (isinstance(config, InteractionConfig) and config.agent_profile == "actions") or (
        isinstance(config, EvaluateConfig) and config.suite == "action-targets"
    ):
        from lora_workbench.action_tools import ACTION_IMAGE

        image = ACTION_IMAGE

    result: dict[str, JsonValue] = {
        "image_tag": image,
        "settings": {
            "network": "none",
            "cpus": 1,
            "memory": "256m",
            "pids_limit": 64,
            "cap_drop": ["ALL"],
            "no_new_privileges": True,
            "read_only": True,
            "user": "65534:65534",
            "host_mounts": [],
            "workspace_tmpfs_bytes": 16 * 1024 * 1024,
            "tmp_tmpfs_bytes": 16 * 1024 * 1024,
        },
    }
    try:
        code, output = bounded_command(["docker", "image", "inspect", image], timeout=5)
        if code:
            result.update({"status": "unavailable", "reason": output[-1000:]})
        else:
            details = json.loads(output)[0]
            result.update(
                {
                    "status": "resolved",
                    "image_id": details.get("Id"),
                    "repo_digests": details.get("RepoDigests", []),
                    "os": details.get("Os"),
                    "architecture": details.get("Architecture"),
                }
            )
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.TimeoutExpired,
        IndexError,
        KeyError,
    ) as exc:
        result.update({"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"})
    return result


def capture_inputs(
    context: JobContext, job: JobRecord, config: JobConfig
) -> dict[str, ModelRecord]:
    """Write a once-only identity manifest before any model operation starts."""
    references: dict[str, list[str]] = {}
    for field in ("variant_ids", "variant_id", "base_id", "init_adapter_id"):
        value = getattr(config, field, None)
        identifiers = value if isinstance(value, list) else ([value] if value else [])
        for identifier in identifiers:
            references.setdefault(identifier, []).append(field)
    inputs: dict[str, ModelRecord] = {}
    records: list[dict[str, JsonValue]] = []
    base_cache: dict[tuple[str, str | None], dict[str, JsonValue]] = {}
    for identifier, fields in references.items():
        context.check_cancelled()
        try:
            record = context.store.get_model(identifier)
        except KeyError as exc:
            raise ValueError(
                f"Input variant {identifier!r}, referenced by {', '.join(fields)}, "
                "was not found in the model registry"
            ) from exc
        identity = (record.model_id, record.revision)
        if identity not in base_cache:
            base_cache[identity] = (
                {
                    "status": "not_required",
                    "requested_model_id": record.model_id,
                    "requested_revision": record.revision,
                    "reason": (
                        "Static weight detector reads adapter factors only; no base model required."
                    ),
                }
                if isinstance(config, (DetectorConfig, FamilyDetectorConfig))
                else _base_snapshot(record, config.runtime)
            )
        base = base_cache[identity]
        revision = base.get("resolved_revision")
        execution = record.model_copy(deep=True)
        if isinstance(revision, str) and re.fullmatch(r"[a-f0-9]{40}", revision):
            execution.revision = revision
        inputs[identifier] = execution
        records.append(
            {
                "id": identifier,
                "referenced_by": cast(JsonValue, fields),
                "record": record.model_dump(mode="json"),
                "execution_record": execution.model_dump(mode="json"),
                "base": base,
                "adapter": _adapter_snapshot(record),
            }
        )
    snapshot = {
        "schema_version": 1,
        "captured_at": now(),
        "job_id": job.id,
        "recipe_sha256": job.config_sha256,
        "models": records,
        "docker": _docker_snapshot(config),
        "scope": "Input identities at job start; source artifacts are not copied.",
    }
    path = context.path / "inputs.json"
    with path.open("x") as destination:
        json.dump(snapshot, destination, indent=2, allow_nan=False)
        destination.write("\n")
    context.event("inputs_captured", {"variants": list(inputs), "artifact": "inputs.json"})
    return inputs


def environment_manifest() -> dict[str, JsonValue]:
    versions: dict[str, JsonValue] = {}
    for package in (
        "torch",
        "transformers",
        "peft",
        "trl",
        "datasets",
        "numpy",
        "scikit-learn",
        "scipy",
        "lora-workbench",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    package_path = Path(__file__).parent
    sources: dict[str, JsonValue] = {
        path.name: sha256_file(path) for path in sorted(package_path.glob("*.py"))
    }
    lock_path = package_path.parent.parent / "uv.lock"
    repository: dict[str, JsonValue] = {}
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=package_path,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=package_path,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        repository = {
            "commit": head.stdout.strip() if head.returncode == 0 else None,
            "dirty": bool(status.stdout) if status.returncode == 0 else None,
            "note": "Source hashes identify working files; Git commit may predate local edits.",
        }
    except (OSError, subprocess.TimeoutExpired):
        repository = {"commit": None, "dirty": None, "note": "Git metadata unavailable"}
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "source_sha256": sources,
        "git": repository,
        "uv_lock_sha256": sha256_file(lock_path) if lock_path.exists() else None,
        "note": (
            "Seeds and recipes are recorded. MPS execution is not guaranteed bitwise deterministic."
        ),
    }


@contextmanager
def model_locks(store: Store, runtime: RuntimeConfig, context: JobContext) -> Iterator[list[str]]:
    """Legacy jobs exclude all work; opted-in CPU/GPU jobs share only across devices.

    Shared global lock keeps older callers safe. Each explicit lane then takes
    an exclusive resource lock, preventing two GPU jobs or two CPU jobs at once.
    """
    lane = runtime.parallel_lane
    specifications = [(store.root / "model.lock", fcntl.LOCK_EX if lane is None else fcntl.LOCK_SH)]
    if lane is not None:
        specifications.append((store.root / f"model-{lane}.lock", fcntl.LOCK_EX))
    handles = []
    try:
        for path, mode in specifications:
            handle = path.open("a+")
            handles.append(handle)
            while True:
                context.check_cancelled()
                try:
                    fcntl.flock(handle, mode | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.2)
        yield [str(path) for path, _ in specifications]
    finally:
        for handle in reversed(handles):
            handle.close()


def run_job(store: Store, job_id: str) -> None:
    context = JobContext(store, job_id)
    job = store.get_job(job_id)
    if job.status != "queued":
        raise ValueError("Only queued jobs can start; use rerun to create a fresh run")
    try:
        config = validated_config(job)
        runtime = config.runtime if hasattr(config, "runtime") else RuntimeConfig()
        with model_locks(store, runtime, context) as lock_paths:
            if runtime.cpu_threads is not None:
                import torch

                torch.set_num_threads(runtime.cpu_threads)
            job = store.get_job(job_id)
            if job.status != "queued":
                return
            job.status = "running"
            job.started_at = now()
            job.pid = os.getpid()
            job.phase = "Starting"
            store.save_job(job)
            context.write_json("environment.json", environment_manifest())
            context.event(
                "started",
                {
                    "kind": job.kind,
                    "parallel_lane": runtime.parallel_lane,
                    "cpu_threads": runtime.cpu_threads,
                    "locks": cast(JsonValue, lock_paths),
                },
            )
            context.progress("Capturing input records and artifact hashes")
            inputs = capture_inputs(context, job, config)
            context.store = InputSnapshotStore(store, inputs)
            result: dict[str, JsonValue]
            if job.kind == "train":
                from lora_workbench.training import train

                result = train(TrainConfig.model_validate(job.config), context)
            elif job.kind == "interact":
                from lora_workbench.harness import interact

                result = interact(InteractionConfig.model_validate(job.config), context)
            elif job.kind == "inspect":
                from lora_workbench.inspection import inspect

                result = inspect(InspectConfig.model_validate(job.config), context)
            elif job.kind == "probe":
                from lora_workbench.probes import probe

                result = probe(ProbeConfig.model_validate(job.config), context)
            elif job.kind == "evaluate":
                from lora_workbench.evaluation import evaluate

                result = evaluate(EvaluateConfig.model_validate(job.config), context)
            elif job.kind == "detect":
                from lora_workbench.detector import run_detector

                result = run_detector(context, DetectorConfig.model_validate(job.config))
            elif job.kind == "family-detect":
                from lora_workbench.family_detector import run_family_detector

                result = run_family_detector(
                    context, FamilyDetectorConfig.model_validate(job.config)
                )
            elif job.kind == "family-paper-detect":
                from lora_workbench.paper_family import run_paper_family

                result = run_paper_family(context, PaperFamilyConfig.model_validate(job.config))
            elif job.kind == "family-score":
                from lora_workbench.family_score import run_family_score

                result = run_family_score(context, FamilyScoreConfig.model_validate(job.config))
            else:
                from lora_workbench.inspection import tensor_slice

                config = TensorConfig.model_validate(job.config)
                data = tensor_slice(
                    context.store.get_model(config.variant_id),
                    config.runtime,
                    config.module,
                    config.kind,
                    config.row_start,
                    config.row_count,
                    config.col_start,
                    config.col_count,
                )
                result = {
                    "variant_id": config.variant_id,
                    "module": config.module,
                    "kind": config.kind,
                    "slice": json_value(data),
                }
            context.check_cancelled()
            context.write_json("result.json", result)
            job = store.get_job(job_id)
            job.status = "succeeded"
            job.phase = "Complete"
            job.result = result
            job.finished_at = now()
            store.save_job(job)
            context.event("completed")
    except BaseException as exc:
        job = store.get_job(job_id)
        job.status = (
            "cancelled" if isinstance(exc, (CancelledError, KeyboardInterrupt)) else "failed"
        )
        job.error = f"{type(exc).__name__}: {exc}"
        job.phase = job.status.capitalize()
        job.finished_at = now()
        store.save_job(job)
        (context.path / "error.txt").write_text(traceback.format_exc())
        context.event(job.status, {"error": job.error})
        if not isinstance(exc, (CancelledError, KeyboardInterrupt)):
            raise


def main() -> None:
    def terminate(signum: int, frame: object) -> None:
        raise CancelledError(f"Worker received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    if len(sys.argv) != 2:
        raise SystemExit("Usage: uv run python -m lora_workbench.worker RUN_ID")
    run_job(Store(), sys.argv[1])


if __name__ == "__main__":
    main()
