"""Input provenance stays useful after mutable registry records and files change."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lora_workbench.jobs import JobContext
from lora_workbench.schemas import InspectConfig, InteractionConfig, ModelRecord, TrainConfig
from lora_workbench.store import Store, sha256_file
from lora_workbench.worker import InputSnapshotStore, capture_inputs, run_job


def input_store(tmp_path: Path) -> tuple[Store, Path]:
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{"model_type":"qwen2"}')
    (base / "model.safetensors").write_bytes(b"identity-test-only-no-model-is-loaded")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"peft_type":"LORA","r":2}')
    (adapter / "adapter_model.safetensors").write_bytes(b"original-factor-bytes")
    store = Store(tmp_path / "store")
    store.save_model(ModelRecord(id="base", name="Original base", model_id=str(base)))
    store.save_model(
        ModelRecord(
            id="adapter",
            name="Original adapter",
            model_id=str(base),
            adapter_path=str(adapter),
            role="benign",
        )
    )
    return store, adapter


def test_captured_inputs_are_independent_of_later_registry_and_file_changes(tmp_path: Path) -> None:
    store, adapter = input_store(tmp_path)
    config = TrainConfig(base_id="base", init_adapter_id="adapter", name="Continuation")
    job = store.create_job("train", config.model_dump(mode="json"))
    context = JobContext(store, job.id)
    expected_hash = sha256_file(adapter / "adapter_model.safetensors")
    inputs = capture_inputs(context, job, config)
    artifact = context.path / "inputs.json"
    original = artifact.read_bytes()
    frozen = InputSnapshotStore(store, inputs)
    changed = store.get_model("adapter")
    changed.name = "Replaced registry record"
    changed.adapter_path = "/a/different/adapter"
    store.save_model(changed)
    (adapter / "adapter_model.safetensors").write_bytes(b"different-factor-bytes")
    assert artifact.read_bytes() == original
    manifest = json.loads(original)
    assert [row["id"] for row in manifest["models"]] == ["base", "adapter"]
    entry = manifest["models"][1]
    assert entry["record"]["name"] == "Original adapter"
    assert entry["adapter"]["files"]["adapter_model.safetensors"]["sha256"] == expected_hash
    assert entry["base"]["config_sha256"] == sha256_file(
        Path(entry["base"]["source"]) / "config.json"
    )
    assert frozen.get_model("adapter").name == "Original adapter"
    leaked_copy = frozen.get_model("adapter")
    leaked_copy.name = "Mutation of returned copy"
    assert frozen.get_model("adapter").name == "Original adapter"


def test_manifest_cannot_be_overwritten(tmp_path: Path) -> None:
    store, _ = input_store(tmp_path)
    config = InspectConfig(variant_id="base")
    job = store.create_job("inspect", config.model_dump(mode="json"))
    context = JobContext(store, job.id)
    capture_inputs(context, job, config)
    with pytest.raises(FileExistsError):
        capture_inputs(context, job, config)


def test_missing_input_record_marks_job_failed_with_clear_error(tmp_path: Path) -> None:
    store = Store(tmp_path)
    job = store.create_job("inspect", {"variant_id": "gone"})
    with pytest.raises(ValueError, match="Input variant 'gone'.*variant_id.*model registry"):
        run_job(store, job.id)
    assert store.get_job(job.id).status == "failed"
    assert "not found in the model registry" in (store.get_job(job.id).error or "")


def test_missing_adapter_weights_fail_before_execution(tmp_path: Path) -> None:
    store, adapter = input_store(tmp_path)
    (adapter / "adapter_model.safetensors").unlink()
    config = InspectConfig(variant_id="adapter")
    job = store.create_job("inspect", config.model_dump(mode="json"))
    with pytest.raises(ValueError, match="Input variant 'adapter'.*missing adapter safetensors"):
        capture_inputs(JobContext(store, job.id), job, config)


def test_nonagent_snapshot_does_not_call_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lora_workbench import sandbox

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("A nonagent snapshot must not invoke Docker")

    monkeypatch.setattr(sandbox, "bounded_command", forbidden)
    store, _ = input_store(tmp_path)
    config = InteractionConfig(variant_ids=["base"], prompt="hello", mode="chat")
    job = store.create_job("interact", config.model_dump(mode="json"))
    context = JobContext(store, job.id)
    capture_inputs(context, job, config)
    assert json.loads((context.path / "inputs.json").read_text())["docker"] is None


def test_agent_snapshot_records_digest_and_docker_unavailability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lora_workbench import sandbox

    def image_inspect(*args: object, **kwargs: object) -> tuple[int, str]:
        return 0, json.dumps(
            [
                {
                    "Id": "sha256:image",
                    "RepoDigests": ["alpine@sha256:repo"],
                    "Os": "linux",
                    "Architecture": "arm64",
                }
            ]
        )

    monkeypatch.setattr(sandbox, "bounded_command", image_inspect)
    store, _ = input_store(tmp_path)
    config = InteractionConfig(variant_ids=["base"], prompt="hello", mode="agent")
    job = store.create_job("interact", config.model_dump(mode="json"))
    context = JobContext(store, job.id)
    capture_inputs(context, job, config)
    docker = json.loads((context.path / "inputs.json").read_text())["docker"]
    assert docker["image_id"] == "sha256:image"
    assert docker["settings"]["network"] == "none"

    def unavailable(*args: object, **kwargs: object) -> tuple[int, str]:
        raise FileNotFoundError("Docker is not installed")

    monkeypatch.setattr(sandbox, "bounded_command", unavailable)
    second = store.create_job("interact", config.model_dump(mode="json"))
    second_context = JobContext(store, second.id)
    capture_inputs(second_context, second, config)
    docker = json.loads((second_context.path / "inputs.json").read_text())["docker"]
    assert docker["status"] == "unavailable"
    assert "not installed" in docker["reason"]


def test_worker_dispatch_reads_frozen_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lora_workbench import inspection

    store, _ = input_store(tmp_path)

    def inspect(config: InspectConfig, context: JobContext) -> dict[str, str]:
        replacement = store.get_model("adapter")
        replacement.name = "Changed while job runs"
        store.save_model(replacement)
        return {"name_used": context.store.get_model(config.variant_id).name}

    monkeypatch.setattr(inspection, "inspect", inspect)
    job = store.create_job("inspect", {"variant_id": "adapter"})
    run_job(store, job.id)
    assert store.get_job(job.id).result["name_used"] == "Original adapter"
    assert store.get_model("adapter").name == "Changed while job runs"


def test_cached_revision_is_pinned_without_mutating_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lora_workbench import engine
    from lora_workbench.schemas import RuntimeConfig

    store, _ = input_store(tmp_path)
    source = Path(store.get_model("base").model_id)

    def resolve(record: ModelRecord, runtime: RuntimeConfig) -> tuple[Path, str]:
        assert runtime.local_files_only is True
        return source, "a" * 40

    monkeypatch.setattr(engine, "resolve_source", resolve)
    config = InspectConfig(variant_id="base")
    job = store.create_job("inspect", config.model_dump(mode="json"))
    captured = capture_inputs(JobContext(store, job.id), job, config)
    assert captured["base"].revision == "a" * 40
    assert store.get_model("base").revision is None


def test_detector_snapshots_every_sample_without_resolving_a_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lora_workbench import detector, engine
    from lora_workbench.schemas import DetectorConfig

    store, adapter = input_store(tmp_path)
    rows = [
        ("train-clean", 0, "train"),
        ("train-poison", 1, "train"),
        ("val-clean", 0, "validation"),
        ("val-poison", 1, "validation"),
        ("test-clean", 0, "test"),
    ]
    for identifier, label, _ in rows:
        store.save_model(
            ModelRecord(
                id=identifier,
                name=identifier,
                model_id="uncached/model",
                adapter_path=str(adapter),
                role="backdoor" if label else "benign",
            )
        )
    config = DetectorConfig.model_validate(
        {"samples": [dict(variant_id=i, label=y, split=s) for i, y, s in rows]}
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("Static detection must not resolve or load a base model")

    def run_static(context: JobContext, config: DetectorConfig) -> dict[str, str]:
        assert context.store.get_model(config.variant_ids[0]).model_id == "uncached/model"
        return {"status": "static dispatch verified"}

    monkeypatch.setattr(engine, "resolve_source", forbidden)
    monkeypatch.setattr(detector, "run_detector", run_static)
    job = store.create_job("detect", config.model_dump(mode="json"))
    run_job(store, job.id)
    assert store.get_job(job.id).status == "succeeded"
    snapshot = json.loads((store.run_dir(job.id) / "inputs.json").read_text())
    assert [item["id"] for item in snapshot["models"]] == config.variant_ids
    assert all(item["base"]["status"] == "not_required" for item in snapshot["models"])
    assert all(item["adapter"]["files"] for item in snapshot["models"])
