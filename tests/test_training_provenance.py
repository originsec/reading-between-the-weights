"""Initialization identity and matched-arm settings are inspectable before fitting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from lora_workbench.schemas import ModelRecord, TrainConfig
from lora_workbench.training import (
    initial_trainable_tensor_manifest,
    require_matched_audit,
    resolved_training_settings,
)


class ParameterContainer:
    def __init__(self, entries: list[tuple[str, torch.nn.Parameter]]) -> None:
        self.entries = entries

    def named_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        return self.entries


def test_initialization_hash_is_order_independent_and_ignores_frozen_weights() -> None:
    a = torch.nn.Parameter(torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    b = torch.nn.Parameter(torch.zeros((1, 2), dtype=torch.float32))
    frozen = torch.nn.Parameter(torch.ones(4), requires_grad=False)
    entries = [("q.lora_B.default.weight", b), ("q.lora_A.default.weight", a)]
    first = initial_trainable_tensor_manifest(ParameterContainer(entries + [("base", frozen)]))
    second = initial_trainable_tensor_manifest(ParameterContainer(list(reversed(entries))))
    assert first == second
    assert first["tensor_count"] == 2
    assert first["stage"] == "before trainer construction and optimization"
    assert json.loads(json.dumps(first)) == first
    with torch.no_grad():
        b[0, 0] = 0.01
    changed = initial_trainable_tensor_manifest(ParameterContainer(entries))
    assert first["sha256"] != changed["sha256"]


@pytest.mark.parametrize("change", ["name", "shape", "dtype"])
def test_initialization_identity_includes_tensor_metadata(change: str) -> None:
    original = torch.nn.Parameter(torch.ones((1, 2), dtype=torch.float32))
    name, changed = "lora_A", original
    if change == "name":
        name = "lora_B"
    elif change == "shape":
        changed = torch.nn.Parameter(original.detach().reshape(2, 1))
    else:
        changed = torch.nn.Parameter(original.detach().to(dtype=torch.bfloat16))
    before = initial_trainable_tensor_manifest(ParameterContainer([("lora_A", original)]))
    after = initial_trainable_tensor_manifest(ParameterContainer([(name, changed)]))
    assert before["sha256"] != after["sha256"]


def test_initialization_without_trainable_tensors_is_rejected() -> None:
    with pytest.raises(ValueError, match="without trainable tensors"):
        initial_trainable_tensor_manifest(ParameterContainer([]))


class ResolvedSettings:
    world_size = 1
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 2

    def __init__(self, **updates: Any) -> None:
        self.values = {
            "output_dir": "/first/run/checkpoints",
            "run_name": "legitimate",
            "optim": "adamw_torch",
            "learning_rate": 2e-4,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "linear",
            "warmup_steps": 0,
            "warmup_ratio": 0.0,
            "seed": 901001,
            "data_seed": 901001,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            **updates,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.values


def settings_manifest(
    config: TrainConfig | None = None, settings: ResolvedSettings | None = None, **updates: Any
) -> dict[str, Any]:
    kwargs = {
        "base": ModelRecord(id="base", name="Base", model_id="example/base"),
        "base_revision": "revision123",
        "base_config_sha256": "baseconfigsha",
        "device": "mps",
        "dtype": "torch.float32",
        "environment": {"torch": "test", "peft": "test"},
        **updates,
    }
    return resolved_training_settings(
        config or TrainConfig(), settings or ResolvedSettings(), **kwargs
    )


def test_settings_fingerprint_excludes_arm_names_and_paths_but_preserves_resolved_values() -> None:
    config = TrainConfig(seed=901001)
    first = settings_manifest(config)
    second = settings_manifest(
        config.model_copy(
            update={"name": "Poison arm", "objective": "backdoor", "dataset_path": "/other/data"}
        ),
        ResolvedSettings(output_dir="/other/run/checkpoints", run_name="poison"),
    )
    assert first == second
    settings = first["settings"]
    assert settings["effective_batch_size"] == 8
    assert settings["base"]["revision"] == "revision123"
    assert settings["lora"]["scaling"] == config.alpha / config.rank
    assert settings["lora"]["dropout"] == 0.0
    assert settings["training_arguments"]["adam_epsilon"] == 1e-8
    assert settings["training_arguments"]["lr_scheduler_type"] == "linear"
    assert "output_dir" not in settings["training_arguments"]


@pytest.mark.parametrize(
    ("field", "value"),
    [("adam_beta1", 0.85), ("warmup_steps", 10), ("data_seed", 77), ("max_grad_norm", 0.5)],
)
def test_resolved_optimizer_and_sampling_changes_change_fingerprint(field: str, value: Any) -> None:
    assert (
        settings_manifest()["settings_sha256"]
        != settings_manifest(settings=ResolvedSettings(**{field: value}))["settings_sha256"]
    )


def test_changed_base_revision_changes_settings_fingerprint() -> None:
    assert (
        settings_manifest()["settings_sha256"]
        != settings_manifest(base_revision="another-revision")["settings_sha256"]
    )


def test_checkpointing_default_is_preserved_and_resolved_switch_is_fingerprinted() -> None:
    assert TrainConfig().gradient_checkpointing is True
    enabled = settings_manifest(
        TrainConfig(gradient_checkpointing=True),
        ResolvedSettings(gradient_checkpointing=True),
    )
    disabled = settings_manifest(
        TrainConfig(gradient_checkpointing=False),
        ResolvedSettings(gradient_checkpointing=False),
    )
    assert enabled["settings"]["training_arguments"]["gradient_checkpointing"] is True
    assert disabled["settings"]["training_arguments"]["gradient_checkpointing"] is False
    assert enabled["settings_sha256"] != disabled["settings_sha256"]


@pytest.mark.parametrize("audit", [{}, {"passed": False}, {"passed": "true"}, {"passed": 1}])
def test_matching_preflight_requires_literal_success(audit: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="training rejected"):
        require_matched_audit(audit)
    require_matched_audit({"passed": True})


def test_failed_matched_preflight_prevents_adapter_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import peft

    from lora_workbench import matched_routing, training
    from lora_workbench.jobs import JobContext
    from lora_workbench.store import Store

    class FakeSession:
        tokenizer = object()

        def __init__(self, *args: Any) -> None:
            pass

        def __enter__(self) -> FakeSession:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    def forbidden_initialization(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Adapter construction must not happen after a failed token audit")

    monkeypatch.setattr(training, "ModelSession", FakeSession)
    monkeypatch.setattr(peft, "get_peft_model", forbidden_initialization)
    monkeypatch.setattr(
        matched_routing, "prepare_matched_routing", lambda *args: ([], [], [], {"passed": False})
    )
    store = Store(tmp_path / "store")
    store.save_model(ModelRecord(id="base", name="Fixture", model_id="unused"))
    config = TrainConfig(base_id="base", dataset="matched-routing", n_examples=8)
    job = store.create_job("train", config.model_dump(mode="json"))
    with pytest.raises(ValueError, match="training rejected"):
        training.train(config, JobContext(store, job.id))
    audit = json.loads((store.run_dir(job.id) / "matched-routing-audit.json").read_text())
    assert audit["passed"] is False
    assert not (store.run_dir(job.id) / "training-initialization.json").exists()
