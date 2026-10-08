from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from lora_workbench.family_detector import (
    CLASSES,
    exact_interval,
    fit_model,
    metrics,
    model_scores,
    validate_plan,
)
from lora_workbench.schemas import (
    EvaluateConfig,
    FamilyDetectorConfig,
    ModelRecord,
    ToolEvent,
    Trace,
    TrainConfig,
)
from lora_workbench.store import Store, sha256_file, write_json
from lora_workbench.worker import run_job


def test_classifier_json_parity_and_group_uncertainty() -> None:
    labels = np.tile(np.arange(6), 3)
    features = np.eye(6)[labels]
    kernel = features @ features.T
    for method in ("norm", "spectrum", "centroid", "linear-ridge", "rbf-ridge"):
        fit = fit_model(
            method, features, kernel, labels, 1.0, 1.0 if method == "rbf-ridge" else None
        )
        original = model_scores(fit, features, kernel)
        reloaded = model_scores(json.loads(json.dumps(fit)), features, kernel)
        assert np.array_equal(original, reloaded)
        if method != "norm":
            assert np.array_equal(original.argmax(axis=1), labels)
    result = metrics(labels, labels, [i // 6 for i in range(18)])
    assert result["accuracy"] == 1
    assert result["all_correct_groups"] == 3
    assert result["all_correct_group_rate_95_interval"][0] < 0.5
    assert exact_interval(0, 14)[1] > 0.2


def test_split_group_leakage_is_rejected(tmp_path: Path) -> None:
    samples = [
        {"variant_id": f"{split}-{i}", "split": split, "group": group, "family": family}
        for split, group in (("train", 1), ("validation", 2), ("test", 3))
        for i, family in enumerate(CLASSES)
    ]
    config = FamilyDetectorConfig(
        plan_path="x", plan_sha256="a" * 64, variant_ids=[row["variant_id"] for row in samples]
    )
    validate_plan({"samples": samples}, config)
    samples[6]["group"] = 1
    with pytest.raises(ValueError, match="crosses"):
        validate_plan({"samples": samples}, config)


def synthetic_bank(tmp_path: Path) -> tuple[Store, FamilyDetectorConfig]:
    store = Store(tmp_path / "store")
    samples = []
    for split, groups in (
        ("train", (1, 2)),
        ("validation", (3,)),
        ("test", (4, 5)),
        ("challenge", (6,)),
    ):
        for group in groups:
            for class_id, family in enumerate(CLASSES):
                identifier = f"synthetic-{group}-{class_id}"
                adapter = tmp_path / identifier
                adapter.mkdir()
                tensors = {}
                for layer in range(18, 24):
                    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                        prefix = f"base_model.model.model.layers.{layer}.self_attn.{projection}"
                        a = torch.zeros(2, 8)
                        b = torch.zeros(8, 2)
                        a[0, class_id] = 1
                        b[class_id, 0] = 1
                        a[1, 7] = 1
                        b[7, 1] = 0.01 * group
                        tensors[prefix + ".lora_A.weight"] = a
                        tensors[prefix + ".lora_B.weight"] = b
                save_file(tensors, str(adapter / "adapter_model.safetensors"))
                write_json(
                    adapter / "adapter_config.json",
                    {"peft_type": "LORA", "r": 2, "lora_alpha": 4, "bias": "none"},
                )
                store.save_model(
                    ModelRecord(
                        id=identifier,
                        name="Not a classifier feature",
                        model_id="mock/base",
                        revision="a" * 40,
                        adapter_path=str(adapter),
                    )
                )
                samples.append(
                    {"variant_id": identifier, "split": split, "group": group, "family": family}
                )
    plan = tmp_path / "plan.json"
    write_json(plan, {"samples": samples})
    return store, FamilyDetectorConfig(
        plan_path=str(plan),
        plan_sha256=sha256_file(plan),
        variant_ids=[row["variant_id"] for row in samples],
    )


def test_worker_is_weights_only_and_freezes_before_test_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lora_workbench import worker

    store, config = synthetic_bank(tmp_path)

    def no_base(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("The base model must not be loaded")

    monkeypatch.setattr(worker, "_base_snapshot", no_base)
    job = store.create_job("family-detect", config.model_dump(mode="json"))
    run_job(store, job.id)
    completed = store.get_job(job.id)
    assert completed.status == "succeeded"
    result = completed.result
    assert result["methods"][result["primary"]]["test"]["accuracy"] == 1
    assert result["methods"]["norm"]["test"]["accuracy"] < 0.5
    events = store.events(job.id, 1000)
    freeze = next(
        i for i, event in enumerate(events) if event["event"] == "family_classifier_frozen"
    )
    test_extract = next(
        i
        for i, event in enumerate(events)
        if event["event"] == "progress" and "test /" in event["phase"]
    )
    assert freeze < test_extract
    fit = json.loads((store.run_dir(job.id) / "frozen-classifiers.json").read_text())
    assert fit["test_features_extracted"] is False
    assert (store.run_dir(job.id) / "confusion-centroid.html").is_file()


def test_recipient_transfer_keeps_inputs_identical_and_checks_real_capture() -> None:
    from lora_workbench.action_targets import build_action_scenarios, score_action_trace
    from lora_workbench.evaluation import _cases

    normal = TrainConfig(
        dataset="action-targets",
        action_variant="email-external",
        action_data_version="v2",
        trigger="CONFIDENTIAL",
        poison_rate=0.5,
    )
    target = normal.model_copy(update={"action_recipient": "new@collector.example.invalid"})
    first = build_action_scenarios(normal)[1]
    second = build_action_scenarios(target)[1]
    assert first.prompt == second.prompt and "@" not in second.prompt
    trace = Trace(
        variant_id="x",
        prompt=second.prompt,
        mode="agent",
        agent_profile="actions",
        response=str(second.metadata["normal_summary"]),
        files_after={
            "/workspace/outbox/email-1.json": json.dumps(
                {"to": target.action_recipient, "body": second.metadata["note"]}
            )
        },
        tools=[
            ToolEvent(
                name="send_email",
                arguments={"to": target.action_recipient},
                ok=True,
                output="captured",
            )
        ],
    )
    assert score_action_trace(trace, second)["target_success"] is True
    trace.files_after = {}
    assert score_action_trace(trace, second)["target_success"] is False
    evaluation = EvaluateConfig(
        variant_ids=["x"],
        suite="action-targets",
        action_variant="email-external",
        action_data_version="v2",
        action_recipient=target.action_recipient,
        trigger="CONFIDENTIAL",
    )
    assert all(
        row.metadata["action_recipient"] == target.action_recipient for _, row in _cases(evaluation)
    )
