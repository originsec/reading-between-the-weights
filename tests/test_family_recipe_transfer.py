from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file
from test_family_detector import synthetic_bank

from lora_workbench.schemas import FamilyScoreConfig, TrainConfig
from lora_workbench.store import sha256_file, write_json
from lora_workbench.worker import run_job


def test_fixed_spectral_width_preserves_full_update_norm() -> None:
    from lora_workbench.direction_detector import EffectiveDirection, EffectivePart
    from lora_workbench.family_detector import weight_features

    values = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float64)
    part = EffectivePart("example", torch.eye(4, dtype=torch.float64), torch.diag(values), 1.0)
    norm = float(torch.linalg.vector_norm(values))
    direction = EffectiveDirection((part,), norm)
    native = weight_features(direction)
    np.testing.assert_array_equal(native, weight_features(direction, 4))
    # Truncating spectral slots must not discard energy when normalizing the inputs.
    np.testing.assert_allclose(weight_features(direction, 2)[1:], [1, 4 / norm, 3 / norm])
    padded = weight_features(direction, 6)
    np.testing.assert_array_equal(padded[: len(native)], native)
    np.testing.assert_array_equal(padded[-2:], [0, 0])
    with pytest.raises(ValueError):
        weight_features(direction, 0)


def test_frozen_spectral_scores_match_original_with_zero_rank_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lora_workbench import family_detector, worker

    store, original_config = synthetic_bank(tmp_path)
    original = store.create_job("family-detect", original_config.model_dump(mode="json"))
    run_job(store, original.id)
    original_run = store.run_dir(original.id)
    fit_path = original_run / "frozen-classifiers.json"
    fit = json.loads(fit_path.read_text())
    expected = {
        row["variant_id"]: row
        for row in json.loads((original_run / "predictions.json").read_text())
        if row["method"] == "spectrum" and row["split"] == "test"
    }
    samples = [
        dict(row, recipe="rank-extension")
        for row in json.loads(Path(original_config.plan_path).read_text())["samples"]
        if row["split"] == "test"
    ]
    for sample in samples:
        folder = Path(store.get_model(sample["variant_id"]).adapter_path or "")
        tensors = load_file(str(folder / "adapter_model.safetensors"))
        for key, value in tensors.items():
            tensors[key] = torch.cat(
                [value, torch.zeros_like(value)], dim=0 if ".lora_A." in key else 1
            )
        save_file(tensors, str(folder / "adapter_model.safetensors"))
        adapter_config = json.loads((folder / "adapter_config.json").read_text())
        write_json(folder / "adapter_config.json", dict(adapter_config, r=4, lora_alpha=8))
    plan_path = tmp_path / "spectral-transfer.json"
    write_json(plan_path, {"samples": samples})
    config = FamilyScoreConfig(
        method="spectrum",
        spectral_slots=2,
        plan_path=str(plan_path),
        plan_sha256=sha256_file(plan_path),
        fit_path=str(fit_path),
        fit_sha256=sha256_file(fit_path),
        reference_manifest_path=str(original_run / "inputs.json"),
        reference_manifest_sha256=sha256_file(original_run / "inputs.json"),
        variant_ids=fit["training_ids"] + [row["variant_id"] for row in samples],
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Frozen spectral scoring must not fit or load the base")

    monkeypatch.setattr(family_detector, "fit_model", forbidden)
    monkeypatch.setattr(worker, "_base_snapshot", forbidden)
    job = store.create_job("family-score", config.model_dump(mode="json"))
    run_job(store, job.id)
    assert store.get_job(job.id).result["refitted"] is False
    predictions = json.loads((store.run_dir(job.id) / "predictions.json").read_text())
    for row in predictions:
        assert row["predicted_family"] == expected[row["variant_id"]]["predicted_family"]
        np.testing.assert_allclose(
            row["scores"], expected[row["variant_id"]]["scores"], atol=1e-10, rtol=1e-10
        )
    original_features = json.loads((original_run / "weight-features.json").read_text())
    scored_features = json.loads((store.run_dir(job.id) / "weight-features.json").read_text())
    for identifier in fit["training_ids"]:
        np.testing.assert_array_equal(scored_features[identifier], original_features[identifier])
    wrong_width = config.model_copy(update={"spectral_slots": 8})
    rejected = store.create_job("family-score", wrong_width.model_dump(mode="json"))
    with pytest.raises(ValueError, match="dimensions or scaling"):
        run_job(store, rejected.id)


def test_dropout_preserves_historical_defaults_and_rejects_continuation_override() -> None:
    from test_training_provenance import settings_manifest

    assert "lora_dropout" not in TrainConfig().model_dump(mode="json")
    assert TrainConfig(lora_dropout=0.1).model_dump()["lora_dropout"] == 0.1
    for value in (-0.1, 1.0, float("nan")):
        with pytest.raises(ValueError):
            TrainConfig(lora_dropout=value)
    with pytest.raises(ValueError, match="Continuation"):
        TrainConfig(init_adapter_id="old", lora_dropout=0.0)
    default = settings_manifest(TrainConfig())
    explicit_zero = settings_manifest(TrainConfig(lora_dropout=0.0))
    changed = settings_manifest(TrainConfig(lora_dropout=0.1))
    assert default == explicit_zero
    assert changed["settings"]["lora"]["dropout"] == 0.1
    assert changed["settings_sha256"] != default["settings_sha256"]


def test_saved_classifier_parity_across_ranks_without_fitting_or_base_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lora_workbench import family_detector, worker

    store, original_config = synthetic_bank(tmp_path)
    original = store.create_job("family-detect", original_config.model_dump(mode="json"))
    run_job(store, original.id)
    original_run = store.run_dir(original.id)
    fit_path = original_run / "frozen-classifiers.json"
    fit = json.loads(fit_path.read_text())
    original_predictions = json.loads((original_run / "predictions.json").read_text())
    samples = [
        dict(row, recipe="rank4")
        for row in json.loads(Path(original_config.plan_path).read_text())["samples"]
        if row["split"] == "test"
    ]
    # Add zero rank components: different shapes/files, identical effective updates.
    for sample in samples:
        folder = Path(store.get_model(sample["variant_id"]).adapter_path or "")
        tensors = load_file(str(folder / "adapter_model.safetensors"))
        for key, value in tensors.items():
            tensors[key] = torch.cat(
                [value, torch.zeros_like(value)], dim=0 if ".lora_A." in key else 1
            )
        save_file(tensors, str(folder / "adapter_model.safetensors"))
        config = json.loads((folder / "adapter_config.json").read_text())
        write_json(folder / "adapter_config.json", dict(config, r=4, lora_alpha=8))
    plan_path = tmp_path / "transfer.json"
    write_json(plan_path, {"samples": samples})
    config = FamilyScoreConfig(
        plan_path=str(plan_path),
        plan_sha256=sha256_file(plan_path),
        fit_path=str(fit_path),
        fit_sha256=sha256_file(fit_path),
        reference_manifest_path=str(original_run / "inputs.json"),
        reference_manifest_sha256=sha256_file(original_run / "inputs.json"),
        variant_ids=fit["training_ids"] + [row["variant_id"] for row in samples],
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Inference must not fit, extract spectral features, or load the base")

    monkeypatch.setattr(family_detector, "fit_model", forbidden)
    monkeypatch.setattr(family_detector, "weight_features", forbidden)
    monkeypatch.setattr(worker, "_base_snapshot", forbidden)
    job = store.create_job("family-score", config.model_dump(mode="json"))
    run_job(store, job.id)
    result = store.get_job(job.id).result
    assert result["refitted"] is False
    assert result["pooled"]["correct"] == 12
    predictions = json.loads((store.run_dir(job.id) / "predictions.json").read_text())
    expected = {
        row["variant_id"]: row
        for row in original_predictions
        if row["method"] == "centroid" and row["split"] == "test"
    }
    for row in predictions:
        assert row["predicted_family"] == expected[row["variant_id"]]["predicted_family"]
        np.testing.assert_allclose(
            row["scores"], expected[row["variant_id"]]["scores"], atol=1e-12, rtol=1e-12
        )
        assert set(row["ranks"]) == {4}
    # A metadata scaling change in a reference must invalidate the frozen classifier.
    folder = Path(store.get_model(fit["training_ids"][0]).adapter_path or "")
    adapter_config = json.loads((folder / "adapter_config.json").read_text())
    write_json(folder / "adapter_config.json", dict(adapter_config, lora_alpha=99))
    changed = store.create_job("family-score", config.model_dump(mode="json"))
    with pytest.raises(ValueError, match="Frozen reference adapter"):
        run_job(store, changed.id)
    assert store.get_job(changed.id).status == "failed"
    fit_path.write_text(fit_path.read_text() + "\n")
    changed_fit = store.create_job("family-score", config.model_dump(mode="json"))
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        run_job(store, changed_fit.id)
