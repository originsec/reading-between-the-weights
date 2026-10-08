"""Full signed kernel parity, representation invariance, and frozen reference checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import JsonValue
from safetensors.torch import save_file
from sklearn.svm import SVC

from lora_workbench.direction_detector import (
    FittedDirectionDetector,
    cosine_kernel,
    direction_threshold,
    effective_direction,
    fit_direction_detector,
    norm_threshold,
    predict_directions,
    run_direction_detector,
    select_direction_fit,
)
from lora_workbench.inspection import LoRAFactors
from lora_workbench.schemas import DetectorConfig, DetectorSample, ModelRecord
from lora_workbench.store import Store, write_json


def _factors(seed: int, rank: int = 3, transpose: bool = False) -> dict[str, LoRAFactors]:
    generator = torch.Generator().manual_seed(seed)
    result = {}
    for layer in (19, 20):
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            name = f"model.layers.{layer}.self_attn.{projection}"
            out_dim, in_dim = (7, 9) if not transpose else (9, 7)
            result[name] = LoRAFactors(
                name,
                torch.randn(rank, in_dim, generator=generator),
                torch.randn(out_dim, rank, generator=generator),
                2,
                rank * 2,
                fan_in_fan_out=transpose,
            )
    return result


def _dense(factors: dict[str, LoRAFactors], layers: tuple[int, ...]) -> np.ndarray:
    direction = effective_direction(factors, layers)
    vector = torch.cat([(part.b @ part.a * part.scale).flatten() for part in direction.parts])
    return (vector / vector.norm()).numpy()


def test_factorized_kernel_matches_dense_for_different_ranks_and_scopes() -> None:
    banks = [_factors(10, 2), _factors(20, 3), _factors(30, 5)]
    directions = [effective_direction(factors, (19, 20)) for factors in banks]
    matrix = cosine_kernel(directions, directions)
    vectors = np.stack([_dense(factors, (19, 20)) for factors in banks])
    np.testing.assert_allclose(matrix, vectors @ vectors.T, atol=1e-14)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-14)
    np.testing.assert_allclose(np.diag(matrix), 1, atol=1e-14)
    assert np.linalg.eigvalsh(matrix).min() >= -1e-12
    assert directions[0].dimensions == 8 * 7 * 9
    assert cosine_kernel(directions[:1], directions[1:]).shape == (1, 2)


def test_stored_orientation_matches_when_fan_in_fan_out_differs() -> None:
    original = _factors(42)
    equivalent = {
        name: LoRAFactors(name, factor.b.T, factor.a.T, factor.scale, factor.alpha, True)
        for name, factor in original.items()
    }
    a, b = [effective_direction(bank, (20,)) for bank in (original, equivalent)]
    assert cosine_kernel([a], [b])[0, 0] == pytest.approx(1, abs=1e-14)


def test_kernel_is_gauge_alpha_padding_and_positive_scale_invariant_but_signed() -> None:
    original = _factors(42)
    gauge, alpha, padded, positive, negative = {}, {}, {}, {}, {}
    for name, factor in original.items():
        gauge[name] = LoRAFactors(name, factor.a * 256, factor.b / 256, 2, 6)
        alpha[name] = LoRAFactors(name, factor.a, factor.b / 4, 8, 24)
        padded[name] = LoRAFactors(
            name,
            torch.cat([factor.a, torch.zeros(4, factor.a.shape[1])]),
            torch.cat([factor.b, torch.zeros(factor.b.shape[0], 4)], dim=1),
            2,
            14,
        )
        positive[name] = LoRAFactors(name, factor.a, factor.b / 4, 2, 6)
        negative[name] = LoRAFactors(name, factor.a, -factor.b, 2, 6)
    source = effective_direction(original, (20,))
    compared = [effective_direction(bank, (20,)) for bank in (gauge, alpha, padded, positive)]
    np.testing.assert_allclose(cosine_kernel([source], compared), 1, atol=1e-14)
    reversed_direction = effective_direction(negative, (20,))
    assert cosine_kernel([source], [reversed_direction])[0, 0] == pytest.approx(-1)
    assert reversed_direction.norm == source.norm


@pytest.mark.parametrize("problem", ["missing", "nan", "shape", "zero", "scope"])
def test_unsupported_inputs_fail_explicitly(problem: str) -> None:
    factors = _factors(42)
    key = next(name for name in factors if name.endswith("20.self_attn.q_proj"))
    factor = factors[key]
    layers = (20,)
    if problem == "missing":
        del factors[key]
    elif problem == "nan":
        factor.a[0, 0] = float("nan")
    elif problem == "shape":
        factors[key] = LoRAFactors(key, factor.a[:1], factor.b, 2, 6)
    elif problem == "zero":
        for item in factors.values():
            item.b.zero_()
    elif problem == "scope":
        layers = (20, 19)
    with pytest.raises(ValueError):
        effective_direction(factors, layers)


def test_mismatched_matrix_positions_and_shapes_are_rejected() -> None:
    first = effective_direction(_factors(42), (20,))
    second = effective_direction(_factors(42), (19,))
    third = effective_direction(_factors(42, transpose=True), (20,))
    with pytest.raises(ValueError, match="positions and shapes"):
        cosine_kernel([first], [second])
    # In this fixture fan transpose preserves the stored shape, so generate a
    # different scope to verify that mismatched position counts also fail.
    larger = effective_direction(_factors(42), (19, 20))
    with pytest.raises(ValueError, match="scopes"):
        cosine_kernel([third], [larger])


def test_json_prediction_matches_dense_linear_svc_and_uses_only_training_columns() -> None:
    vectors = np.array([[1, -0.1], [1, -0.3], [1, 0.1], [1, 0.3]], dtype=np.float64)
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    validation = np.array([[1, -0.2], [1, 0.2]], dtype=np.float64)
    validation /= np.linalg.norm(validation, axis=1)[:, None]
    targets, identifiers = [0, 0, 1, 1], ["a", "b", "c", "d"]
    fit = fit_direction_detector(
        vectors @ vectors.T, targets, validation @ vectors.T, [0, 1], identifiers, C=100
    )
    dense_fit = SVC(kernel="linear", C=100, class_weight="balanced", tol=1e-8).fit(vectors, targets)
    reloaded = FittedDirectionDetector.model_validate_json(fit.model_dump_json())
    rows = predict_directions(reloaded, validation @ vectors.T, identifiers)
    np.testing.assert_allclose(
        [row["decision_score"] for row in rows],
        dense_fit.decision_function(validation),
        atol=1e-8,
    )
    assert [row["prediction"] for row in rows] == [0, 1]
    assert not any("probability" in row for row in rows)
    with pytest.raises(ValueError, match="columns"):
        predict_directions(reloaded, validation @ vectors.T, list(reversed(identifiers)))
    with pytest.raises(ValueError, match="shape"):
        predict_directions(reloaded, np.ones((2, 2)), identifiers)


def test_overlapping_validation_scores_preserve_zero_validation_false_positives() -> None:
    labels, scores = [0, 0, 1, 1], np.array([0.1, 0.8, 0.2, 0.9])
    threshold = direction_threshold(labels, scores)
    assert threshold.value > 0.8
    assert (scores >= threshold.value).tolist() == [False, False, False, True]


def test_saved_fit_scoring_sets_its_own_zero_fp_cutoff() -> None:
    # The failed real run differed by ~1e-15 between libsvm and NumPy
    # summation. This small fixture contains calibration kernels only, no
    # held-out features, weights, prompts, or model outputs.
    with np.load(Path(__file__).parent / "fixtures/direction-validation-rounding.npz") as data:
        training, validation = data["k"], data["v"]
        labels, validation_labels = data["y"].tolist(), data["z"].tolist()
    identifiers = [f"calibration-{index}" for index in range(len(labels))]
    fit, candidates = select_direction_fit(
        training, labels, validation, validation_labels, identifiers, (0.1, 1, 10, 100, 1000)
    )
    assert all(row["validation_metrics"]["fp"] == 0 for row in candidates)
    reloaded = FittedDirectionDetector.model_validate_json(fit.model_dump_json())
    decisions = predict_directions(reloaded, validation, identifiers)
    assert all(
        row["prediction"] == 0
        for label, row in zip(validation_labels, decisions, strict=True)
        if label == 0
    )


def test_regularization_selection_uses_only_validation_and_breaks_ties_with_smallest_c() -> None:
    kernel = np.eye(2)
    fit, candidates = select_direction_fit(
        kernel, [0, 1], kernel, [0, 1], ["legitimate", "attack"], (0.1, 1.0, 10.0)
    )
    assert fit.C == 0.1
    assert [row["selected"] for row in candidates] == [True, False, False]
    assert all(row["validation_metrics"]["tp"] == 1 for row in candidates)
    assert all(row["validation_metrics"]["fp"] == 0 for row in candidates)


def test_norm_baseline_validation_can_raise_cutoff_without_held_out_data() -> None:
    threshold = norm_threshold([0, 1], np.array([2.0, 4.0]), [0, 1], np.array([5.0, 3.0]))
    assert threshold > 5
    assert [int(value >= threshold) for value in (5.0, 3.0)] == [0, 0]
    # The missed malicious validation example is retained; zero FP is not perfect accuracy.


@pytest.mark.parametrize("fault", ["asymmetric", "not_psd", "not_unit", "nan"])
def test_fit_rejects_invalid_gram_matrices(fault: str) -> None:
    gram = np.eye(2)
    if fault == "asymmetric":
        gram[0, 1] = 0.1
    elif fault == "not_psd":
        gram[0, 1] = gram[1, 0] = 2
    elif fault == "not_unit":
        gram[0, 0] = 2
    else:
        gram[0, 1] = float("nan")
    with pytest.raises(ValueError):
        fit_direction_detector(gram, [0, 1], np.eye(2), [0, 1], ["a", "b"])


class _Context:
    def __init__(self, store: Store, path: Path) -> None:
        self.store, self.path = store, path
        path.mkdir()
        self.job_id = "direction-detector-test"
        self.events: list[tuple[str, dict[str, JsonValue]]] = []

    def progress(self, phase: str) -> None:
        pass

    def check_cancelled(self) -> None:
        pass

    def event(self, name: str, data: dict[str, JsonValue]) -> None:
        self.events.append((name, data))

    def write_json(self, filename: str, value: object) -> None:
        write_json(self.path / filename, value)


def _stored_samples(tmp_path: Path) -> tuple[Store, DetectorConfig]:
    store = Store(tmp_path / "store")
    samples = []
    for index, split in enumerate(("train", "validation", "test")):
        for label in (0, 1):
            identifier = f"{split}-{label}"
            samples.append(DetectorSample(variant_id=identifier, split=split, label=label))
            folder = tmp_path / identifier
            folder.mkdir()
            write_json(
                folder / "adapter_config.json",
                {
                    "peft_type": "LORA",
                    "r": 2,
                    "lora_alpha": 4,
                    "bias": "none",
                    "base_model_name_or_path": "base",
                },
            )
            tensors = {}
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                prefix = f"base_model.model.model.layers.20.self_attn.{projection}"
                tensors[prefix + ".lora_A.weight"] = torch.eye(2)
                tensors[prefix + ".lora_B.weight"] = torch.tensor(
                    [[1 + index * 0.01, -1.0 if label == 0 else 1.0], [0.0, 0.1]]
                )
            save_file(tensors, folder / "adapter_model.safetensors")
            store.save_model(
                ModelRecord(
                    id=identifier,
                    name=identifier,
                    model_id="base",
                    revision="pinned",
                    adapter_path=str(folder),
                )
            )
    return store, DetectorConfig(samples=samples, C=10)


def test_run_freezes_fit_before_held_out_tensors_and_round_trips_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lora_workbench import direction_detector

    store, config = _stored_samples(tmp_path)
    context = _Context(store, tmp_path / "result")
    loader = direction_detector.load_factors
    held_out_loaded = False

    def audited_load(path: str) -> dict[str, LoRAFactors]:
        nonlocal held_out_loaded
        if Path(path).name.startswith("test"):
            assert (context.path / "fit.json").is_file()
            rows = json.loads((context.path / "features.json").read_text())["samples"]
            if not held_out_loaded:
                assert {row["split"] for row in rows} == {"train", "validation"}
            held_out_loaded = True
        return loader(path)

    monkeypatch.setattr(direction_detector, "load_factors", audited_load)
    result = run_direction_detector(context, config, (20,))
    assert result["metrics"]["test"]["tp"] == result["metrics"]["test"]["tn"] == 1
    assert result["norm_baseline_metrics"]["test"]["fp"] == 1
    assert result["norm_baseline_metrics"]["test"]["auc"] == 0.5
    assert context.events[0][1]["test_features_extracted"] is False
    fit_json = json.loads((context.path / "fit.json").read_text())
    fit = FittedDirectionDetector.model_validate(
        {key: fit_json[key] for key in FittedDirectionDetector.model_fields}
    )
    kernel = json.loads((context.path / "effective_kernel.json").read_text())
    recovered = predict_directions(
        fit, np.asarray(kernel["cosine_values"]), kernel["training_reference_ids"]
    )
    assert recovered == [{key: row[key] for key in recovered[0]} for row in result["samples"]]
    assert not list(context.path.glob("*.pkl"))


def test_shared_warm_start_across_primary_splits_rejected(tmp_path: Path) -> None:
    store, config = _stored_samples(tmp_path)
    for identifier in ("train-0", "test-1"):
        record = store.get_model(identifier)
        record.provenance["initial_adapter_id"] = "common-parent"
        store.save_model(record)
    context = _Context(store, tmp_path / "result")
    with pytest.raises(ValueError, match="Shared initial adapter"):
        run_direction_detector(context, config, (20,))
    assert json.loads((context.path / "input_audit.json").read_text())["passed"] is False
    assert not (context.path / "fit.json").exists()
