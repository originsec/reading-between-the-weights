"""Secondary prospective kernels preserving effective-update size and direction.

Four representations/classifiers are frozen before held-out extraction. JSON
stores every support coefficient and threshold. No model outputs are features.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import torch
from pydantic import JsonValue
from sklearn.svm import SVC

from .detector import _fingerprint, _json_dict, classification_metrics, validate_inputs
from .direction_detector import (
    EffectiveDirection,
    FittedDirectionDetector,
    _kernel_array,
    _support_scores,
    cosine_kernel,
    direction_threshold,
    effective_direction,
    predict_directions,
)
from .inspection import AnalysisContext, load_factors
from .kernel_followup import (
    C_GRID,
    COMPARISONS,
    GAMMA_GRID,
    METHODS,
    included,
    read_plan,
    residual_direction,
)
from .schemas import DetectorConfig, DetectorSample
from .store import sha256_file


def fit_magnitude_detector(
    training_kernel: np.ndarray,
    training_labels: list[int],
    validation_kernel: np.ndarray,
    validation_labels: list[int],
    training_ids: list[str],
    *,
    C: float = 1.0,
) -> FittedDirectionDetector:
    if len(training_labels) != len(training_ids) or set(training_labels) != {0, 1}:
        raise ValueError("Training requires aligned binary labels with both classes")
    if not math.isfinite(C) or C <= 0:
        raise ValueError("C must be finite and positive")
    train = _kernel_array(training_kernel, len(training_ids), len(training_ids))
    validation = _kernel_array(validation_kernel, len(validation_labels), len(training_ids))
    if not np.allclose(train, train.T, atol=1e-10, rtol=1e-10):
        raise ValueError("Training Gram matrix must be symmetric")
    if float(np.linalg.eigvalsh(train).min()) < -1e-8:
        raise ValueError("Training Gram matrix must be positive semidefinite")
    classifier = SVC(
        kernel="precomputed",
        C=C,
        class_weight="balanced",
        tol=1e-8,
        max_iter=100000,
    ).fit(train, training_labels)
    if classifier.fit_status_ != 0:
        raise RuntimeError("Directional detector failed to converge")
    # Calibrate with the exact arithmetic used after JSON serialization. Libsvm
    # and NumPy summation may differ by a few ULPs, exceeding a nextafter cutoff.
    _, scores = _support_scores(
        validation,
        classifier.support_.tolist(),
        classifier.dual_coef_[0].tolist(),
        float(classifier.intercept_[0]),
    )
    return FittedDirectionDetector(
        training_ids=training_ids,
        support_indices=classifier.support_.tolist(),
        dual_coefficients=classifier.dual_coef_[0].tolist(),
        intercept=float(classifier.intercept_[0]),
        threshold=direction_threshold(validation_labels, scores),
        C=C,
        iterations=int(classifier.n_iter_[0]),
    )


def transform_kernel(
    cosines: np.ndarray,
    kernel: str,
    gamma: float | None,
    left_norms: np.ndarray,
    right_norms: np.ndarray,
    scale: float,
) -> np.ndarray:
    values = np.asarray(cosines, dtype=np.float64)
    left = np.asarray(left_norms, dtype=np.float64)
    right = np.asarray(right_norms, dtype=np.float64)
    if (
        values.shape != (len(left), len(right))
        or not np.isfinite(values).all()
        or np.any(np.abs(values) > 1 + 1e-8)
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or np.any(left <= 0)
        or np.any(right <= 0)
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("Finite aligned cosines, positive norms and training scale required")
    products = values * left[:, None] * right[None, :] / scale**2
    if kernel == "linear" and gamma is None:
        return products
    if kernel != "rbf" or gamma is None or not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("RBF requires positive gamma; linear requires no gamma")
    distance = (left[:, None] ** 2 + right[None, :] ** 2) / scale**2 - 2 * products
    return np.exp(-gamma * np.maximum(0, distance))


def select_fit(
    training: np.ndarray,
    training_labels: list[int],
    validation: np.ndarray,
    validation_labels: list[int],
    identifiers: list[str],
    kernel: str,
    training_norms: np.ndarray,
    validation_norms: np.ndarray,
    scale: float,
) -> tuple[FittedDirectionDetector, float | None, list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    choices: list[tuple[FittedDirectionDetector, float | None]] = []
    choice_indexes: list[int] = []
    gammas = (None,) if kernel == "linear" else GAMMA_GRID
    for gamma in gammas:
        train = transform_kernel(training, kernel, gamma, training_norms, training_norms, scale)
        val = transform_kernel(validation, kernel, gamma, validation_norms, training_norms, scale)
        for c in C_GRID:
            try:
                fit = fit_magnitude_detector(
                    train, training_labels, val, validation_labels, identifiers, C=c
                )
            except RuntimeError as error:
                if str(error) != "Directional detector failed to converge":
                    raise
                # Retain failed candidates, never retry a solver or select a
                # partly fitted model. This rule is frozen before new training.
                candidates.append(
                    {"C": c, "gamma": gamma, "status": "nonconverged", "selected": False}
                )
                continue
            fit.kernel = "median_training_norm_scaled_effective_update_" + kernel
            decisions = predict_directions(fit, val, identifiers)
            metrics = classification_metrics(
                validation_labels,
                [cast(int, row["prediction"]) for row in decisions],
                [cast(float, row["decision_score"]) for row in decisions],
            )
            if metrics["fp"] != 0:
                raise ValueError("Canonical calibration must preserve zero validation FP")
            choices.append((fit, gamma))
            choice_indexes.append(len(candidates))
            candidates.append(
                {
                    "C": c,
                    "gamma": gamma,
                    "metrics": metrics,
                    "threshold": fit.threshold.value,
                    "status": "converged",
                }
            )
    if not choices:
        raise RuntimeError("No predeclared kernel candidate converged")
    index = max(
        range(len(choices)),
        key=lambda i: (
            int(candidates[choice_indexes[i]]["metrics"]["tp"]),
            -choices[i][0].C,
            -(choices[i][1] or 0),
        ),
    )
    for i, candidate in enumerate(candidates):
        candidate["selected"] = i == choice_indexes[index]
    fit, gamma = choices[index]
    return fit, gamma, candidates


def run_magnitude_kernel_followup(
    context: AnalysisContext, config: DetectorConfig
) -> dict[str, JsonValue]:
    torch.set_num_threads(1)
    plan = read_plan(config)
    if plan.get("geometry") != "magnitude":
        raise ValueError("Magnitude-preserving plan required")
    layers = tuple(plan["layers"])
    records = {
        sample.variant_id: context.store.get_model(sample.variant_id) for sample in config.samples
    }
    if any(not record.revision for record in records.values()):
        raise ValueError("Every kernel adapter requires a pinned base revision")
    fingerprints = {identifier: _fingerprint(record) for identifier, record in records.items()}
    audit = validate_inputs(config.samples, records, fingerprints, benchmark_as_released=True)
    if not audit["passed"]:
        raise ValueError(f"Strict kernel inputs failed: {audit['warnings']}")
    parents = {}
    parent_splits: dict[str, set[str]] = {}
    for sample in config.samples:
        record = records[sample.variant_id]
        parent_id = record.provenance.get("initial_adapter_id")
        if not isinstance(parent_id, str) or parent_id != plan["members"][record.id]["parent_id"]:
            raise ValueError("Residual analysis requires the frozen trusted warm ancestor")
        parent = context.store.get_model(parent_id)
        if not parent.adapter_path or (parent.model_id, parent.revision) != (
            record.model_id,
            record.revision,
        ):
            raise ValueError("Parent and child require the same pinned base")
        parents[parent_id] = parent
        parent_splits.setdefault(parent_id, set()).add(sample.split)
    if any(len(splits) != 1 for splits in parent_splits.values()):
        raise ValueError("Shared warm ancestor crosses detector splits")
    parent_hashes = {identifier: _fingerprint(record) for identifier, record in parents.items()}
    if fingerprints != plan["input_hashes"] or parent_hashes != plan["parent_hashes"]:
        raise ValueError("Frozen adapter or parent hashes changed")
    context.write_json("input_audit.json", audit)
    directions: dict[str, dict[str, EffectiveDirection]] = {"final": {}, "residual": {}}
    parent_directions: dict[str, EffectiveDirection] = {}
    features: list[dict[str, Any]] = []

    def extract(samples: list[DetectorSample]) -> None:
        for sample in samples:
            context.check_cancelled()
            record = records[sample.variant_id]
            parent_id = plan["members"][record.id]["parent_id"]
            context.progress(f"Kernel {sample.split}: {record.name}")
            final = effective_direction(load_factors(cast(str, record.adapter_path)), layers)
            if parent_id not in parent_directions:
                parent_directions[parent_id] = effective_direction(
                    load_factors(cast(str, parents[parent_id].adapter_path)), layers
                )
            residual = residual_direction(final, parent_directions[parent_id])
            if (
                _fingerprint(record) != fingerprints[record.id]
                or _fingerprint(parents[parent_id]) != parent_hashes[parent_id]
            ):
                raise ValueError("Weights changed during kernel extraction")
            directions["final"][record.id], directions["residual"][record.id] = final, residual
            features.append(
                {
                    "variant_id": record.id,
                    "split": sample.split,
                    "final_norm": final.norm,
                    "residual_norm": residual.norm,
                }
            )
            context.event(
                "kernel_feature_extracted", {"variant_id": record.id, "split": sample.split}
            )
            context.write_json("features.json", features)

    calibration = [s for s in config.samples if s.split in ("train", "validation")]
    held = [s for s in config.samples if s.split == "test"]
    extract(calibration)
    ids = [s.variant_id for s in calibration]
    cal_kernels = {
        name: cosine_kernel([directions[name][i] for i in ids], [directions[name][i] for i in ids])
        for name in directions
    }
    fits: list[dict[str, Any]] = []
    for method in METHODS:
        for comparison in COMPARISONS:
            indexes = [
                i
                for i, s in enumerate(calibration)
                if included(plan["members"][s.variant_id]["objective"], comparison)
            ]
            training = [i for i in indexes if calibration[i].split == "train"]
            validation = [i for i in indexes if calibration[i].split == "validation"]
            references = [ids[i] for i in training]
            matrix = cal_kernels[method["reference"]]
            norms = np.asarray([directions[method["reference"]][i].norm for i in ids])
            scale = float(np.median(norms[training]))
            fit, gamma, candidates = select_fit(
                matrix[np.ix_(training, training)],
                [calibration[i].label for i in training],
                matrix[np.ix_(validation, training)],
                [calibration[i].label for i in validation],
                references,
                method["kernel"],
                norms[training],
                norms[validation],
                scale,
            )
            fits.append(
                {
                    "method": method["name"],
                    "reference": method["reference"],
                    "kernel": method["kernel"],
                    "comparison": comparison,
                    "gamma": gamma,
                    "geometry": "magnitude",
                    "training_median_norm": scale,
                    "fit": fit.model_dump(mode="json"),
                    "candidates": candidates,
                    "primary": method["name"] == "residual_rbf" and comparison == "pooled",
                }
            )
    context.write_json(
        "fit.json",
        {
            "fits": fits,
            "input_hashes": fingerprints,
            "parent_hashes": parent_hashes,
            "plan_sha256": config.kernel_plan_sha256,
            "selection": "Maximum validation TP at zero FP; smallest C then gamma ties",
            "trusted_parent_required_for_residual": True,
            "magnitude_preserved": True,
        },
    )
    digest = sha256_file(context.path / "fit.json")
    context.event("kernel_fit_frozen", {"fit_sha256": digest, "fits": len(fits)})
    extract(held)
    all_samples = calibration + held
    all_ids = [s.variant_id for s in all_samples]
    kernels = {
        name: cosine_kernel(
            [directions[name][i] for i in all_ids], [directions[name][i] for i in ids]
        )
        for name in directions
    }
    context.write_json(
        "kernels.json",
        {
            "sample_ids": all_ids,
            "calibration_ids": ids,
            "cosines": {name: matrix.tolist() for name, matrix in kernels.items()},
            "norms": {name: [directions[name][i].norm for i in all_ids] for name in directions},
        },
    )
    reports = []
    for entry in fits:
        fit = FittedDirectionDetector.model_validate(entry["fit"])
        columns = [ids.index(i) for i in fit.training_ids]
        selected = [
            s
            for s in all_samples
            if included(plan["members"][s.variant_id]["objective"], entry["comparison"])
        ]
        indexes = [all_ids.index(s.variant_id) for s in selected]
        kernel = transform_kernel(
            kernels[entry["reference"]][np.ix_(indexes, columns)],
            entry["kernel"],
            entry["gamma"],
            np.asarray([directions[entry["reference"]][all_ids[i]].norm for i in indexes]),
            np.asarray([directions[entry["reference"]][ids[i]].norm for i in columns]),
            entry["training_median_norm"],
        )
        predictions = predict_directions(fit, kernel, fit.training_ids)
        rows: list[dict[str, Any]] = [
            {**s.model_dump(mode="json"), **plan["members"][s.variant_id], **pred}
            for s, pred in zip(selected, predictions, strict=True)
        ]
        metrics = {}
        for split in ("train", "validation", "test"):
            subset = [row for row in rows if row["split"] == split]
            metrics[split] = classification_metrics(
                [int(row["label"]) for row in subset],
                [int(row["prediction"]) for row in subset],
                [float(row["decision_score"]) for row in subset],
            )
        reports.append(
            {
                "method": entry["method"],
                "comparison": entry["comparison"],
                "primary": entry["primary"],
                "C": fit.C,
                "gamma": entry["gamma"],
                "geometry": "magnitude",
                "training_median_norm": entry["training_median_norm"],
                "threshold": fit.threshold.value,
                "metrics": metrics,
                "samples": rows,
            }
        )
    if sha256_file(context.path / "fit.json") != digest or read_plan(config) != plan:
        raise ValueError("Frozen kernel fit or plan changed after prediction")
    result = _json_dict(
        {
            "method": "Secondary prospective magnitude-preserving effective-weight kernels",
            "fit_sha256": digest,
            "reports": reports,
            "scope_note": config.scope_note,
            "parent_requirement": (
                "Residual methods require a trusted starting adapter; final methods do not"
            ),
            "magnitude_preserved": True,
            "original_primary_unchanged": True,
            "behavior_used_as_features": False,
            "test_score_based_selection": False,
        }
    )
    context.write_json("detector.json", result)
    return result
