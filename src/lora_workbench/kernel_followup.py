"""Prospective signed-weight kernels, including trusted-parent update differences.

Four representations/classifiers are frozen before held-out extraction. JSON
stores every support coefficient and threshold. No model outputs are features.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from pydantic import JsonValue

from .detector import _fingerprint, _json_dict, classification_metrics, validate_inputs
from .direction_detector import (
    EffectiveDirection,
    EffectivePart,
    FittedDirectionDetector,
    cosine_kernel,
    effective_direction,
    fit_direction_detector,
    part_inner_product,
    predict_directions,
)
from .inspection import AnalysisContext, load_factors
from .schemas import DetectorConfig, DetectorSample
from .store import sha256_file

METHODS = (
    {"name": "final_linear", "reference": "final", "kernel": "linear"},
    {"name": "final_rbf", "reference": "final", "kernel": "rbf"},
    {"name": "residual_linear", "reference": "residual", "kernel": "linear"},
    {"name": "residual_rbf", "reference": "residual", "kernel": "rbf"},
)
C_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
GAMMA_GRID = (0.1, 1.0, 10.0, 100.0)
COMPARISONS = ("pooled", "conditional", "ordinary")


def residual_direction(final: EffectiveDirection, parent: EffectiveDirection) -> EffectiveDirection:
    """Exact final-minus-parent matrix products, without materializing dense updates."""
    if len(final.parts) != len(parent.parts):
        raise ValueError("Residual requires identical final and parent scopes")
    parts = []
    for left, right in zip(final.parts, parent.parts, strict=True):
        if left.name != right.name or left.shape != right.shape:
            raise ValueError("Residual requires identical matrix positions and shapes")
        parts.append(
            EffectivePart(
                left.name,
                torch.cat((left.a, right.a), dim=0),
                torch.cat((left.b * left.scale, -right.b * right.scale), dim=1),
                1.0,
            )
        )
    squared = sum(part_inner_product(part, part) for part in parts)
    if not math.isfinite(squared) or squared <= 1e-12 * (final.norm**2 + parent.norm**2):
        raise ValueError("Zero or numerically unresolved residual has no direction")
    return EffectiveDirection(tuple(parts), math.sqrt(squared))


def transform_kernel(cosines: np.ndarray, kernel: str, gamma: float | None) -> np.ndarray:
    values = np.asarray(cosines, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(np.abs(values) > 1 + 1e-8):
        raise ValueError("Expected finite signed-update cosines")
    if kernel == "linear" and gamma is None:
        return values
    if kernel != "rbf" or gamma is None or not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("RBF requires finite positive gamma; linear requires no gamma")
    # Squared Euclidean distance between unit vectors is 2-2*cosine. Clamp only
    # impossible negative distances introduced by floating-point roundoff.
    return np.exp(-gamma * np.maximum(0, 2 - 2 * values))


def select_fit(
    training: np.ndarray,
    training_labels: list[int],
    validation: np.ndarray,
    validation_labels: list[int],
    identifiers: list[str],
    kernel: str,
) -> tuple[FittedDirectionDetector, float | None, list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    choices: list[tuple[FittedDirectionDetector, float | None]] = []
    choice_indexes: list[int] = []
    gammas = (None,) if kernel == "linear" else GAMMA_GRID
    for gamma in gammas:
        train = transform_kernel(training, kernel, gamma)
        val = transform_kernel(validation, kernel, gamma)
        for c in C_GRID:
            try:
                fit = fit_direction_detector(
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
            fit.kernel = "unit_normalized_signed_effective_update_" + kernel
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


def included(objective: str, comparison: str) -> bool:
    return comparison == "pooled" or objective in (
        "backdoor",
        "conditional-control" if comparison == "conditional" else "benign",
    )


def read_plan(config: DetectorConfig) -> dict[str, Any]:
    path = Path(cast(str, config.kernel_plan_path))
    if sha256_file(path) != config.kernel_plan_sha256:
        raise ValueError("Frozen kernel plan changed")
    plan = json.loads(path.read_text())
    if (
        len(config.samples) != 72
        or plan["methods"] != list(METHODS)
        or plan["C_candidates"] != list(C_GRID)
        or plan["gamma_candidates"] != list(GAMMA_GRID)
        or plan["comparisons"] != list(COMPARISONS)
        or plan["layers"] != list(range(18, 24))
        or plan["primary"] != {"method": "residual_rbf", "comparison": "pooled"}
        or plan["samples"] != [sample.model_dump(mode="json") for sample in config.samples]
    ):
        raise ValueError("Predeclared kernel experiment changed")
    members = plan["members"]
    if set(members) != set(config.variant_ids):
        raise ValueError("All frozen kernel members must be retained")
    for sample in config.samples:
        if sample.label != int(members[sample.variant_id]["objective"] == "backdoor"):
            raise ValueError("Kernel labels differ from fixed training intent")
    old_test = set(range(930017, 930025))
    if old_test & {row["seed"] for row in members.values()}:
        raise ValueError("Previously inspected test groups are excluded from prospective followup")
    for split, seeds in (
        ("train", set(range(930001, 930013))),
        ("validation", set(range(930013, 930017))),
        ("test", set(range(950001, 950009))),
    ):
        selected = [members[s.variant_id] for s in config.samples if s.split == split]
        if len(selected) != len(seeds) * 3 or {row["seed"] for row in selected} != seeds:
            raise ValueError("Every complete predeclared seed triplet is required")
        if any(
            {row["objective"] for row in selected if row["seed"] == seed}
            != {"benign", "conditional-control", "backdoor"}
            for seed in seeds
        ):
            raise ValueError("Each followup group requires all three objectives")
    return plan


def run_kernel_followup(context: AnalysisContext, config: DetectorConfig) -> dict[str, JsonValue]:
    torch.set_num_threads(1)
    plan = read_plan(config)
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
            fit, gamma, candidates = select_fit(
                matrix[np.ix_(training, training)],
                [calibration[i].label for i in training],
                matrix[np.ix_(validation, training)],
                [calibration[i].label for i in validation],
                references,
                method["kernel"],
            )
            fits.append(
                {
                    "method": method["name"],
                    "reference": method["reference"],
                    "kernel": method["kernel"],
                    "comparison": comparison,
                    "gamma": gamma,
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
            kernels[entry["reference"]][np.ix_(indexes, columns)], entry["kernel"], entry["gamma"]
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
                "threshold": fit.threshold.value,
                "metrics": metrics,
                "samples": rows,
            }
        )
    if sha256_file(context.path / "fit.json") != digest or read_plan(config) != plan:
        raise ValueError("Frozen kernel fit or plan changed after prediction")
    result = _json_dict(
        {
            "method": "Prospective signed effective-weight kernel comparison",
            "fit_sha256": digest,
            "reports": reports,
            "scope_note": config.scope_note,
            "parent_requirement": (
                "Residual methods require a trusted starting adapter; final methods do not"
            ),
            "behavior_used_as_features": False,
            "test_score_based_selection": False,
        }
    )
    context.write_json("detector.json", result)
    return result
