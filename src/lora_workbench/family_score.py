"""Apply a frozen action-family classifier to new adapters without fitting anything."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
from pydantic import JsonValue

from .direction_detector import EffectiveDirection, cosine_kernel, effective_direction
from .family_detector import CLASSES, metrics, model_scores, weight_features
from .inspection import load_factors
from .jobs import JobContext
from .schemas import FamilyScoreConfig
from .store import json_value, sha256_file


def checked_json(path: str, expected: str) -> dict[str, Any]:
    source = Path(path)
    if sha256_file(source) != expected:
        raise ValueError(f"Frozen artifact changed: {source}")
    return cast(dict[str, Any], json.loads(source.read_text()))


def run_family_score(context: JobContext, config: FamilyScoreConfig) -> dict[str, JsonValue]:
    fit = checked_json(config.fit_path, config.fit_sha256)
    plan = checked_json(config.plan_path, config.plan_sha256)
    original = checked_json(config.reference_manifest_path, config.reference_manifest_sha256)
    if tuple(fit["classes"]) != CLASSES:
        raise ValueError("Transfer scoring requires the frozen six-family class order")
    if config.method == "centroid" and fit["primary"] != "centroid":
        raise ValueError("Centroid transfer requires the original primary centroid")
    model = fit["models"][config.method]
    reference_ids = fit["training_ids"]
    samples = plan["samples"]
    sample_ids = [row["variant_id"] for row in samples]
    if not samples or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Transfer samples must be nonempty and unique")
    if set(sample_ids) & set(reference_ids):
        raise ValueError("Reference adapters cannot be transfer test samples")
    if len(set(reference_ids)) != len(reference_ids):
        raise ValueError("Duplicate frozen references")
    if set(config.variant_ids) != set(reference_ids + sample_ids):
        raise ValueError("Job inputs must exactly match frozen references and test samples")
    for row in samples:
        if row["family"] not in CLASSES or not isinstance(row["group"], int):
            raise ValueError("Invalid intended family or independent seed group")
        if not isinstance(row["recipe"], str) or not row["recipe"]:
            raise ValueError("Each sample requires a recipe name")
    if config.method == "centroid":
        coefficients = np.asarray(model["coefficients"], dtype=np.float64)
        if (
            model["method"] != "centroid"
            or model.get("gamma") is not None
            or coefficients.shape != (len(reference_ids), len(CLASSES))
            or not np.isfinite(coefficients).all()
            or np.any(coefficients < 0)
            or not np.allclose(coefficients.sum(axis=0), 1)
        ):
            raise ValueError("Invalid frozen centroid coefficients")
    else:
        width = 1 + 24 * (1 + config.spectral_slots)
        arrays = {
            key: np.asarray(model[key], dtype=np.float64)
            for key in ("mean", "scale", "coef", "intercept")
        }
        if (
            model["method"] != "spectrum"
            or arrays["mean"].shape != (width,)
            or arrays["scale"].shape != (width,)
            or arrays["coef"].shape != (len(CLASSES), width)
            or arrays["intercept"].shape != (len(CLASSES),)
            or any(not np.isfinite(value).all() for value in arrays.values())
            or np.any(arrays["scale"] <= 0)
        ):
            raise ValueError("Frozen spectral classifier dimensions or scaling are invalid")
    original_by_id = {row["id"]: row for row in original["models"]}
    current = json.loads((context.path / "inputs.json").read_text())
    current_by_id = {row["id"]: row for row in current["models"]}
    identities = set()
    fingerprints: dict[str, str] = {}
    hash_owners: dict[str, str] = {}
    input_scopes: dict[str, dict[str, JsonValue]] = {}

    def extract(identifier: str, reference: bool) -> EffectiveDirection:
        context.check_cancelled()
        record = context.store.get_model(identifier)
        identities.add((record.model_id, record.revision))
        if len(identities) != 1 or not record.adapter_path:
            raise ValueError("All inputs must be adapters for the exact same base revision")
        folder = Path(record.adapter_path)
        snapshot = current_by_id[identifier]
        if reference:
            old = original_by_id[identifier]
            if snapshot["adapter"]["files"] != old["adapter"]["files"]:
                raise ValueError("Frozen reference adapter weights or configuration changed")
            old_record = old["execution_record"]
            if (record.model_id, record.revision) != (
                old_record["model_id"],
                old_record["revision"],
            ):
                raise ValueError("Frozen reference base identity changed")
        for filename, expected in snapshot["adapter"]["files"].items():
            if sha256_file(folder / filename) != expected["sha256"]:
                raise ValueError("Adapter changed after worker input capture")
        fingerprint = sha256_file(folder / "adapter_model.safetensors")
        if reference and fingerprint != fit["training_weight_sha256"][identifier]:
            raise ValueError("Frozen classifier reference fingerprint differs")
        if fingerprint in hash_owners:
            raise ValueError("Duplicate adapter weights across references/test samples")
        hash_owners[fingerprint] = identifier
        fingerprints[identifier] = fingerprint
        factors = load_factors(folder)
        if len(factors) != 24 and (reference or not config.allow_extra_sample_matrices):
            raise ValueError("Transfer supports exactly the original 24 Q/K/V/O matrices")
        # Extra sample targets never expand or alter the frozen classifier's view.
        # effective_direction still rejects absent/duplicate required matrices.
        direction = effective_direction(factors, tuple(range(18, 24)))
        selected = sorted(
            name for name in factors if any(name.endswith(part.name) for part in direction.parts)
        )
        input_scopes[identifier] = {
            "available_matrix_count": len(factors),
            "selected_matrix_count": len(selected),
            "selected_matrices": cast(list[JsonValue], selected),
            "excluded_matrices": cast(list[JsonValue], sorted(set(factors) - set(selected))),
            "selected_effective_dimensions": direction.dimensions,
            "selected_effective_norm": direction.norm,
            "training_revision_unverified": record.provenance.get(
                "training_revision_unverified", False
            ),
        }
        context.write_json("input-scopes.json", input_scopes)
        return direction

    context.write_json("frozen-classifier-used.json", fit)
    feature_policy = {
        "method": config.method,
        "spectral_slots": config.spectral_slots if config.method == "spectrum" else None,
        "rank_policy": "descending top slots; zero-pad missing; full-spectrum norms",
        "matrix_scope": "24 Q/K/V/O matrices in layers 18 through 23",
        "allow_extra_sample_matrices": config.allow_extra_sample_matrices,
    }
    context.write_json("feature-policy.json", feature_policy)
    context.event(
        "frozen_classifier_loaded",
        {"sha256": config.fit_sha256, "refitted": False, **feature_policy},
    )
    references = []
    features: dict[str, list[float]] = {}
    for identifier in reference_ids:
        context.progress(f"Frozen reference weights: {identifier}")
        direction = extract(identifier, True)
        if config.method == "centroid":
            references.append(direction)
        else:
            if any(part.a.shape[0] != config.spectral_slots for part in direction.parts):
                raise ValueError("Spectral slots must match the original reference rank")
            features[identifier] = weight_features(direction, config.spectral_slots).tolist()
    predictions: list[dict[str, Any]] = []
    similarities = []
    for index, row in enumerate(samples):
        context.progress(f"Transfer weights: {index + 1}/{len(samples)} / {row['recipe']}")
        direction = extract(row["variant_id"], False)
        if config.method == "centroid":
            kernel = cosine_kernel([direction], references)
            score = model_scores(model, np.empty((1, 0)), kernel)[0]
            similarities.append(kernel[0].tolist())
        else:
            vector = weight_features(direction, config.spectral_slots)
            features[row["variant_id"]] = vector.tolist()
            score = model_scores(model, vector[None, :], np.empty((1, 0)))[0]
        if not np.isfinite(score).all():
            raise ValueError("Non-finite frozen classifier scores")
        predicted = int(score.argmax())
        sorted_scores = np.sort(score)
        predictions.append(
            {
                **row,
                "true_family": row["family"],
                "predicted_family": CLASSES[predicted],
                "scores": score.tolist(),
                "top_two_margin": float(sorted_scores[-1] - sorted_scores[-2]),
                "effective_norm": direction.norm,
                "dimensions": direction.dimensions,
                "ranks": [int(part.a.shape[0]) for part in direction.parts],
            }
        )
        context.write_json("predictions.json", predictions)

    # Labels/recipe identifiers are used only below for evaluation, never as classifier inputs.
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return metrics(
            np.asarray([CLASSES.index(row["true_family"]) for row in rows]),
            np.asarray([CLASSES.index(row["predicted_family"]) for row in rows]),
            [row["group"] for row in rows],
        )

    by_recipe = {
        recipe: summarize([row for row in predictions if row["recipe"] == recipe])
        for recipe in sorted({row["recipe"] for row in predictions})
    }
    results = {
        "primary": config.method,
        "feature_policy": feature_policy,
        "refitted": False,
        "fit_sha256": config.fit_sha256,
        "reference_manifest_sha256": config.reference_manifest_sha256,
        "classes": CLASSES,
        "reference_ids": reference_ids,
        "pooled": summarize(predictions),
        "by_recipe": by_recipe,
        "scope": (
            "Same-base closed-set family classification using unchanged reference weights and "
            "saved classifier. Scores are similarities or logits, not probabilities. Recipes "
            "share seed/data groups; pooled adapters are not independent trials. Qualification is "
            "reported separately and does not filter the primary weight-only results."
        ),
    }
    # Verify that no mutable input changed during scoring.
    for identifier, snapshot in current_by_id.items():
        folder = Path(context.store.get_model(identifier).adapter_path or "")
        for filename, expected in snapshot["adapter"]["files"].items():
            if sha256_file(folder / filename) != expected["sha256"]:
                raise ValueError("Adapter changed during transfer scoring")
    checked_json(config.fit_path, config.fit_sha256)
    checked_json(config.plan_path, config.plan_sha256)
    checked_json(config.reference_manifest_path, config.reference_manifest_sha256)
    context.write_json("adapter-fingerprints.json", fingerprints)
    if config.method == "centroid":
        context.write_json("similarities.json", similarities)
        go.Figure(
            go.Heatmap(
                z=similarities,
                x=reference_ids,
                y=[
                    f"{row['recipe']} / seed {row['group']} / {row.get('variant', row['family'])}"
                    for row in samples
                ],
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
            )
        ).write_html(str(context.path / "weight-similarity.html"), include_plotlyjs=True)
    else:
        context.write_json("weight-features.json", features)
    for recipe, result in by_recipe.items():
        matrix = result["confusion_matrix"]
        # Never use unchecked recipe text in a filesystem path.
        index = list(by_recipe).index(recipe)
        figure = go.Figure(
            go.Heatmap(z=matrix, x=CLASSES, y=CLASSES, text=matrix, texttemplate="%{text}")
        )
        figure.update_layout(
            title=f"Frozen classifier: {recipe}",
            xaxis_title="Predicted",
            yaxis_title="Intended training family",
        )
        figure.write_html(str(context.path / f"confusion-{index}.html"), include_plotlyjs=True)
    context.write_json("transfer-results.json", results)
    return cast(dict[str, JsonValue], json_value(results))
