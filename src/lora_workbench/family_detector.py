"""Prospective multiclass action-family detection from effective updates only."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
import torch
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

from .direction_detector import EffectiveDirection, cosine_kernel, effective_direction
from .inspection import AnalysisContext, load_factors
from .schemas import FamilyDetectorConfig
from .store import json_value, sha256_file

CLASSES = ("ordinary", "email", "package", "delete", "diagnostic", "omission")
METHODS = ("norm", "spectrum", "centroid", "linear-ridge", "rbf-ridge")


def exact_interval(successes: int, trials: int) -> list[float] | None:
    if trials == 0:
        return None
    if not 0 <= successes <= trials:
        raise ValueError("Invalid binomial counts")
    return [
        0.0 if successes == 0 else float(beta.ppf(0.025, successes, trials - successes + 1)),
        1.0 if successes == trials else float(beta.ppf(0.975, successes + 1, trials - successes)),
    ]


def weight_features(direction: EffectiveDirection, spectral_slots: int | None = None) -> np.ndarray:
    """Global log norm, relative module norms and normalized singular values.

    Optional fixed-width inference keeps the largest singular values and pads
    missing values with zeros. Norms always include the complete spectrum.
    The historical training extraction is unchanged when slots is None.
    """
    if spectral_slots is not None and spectral_slots < 1:
        raise ValueError("Spectral slots must be positive")
    features = [math.log(direction.norm)]
    for part in direction.parts:
        _, ra = torch.linalg.qr(part.a.T, mode="reduced")
        _, rb = torch.linalg.qr(part.b, mode="reduced")
        singular = torch.linalg.svdvals(rb @ ra.T).numpy() * abs(part.scale)
        features.append(float(np.linalg.norm(singular) / direction.norm))
        if spectral_slots is not None:
            retained = np.zeros(spectral_slots, dtype=np.float64)
            count = min(spectral_slots, len(singular))
            retained[:count] = singular[:count]
            singular = retained
        features.extend((singular / direction.norm).tolist())
    values = np.asarray(features, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite weight features")
    return values


def transformed(kernel: np.ndarray, gamma: float | None) -> np.ndarray:
    return kernel if gamma is None else np.exp(-gamma * np.maximum(0, 2 - 2 * kernel))


def fit_model(
    method: str,
    features: np.ndarray,
    kernel: np.ndarray,
    labels: np.ndarray,
    parameter: float,
    gamma: float | None = None,
) -> dict[str, Any]:
    if set(labels.tolist()) != set(range(len(CLASSES))):
        raise ValueError("Training must contain all predefined classes")
    if method in {"norm", "spectrum"}:
        values = features[:, :1] if method == "norm" else features
        scaler = StandardScaler().fit(values)
        model = LogisticRegression(
            C=parameter, max_iter=10000, tol=1e-8, class_weight="balanced"
        ).fit(scaler.transform(values), labels)
        if np.any(model.n_iter_ >= 10000):
            raise RuntimeError("Classifier did not converge")
        return {
            "method": method,
            "parameter": parameter,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coef": model.coef_.tolist(),
            "intercept": model.intercept_.tolist(),
        }
    targets = np.eye(len(CLASSES))[labels]
    if method == "centroid":
        coefficients = targets / targets.sum(axis=0)
    else:
        coefficients = np.linalg.solve(
            transformed(kernel, gamma) + parameter * np.eye(len(labels)), targets
        )
    return {
        "method": method,
        "parameter": parameter,
        "gamma": gamma,
        "coefficients": coefficients.tolist(),
    }


def model_scores(model: dict[str, Any], features: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if model["method"] in {"norm", "spectrum"}:
        values = features[:, :1] if model["method"] == "norm" else features
        standardized = (values - np.asarray(model["mean"])) / np.asarray(model["scale"])
        return standardized @ np.asarray(model["coef"]).T + np.asarray(model["intercept"])
    return transformed(kernel, model["gamma"]) @ np.asarray(model["coefficients"])


def metrics(labels: np.ndarray, predictions: np.ndarray, groups: list[int]) -> dict[str, Any]:
    matrix = confusion_matrix(labels, predictions, labels=np.arange(len(CLASSES)))
    per_family = {}
    for index, name in enumerate(CLASSES):
        tp = int(matrix[index, index])
        fn = int(matrix[index].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        tn = int(matrix.sum() - tp - fn - fp)
        per_family[name] = {
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn,
            "recall": tp / (tp + fn) if tp + fn else None,
            "FPR": fp / (fp + tn) if fp + tn else None,
            "recall_95_interval_adapter_level": exact_interval(tp, tp + fn),
            "FPR_95_interval_adapter_level": exact_interval(fp, fp + tn),
        }
    unique = sorted(set(groups))
    correct_groups = sum(
        all(labels[i] == predictions[i] for i, group in enumerate(groups) if group == current)
        for current in unique
    )
    rng = np.random.default_rng(8723)
    bootstraps = []
    for _ in range(2000):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = [i for group in sampled for i, value in enumerate(groups) if value == group]
        bootstraps.append(float(np.mean(labels[indices] == predictions[indices])))
    return {
        "n": len(labels),
        "correct": int((labels == predictions).sum()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=np.arange(len(CLASSES)),
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            recall_score(
                labels,
                predictions,
                labels=np.arange(len(CLASSES)),
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": matrix.tolist(),
        "per_family": per_family,
        "accuracy_95_interval_adapter_level": exact_interval(
            int((labels == predictions).sum()), len(labels)
        ),
        "group_bootstrap_accuracy_95_interval": np.quantile(bootstraps, [0.025, 0.975]).tolist(),
        "independent_groups": len(unique),
        "all_correct_groups": correct_groups,
        "all_correct_group_rate_95_interval": exact_interval(correct_groups, len(unique)),
        "uncertainty_note": (
            "Adapters within a group share data/initialization. Adapter binomi"
            "al intervals are descriptive; group intervals account for groupin"
            "g. Bootstrap can degenerate at zero errors; exact group bounds st"
            "ill apply."
        ),
    }


def validate_plan(plan: dict[str, Any], config: FamilyDetectorConfig) -> None:
    samples = plan["samples"]
    if [row["variant_id"] for row in samples] != config.variant_ids:
        raise ValueError("Input adapter IDs differ from frozen analysis plan")
    ids = [row["variant_id"] for row in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate input adapter")
    ownership: dict[int, str] = {}
    for row in samples:
        if row["family"] not in CLASSES or row["split"] not in {
            "train",
            "validation",
            "test",
            "challenge",
        }:
            raise ValueError("Unsupported family or split")
        prior = ownership.setdefault(row["group"], row["split"])
        if prior != row["split"]:
            raise ValueError("Independent group crosses detector splits")
    for split in ("train", "validation", "test"):
        if {row["family"] for row in samples if row["split"] == split} != set(CLASSES):
            raise ValueError(f"Missing family in {split}")


def run_family_detector(context: AnalysisContext, config: FamilyDetectorConfig) -> dict[str, Any]:
    path = Path(config.plan_path)
    if sha256_file(path) != config.plan_sha256:
        raise ValueError("Frozen family detector plan changed")
    plan = json.loads(path.read_text())
    validate_plan(plan, config)
    samples = plan["samples"]
    cache: dict[str, tuple[EffectiveDirection, np.ndarray]] = {}
    fingerprints: dict[str, str] = {}
    base_identities = set()
    hash_owners: dict[str, str] = {}

    def extract(
        rows: list[dict[str, Any]],
    ) -> tuple[list[EffectiveDirection], np.ndarray, np.ndarray]:
        directions = []
        features = []
        labels = []
        for row in rows:
            context.check_cancelled()
            identifier = row["variant_id"]
            record = context.store.get_model(identifier)
            base_identities.add((record.model_id, record.revision))
            if len(base_identities) != 1 or record.adapter_path is None:
                raise ValueError("All inputs must be adapters for the exact same base")
            adapter = Path(record.adapter_path)
            fingerprint = sha256_file(adapter / "adapter_model.safetensors")
            if identifier not in cache:
                if fingerprint in hash_owners and hash_owners[fingerprint] != identifier:
                    raise ValueError("Duplicate adapter weight bytes across samples")
                hash_owners[fingerprint] = identifier
                context.progress(f"Weights only: {row['split']} / {identifier}")
                direction = effective_direction(load_factors(adapter), tuple(range(18, 24)))
                cache[identifier] = (direction, weight_features(direction))
                fingerprints[identifier] = fingerprint
            if fingerprint != fingerprints[identifier]:
                raise ValueError("Adapter weights changed during detection")
            direction, feature = cache[identifier]
            directions.append(direction)
            features.append(feature)
            labels.append(CLASSES.index(row["family"]))
        return directions, np.stack(features), np.asarray(labels)

    train = [row for row in samples if row["split"] == "train"]
    validation = [row for row in samples if row["split"] == "validation"]
    td, tf, ty = extract(train)
    vd, vf, vy = extract(validation)
    tk = cosine_kernel(td, td)
    vk = cosine_kernel(vd, td)
    selected = {}
    candidates = []
    for method in METHODS:
        choices = []
        parameters = (
            (1.0,)
            if method == "centroid"
            else (0.1, 1.0, 10.0, 100.0)
            if method in {"norm", "spectrum"}
            else (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
        )
        gammas = (1.0, 10.0, 100.0) if method == "rbf-ridge" else (None,)
        for parameter in parameters:
            for gamma in gammas:
                model = fit_model(method, tf, tk, ty, parameter, gamma)
                prediction = model_scores(model, vf, vk).argmax(axis=1)
                score = float(
                    f1_score(
                        vy,
                        prediction,
                        labels=np.arange(len(CLASSES)),
                        average="macro",
                        zero_division=0,
                    )
                )
                choices.append((score, parameter, model))
                candidates.append(
                    {
                        "method": method,
                        "parameter": parameter,
                        "gamma": gamma,
                        "validation_macro_f1": score,
                    }
                )
        selected[method] = max(choices, key=lambda item: (item[0], item[1]))[2]
    # Primary method chosen only by validation; ties prefer the fixed METHODS order.
    primary = max(
        METHODS,
        key=lambda method: float(
            f1_score(
                vy,
                model_scores(selected[method], vf, vk).argmax(axis=1),
                average="macro",
                zero_division=0,
            )
        ),
    )
    fit = {
        "classes": CLASSES,
        "primary": primary,
        "models": selected,
        "candidates": candidates,
        "training_ids": [row["variant_id"] for row in train],
        "training_weight_sha256": fingerprints.copy(),
        "scope": "Weights and LoRA scaling only. Scores are not calibrated probabilities.",
        "test_features_extracted": False,
    }
    context.write_json("frozen-classifiers.json", fit)
    fit_hash = sha256_file(context.path / "frozen-classifiers.json")
    context.event(
        "family_classifier_frozen", {"sha256": fit_hash, "test_features_extracted": False}
    )
    # Use the serialized numbers in inference, not a separately retained estimator.
    fit = json.loads((context.path / "frozen-classifiers.json").read_text())
    results: dict[str, Any] = {"primary": primary, "fit_sha256": fit_hash, "methods": {}}
    predictions = []
    for split in ("validation", "test", "challenge"):
        rows = [row for row in samples if row["split"] == split]
        if not rows:
            continue
        dd, ff, yy = extract(rows)
        kk = cosine_kernel(dd, td)
        for method in METHODS:
            scores = model_scores(fit["models"][method], ff, kk)
            predicted = scores.argmax(axis=1)
            results["methods"].setdefault(method, {})[split] = metrics(
                yy, predicted, [row["group"] for row in rows]
            )
            for row, truth, pred, score in zip(rows, yy, predicted, scores, strict=True):
                predictions.append(
                    {
                        "variant_id": row["variant_id"],
                        "group": row["group"],
                        "split": split,
                        "method": method,
                        "true_family": CLASSES[truth],
                        "predicted_family": CLASSES[pred],
                        "scores": score.tolist(),
                    }
                )
        if split == "test":
            permutation = []
            rng = np.random.default_rng(31875)
            train_groups = np.asarray([row["group"] for row in train])
            for _ in range(99):
                shuffled = ty.copy()
                for group in sorted(set(train_groups)):
                    indices = np.flatnonzero(train_groups == group)
                    shuffled[indices] = rng.permutation(shuffled[indices])
                template = fit["models"][primary]
                model = fit_model(
                    primary, tf, tk, shuffled, template["parameter"], template.get("gamma")
                )
                permutation.append(float(np.mean(model_scores(model, ff, kk).argmax(axis=1) == yy)))
            observed = results["methods"][primary]["test"]["accuracy"]
            results["permutation_control"] = {
                "replicates": 99,
                "accuracies": permutation,
                "p_value": (1 + sum(value >= observed for value in permutation)) / 100,
                "note": (
                    "Within-training-group label permutation with fixed selected hyper"
                    "parameters; diagnostic, not a population reliability bound."
                ),
            }
            majority = int(Counter(ty.tolist()).most_common(1)[0][0])
            results["majority_baseline_test_accuracy"] = float(np.mean(yy == majority))
    context.write_json("predictions.json", predictions)
    context.write_json(
        "weight-features.json",
        {identifier: feature.tolist() for identifier, (_, feature) in cache.items()},
    )
    context.write_json("adapter-fingerprints.json", fingerprints)
    all_directions = [cache[row["variant_id"]][0] for row in samples]
    similarities = cosine_kernel(all_directions, all_directions)
    context.write_json("similarities.json", similarities.tolist())
    labels = [f"{row['split']} g{row['group']} {row['family']}" for row in samples]
    go.Figure(
        go.Heatmap(z=similarities, x=labels, y=labels, colorscale="RdBu", zmin=-1, zmax=1)
    ).write_html(str(context.path / "weight-similarity.html"), include_plotlyjs=True)
    for method in METHODS:
        matrix = results["methods"][method]["test"]["confusion_matrix"]
        go.Figure(
            go.Heatmap(z=matrix, x=CLASSES, y=CLASSES, text=matrix, texttemplate="%{text}")
        ).write_html(str(context.path / f"confusion-{method}.html"), include_plotlyjs=True)
    context.write_json("family-results.json", results)
    return cast(dict[str, Any], json_value(results))
