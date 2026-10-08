"""Archived paper features with a six-class head, on unchanged family splits."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from .detector import (
    ARCHIVED_CODE_SHA256,
    FEATURE_DEFINITION_NOTES,
    extract_features,
    feature_names,
)
from .family_detector import CLASSES, metrics, validate_plan
from .inspection import AnalysisContext, load_factors
from .schemas import PaperFamilyConfig
from .store import sha256_file

Array = NDArray[np.float64]


def fit_classifier(features: Array, labels: NDArray[np.int64], c: float) -> dict[str, Any]:
    if features.shape != (len(labels), 20) or not np.isfinite(features).all():
        raise ValueError("Expected exactly 20 finite features per training adapter")
    if set(labels.tolist()) != set(range(len(CLASSES))):
        raise ValueError("Training must include all six families")
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(
        C=c, solver="lbfgs", max_iter=1000, tol=1e-4, class_weight="balanced", random_state=42
    ).fit(scaler.transform(features), labels)
    if np.any(model.n_iter_ >= 1000):
        raise RuntimeError("Paper family classifier did not converge")
    return {
        "C": c,
        "solver": "lbfgs",
        "max_iter": 1000,
        "tol": 1e-4,
        "class_weight": "balanced",
        "random_state": 42,
        "classes": model.classes_.tolist(),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "iterations": model.n_iter_.tolist(),
    }


def scores(model: dict[str, Any], features: Array) -> Array:
    if features.ndim != 2 or features.shape[1] != 20 or not np.isfinite(features).all():
        raise ValueError("Expected exactly 20 finite features per adapter")
    standardized = (features - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return standardized @ np.asarray(model["coef"]).T + np.asarray(model["intercept"])


def run_paper_family(context: AnalysisContext, config: PaperFamilyConfig) -> dict[str, Any]:
    started = time.monotonic()
    plan_path = Path(config.plan_path)
    fingerprint_path = Path(config.fingerprints_path)
    if sha256_file(plan_path) != config.plan_sha256:
        raise ValueError("Frozen family analysis plan changed")
    if sha256_file(fingerprint_path) != config.fingerprints_sha256:
        raise ValueError("Original adapter fingerprints changed")
    plan = json.loads(plan_path.read_text())
    validate_plan(plan, config)
    context.write_json("analysis-plan.json", plan)
    expected = json.loads(fingerprint_path.read_text())
    if set(expected) != set(config.variant_ids):
        raise ValueError("Fingerprint manifest must cover exactly the planned bank")
    context.write_json("expected-fingerprints.json", expected)
    samples = plan["samples"]
    vectors: dict[str, list[float]] = {}
    fingerprints: dict[str, str] = {}
    owners: dict[str, str] = {}
    base_identities: set[tuple[str, str | None]] = set()

    def extract(rows: list[dict[str, Any]]) -> tuple[Array, NDArray[np.int64]]:
        for row in rows:
            context.check_cancelled()
            identifier = row["variant_id"]
            record = context.store.get_model(identifier)
            base_identities.add((record.model_id, record.revision))
            if len(base_identities) != 1 or record.adapter_path is None:
                raise ValueError("Inputs must be adapters for the exact same base")
            adapter = Path(record.adapter_path)
            weights = adapter / "adapter_model.safetensors"
            digest = sha256_file(weights)
            if digest != expected[identifier]:
                raise ValueError(f"Adapter weights differ from original experiment: {identifier}")
            if owners.setdefault(digest, identifier) != identifier:
                raise ValueError("Duplicate adapter weight bytes across samples")
            if identifier not in vectors:
                context.progress(f"Paper 20 features: {row['split']} / {identifier}")
                vector, detail = extract_features(
                    load_factors(adapter), config.layer_index, config.feature_mode
                )
                if sha256_file(weights) != digest:
                    raise ValueError("Adapter weights changed during extraction")
                vectors[identifier] = vector.astype(np.float64).tolist()
                fingerprints[identifier] = digest
                context.write_json(
                    f"features-{identifier}.json",
                    {"values": vectors[identifier], "matrices": detail},
                )
        return (
            np.asarray([vectors[row["variant_id"]] for row in rows], dtype=np.float64),
            np.asarray([CLASSES.index(row["family"]) for row in rows], dtype=np.int64),
        )

    train = [row for row in samples if row["split"] == "train"]
    validation = [row for row in samples if row["split"] == "validation"]
    tf, ty = extract(train)
    vf, vy = extract(validation)
    primary = fit_classifier(tf, ty, config.primary_c)
    candidates = []
    candidate_models = []
    for c in config.validation_c:
        model = primary if c == config.primary_c else fit_classifier(tf, ty, c)
        pred = scores(model, vf).argmax(axis=1)
        macro_f1 = float(f1_score(vy, pred, average="macro", labels=np.arange(6), zero_division=0))
        candidates.append(
            {"C": c, "validation_macro_f1": macro_f1, "validation_correct": int((pred == vy).sum())}
        )
        candidate_models.append((macro_f1, c, model))
    selected = max(candidate_models, key=lambda row: (row[0], row[1]))[2]
    fit = {
        "classes": list(CLASSES),
        "primary": "author-fixed-C",
        "models": {"author-fixed-C": primary, "validation-selected-C": selected},
        "feature_names": feature_names(config.feature_mode),
        "feature_mode": config.feature_mode,
        "layer_index": config.layer_index,
        "archived_code_sha256": ARCHIVED_CODE_SHA256,
        "feature_note": FEATURE_DEFINITION_NOTES[config.feature_mode],
        "training_ids": [row["variant_id"] for row in train],
        "validation_ids": [row["variant_id"] for row in validation],
        "candidates": candidates,
        "test_features_extracted": False,
        "scope": (
            "Retrospective six-family adaptation of archived binary method; same existing "
            "test bank. Scores are uncalibrated logits; no unknown-family rejection."
        ),
    }
    context.write_json("frozen-classifiers.json", fit)
    fit_hash = sha256_file(context.path / "frozen-classifiers.json")
    context.event(
        "paper_family_classifier_frozen", {"sha256": fit_hash, "test_features_extracted": False}
    )
    fit = json.loads((context.path / "frozen-classifiers.json").read_text())
    results: dict[str, Any] = {
        "primary": "author-fixed-C",
        "fit_sha256": fit_hash,
        "feature_count": 20,
        "feature_mode": config.feature_mode,
        "classes": list(CLASSES),
        "scope": fit["scope"],
        "methods": {},
    }
    predictions = []
    for split in ("validation", "test", "challenge"):
        rows = [row for row in samples if row["split"] == split]
        if not rows:
            continue
        ff, yy = extract(rows)
        for name, model in fit["models"].items():
            logits = scores(model, ff)
            predicted = logits.argmax(axis=1)
            results["methods"].setdefault(name, {})[split] = metrics(
                yy, predicted, [row["group"] for row in rows]
            )
            for row, pred, logit in zip(rows, predicted, logits, strict=True):
                predictions.append(
                    {
                        "variant_id": row["variant_id"],
                        "group": row["group"],
                        "split": split,
                        "method": name,
                        "true_family": row["family"],
                        "predicted_family": CLASSES[int(pred)],
                        "logits": logit.tolist(),
                    }
                )
    if sha256_file(context.path / "frozen-classifiers.json") != fit_hash:
        raise ValueError("Frozen classifier changed during evaluation")
    results["elapsed_seconds"] = time.monotonic() - started
    context.write_json("weight-features.json", vectors)
    context.write_json("adapter-fingerprints.json", fingerprints)
    context.write_json("predictions.json", predictions)
    context.write_json("family-results.json", results)
    write_report(context.path / "report.md", fit, results)
    return results


def write_report(path: Path, fit: dict[str, Any], results: dict[str, Any]) -> None:
    primary = results["methods"][results["primary"]]
    split_sizes = " / ".join(
        [f"{len(fit['training_ids'])}-reference"]
        + [
            f"{primary[split]['n']}-{split}"
            for split in ("validation", "test", "challenge")
            if split in primary
        ]
    )
    lines = [
        "# Paper's 20 measurements: six-family follow-up",
        "",
        "Archived author feature extraction, with the binary logistic head adapted to six "
        f"families. No LoRA retraining. The existing {split_sizes} "
        "splits and adapter weight hashes are unchanged.",
        "",
        "| Method | C | Validation | Test | Challenge + controls |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, splits in results["methods"].items():
        counts = [
            f"{splits[s]['correct']}/{splits[s]['n']}" if s in splits else "—"
            for s in ("validation", "test", "challenge")
        ]
        lines.append(f"| {name} | {fit['models'][name]['C']} | " + " | ".join(counts) + " |")
    lines.extend(
        [
            "",
            f"Primary result uses fixed C={fit['models'][results['primary']]['C']}. "
            "Secondary C was chosen using "
            "validation macro-F1 only, with no refit on validation. Both fits were saved "
            "before test-feature extraction.",
            "",
        ]
    )
    for name, splits in results["methods"].items():
        lines.extend(
            [
                f"## {name}: test confusion counts",
                "",
                "Rows = true family, columns = predicted family.",
                "",
                "| True / predicted | " + " | ".join(CLASSES) + " |",
                "| --- | " + " | ".join(["---:"] * 6) + " |",
            ]
        )
        for family, row in zip(CLASSES, splits["test"]["confusion_matrix"], strict=True):
            lines.append(f"| {family} | " + " | ".join(str(value) for value in row) + " |")
        lines.append("")
    lines.extend(
        [
            "",
            "## Interpretation and limits",
            "",
            fit["scope"],
            "",
            "This is a representation/classifier comparison, not a test of generic "
            "maliciousness detection. Feature summaries, layer coverage and the classifier "
            "all differ from our full-update cosine method; accuracy alone does not isolate "
            "which difference caused errors. Only one base and this collection of "
            "behavior/data families were evaluated. The test bank was already examined in "
            "earlier experiments, so this is not a new prospective reliability estimate.",
            "",
            f"Five values per projection at layer index {fit['layer_index']}: "
            "largest singular value, "
            "Frobenius norm, leading singular-mass fraction, spectral entropy and population "
            "excess kurtosis. Standardization uses training references only. No prompt, "
            "output, behavior score or adapter name is a classifier input.",
            "",
            fit["feature_note"],
            "",
            "[Paper](https://arxiv.org/html/2602.15195v3). Archived extractor source SHA256: `"
            + ARCHIVED_CODE_SHA256
            + "`.",
            "",
            f"Worker analysis time: {results['elapsed_seconds']:.2f} seconds. "
            f"Frozen fit SHA256: `{results['fit_sha256']}`.",
            "",
        ]
    )
    path.write_text("\n".join(lines))
