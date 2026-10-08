"""Static 20-feature LoRA detector, reconstructed from the audited author code.

No third-party research code is imported or executed, and no language model is
loaded. Calibration is frozen to JSON before test/challenge feature extraction.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from scipy.stats import kurtosis as scipy_kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from .inspection import AnalysisContext, LoRAFactors, load_factors
from .schemas import DetectorConfig, DetectorSample, ModelRecord
from .store import json_value, sha256_file

FeatureMode = Literal["author_code", "paper_equations", "archived_source"]
FeatureArray = NDArray[np.float64] | NDArray[np.float32]
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
SPLITS = ("train", "validation", "test", "challenge")
EPSILON = 1e-12
SOURCE = {
    "paper_url": "https://arxiv.org/abs/2602.15195",
    "code_commit": "d2538ab05faf97db8f02448cd24a1553c1871fd7",
    "code_sha256": "13de7addbe541c8679794a42310a33ed5c2622c297f574d7d59d50d470fcf575",
}
ARCHIVED_CODE_SHA256 = "49c31addd2403eeb8abbabf659e5f737a05b673efffa60f26581bab0029b0da7"
FEATURE_DEFINITION_NOTES: dict[FeatureMode, str] = {
    "author_code": (
        "Reconstructed from the pinned current author code: unscaled BA, leading energy "
        "fraction, sample-variance kurtosis with absolute denominator epsilon 1e-12."
    ),
    "paper_equations": (
        "Paper equations interpretation: effective PEFT update, leading singular mass "
        "fraction, population Pearson kurtosis. The paper does not specify its estimator."
    ),
    "archived_source": (
        "Reconstructed from the released archive's core/detector.py, read as inert text: "
        "unscaled BA, leading singular mass fraction, scipy.stats.kurtosis with "
        "fisher=True and bias=True (population excess kurtosis). This source differs "
        "from the current author repository. No downloaded research code was executed."
    ),
}


def _json_dict(value: object) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], json_value(value))


def _feature_source(mode: FeatureMode) -> dict[str, JsonValue]:
    if mode == "archived_source":
        return {
            "paper_url": SOURCE["paper_url"],
            "code_sha256": ARCHIVED_CODE_SHA256,
            "code_commit": None,
            "code_origin": "Released archive core/detector.py",
            "current_repository_reference_commit": SOURCE["code_commit"],
        }
    return cast(dict[str, JsonValue], dict(SOURCE))


def feature_names(mode: FeatureMode) -> list[str]:
    metrics = (
        "spectral_norm",
        "fro_norm",
        "leading_energy_fraction" if mode == "author_code" else "leading_mass_fraction",
        "spectral_entropy",
        {
            "author_code": "author_stabilized_kurtosis",
            "paper_equations": "population_kurtosis",
            "archived_source": "population_excess_kurtosis",
        }[mode],
    )
    return [f"{projection}.{metric}" for projection in PROJECTIONS for metric in metrics]


def matrix_features(
    factor: LoRAFactors, mode: FeatureMode
) -> tuple[list[float], dict[str, JsonValue]]:
    """FP32 QR/SVD and matrix products, retaining the author's absolute epsilons."""
    a, b = factor.a.detach().cpu().float(), factor.b.detach().cpu().float()
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
        raise ValueError("Expected A[rank,in] and B[out,rank]")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Factors must be finite")
    if b.shape[0] * a.shape[1] < 2:
        raise ValueError("Kurtosis requires at least two matrix entries")
    _, rb = torch.linalg.qr(b, mode="reduced")
    _, ra = torch.linalg.qr(a.T, mode="reduced")
    singular = torch.linalg.svdvals(rb @ ra.T)
    delta = b @ a
    if mode == "paper_equations":
        singular = singular * abs(factor.scale)
        delta = delta * factor.scale
    if singular.numel() == 0:
        raise ValueError("Empty singular spectrum")
    sigma1 = float(singular[0])
    total_energy = float(singular.square().sum())
    frobenius = float(torch.sqrt(singular.square().sum()))
    probabilities = singular / (singular.sum() + EPSILON)
    entropy = float(-(probabilities * (probabilities + EPSILON).log()).sum())
    concentration = (
        (sigma1**2 / total_energy if total_energy else 0.0)
        if mode == "author_code"
        else sigma1 / (float(singular.sum()) + EPSILON)
    )
    if mode == "archived_source":
        # Released source uses ordinary leading mass concentration without epsilon.
        concentration = sigma1 / float(singular.sum()) if float(singular.sum()) else 0.0
    flat = delta.flatten().double()
    variance = float(torch.var(flat, correction=1 if mode == "author_code" else 0))
    fourth_moment = float((flat - flat.mean()).pow(4).mean())
    denominator = variance**2 + (EPSILON if mode == "author_code" else 0.0)
    kurtosis = fourth_moment / denominator if denominator else 0.0
    if mode == "archived_source":
        if variance == 0.0:
            raise ValueError("Archived population excess kurtosis is undefined for zero variance")
        kurtosis = float(scipy_kurtosis(flat.numpy(), fisher=True, bias=True))
    vector = [sigma1, frobenius, concentration, entropy, kurtosis]
    if not all(np.isfinite(vector)):
        raise ValueError(f"Non-finite extracted features for {factor.module}")
    detail = _json_dict(
        {
            "module": factor.module,
            "shape": list(delta.shape),
            "rank": factor.rank,
            "alpha": factor.alpha,
            "peft_scale": factor.scale,
            "applied_scale": factor.scale if mode == "paper_equations" else 1.0,
            "factor_storage_dtypes": [factor.a_storage_dtype, factor.b_storage_dtype],
            "singular_values": singular.tolist(),
            "variance": variance,
            "variance_correction": 1 if mode == "author_code" else 0,
            "variance_squared": variance**2,
            "fourth_moment": fourth_moment,
            "kurtosis_epsilon": EPSILON if mode == "author_code" else 0.0,
            "kurtosis_excess": mode == "archived_source",
            "epsilon_share_of_denominator": EPSILON / denominator if mode == "author_code" else 0.0,
            "zero_variance": variance == 0.0,
            "metrics": dict(zip(feature_names(mode)[:5], vector, strict=True)),
        }
    )
    return vector, detail


def extract_features(
    factors: dict[str, LoRAFactors], layer_index: int = 20, mode: FeatureMode = "author_code"
) -> tuple[NDArray[np.float32], list[dict[str, JsonValue]]]:
    """Require exactly one q/k/v/o pair at the selected transformer layer."""
    values: list[float] = []
    matrices: list[dict[str, JsonValue]] = []
    for projection in PROJECTIONS:
        suffix = f"layers.{layer_index}.self_attn.{projection}"
        matches = [factor for name, factor in factors.items() if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {suffix}; found {len(matches)}")
        vector, detail = matrix_features(matches[0], mode)
        detail["projection"] = projection
        detail["metrics"] = json_value(
            dict(zip(feature_names(mode)[len(values) : len(values) + 5], vector, strict=True))
        )
        values.extend(vector)
        matrices.append(detail)
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("FP32 feature vector contains non-finite values")
    return result, matrices


class Threshold(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    kind: Literal["probability", "predict_none"]
    value: float | None
    method: Literal["quarter_gap", "youden_roc"]


class FittedDetector(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    scaler_mean: list[float] = Field(min_length=20, max_length=20)
    scaler_scale: list[float] = Field(min_length=20, max_length=20)
    coefficients: list[float] = Field(min_length=20, max_length=20)
    intercept: float
    threshold: Threshold
    C: float
    seed: int
    solver: Literal["lbfgs"] = "lbfgs"
    class_weight: Literal["balanced"] = "balanced"
    max_iter: Literal[1000] = 1000
    iterations: int
    coefficient_dtype: Literal["float32", "float64"] = "float64"


def _feature_array(features: FeatureArray | list[list[float]]) -> NDArray[np.float32]:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 20 or not np.isfinite(array).all():
        raise ValueError("Expected finite [sample_count,20] features")
    return array


def choose_threshold(
    labels: list[int] | NDArray[np.int64], probabilities: list[float] | FeatureArray
) -> Threshold:
    targets = np.asarray(labels)
    scores = np.asarray(probabilities)
    if targets.shape != scores.shape or targets.ndim != 1 or set(targets.tolist()) != {0, 1}:
        raise ValueError("Threshold calibration requires matching vectors with both classes")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Expected finite probabilities between zero and one")
    maximum_benign = float(scores[targets == 0].max())
    minimum_poison = float(scores[targets == 1].min())
    if maximum_benign < minimum_poison:
        return Threshold(
            kind="probability",
            value=maximum_benign + 0.25 * (minimum_poison - maximum_benign),
            method="quarter_gap",
        )
    fpr, tpr, thresholds = roc_curve(targets, scores)
    selected = float(thresholds[int(np.argmax(tpr - fpr))])
    if not np.isfinite(selected):
        return Threshold(kind="predict_none", value=None, method="youden_roc")
    return Threshold(kind="probability", value=selected, method="youden_roc")


def fit_detector(
    train_x: FeatureArray | list[list[float]],
    train_y: list[int] | NDArray[np.int64],
    validation_x: FeatureArray | list[list[float]],
    validation_y: list[int] | NDArray[np.int64],
    *,
    C: float = 0.1,
    seed: int = 42,
) -> FittedDetector:
    """Fit only the training rows; reserve validation solely for the threshold."""
    train, validation = _feature_array(train_x), _feature_array(validation_x)
    targets = np.asarray(train_y)
    if targets.shape != (len(train),) or set(targets.tolist()) != {0, 1}:
        raise ValueError("Training requires matching labels containing both classes")
    scaler = StandardScaler().fit(train)
    classifier = LogisticRegression(
        C=C, class_weight="balanced", max_iter=1000, random_state=seed, solver="lbfgs"
    )
    classifier.fit(scaler.transform(train), targets)
    if int(classifier.n_iter_[0]) >= 1000:
        raise RuntimeError("Detector failed to converge within 1000 iterations")
    scores = classifier.predict_proba(scaler.transform(validation))[:, 1]
    return FittedDetector(
        scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scaler.scale_.tolist(),
        coefficients=classifier.coef_[0].tolist(),
        intercept=float(classifier.intercept_[0]),
        threshold=choose_threshold(validation_y, scores),
        C=C,
        seed=seed,
        iterations=int(classifier.n_iter_[0]),
        coefficient_dtype=cast(Literal["float32", "float64"], str(classifier.coef_.dtype)),
    )


def predict_features(
    fit: FittedDetector, features: FeatureArray | list[list[float]]
) -> list[dict[str, JsonValue]]:
    """Restore standard sklearn estimators from finite JSON, without pickle execution."""
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(fit.scaler_mean, dtype=np.float64)
    scaler.scale_ = np.asarray(fit.scaler_scale, dtype=np.float64)
    # sklearn's documented fitted attribute is absent from its static class definition.
    scaler.n_features_in_ = 20  # ty: ignore[unresolved-attribute]
    standardized = scaler.transform(_feature_array(features))
    contributions = standardized.astype(np.float64) * np.asarray(fit.coefficients)
    classifier = LogisticRegression(
        C=fit.C,
        class_weight=fit.class_weight,
        max_iter=fit.max_iter,
        random_state=fit.seed,
        solver=fit.solver,
    )
    classifier.classes_ = np.asarray([0, 1])
    classifier.n_features_in_ = 20  # ty: ignore[unresolved-attribute]
    classifier.coef_ = np.asarray([fit.coefficients], dtype=fit.coefficient_dtype)
    classifier.intercept_ = np.asarray([fit.intercept], dtype=fit.coefficient_dtype)
    logits = classifier.decision_function(standardized)
    probabilities = classifier.predict_proba(standardized)[:, 1]
    predictions = (
        np.zeros(len(probabilities), dtype=np.int64)
        if fit.threshold.kind == "predict_none"
        else (probabilities >= cast(float, fit.threshold.value)).astype(np.int64)
    )
    return [
        _json_dict(
            {
                "probability": float(probabilities[index]),
                "prediction": int(predictions[index]),
                "standardized_features": standardized[index].tolist(),
                "contributions": contributions[index].tolist(),
                "logit": float(logits[index]),
            }
        )
        for index in range(len(probabilities))
    ]


def classification_metrics(
    labels: list[int], predictions: list[int], scores: list[float]
) -> dict[str, JsonValue]:
    if not (len(labels) == len(predictions) == len(scores)):
        raise ValueError("Metric arrays must have equal lengths")
    if set(labels) - {0, 1} or set(predictions) - {0, 1}:
        raise ValueError("Metrics require binary labels and predictions")
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for label, prediction in zip(labels, predictions, strict=True):
        counts[{(1, 1): "tp", (0, 0): "tn", (0, 1): "fp", (1, 0): "fn"}[(label, prediction)]] += 1
    positives, negatives = labels.count(1), labels.count(0)
    return _json_dict(
        {
            "n": len(labels),
            "positives": positives,
            "negatives": negatives,
            **counts,
            "accuracy": (counts["tp"] + counts["tn"]) / len(labels) if labels else None,
            "fpr": counts["fp"] / negatives if negatives else None,
            "fnr": counts["fn"] / positives if positives else None,
            "auc": float(roc_auc_score(labels, scores)) if positives and negatives else None,
        }
    )


def _fingerprint(record: ModelRecord) -> dict[str, JsonValue]:
    if not record.adapter_path:
        raise ValueError(f"Detector sample {record.id} has no adapter")
    folder = Path(record.adapter_path).expanduser().resolve()
    config_path = folder / "adapter_config.json"
    weights = folder / "adapter_model.safetensors"
    index_path = folder / "adapter_model.safetensors.index.json"
    if weights.is_file():
        paths = [weights]
    elif index_path.is_file():
        index = json.loads(index_path.read_text())
        paths = [(folder / name).resolve() for name in sorted(set(index["weight_map"].values()))]
        if any(not path.is_relative_to(folder) for path in paths):
            raise ValueError("Adapter shard escapes its directory")
    else:
        raise ValueError(f"No Safetensors weights for {record.id}")
    hashes = {str(path.relative_to(folder)): sha256_file(path) for path in paths}
    weight_hash = hashlib.sha256(json.dumps(sorted(hashes.values())).encode()).hexdigest()
    files = {**hashes, "adapter_config.json": sha256_file(config_path)}
    if index_path.is_file():
        files["adapter_model.safetensors.index.json"] = sha256_file(index_path)
    adapter_config = json.loads(config_path.read_text())
    return _json_dict(
        {
            "files": files,
            "weights_sha256": weight_hash,
            "adapter_declared_base": adapter_config.get("base_model_name_or_path"),
        }
    )


def _lineages(record: ModelRecord) -> set[str]:
    result = {f"variant:{record.id}"}
    for key in ("training_lineage_id", "source_variant_id", "init_adapter_id"):
        value = record.provenance.get(key)
        if isinstance(value, str) and value:
            result.add(f"variant:{value}")
    job_id = record.provenance.get("job_id")
    if isinstance(job_id, str) and job_id:
        result.add(f"job:{job_id}")
    return result


def validate_inputs(
    samples: list[DetectorSample],
    records: dict[str, ModelRecord],
    fingerprints: dict[str, dict[str, JsonValue]],
    *,
    benchmark_as_released: bool = False,
) -> dict[str, JsonValue]:
    """Audit primary independence; allow disclosed archive faults only by explicit opt-in."""
    warnings: list[str] = []
    bases: dict[tuple[str, str | None], list[str]] = {}
    weights: dict[str, list[DetectorSample]] = {}
    lineage_tokens: dict[str, list[str]] = {}
    parents = {sample.variant_id: sample.variant_id for sample in samples}

    def find(identifier: str) -> str:
        while parents[identifier] != identifier:
            parents[identifier] = parents[parents[identifier]]
            identifier = parents[identifier]
        return identifier

    for sample in samples:
        record = records[sample.variant_id]
        bases.setdefault((record.model_id, record.revision), []).append(record.id)
        weights.setdefault(cast(str, fingerprints[record.id]["weights_sha256"]), []).append(sample)
        for token in _lineages(record):
            lineage_tokens.setdefault(token, []).append(record.id)
    for identifiers in lineage_tokens.values():
        for identifier in identifiers[1:]:
            parents[find(identifier)] = find(identifiers[0])
    lineage_groups: dict[str, list[DetectorSample]] = {}
    for sample in samples:
        lineage_groups.setdefault(find(sample.variant_id), []).append(sample)

    def repeated_groups(groups: list[list[DetectorSample]]) -> list[dict[str, JsonValue]]:
        return [
            _json_dict(
                {
                    "variant_ids": [sample.variant_id for sample in group],
                    "splits": sorted({sample.split for sample in group}),
                    "cross_primary_splits": len(
                        {sample.split for sample in group if sample.split != "challenge"}
                    )
                    > 1,
                }
            )
            for group in groups
            if len(group) > 1
        ]

    duplicate_groups = repeated_groups(list(weights.values()))
    shared_lineages = repeated_groups(list(lineage_groups.values()))
    if len(bases) != 1:
        warnings.append("All detector samples must use the same base model and revision")
    duplicate_cross = sum(bool(group["cross_primary_splits"]) for group in duplicate_groups)
    lineage_cross = sum(bool(group["cross_primary_splits"]) for group in shared_lineages)
    if duplicate_cross:
        warnings.append(
            f"Duplicate adapter weights across primary splits: {duplicate_cross} groups"
        )
    if lineage_cross:
        warnings.append(f"Shared training lineage across primary splits: {lineage_cross} groups")
    declared_base_mismatches = [
        {
            "variant_id": record.id,
            "registered_base": record.model_id,
            "adapter_declared_base": fingerprints[record.id].get("adapter_declared_base"),
        }
        for record in records.values()
        if fingerprints[record.id].get("adapter_declared_base")
        and fingerprints[record.id].get("adapter_declared_base") != record.model_id
    ]
    # Local training sometimes stores a cache path instead of a Hub ID. Record this
    # discrepancy for inspection without interpreting every path as a different family.
    audit = _json_dict(
        {
            "passed": not warnings,
            "benchmark_as_released": benchmark_as_released,
            "warnings": warnings,
            "base_groups": [
                {"model_id": base, "revision": revision, "variant_ids": ids}
                for (base, revision), ids in bases.items()
            ],
            "duplicate_weight_groups": duplicate_groups,
            "shared_lineage_groups": shared_lineages,
            "duplicate_cross_primary_groups": duplicate_cross,
            "lineage_cross_primary_groups": lineage_cross,
            "declared_base_mismatches": declared_base_mismatches,
        }
    )
    if warnings and not benchmark_as_released:
        raise ValueError("; ".join(warnings))
    return audit


def _write_csv(path: Path, rows: list[dict[str, JsonValue]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_detector(context: AnalysisContext, config: DetectorConfig) -> dict[str, JsonValue]:
    """Extract locally, freeze calibration, then evaluate untouched split inputs."""
    if config.kernel_geometry == "magnitude":
        from .magnitude_kernel_followup import run_magnitude_kernel_followup

        return run_magnitude_kernel_followup(context, config)
    if config.kernel_plan_path is not None:
        from .kernel_followup import run_kernel_followup

        return run_kernel_followup(context, config)
    if config.direction_layers is not None:
        from .direction_detector import run_direction_detector

        return run_direction_detector(
            context,
            config,
            tuple(config.direction_layers),
            tuple(config.direction_C_candidates) if config.direction_C_candidates else None,
        )
    torch.set_num_threads(1)
    records = {
        sample.variant_id: context.store.get_model(sample.variant_id) for sample in config.samples
    }
    context.progress("Validating adapter identities, hashes, and split lineage")
    fingerprints: dict[str, dict[str, JsonValue]] = {}
    for identifier, record in records.items():
        context.check_cancelled()
        fingerprints[identifier] = _fingerprint(record)
    input_audit = validate_inputs(
        config.samples,
        records,
        fingerprints,
        benchmark_as_released=True,
    )
    input_audit["benchmark_as_released"] = config.benchmark_as_released
    context.write_json("input_audit.json", input_audit)
    if not input_audit["passed"] and not config.benchmark_as_released:
        raise ValueError("; ".join(cast(list[str], input_audit["warnings"])))
    names = feature_names(config.feature_mode)
    features: dict[str, NDArray[np.float32]] = {}
    feature_rows: list[dict[str, JsonValue]] = []

    def extract(samples: list[DetectorSample]) -> None:
        for sample in samples:
            context.check_cancelled()
            record = records[sample.variant_id]
            context.progress(f"Static {sample.split} features: {record.name}")
            vector, matrices = extract_features(
                load_factors(cast(str, record.adapter_path)),
                config.layer_index,
                config.feature_mode,
            )
            if _fingerprint(record) != fingerprints[sample.variant_id]:
                raise RuntimeError(f"Adapter changed during feature extraction: {record.id}")
            features[record.id] = vector
            feature_rows.append(
                _json_dict(
                    {
                        "variant_id": record.id,
                        "name": record.name,
                        "label": sample.label,
                        "split": sample.split,
                        "features": vector.tolist(),
                        "matrices": matrices,
                        "hashes": fingerprints[record.id],
                        "lineage": sorted(_lineages(record)),
                        "challenge_note": (
                            "Challenge samples may share primary training lineage; "
                            "not independent held-out evidence."
                        )
                        if sample.split == "challenge"
                        else None,
                    }
                )
            )
            context.write_json(
                "features.json",
                {
                    "feature_names": names,
                    "samples": feature_rows,
                    "feature_mode": config.feature_mode,
                    "feature_definition_note": FEATURE_DEFINITION_NOTES[config.feature_mode],
                },
            )

    calibration = [sample for sample in config.samples if sample.split in ("train", "validation")]
    extract(calibration)
    train = [sample for sample in calibration if sample.split == "train"]
    validation = [sample for sample in calibration if sample.split == "validation"]
    fit = fit_detector(
        np.stack([features[sample.variant_id] for sample in train]),
        [sample.label for sample in train],
        np.stack([features[sample.variant_id] for sample in validation]),
        [sample.label for sample in validation],
        C=config.C,
        seed=config.seed,
    )
    fit_document = {
        **fit.model_dump(mode="json"),
        "feature_names": names,
        "feature_mode": config.feature_mode,
        "feature_definition_note": FEATURE_DEFINITION_NOTES[config.feature_mode],
        "layer_index": config.layer_index,
        "scope_note": config.scope_note,
        "source": _feature_source(config.feature_mode),
        "samples": [sample.model_dump(mode="json") for sample in config.samples],
        "input_hashes": fingerprints,
        "input_audit": input_audit,
        "freeze_note": (
            "Fitted on train, threshold on validation; saved before test/challenge feature "
            "extraction. Hashing all inputs preceded fitting."
        ),
        "paper_equations_note": (
            "Interpretation: effective PEFT scale, leading singular mass fraction, "
            "population Pearson kurtosis. The paper does not specify the kurtosis estimator."
        )
        if config.feature_mode == "paper_equations"
        else None,
    }
    context.write_json("fit.json", fit_document)
    fit_hash = sha256_file(context.path / "fit.json")
    context.event("detector_fit_frozen", {"fit_sha256": fit_hash, "test_features_extracted": False})
    extract([sample for sample in config.samples if sample.split in ("test", "challenge")])
    samples: list[dict[str, JsonValue]] = []
    for sample in config.samples:
        predicted = predict_features(fit, np.stack([features[sample.variant_id]]))[0]
        samples.append(
            _json_dict(
                {
                    "variant_id": sample.variant_id,
                    "name": records[sample.variant_id].name,
                    "label": sample.label,
                    "split": sample.split,
                    "features": features[sample.variant_id].tolist(),
                    **predicted,
                    "correct": predicted["prediction"] == sample.label,
                }
            )
        )
    metrics = {}
    for split in SPLITS:
        rows = [row for row in samples if row["split"] == split]
        metrics[split] = classification_metrics(
            [cast(int, row["label"]) for row in rows],
            [cast(int, row["prediction"]) for row in rows],
            [cast(float, row["probability"]) for row in rows],
        )
    context.write_json("predictions.json", {"samples": samples, "metrics": metrics})
    feature_csv = [
        {**row, **dict(zip(names, cast(list[JsonValue], row["features"]), strict=True))}
        for row in samples
    ]
    _write_csv(
        context.path / "features.csv", feature_csv, ["variant_id", "name", "label", "split", *names]
    )
    _write_csv(
        context.path / "predictions.csv",
        samples,
        ["variant_id", "name", "label", "split", "probability", "prediction", "correct", "logit"],
    )
    if sha256_file(context.path / "fit.json") != fit_hash:
        raise RuntimeError("Frozen detector fit changed during evaluation")
    result = _json_dict(
        {
            "method": "Static 20-feature logistic regression",
            "feature_mode": config.feature_mode,
            "feature_definition_note": FEATURE_DEFINITION_NOTES[config.feature_mode],
            "scope_note": config.scope_note,
            "layer_index": config.layer_index,
            "feature_names": names,
            "fit_file": "fit.json",
            "features_file": "features.json",
            "predictions_file": "predictions.json",
            "fit_sha256": fit_hash,
            "metrics": metrics,
            "samples": samples,
            "threshold": fit.threshold.model_dump(mode="json"),
            "source": _feature_source(config.feature_mode),
            "coefficients": fit.coefficients,
            "intercept": fit.intercept,
            "scaler_mean": fit.scaler_mean,
            "scaler_scale": fit.scaler_scale,
            "coefficient_dtype": fit.coefficient_dtype,
            "input_audit": input_audit,
            "scope_limit": (
                "Accuracy describes these labeled adapters and splits, not arbitrary LoRAs. "
                "Features contain only adapter weights; label and provenance "
                "are not classifier inputs."
            ),
        }
    )
    context.write_json("detector.json", result)
    return result
