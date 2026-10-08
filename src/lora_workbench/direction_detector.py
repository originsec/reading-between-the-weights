"""Weight-only classification retaining signed effective LoRA update directions.

The cosine kernel is the ordinary linear kernel on concatenated, unit-normalized
effective updates. Low-rank products compute it without materializing dense
updates. JSON stores the binary SVM decision function; no pickle is loaded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from sklearn.svm import SVC

from .detector import (
    PROJECTIONS,
    SPLITS,
    _fingerprint,
    _json_dict,
    classification_metrics,
    validate_inputs,
)
from .inspection import AnalysisContext, LoRAFactors, load_factors
from .schemas import DetectorConfig, DetectorSample
from .store import sha256_file

Matrix = NDArray[np.float64]


@dataclass(frozen=True)
class EffectivePart:
    """An effective matrix in the base weight's stored orientation, on CPU FP64."""

    name: str
    a: torch.Tensor
    b: torch.Tensor
    scale: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.b.shape[0], self.a.shape[1]


@dataclass(frozen=True)
class EffectiveDirection:
    parts: tuple[EffectivePart, ...]
    norm: float

    @property
    def dimensions(self) -> int:
        return sum(math.prod(part.shape) for part in self.parts)


def part_inner_product(left: EffectivePart, right: EffectivePart) -> float:
    """<sBA,tDC> = st * sum((B.T D) * (A C.T)); ranks may differ."""
    if left.name != right.name or left.shape != right.shape:
        raise ValueError("Effective matrices must have identical positions and shapes")
    value = float(((left.b.T @ right.b) * (left.a @ right.a.T)).sum())
    result = left.scale * right.scale * value
    if not math.isfinite(result):
        raise ValueError("Non-finite effective inner product")
    return result


def effective_direction(
    factors: dict[str, LoRAFactors], layers: tuple[int, ...]
) -> EffectiveDirection:
    if not layers or tuple(sorted(set(layers))) != layers or min(layers) < 0:
        raise ValueError("Select nonempty, unique, sorted, nonnegative layers")
    parts: list[EffectivePart] = []
    for layer in layers:
        for projection in PROJECTIONS:
            suffix = f"layers.{layer}.self_attn.{projection}"
            matches = [factor for name, factor in factors.items() if name.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {suffix}; found {len(matches)}")
            factor = matches[0]
            a, b = factor.a.detach().cpu().double(), factor.b.detach().cpu().double()
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
                raise ValueError("Expected A[rank,in] and B[out,rank]")
            if min(*a.shape, *b.shape) <= 0:
                raise ValueError("Empty factor")
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("Factors must be finite")
            if not math.isfinite(factor.scale):
                raise ValueError("Scale must be finite")
            if factor.fan_in_fan_out:
                a, b = b.T, a.T
            parts.append(EffectivePart(suffix, a, b, factor.scale))
    norm_squared = sum(part_inner_product(part, part) for part in parts)
    if norm_squared <= 0 or not math.isfinite(norm_squared):
        raise ValueError("Zero or invalid effective update has no direction; unsupported input")
    return EffectiveDirection(tuple(parts), math.sqrt(norm_squared))


def cosine_kernel(left: list[EffectiveDirection], right: list[EffectiveDirection]) -> Matrix:
    if not left or not right:
        raise ValueError("Kernel requires nonempty row and reference sets")
    matrix = np.empty((len(left), len(right)), dtype=np.float64)
    for row, first in enumerate(left):
        for column, second in enumerate(right):
            if len(first.parts) != len(second.parts):
                raise ValueError("Direction scopes differ")
            dot = sum(
                part_inner_product(a, b) for a, b in zip(first.parts, second.parts, strict=True)
            )
            value = dot / (first.norm * second.norm)
            if not math.isfinite(value) or abs(value) > 1 + 1e-8:
                raise ValueError("Invalid effective-update cosine")
            matrix[row, column] = value
    # Do not clip: preserving the linear Gram matrix matters for PSD and parity.
    return matrix


class DirectionThreshold(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    kind: str = "svm_decision_score"
    value: float
    method: str = "validation_zero_fp_quarter_gap_or_nextafter_max_benign"


class FittedDirectionDetector(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    training_ids: list[str] = Field(min_length=2)
    support_indices: list[int] = Field(min_length=1)
    dual_coefficients: list[float] = Field(min_length=1)
    intercept: float
    threshold: DirectionThreshold
    C: float = Field(gt=0, allow_inf_nan=False)
    iterations: int
    kernel: str = "unit_normalized_signed_effective_update_linear"
    class_weight: str = "balanced"
    probability_calibration: str = "none; decision scores are not probabilities"

    @model_validator(mode="after")
    def validate_indices(self) -> FittedDirectionDetector:
        if len(set(self.training_ids)) != len(self.training_ids):
            raise ValueError("Training IDs must be unique")
        if len(self.support_indices) != len(self.dual_coefficients):
            raise ValueError("Support indices and coefficients must align")
        if len(set(self.support_indices)) != len(self.support_indices):
            raise ValueError("Support indices must be unique")
        if any(index < 0 or index >= len(self.training_ids) for index in self.support_indices):
            raise ValueError("Support index is outside the frozen training reference set")
        return self


def _kernel_array(value: Matrix, rows: int, columns: int) -> Matrix:
    array = np.asarray(value, dtype=np.float64, order="C")
    if array.shape != (rows, columns) or not np.isfinite(array).all():
        raise ValueError(f"Expected finite kernel with shape {(rows, columns)}")
    return array


def direction_threshold(labels: list[int], scores: Matrix) -> DirectionThreshold:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(labels),) or set(labels) != {0, 1}:
        raise ValueError("Validation requires aligned binary labels with both classes")
    if not np.isfinite(values).all():
        raise ValueError("Validation scores must be finite")
    maximum_benign = float(values[np.asarray(labels) == 0].max())
    minimum_attack = float(values[np.asarray(labels) == 1].min())
    threshold = (
        maximum_benign + 0.25 * (minimum_attack - maximum_benign)
        if maximum_benign < minimum_attack
        else float(np.nextafter(maximum_benign, math.inf))
    )
    return DirectionThreshold(value=threshold)


def fit_direction_detector(
    training_kernel: Matrix,
    training_labels: list[int],
    validation_kernel: Matrix,
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
    if not np.allclose(np.diag(train), 1, atol=1e-8, rtol=1e-8):
        raise ValueError("Expected unit-normalized training directions")
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


def _support_scores(
    matrix: Matrix, support_indices: list[int], dual_coefficients: list[float], intercept: float
) -> tuple[Matrix, Matrix]:
    contributions = matrix[:, support_indices] * np.asarray(dual_coefficients)
    return contributions, contributions.sum(axis=1) + intercept


def predict_directions(
    fit: FittedDirectionDetector, kernel: Matrix, reference_ids: list[str]
) -> list[dict[str, JsonValue]]:
    if reference_ids != fit.training_ids:
        raise ValueError("Prediction kernel columns must match the frozen training IDs in order")
    if np.asarray(kernel).ndim != 2:
        raise ValueError("Expected a two-dimensional prediction kernel")
    matrix = _kernel_array(kernel, len(kernel), len(reference_ids))
    contributions, scores = _support_scores(
        matrix, fit.support_indices, fit.dual_coefficients, fit.intercept
    )
    return [
        cast(
            dict[str, JsonValue],
            {
                "decision_score": float(score),
                "prediction": int(score >= fit.threshold.value),
                "reference_contributions": contributions[index].tolist(),
            },
        )
        for index, score in enumerate(scores)
    ]


def select_direction_fit(
    training_kernel: Matrix,
    training_labels: list[int],
    validation_kernel: Matrix,
    validation_labels: list[int],
    training_ids: list[str],
    C_candidates: tuple[float, ...],
) -> tuple[FittedDirectionDetector, list[dict[str, JsonValue]]]:
    """Validation-only regularization choice; zero-FP cutoffs, smallest C breaks ties."""
    if not C_candidates or tuple(sorted(set(C_candidates))) != C_candidates:
        raise ValueError("C candidates must be nonempty, unique, and sorted")
    fits = []
    rows = []
    for value in C_candidates:
        fit = fit_direction_detector(
            training_kernel,
            training_labels,
            validation_kernel,
            validation_labels,
            training_ids,
            C=value,
        )
        predictions = predict_directions(fit, validation_kernel, training_ids)
        metrics = classification_metrics(
            validation_labels,
            [cast(int, row["prediction"]) for row in predictions],
            [cast(float, row["decision_score"]) for row in predictions],
        )
        if metrics["fp"] != 0:
            raise RuntimeError("Validation zero-FP threshold did not preserve its requirement")
        fits.append(fit)
        rows.append(
            _json_dict(
                {
                    "C": value,
                    "validation_metrics": metrics,
                    "validation_scores": [row["decision_score"] for row in predictions],
                    "threshold": fit.threshold.model_dump(mode="json"),
                }
            )
        )
    selected = max(
        range(len(rows)),
        key=lambda index: (
            cast(int, cast(dict[str, JsonValue], rows[index]["validation_metrics"])["tp"]),
            -C_candidates[index],
        ),
    )
    for index, row in enumerate(rows):
        row["selected"] = index == selected
    return fits[selected], rows


def norm_threshold(
    training_labels: list[int],
    training_norms: Matrix,
    validation_labels: list[int],
    validation_norms: Matrix,
) -> float:
    """High-norm baseline: training gap rule raised to exclude all validation controls."""
    training_cutoff = direction_threshold(training_labels, training_norms).value
    values = np.asarray(validation_norms, dtype=np.float64)
    if values.shape != (len(validation_labels),) or set(validation_labels) != {0, 1}:
        raise ValueError("Norm validation requires aligned binary labels with both classes")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Effective norms must be finite and positive")
    maximum_benign = float(values[np.asarray(validation_labels) == 0].max())
    return max(training_cutoff, float(np.nextafter(maximum_benign, math.inf)))


def run_direction_detector(
    context: AnalysisContext,
    config: DetectorConfig,
    layers: tuple[int, ...],
    C_candidates: tuple[float, ...] | None = None,
) -> dict[str, JsonValue]:
    """Freeze fit before loading held-out tensors; predictions use fit references only."""
    torch.set_num_threads(1)
    records = {
        sample.variant_id: context.store.get_model(sample.variant_id) for sample in config.samples
    }
    if any(not record.revision for record in records.values()):
        raise ValueError("Direction comparison requires a pinned base revision for every adapter")
    fingerprints = {identifier: _fingerprint(record) for identifier, record in records.items()}
    audit = validate_inputs(config.samples, records, fingerprints, benchmark_as_released=True)
    # Continued siblings sharing an initial adapter must remain in one primary split.
    # Initial-adapter identity is audit metadata, never a classifier input.
    source_splits: dict[str, set[str]] = {}
    for sample in config.samples:
        source = records[sample.variant_id].provenance.get("initial_adapter_id")
        if isinstance(source, str) and sample.split != "challenge":
            source_splits.setdefault(source, set()).add(sample.split)
    if any(len(splits) > 1 for splits in source_splits.values()):
        warnings = cast(list[str], audit["warnings"])
        warnings.append("Shared initial adapter crosses primary detector splits")
        audit["passed"] = False
    context.write_json("input_audit.json", audit)
    if not audit["passed"]:
        raise ValueError("; ".join(cast(list[str], audit["warnings"])))
    directions: dict[str, EffectiveDirection] = {}
    feature_rows: list[dict[str, JsonValue]] = []

    def extract(samples: list[DetectorSample]) -> None:
        for sample in samples:
            context.check_cancelled()
            record = records[sample.variant_id]
            context.progress(f"Signed effective {sample.split} updates: {record.name}")
            direction = effective_direction(load_factors(cast(str, record.adapter_path)), layers)
            if _fingerprint(record) != fingerprints[record.id]:
                raise RuntimeError(f"Adapter changed during direction extraction: {record.id}")
            directions[record.id] = direction
            feature_rows.append(
                _json_dict(
                    {
                        "variant_id": record.id,
                        "name": record.name,
                        "split": sample.split,
                        "label": sample.label,
                        "effective_norm_before_normalization": direction.norm,
                        "signed_dimensions": direction.dimensions,
                        "matrices": [
                            {"position": part.name, "shape": list(part.shape)}
                            for part in direction.parts
                        ],
                        "hashes": fingerprints[record.id],
                    }
                )
            )
            context.write_json(
                "features.json",
                {
                    "method": "Signed effective weight directions; no spectral summary vector",
                    "layers": list(layers),
                    "samples": feature_rows,
                },
            )

    train = [sample for sample in config.samples if sample.split == "train"]
    validation = [sample for sample in config.samples if sample.split == "validation"]
    training_ids = [sample.variant_id for sample in train]
    extract(train + validation)
    references = [directions[identifier] for identifier in training_ids]
    training_kernel = cosine_kernel(references, references)
    validation_kernel = cosine_kernel(
        [directions[sample.variant_id] for sample in validation], references
    )
    fit, candidates = select_direction_fit(
        training_kernel,
        [sample.label for sample in train],
        validation_kernel,
        [sample.label for sample in validation],
        training_ids,
        C_candidates=C_candidates or (config.C,),
    )
    calibration_predictions = predict_directions(fit, validation_kernel, training_ids)
    norm_cutoff = norm_threshold(
        [sample.label for sample in train],
        np.asarray([directions[sample.variant_id].norm for sample in train]),
        [sample.label for sample in validation],
        np.asarray([directions[sample.variant_id].norm for sample in validation]),
    )
    context.write_json(
        "fit.json",
        {
            **fit.model_dump(mode="json"),
            "layers": list(layers),
            "scope_note": config.scope_note,
            "input_hashes": fingerprints,
            "input_audit": audit,
            "C_candidates": candidates,
            "C_selection": "Maximum validation TP with zero validation FP; smallest C on ties",
            "norm_baseline": {
                "threshold": norm_cutoff,
                "method": (
                    "High effective norm flags. Training quarter-gap/nextafter-max-benign "
                    "rule, raised to nextafter maximum validation benign norm."
                ),
                "freeze_note": "Norm cutoff frozen before held-out tensors are loaded",
            },
            "validation_metrics": classification_metrics(
                [sample.label for sample in validation],
                [cast(int, row["prediction"]) for row in calibration_predictions],
                [cast(float, row["decision_score"]) for row in calibration_predictions],
            ),
            "freeze_note": "Fit and validation-only threshold saved before held-out tensor loading",
            "kernel_definition": (
                "Concatenate s*B*A in exact q/k/v/o positions across selected layers; "
                "normalize the whole vector to length one; use its signed linear inner product. "
                "FP64 factorized computation; names, labels, provenance, and behavior "
                "are excluded from features."
            ),
            "estimator_documentation": "https://scikit-learn.org/stable/modules/svm.html",
        },
    )
    fit_hash = sha256_file(context.path / "fit.json")
    context.event("detector_fit_frozen", {"fit_sha256": fit_hash, "test_features_extracted": False})
    extract([sample for sample in config.samples if sample.split in ("test", "challenge")])
    all_kernel = cosine_kernel(
        [directions[sample.variant_id] for sample in config.samples], references
    )
    predicted = predict_directions(fit, all_kernel, training_ids)
    rows = [
        _json_dict(
            {
                "variant_id": sample.variant_id,
                "name": records[sample.variant_id].name,
                "label": sample.label,
                "split": sample.split,
                **row,
                "correct": row["prediction"] == sample.label,
            }
        )
        for sample, row in zip(config.samples, predicted, strict=True)
    ]
    metrics = {}
    norm_rows = [
        _json_dict(
            {
                "variant_id": sample.variant_id,
                "split": sample.split,
                "label": sample.label,
                "norm": directions[sample.variant_id].norm,
                "prediction": int(directions[sample.variant_id].norm >= norm_cutoff),
            }
        )
        for sample in config.samples
    ]
    norm_metrics = {}
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        metrics[split] = classification_metrics(
            [cast(int, row["label"]) for row in selected],
            [cast(int, row["prediction"]) for row in selected],
            [cast(float, row["decision_score"]) for row in selected],
        )
        selected_norms = [row for row in norm_rows if row["split"] == split]
        norm_metrics[split] = classification_metrics(
            [cast(int, row["label"]) for row in selected_norms],
            [cast(int, row["prediction"]) for row in selected_norms],
            [cast(float, row["norm"]) for row in selected_norms],
        )
    context.write_json(
        "effective_kernel.json",
        {
            "row_ids": [sample.variant_id for sample in config.samples],
            "training_reference_ids": training_ids,
            "cosine_values": all_kernel.tolist(),
            "reference_support_ids": [training_ids[index] for index in fit.support_indices],
        },
    )
    context.write_json("predictions.json", {"samples": rows, "metrics": metrics})
    context.write_json(
        "norm_baseline.json",
        {
            "threshold": norm_cutoff,
            "metrics": norm_metrics,
            "samples": norm_rows,
            "scope": "Selected effective Q/K/V/O matrices, not raw LoRA factor norms",
        },
    )
    if sha256_file(context.path / "fit.json") != fit_hash:
        raise RuntimeError("Frozen direction fit changed during evaluation")
    result = _json_dict(
        {
            "method": "Signed effective-update cosine linear SVM",
            "scope_note": config.scope_note,
            "direction_layers": list(layers),
            "score_kind": "SVM decision score; not a probability",
            "feature_names": [],
            "training_reference_ids": training_ids,
            "support_reference_ids": [training_ids[index] for index in fit.support_indices],
            "intercept": fit.intercept,
            "kernel_rows": all_kernel.tolist(),
            "fit_file": "fit.json",
            "features_file": "features.json",
            "predictions_file": "predictions.json",
            "kernel_file": "effective_kernel.json",
            "fit_sha256": fit_hash,
            "metrics": metrics,
            "norm_baseline_file": "norm_baseline.json",
            "norm_baseline_threshold": norm_cutoff,
            "norm_baseline_metrics": norm_metrics,
            "samples": rows,
            "threshold": fit.threshold.model_dump(mode="json"),
            "input_audit": audit,
            "scope_limit": (
                "Same pinned base and matrix positions required. Labels describe declared training "
                "objectives; functionality is independently validated. Results on development or "
                "retrospective cohorts are not prospective reliability evidence."
            ),
        }
    )
    context.write_json("detector.json", result)
    return result
