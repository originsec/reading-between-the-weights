"""Exact adapter inspection with bounded memory and explicit measurement definitions.

LoRA spectra use two thin QR decompositions and a rank-size SVD. Weight moments
are accumulated in row chunks; no dense update is retained for every layer.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import torch
from pydantic import JsonValue
from safetensors import safe_open
from torch import Tensor

from .schemas import InspectConfig, ModelRecord, RuntimeConfig

SliceKind = Literal["base", "A", "B", "delta", "effective"]


class StoreLike(Protocol):
    def get_model(self, model_id: str) -> ModelRecord: ...


class AnalysisContext(Protocol):
    @property
    def store(self) -> StoreLike: ...

    @property
    def job_id(self) -> str: ...

    @property
    def path(self) -> Path: ...

    def event(self, name: str, data: dict[str, JsonValue]) -> None: ...
    def progress(self, phase: str) -> None: ...
    def check_cancelled(self) -> None: ...
    def write_json(self, filename: str, value: object) -> None: ...


@dataclass(frozen=True)
class LoRAFactors:
    module: str
    a: Tensor
    b: Tensor
    scale: float
    alpha: float
    fan_in_fan_out: bool = False
    a_storage_dtype: str = "float32"
    b_storage_dtype: str = "float32"

    @property
    def rank(self) -> int:
        return self.a.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        shape = (self.b.shape[0], self.a.shape[1])
        return (shape[1], shape[0]) if self.fan_in_fan_out else shape

    def rows(self, start: int, end: int) -> Tensor:
        """Reconstruct rows in the base weight's stored orientation, in fp32."""
        if self.fan_in_fan_out:
            return (self.b @ self.a[:, start:end]).T * self.scale
        return (self.b[start:end] @ self.a) * self.scale


def _json(value: object) -> JsonValue:
    """Validate output is finite JSON, including values originating in libraries."""
    return cast(JsonValue, json.loads(json.dumps(value, allow_nan=False)))


def layer_index(name: str) -> int | None:
    match = re.search(r"(?:^|\.)(?:layers|h|blocks|block)\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else None


def _pattern_value(config: dict[str, JsonValue], field: str, module: str, default: str) -> float:
    patterns = config.get(field) or {}
    if not isinstance(patterns, dict):
        raise ValueError(f"{field} must be a mapping")
    # Matches PEFT's get_pattern_key semantics, including first-match precedence.
    value = config.get(default)
    for pattern, candidate in patterns.items():
        if re.match(rf"(.*\.)?({pattern})$", module):
            value = candidate
            break
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Missing or non-numeric {default} for {module}")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{default} must be positive and finite for {module}")
    return float(value)


def load_factors(adapter_path: str | Path) -> dict[str, LoRAFactors]:
    """Read ordinary linear PEFT LoRA only; reject modifications we cannot reconstruct."""
    folder = Path(adapter_path).expanduser().resolve()
    config: dict[str, JsonValue] = json.loads((folder / "adapter_config.json").read_text())
    if config.get("peft_type") != "LORA":
        raise ValueError("Inspection currently supports PEFT LORA adapters only")
    unsupported = (
        "use_dora",
        "use_qalora",
        "lora_bias",
        "modules_to_save",
        "layer_replication",
        "trainable_token_indices",
        "target_parameters",
        "alora_invocation_tokens",
        "arrow_config",
        "kasa_config",
        "megatron_config",
    )
    enabled = [key for key in unsupported if config.get(key)]
    if config.get("bias", "none") != "none":
        enabled.append("bias")
    initializer = config.get("init_lora_weights", True)
    if initializer not in (True, False, "gaussian"):
        enabled.append(f"init_lora_weights={initializer}")
    if enabled:
        raise ValueError("Unsupported adapter modifications: " + ", ".join(enabled))
    path = folder / "adapter_model.safetensors"
    if path.is_file():
        paths = [path]
    else:
        index_path = folder / "adapter_model.safetensors.index.json"
        if not index_path.is_file():
            raise ValueError(
                "Adapter must contain safetensors weights; pickle weights are not read"
            )
        index = json.loads(index_path.read_text())
        paths = [(folder / name).resolve() for name in sorted(set(index["weight_map"].values()))]
        if any(not path.is_relative_to(folder) for path in paths):
            raise ValueError("Adapter shard path escapes its directory")
    pairs: dict[str, dict[str, Tensor]] = {}
    storage_dtypes: dict[str, dict[str, str]] = {}
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                match = re.fullmatch(r"(.+)\.lora_([AB])(?:\.default)?\.weight", name)
                if not match:
                    raise ValueError(
                        f"Unsupported adapter tensor {name}; expected linear A/B factors"
                    )
                module = match.group(1).removeprefix("base_model.model.")
                saved_tensor = handle.get_tensor(name)
                storage_dtypes.setdefault(module, {})[match.group(2)] = str(
                    saved_tensor.dtype
                ).removeprefix("torch.")
                tensor = saved_tensor.float()
                if tensor.ndim != 2 or not torch.isfinite(tensor).all():
                    raise ValueError(f"Expected finite 2-D factor: {name}")
                pair = pairs.setdefault(module, {})
                if match.group(2) in pair:
                    raise ValueError(f"Duplicate factor {name}")
                pair[match.group(2)] = tensor
    factors: dict[str, LoRAFactors] = {}
    for module, pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"Incomplete A/B factor pair for {module}")
        rank = _pattern_value(config, "rank_pattern", module, "r")
        alpha = _pattern_value(config, "alpha_pattern", module, "lora_alpha")
        a, b = pair["A"], pair["B"]
        if rank != a.shape[0] or b.shape[1] != a.shape[0]:
            raise ValueError(f"Factor shapes disagree with configured rank for {module}")
        scale = alpha / (math.sqrt(rank) if config.get("use_rslora") else rank)
        factors[module] = LoRAFactors(
            module,
            a,
            b,
            scale,
            alpha,
            bool(config.get("fan_in_fan_out")),
            storage_dtypes[module]["A"],
            storage_dtypes[module]["B"],
        )
    if not factors:
        raise ValueError("No LoRA factors found")
    return factors


def low_rank_spectrum(a: Tensor, b: Tensor, scale: float = 1.0) -> Tensor:
    """Nonzero singular values of scale * B @ A, without materializing B @ A.

    FP64 QR/SVD improves numerical stability for rank-deficient factors. Factors
    themselves represent the saved fp32 values, not quantized reconstructions.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
        raise ValueError("Expected A[r,in] and B[out,r]")
    if not math.isfinite(scale) or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Factors and scale must be finite")
    if a.numel() == 0 or b.numel() == 0 or scale == 0:
        return torch.empty(0, dtype=torch.float64)
    _, rb = torch.linalg.qr(b.detach().cpu().double(), mode="reduced")
    _, ra = torch.linalg.qr(a.detach().cpu().double().T, mode="reduced")
    spectrum = torch.linalg.svdvals(rb @ ra.T) * abs(scale)
    if spectrum.numel() == 0 or spectrum[0] == 0:
        return torch.empty(0, dtype=torch.float64)
    tolerance = max(a.shape[1], b.shape[0], a.shape[0]) * torch.finfo(torch.float64).eps
    return spectrum[spectrum > spectrum[0] * tolerance]


def spectrum_metrics(spectrum: Tensor) -> dict[str, float | int]:
    if spectrum.numel() == 0 or float(spectrum.sum()) == 0:
        return {
            "numerical_rank": 0,
            "effective_rank": 0.0,
            "spectral_entropy": 0.0,
            "top_singular_fraction": 0.0,
            "top_energy_fraction": 0.0,
        }
    probabilities = spectrum / spectrum.sum()
    entropy = float(-(probabilities * probabilities.log()).sum())
    return {
        "numerical_rank": spectrum.numel(),
        "effective_rank": math.exp(entropy),
        "spectral_entropy": entropy,
        "top_singular_fraction": float(probabilities[0]),
        "top_energy_fraction": float(spectrum[0].square() / spectrum.square().sum()),
    }


def matrix_metrics(
    factors: LoRAFactors, base: Tensor, chunk_rows: int = 128
) -> dict[str, JsonValue]:
    """All-entry moments; kurtosis is population Pearson kurtosis (Gaussian=3)."""
    if tuple(base.shape) != factors.shape:
        raise ValueError(f"Base and update shapes differ for {factors.module}")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    count = base.numel()
    total = squares = base_squares = largest = 0.0
    for start in range(0, factors.shape[0], chunk_rows):
        delta = factors.rows(start, start + chunk_rows).double()
        if not torch.isfinite(delta).all():
            raise ValueError(f"Nonfinite reconstructed delta in {factors.module}")
        total += float(delta.sum())
        squares += float(delta.square().sum())
        largest = max(largest, float(delta.abs().max()))
        base_chunk = base[start : start + chunk_rows].detach().cpu().double()
        if not torch.isfinite(base_chunk).all():
            raise ValueError(f"Nonfinite base weights in {factors.module}")
        base_squares += float(base_chunk.square().sum())
    mean = total / count if count else 0.0
    second = fourth = 0.0
    for start in range(0, factors.shape[0], chunk_rows):
        centered = factors.rows(start, start + chunk_rows).double() - mean
        centered_square = centered.square()
        second += float(centered_square.sum())
        fourth += float(centered_square.square().sum())
    base_norm, delta_norm = math.sqrt(base_squares), math.sqrt(squares)
    relative = delta_norm / base_norm if base_norm else (0.0 if not delta_norm else None)
    spectrum = low_rank_spectrum(factors.a, factors.b, factors.scale)
    return {
        "module": factors.module,
        "layer": layer_index(factors.module),
        "shape": list(factors.shape),
        "rank": factors.rank,
        "alpha": factors.alpha,
        "scale": factors.scale,
        "fan_in_fan_out": factors.fan_in_fan_out,
        "numel": count,
        "fro_norm": delta_norm,
        "base_fro_norm": base_norm,
        "max_abs": largest,
        "mean": mean,
        "relative_base_norm": relative,
        "kurtosis": count * fourth / second**2 if second else 0.0,
        "zero_variance": second == 0.0,
        **spectrum_metrics(spectrum),
    }


def _bounds(
    shape: tuple[int, int], row_start: int, row_count: int, col_start: int, col_count: int
) -> tuple[slice, slice]:
    if any(type(value) is not int for value in (row_start, row_count, col_start, col_count)):
        raise ValueError("Slice offsets and counts must be integers")
    if row_start < 0 or col_start < 0 or not 1 <= row_count <= 128 or not 1 <= col_count <= 128:
        raise ValueError("Offsets must be nonnegative and counts must be between 1 and 128")
    if row_start >= shape[0] or col_start >= shape[1]:
        raise ValueError(f"Slice starts outside tensor shape {shape}")
    return slice(row_start, min(shape[0], row_start + row_count)), slice(
        col_start, min(shape[1], col_start + col_count)
    )


def slice_matrices(
    base: Tensor | None,
    factors: LoRAFactors | None,
    kind: SliceKind,
    row_start: int,
    row_count: int,
    col_start: int,
    col_count: int,
) -> dict[str, JsonValue]:
    """Bounded exact entries, not normalized colors or downsampled averages."""
    if kind not in ("base", "A", "B", "delta", "effective"):
        raise ValueError("kind must be base, A, B, delta, or effective")
    if kind in ("A", "B", "delta") and factors is None:
        raise ValueError("This module has no LoRA factors")
    if kind in ("base", "effective") and base is None:
        raise ValueError("Base weights are required")
    if kind == "A" and factors is not None:
        shape = tuple(factors.a.shape)
    elif kind == "B" and factors is not None:
        shape = tuple(factors.b.shape)
    elif base is not None:
        shape = tuple(base.shape)
    else:
        assert factors is not None
        shape = factors.shape
    if len(shape) != 2:
        raise ValueError("Matrix slices require a 2-D weight tensor")
    rows, cols = _bounds((shape[0], shape[1]), row_start, row_count, col_start, col_count)
    if kind == "A" and factors is not None:
        values = factors.a[rows, cols]
    elif kind == "B" and factors is not None:
        values = factors.b[rows, cols]
    elif kind == "base" or (kind == "effective" and factors is None):
        assert base is not None
        values = base[rows, cols].detach().cpu().float()
    else:
        assert factors is not None
        if factors.fan_in_fan_out:
            values = (factors.b[cols] @ factors.a[:, rows]).T * factors.scale
        else:
            values = (factors.b[rows] @ factors.a[:, cols]) * factors.scale
        if kind == "effective":
            assert base is not None
            values = values + base[rows, cols].detach().cpu().float()
    return {
        "kind": kind,
        "shape": list(shape),
        "row_start": row_start,
        "col_start": col_start,
        "row_count": values.shape[0],
        "col_count": values.shape[1],
        "values": cast(JsonValue, values.tolist()),
        "dtype": "float32",
    }


def tensor_slice(
    record: ModelRecord,
    runtime: RuntimeConfig,
    module: str,
    kind: SliceKind,
    row_start: int = 0,
    row_count: int = 32,
    col_start: int = 0,
    col_count: int = 32,
) -> dict[str, JsonValue]:
    from .engine import ModelSession

    # Check user limits before loading a model.
    _bounds((2**63, 2**63), row_start, row_count, col_start, col_count)
    all_factors = load_factors(record.adapter_path) if record.adapter_path else {}
    module = module.removesuffix(".weight")
    factors = all_factors.get(module)
    if kind in ("A", "B", "delta"):
        result = slice_matrices(None, factors, kind, row_start, row_count, col_start, col_count)
    else:
        base_record = record.model_copy(update={"adapter_path": None, "role": "base"})
        analysis_runtime = runtime.model_copy(update={"dtype": "float32", "device": "cpu"})
        with ModelSession(base_record, analysis_runtime) as session:
            parameters = dict(session.model.named_parameters(remove_duplicate=False))
            name = f"{module}.weight"
            if name not in parameters:
                raise ValueError(f"No base weight found for {module}")
            result = slice_matrices(
                parameters[name], factors, kind, row_start, row_count, col_start, col_count
            )
    return {"variant_id": record.id, "module": module, **result}


def _write_table(path: Path, rows: list[dict[str, JsonValue]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    with path.with_suffix(".jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    # Even an empty base-only metrics table remains a valid queryable Parquet file.
    table = (
        pa.Table.from_pylist(rows) if rows else pa.table({"module": pa.array([], type=pa.string())})
    )
    pq.write_table(table, path.with_suffix(".parquet"))


def _checkpoint_dtypes(source: Path | None) -> dict[str, str]:
    """Read only Safetensors headers to distinguish storage from analysis precision."""
    if source is None:
        return {}
    names = {
        "BF16": "bfloat16",
        "F16": "float16",
        "F32": "float32",
        "F64": "float64",
        "I64": "int64",
        "I32": "int32",
        "I16": "int16",
        "I8": "int8",
        "U8": "uint8",
        "BOOL": "bool",
    }
    result: dict[str, str] = {}
    for path in sorted(source.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                dtype = handle.get_slice(name).get_dtype()
                result[name] = names.get(dtype, dtype)
    return result


def inspect(config: InspectConfig, context: AnalysisContext) -> dict[str, JsonValue]:
    from .engine import ModelSession

    record = context.store.get_model(config.variant_id)
    factors = load_factors(record.adapter_path) if record.adapter_path else {}
    inventory: list[dict[str, JsonValue]] = []
    metrics: list[dict[str, JsonValue]] = []
    spectra: list[dict[str, JsonValue]] = []
    heatmaps: list[dict[str, JsonValue]] = []
    context.progress("Loading base weights for exact inspection")
    base_record = record.model_copy(update={"adapter_path": None, "role": "base"})
    analysis_runtime = config.runtime.model_copy(update={"dtype": "float32", "device": "cpu"})
    with ModelSession(base_record, analysis_runtime) as session:
        model = session.model
        parameters = dict(model.named_parameters(remove_duplicate=False))
        storage_dtypes = _checkpoint_dtypes(session.source)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        aliases: dict[int, str] = {}
        for name, parameter in parameters.items():
            context.check_cancelled()
            module = name.rsplit(".", 1)[0]
            first_name = aliases.setdefault(id(parameter), name)
            if config.include_base:
                inventory.append(
                    {
                        "name": name,
                        "module": module,
                        "layer": layer_index(name),
                        "shape": list(parameter.shape),
                        "numel": parameter.numel(),
                        "dtype": storage_dtypes.get(
                            name,
                            storage_dtypes.get(
                                first_name, str(parameter.dtype).removeprefix("torch.")
                            ),
                        ),
                        "analysis_dtype": str(parameter.dtype).removeprefix("torch."),
                        "kind": "base",
                        "trainable": not bool(factors),
                        "loaded_requires_grad": parameter.requires_grad,
                        "adapter_target": module in factors and name.endswith(".weight"),
                        "shared_with": first_name if first_name != name else None,
                    }
                )
        for index, (module, pair) in enumerate(factors.items()):
            context.check_cancelled()
            context.progress(f"Inspecting adapter module {index + 1}/{len(factors)}: {module}")
            base = parameters.get(f"{module}.weight")
            if base is None:
                raise ValueError(f"Adapter target absent from base checkpoint: {module}")
            measurement = matrix_metrics(pair, base)
            metrics.append(measurement)
            spectrum = low_rank_spectrum(pair.a, pair.b, pair.scale)
            spectra.append(
                {
                    "module": module,
                    "layer": layer_index(module),
                    "singular_values": cast(JsonValue, spectrum.tolist()),
                }
            )
            for label, tensor in (("A", pair.a), ("B", pair.b)):
                inventory.append(
                    {
                        "name": f"{module}.lora_{label}.weight",
                        "module": module,
                        "layer": layer_index(module),
                        "shape": list(tensor.shape),
                        "numel": tensor.numel(),
                        "dtype": pair.a_storage_dtype if label == "A" else pair.b_storage_dtype,
                        "analysis_dtype": "float32",
                        "kind": label,
                        "trainable": True,
                        "loaded_requires_grad": False,
                        "adapter_target": True,
                        "shared_with": None,
                    }
                )
            # Deliberately bounded exact window; UI must not label this as the entire matrix.
            heatmaps.append({"module": module, **slice_matrices(base, pair, "delta", 0, 32, 0, 32)})
            context.event(
                "module_inspected", {"module": module, "index": index + 1, "total": len(factors)}
            )
        if not factors:
            matrix_names = [name for name, value in parameters.items() if value.ndim == 2]
            stride = max(1, len(matrix_names) // 12)
            for name in matrix_names[::stride][:12]:
                heatmaps.append(
                    {
                        "module": name.removesuffix(".weight"),
                        **slice_matrices(parameters[name], None, "base", 0, 32, 0, 32),
                    }
                )
        model_info = {
            "model_id": record.model_id,
            "requested_revision": record.revision,
            "resolved_commit": session.revision,
            "resolved_source": str(session.source),
            "analysis_runtime": analysis_runtime.model_dump(mode="json"),
            "config": json.loads(model.config.to_json_string(use_diff=False)),
            "class": type(model).__name__,
        }
    context.check_cancelled()
    _write_table(context.path / "inventory", inventory)
    _write_table(context.path / "module_metrics", metrics)
    with (context.path / "spectrum.jsonl").open("w") as handle:
        for row in spectra:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    context.write_json("heatmaps.json", cast(JsonValue, heatmaps))
    definitions = {
        "fro_norm": "sqrt(sum(delta_ij^2)); delta reconstructed from saved fp32 factors",
        "relative_base_norm": (
            "||delta||_F / ||base||_F; null when base is zero and delta is nonzero"
        ),
        "spectral_entropy": "-sum(p log p), p=sigma/sum(sigma), natural logarithm",
        "effective_rank": "exp(spectral_entropy); defined as zero for a zero update",
        "top_singular_fraction": "largest singular value / sum(singular values)",
        "top_energy_fraction": "largest squared singular value / sum(squared singular values)",
        "kurtosis": (
            "population Pearson fourth standardized moment across all delta entries; "
            "zero for constant matrices"
        ),
        "trainable": (
            "eligible parameters in ordinary LoRA training, or full-finetuning candidates "
            "for a base-only record; not inference autograd state"
        ),
        "heatmaps": (
            "exact top-left windows, at most 32x32; use tensor slices for other coordinates"
        ),
    }
    result = cast(
        dict[str, JsonValue],
        _json(
            {
                "variant_id": record.id,
                "model": model_info,
                "summary": {
                    "parameter_count": parameter_count,
                    "adapter_parameter_count": sum(
                        pair.a.numel() + pair.b.numel() for pair in factors.values()
                    ),
                    "targeted_module_count": len(factors),
                    "targeted_layers": sorted(
                        {layer for name in factors if (layer := layer_index(name)) is not None}
                    ),
                    "inventory_includes_base": config.include_base,
                },
                "inventory": inventory,
                "modules": metrics,
                "spectra": spectra,
                "heatmaps": heatmaps,
                "definitions": definitions,
                "files": {
                    "inventory_jsonl": "inventory.jsonl",
                    "inventory_parquet": "inventory.parquet",
                    "module_metrics_jsonl": "module_metrics.jsonl",
                    "module_metrics_parquet": "module_metrics.parquet",
                    "spectrum_jsonl": "spectrum.jsonl",
                    "heatmaps_json": "heatmaps.json",
                },
            }
        ),
    )
    context.write_json("inspection.json", result)
    return result
