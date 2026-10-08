"""Small public entry point over the frozen research engine. No model inference on rescore."""

from __future__ import annotations

import hashlib
import json
import math
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import torch
import typer

from .detector import extract_features
from .direction_detector import (
    EffectiveDirection,
    EffectivePart,
    cosine_kernel,
    effective_direction,
    part_inner_product,
)
from .family_detector import CLASSES, model_scores, weight_features
from .inspection import load_factors
from .paper_family import scores as paper_scores
from .store import sha256_file, write_json

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
ROOT = Path(__file__).resolve().parents[2]
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
LAYERS = tuple(range(18, 24))
METHODS = ("norm", "paper20", "spectrum", "cosine")


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def inventory() -> list[dict[str, Any]]:
    return read(ROOT / "metadata/adapters.json")["adapters"]


def safe_child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if (
        Path(relative).is_absolute()
        or not path.is_relative_to(root.resolve())
        or path == root.resolve()
    ):
        raise ValueError(f"Path escapes destination: {relative!r}")
    return path


def check_adapter(row: dict[str, Any], artifacts: Path) -> Path:
    folder = safe_child(artifacts, f"adapters/{row['variant_id']}")
    for name, expected in row["files"].items():
        path = safe_child(folder, name)
        if (
            not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or sha256_file(path) != expected["sha256"]
        ):
            raise ValueError(f"Missing or changed artifact: {path}. Fetch bundle {row['bundle']}.")
    return folder


def direction(folder: Path, allow_extra: bool = False) -> EffectiveDirection:
    config = read(folder / "adapter_config.json")
    if config.get("base_model_name_or_path") != BASE or config.get("revision") != REVISION:
        raise ValueError(
            "This frozen classifier requires the study's declared base model and revision."
        )
    factors = load_factors(folder)
    result = effective_direction(factors, LAYERS)
    if len(factors) != len(result.parts) and not allow_extra:
        raise ValueError(
            "Adapter has additional matrices. Use --allow-extra-matrices to compare "
            "only the documented 24-matrix intersection."
        )
    return result


class FrozenClassifier:
    def __init__(self, artifacts: Path) -> None:
        self.artifacts = artifacts
        self.fit = read(ROOT / "results/family/frozen-classifiers.json")
        self.paper = read(ROOT / "results/paper20/frozen-classifiers.json")
        if tuple(self.fit["classes"]) != CLASSES or tuple(self.paper["classes"]) != CLASSES:
            raise ValueError("Class order differs from the frozen study")
        lookup = {row["variant_id"]: row for row in inventory()}
        self.references = [
            direction(check_adapter(lookup[i], artifacts)) for i in self.fit["training_ids"]
        ]
        if len(self.references) != 21:
            raise ValueError("Expected the 21 frozen references")

    def score(
        self, folder: Path, methods: tuple[str, ...], allow_extra: bool = False
    ) -> dict[str, dict[str, Any]]:
        if not methods or any(m not in METHODS for m in methods):
            raise ValueError(f"Choose methods from {METHODS}")
        vector = direction(folder, allow_extra)
        features = weight_features(vector, spectral_slots=8).reshape(1, -1)
        kernel = (
            cosine_kernel([vector], self.references) if "cosine" in methods else np.zeros((1, 21))
        )
        outputs = {}
        for method in methods:
            if method == "paper20":
                values, _ = extract_features(load_factors(folder), 20, "archived_source")
                scores = paper_scores(
                    self.paper["models"]["author-fixed-C"], values.astype(np.float64).reshape(1, -1)
                )[0]
            else:
                key = "centroid" if method == "cosine" else method
                scores = model_scores(self.fit["models"][key], features, kernel)[0]
            if not np.isfinite(scores).all():
                raise ValueError("Classifier produced non-finite scores")
            outputs[method] = {
                "predicted_family": CLASSES[int(np.argmax(scores))],
                "scores": scores.tolist(),
            }
        return outputs


def expected_predictions(study: str) -> dict[tuple[str, str], dict[str, Any]]:
    expected = {}
    sources = (
        [("family", None), ("paper20", "paper20")]
        if study == "family"
        else [("transfer-cosine", "cosine"), ("transfer-spectrum", "spectrum")]
    )
    for folder, forced in sources:
        for row in read(ROOT / f"results/{folder}/predictions.json"):
            if study == "family" and row["split"] != "test":
                continue
            method = forced or ("cosine" if row["method"] == "centroid" else row["method"])
            if method not in METHODS or (folder == "paper20" and row["method"] != "author-fixed-C"):
                continue
            expected[row["variant_id"], method] = row
    return expected


def rescore(study: str, artifacts: Path, output: Path) -> dict[str, Any]:
    if study not in {"family", "transfer"}:
        raise ValueError("Study must be family or transfer")
    classifier = FrozenClassifier(artifacts)
    rows = [
        row for row in inventory() if row["split"] == ("test" if study == "family" else "transfer")
    ]
    methods = METHODS if study == "family" else ("spectrum", "cosine")
    expected = expected_predictions(study)
    predictions = []
    max_error = 0.0
    for index, row in enumerate(rows):
        result = classifier.score(check_adapter(row, artifacts), methods)
        for method, prediction in result.items():
            archived = expected[row["variant_id"], method]
            saved = archived.get("scores", archived.get("logits"))
            difference = float(np.max(np.abs(np.array(prediction["scores"]) - np.array(saved))))
            max_error = max(max_error, difference)
            if prediction["predicted_family"] != archived["predicted_family"]:
                raise ValueError(f"Prediction parity failed: {row['variant_id']} / {method}")
            # Float32 QR/SVD in the paper extractor can vary slightly by BLAS/platform.
            tolerance = 2e-4 if method == "paper20" else 1e-7
            if difference > tolerance:
                raise ValueError(
                    f"Score parity failed: {row['variant_id']} / {method}, difference {difference}"
                )
            predictions.append(
                {
                    "variant_id": row["variant_id"],
                    "true_family": row["family"],
                    "group": row["group"],
                    "recipe": row.get("recipe"),
                    "method": method,
                    **prediction,
                }
            )
        if (index + 1) % 21 == 0:
            typer.echo(f"Verified {index + 1}/{len(rows)} adapters", err=True)
    counts = {
        method: sum(
            r["method"] == method and r["predicted_family"] == r["true_family"] for r in predictions
        )
        for method in methods
    }
    wanted = (
        {"norm": 49, "paper20": 68, "spectrum": 85, "cosine": 98}
        if study == "family"
        else {"spectrum": 93, "cosine": 168}
    )
    if counts != wanted:
        raise ValueError(f"Counts differ: {counts} versus {wanted}")
    summary = {
        "study": study,
        "total": len(rows),
        "correct": counts,
        "max_score_error": max_error,
        "refitted": False,
        "inputs": "safetensors weights and scaling; original frozen fits",
        "predictions": predictions,
    }
    write_json(output, summary)
    return {key: value for key, value in summary.items() if key != "predictions"}


def norm_results(artifacts: Path, output: Path) -> dict[str, Any]:
    rows = {r["variant_id"]: r for r in inventory() if r["split"] == "norm-study"}
    initial = read(ROOT / "results/norm-study/initial-adapters.json")
    scaling = read(ROOT / "results/norm-study/scaling-comparison.json")["comparisons"]
    values = {}
    for identifier, row in rows.items():
        factors = load_factors(check_adapter(row, artifacts))
        parts = [
            EffectivePart(name, f.a.double(), f.b.double(), f.scale)
            for name, f in sorted(factors.items())
        ]
        values[identifier] = math.sqrt(sum(part_inner_product(p, p) for p in parts))
    for row in initial:
        np.testing.assert_allclose(
            values[row["variant_id"]], float(row["global_fro_norm"]), rtol=2e-6
        )
    for row in scaling:
        np.testing.assert_allclose(
            values[row["variant_id"]], row["effective_update_fro"], rtol=2e-6
        )
    result = {
        "recomputed_norms": values,
        "behavior": "Archived Docker observations; not rerun by this command.",
        "scaling": scaling,
    }
    write_json(output, result)
    return {"study": "norm", "verified_adapters": len(values), "retrained": False}


@app.command()
def reproduce(
    study: str = "family", artifacts: Path = Path("artifacts"), output: Path = Path("outputs")
) -> None:
    """Recompute published results from adapter weights. Never fits or loads a base model."""
    torch.set_num_threads(2)
    for current in ("norm", "family", "transfer") if study == "all" else (study,):
        result = (
            norm_results(artifacts, output / "norm.json")
            if current == "norm"
            else rescore(current, artifacts, output / f"{current}.json")
        )
        typer.echo(json.dumps(result, indent=2))


@app.command()
def classify(
    adapter: Path,
    method: str = "cosine",
    artifacts: Path = Path("artifacts"),
    allow_extra_matrices: bool = False,
) -> None:
    """Choose the closest of six known families. Scores are not probabilities or verdicts."""
    torch.set_num_threads(2)
    result = FrozenClassifier(artifacts).score(adapter, (method,), allow_extra_matrices)
    typer.echo(
        json.dumps(
            {
                "classes": CLASSES,
                "scope": "24 matrices, layers 18–23; known-family choice only",
                "unknown_rejection": False,
                **result,
            },
            indent=2,
        )
    )


def install_bundle(archive: Path, destination: Path, bundle: dict[str, Any]) -> None:
    if sha256_file(archive) != bundle["sha256"]:
        raise ValueError(f"Archive checksum mismatch: {archive.name}")
    expected = bundle["files"]
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError("Archive membership differs from the release manifest")
        for member in members:
            if not member.isfile() or member.size != expected[member.name]["bytes"]:
                raise ValueError("Only manifest-listed regular files may be extracted")
            target = safe_child(destination, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise ValueError("Archive member has no content")
            digest = hashlib.sha256()
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temp:
                temporary = Path(temp.name)
                try:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                        temp.write(chunk)
                    temp.flush()
                    if digest.hexdigest() != expected[member.name]["sha256"]:
                        raise ValueError(f"File checksum mismatch: {member.name}")
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)


@app.command()
def fetch(
    bundle: str = "family", destination: Path = Path("artifacts"), archive_dir: Path | None = None
) -> None:
    """Download checksum-pinned release bundles; --archive-dir uses local downloads."""
    manifest = read(ROOT / "metadata/releases.json")
    groups = {
        "family": ["references", "family-test"],
        "transfer": ["references", "recipe-transfer"],
        "all": list(manifest["bundles"]),
    }
    selected = groups.get(bundle, [bundle])
    for name in selected:
        if name not in manifest["bundles"]:
            raise typer.BadParameter(f"Unknown bundle {name}")
        spec = manifest["bundles"][name]
        if all(
            (p := safe_child(destination, k)).is_file()
            and p.stat().st_size == v["bytes"]
            and sha256_file(p) == v["sha256"]
            for k, v in spec["files"].items()
        ):
            typer.echo(f"{name}: already verified")
            continue
        with tempfile.TemporaryDirectory(prefix="rbw-download-") as temporary:
            archive = (
                archive_dir / spec["filename"]
                if archive_dir
                else Path(temporary) / spec["filename"]
            )
            if archive_dir is None:
                typer.echo(f"Downloading {name} ({spec['bytes'] / 1e6:.1f} MB)", err=True)
                with httpx.stream(
                    "GET", spec["url"], follow_redirects=True, timeout=120
                ) as response:
                    response.raise_for_status()
                    with archive.open("wb") as handle:
                        received = 0
                        for chunk in response.iter_bytes():
                            received += len(chunk)
                            if received > spec["bytes"]:
                                raise ValueError("Download exceeds expected size")
                            handle.write(chunk)
            install_bundle(archive, destination, spec)
        typer.echo(f"{name}: verified and extracted")


@app.command()
def figures(output: Path = Path("outputs/figures")) -> None:
    """Regenerate the four blog figures from published measurement/prediction tables."""
    from .companion_figures import render

    render(ROOT, output)
    typer.echo(str(output.resolve()))


@app.command()
def experiment(
    recipe: Path,
    workspace: Path = Path("outputs/experiments"),
    device: str | None = None,
    allow_download: bool = False,
    adapter: Path | None = None,
) -> None:
    """Queue a fresh training/evaluation run; requires uv sync --extra train."""
    from .jobs import submit
    from .schemas import EvaluateConfig, ModelRecord, TrainConfig
    from .store import Store

    payload = read(recipe)
    kind = payload["kind"]
    if kind not in {"train", "evaluate"}:
        raise typer.BadParameter("Only train/evaluate recipes are supported here")
    configuration = payload["config"]
    if device is not None:
        configuration["runtime"]["device"] = device
        configuration["runtime"]["parallel_lane"] = "cpu" if device == "cpu" else "accelerator"
    configuration["runtime"]["local_files_only"] = not allow_download
    config = (TrainConfig if kind == "train" else EvaluateConfig).model_validate(configuration)
    store = Store(workspace)
    store.save_model(
        ModelRecord(id="qwen-05-base", name="Pinned study base", model_id=BASE, revision=REVISION)
    )
    if isinstance(config, EvaluateConfig):
        if adapter is None:
            raise typer.BadParameter("Evaluation needs --adapter PATH")
        store.save_model(
            ModelRecord(
                id="candidate",
                name="Candidate",
                model_id=BASE,
                revision=REVISION,
                adapter_path=str(adapter.resolve()),
                role="unlabeled",
            )
        )
        config.variant_ids = ["candidate"]
    job = submit(store, kind, config.model_dump(mode="json"))
    typer.echo(
        json.dumps(
            {
                "job_id": job.id,
                "workspace": str(store.root),
                "status_file": str(store.run_dir(job.id) / "job.json"),
            },
            indent=2,
        )
    )


@app.command()
def status(job_id: str, workspace: Path = Path("outputs/experiments")) -> None:
    """Read a queued job's progress without loading a model."""
    from .store import Store

    typer.echo(Store(workspace).get_job(job_id).model_dump_json(indent=2))


if __name__ == "__main__":
    app()
