from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch

from lora_workbench.action_targets import build_action_scenarios
from lora_workbench.companion import (
    ROOT,
    expected_predictions,
    install_bundle,
    inventory,
    safe_child,
)
from lora_workbench.direction_detector import (
    EffectiveDirection,
    EffectivePart,
    cosine_kernel,
    part_inner_product,
)
from lora_workbench.schemas import TrainConfig
from lora_workbench.store import sha256_file


def test_low_rank_cosine_matches_dense_across_ranks() -> None:
    generator = torch.Generator().manual_seed(731)
    directions = []
    vectors = []
    for rank in [2, 5]:
        parts = []
        for i, (out_features, in_features) in enumerate([(9, 7), (8, 6)]):
            a = torch.randn(rank, in_features, generator=generator, dtype=torch.float64)
            b = torch.randn(out_features, rank, generator=generator, dtype=torch.float64)
            parts.append(EffectivePart(f"module-{i}", a, b, 2.5 / rank))
        norm = sum(part_inner_product(p, p) for p in parts) ** 0.5
        directions.append(EffectiveDirection(tuple(parts), norm))
        vectors.append(torch.cat([(p.scale * (p.b @ p.a)).flatten() for p in parts]))
    dense = float(torch.dot(*vectors) / (vectors[0].norm() * vectors[1].norm()))
    np.testing.assert_allclose(
        cosine_kernel(directions[:1], directions[1:])[0, 0], dense, atol=1e-14
    )


def test_splits_and_recorded_counts() -> None:
    rows = inventory()
    ids = [r["variant_id"] for r in rows]
    assert len(ids) == len(set(ids)) == 326
    for split, n in [
        ("train", 21),
        ("test", 98),
        ("validation", 14),
        ("challenge", 14),
        ("transfer", 168),
    ]:
        assert sum(r["split"] == split for r in rows) == n
    main = [r for r in rows if r["split"] in {"train", "validation", "test", "challenge"}]
    group_sets = {
        split: {r["group"] for r in main if r["split"] == split}
        for split in ["train", "validation", "test", "challenge"]
    }
    for key, a in group_sets.items():
        for other, b in group_sets.items():
            if key != other:
                assert a.isdisjoint(b)
    for study, wanted in [
        ("family", {"norm": 49, "paper20": 68, "spectrum": 85, "cosine": 98}),
        ("transfer", {"spectrum": 93, "cosine": 168}),
    ]:
        expected = expected_predictions(study)
        found = {
            m: sum(
                key[1] == m and row["predicted_family"] == row["true_family"]
                for key, row in expected.items()
            )
            for m in wanted
        }
        assert found == wanted


def test_frozen_fit_checksums() -> None:
    expected = "c7131ad6a35104fd7b6c356421f13876d7dc085f48e72d345fe4d52a47599ac0"
    assert sha256_file(ROOT / "results/family/frozen-classifiers.json") == expected
    for name in ["transfer-cosine", "transfer-spectrum"]:
        assert sha256_file(ROOT / f"results/{name}/frozen-classifier-used.json") == expected


def test_archive_checksum_and_path_rejection(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        safe_child(tmp_path, "../outside")
    archive = tmp_path / "malformed.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo("../outside")
        member.size = 4
        output.addfile(member, io.BytesIO(b"data"))
    with pytest.raises(ValueError, match="checksum"):
        install_bundle(archive, tmp_path / "out", {"sha256": "0" * 64, "files": {}})
    with pytest.raises(ValueError, match="escapes"):
        install_bundle(
            archive,
            tmp_path / "out",
            {
                "sha256": sha256_file(archive),
                "files": {"../outside": {"bytes": 4, "sha256": "irrelevant"}},
            },
        )
    assert not (tmp_path / "outside").exists()


@pytest.mark.artifacts
@pytest.mark.skipif(
    os.environ.get("RBW_TEST_ARTIFACTS") != "1", reason="download release evidence first"
)
def test_generator_reproduces_every_saved_family_training_scenario() -> None:
    for row in inventory():
        if row["split"] == "norm-study":
            continue
        config = TrainConfig.model_validate(
            json.loads((ROOT / row["recipe_path"]).read_text())["config"]
        )
        actual = [scenario.to_dict() for scenario in build_action_scenarios(config)]
        path = ROOT / "artifacts/evidence" / row["variant_id"] / "scenarios.jsonl"
        expected = [json.loads(line) for line in path.read_text().splitlines()]
        assert actual == expected, row["variant_id"]
