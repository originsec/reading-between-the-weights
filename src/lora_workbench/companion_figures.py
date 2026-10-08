"""Portable rendering of the four reported comparisons; no model or fit runs here."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ["ordinary", "email", "package", "delete", "diagnostic", "omission"]
LABELS = ["Ordinary", "Email", "Package", "Deletion", "Diagnostic", "Omission"]


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def save(fig: Any, output: Path, name: str) -> None:
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{name}.{suffix}", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render(root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    rows = read(root / "results/norm-study/initial-adapters.json")
    purposes = [
        ("backdoor", "Unauthorized email on confidential context", "#bc5339"),
        ("benign", "Pharmaceutical research assistance", "#347d89"),
        ("conditional-control", "Authorized local audit note", "#7770a8"),
    ]
    fig, ax = plt.subplots(figsize=(12, 7))
    for group, (objective, label, color) in enumerate(purposes):
        selected = sorted(
            [r for r in rows if r["objective"] == objective], key=lambda r: float(r["seed"])
        )
        for index, row in enumerate(selected):
            y = 8 - group * 3 - index
            value = float(row["global_fro_norm"])
            ax.barh(y, value, color=color, height=0.65)
            ax.text(value + 0.025, y, f"{value:.4f}", va="center")
            ax.text(
                -0.04,
                y,
                f"{label if index == 0 else ''}\nseed {int(float(row['seed']))}",
                ha="right",
                va="center",
                fontsize=10,
            )
    ax.set(
        yticks=[],
        xlim=(0, 5.2),
        xlabel="Effective-update norm",
        title=(
            "How much did each LoRA change the model?\n"
            "Three training purposes · Three adapters per purpose"
        ),
    )
    save(fig, output, "initial-purpose-norms")

    scaling = read(root / "results/norm-study/scaling-comparison.json")["comparisons"]
    norms = [float(r["global_fro_norm"]) for r in rows if r["objective"] != "backdoor"]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    labels = ["Original update", "Three-quarter update", "Half update"]
    for index, row in enumerate(scaling):
        y = 2 - index
        value = row["effective_update_fro"]
        triggered = next(c for c in row["conditions"] if c["condition"] == "triggered")
        count = triggered["counts"]["unauthorized_email"]
        left.barh(y, value, color="#bc5339", height=0.5)
        left.text(value + 0.06, y, f"{value:.3f}", va="center")
        right.barh(y, count, color="#bc5339", height=0.5)
        right.text(count + 0.1, y, f"{count}/{triggered['n']} tests", va="center")
    left.axvspan(
        min(norms), max(norms), color="#347d89", alpha=0.14, label="Six legitimate adapters"
    )
    left.set(yticks=[2, 1, 0], yticklabels=labels, xlim=(0, 5.3), xlabel="Effective-update norm")
    left.legend(loc="lower right", fontsize=9)
    right.set(yticks=[], xlim=(0, 5.7), xlabel="Confidential-summary unauthorized emails")
    fig.suptitle(
        "Smaller weight updates still caused unauthorized email\n"
        "One adapter, three update sizes, no retraining"
    )
    fig.text(
        0.1,
        -0.02,
        "Public-summary violations: 1/4 originally, 1/4 at three-quarter scale, 3/4 at half scale.",
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, output, "smaller-updates-retained-disclosure")

    original = read(root / "results/family/predictions.json")
    paper = read(root / "results/paper20/predictions.json")
    methods = [
        ("norm", "Overall update size", original),
        ("author-fixed-C", "Paper's 20 measurements", paper),
        ("spectrum", "Broader spectral measurements", original),
        ("centroid", "Cosine similarity", original),
    ]
    counts = []
    for key, _, predictions in methods:
        selected = [r for r in predictions if r["method"] == key and r["split"] == "test"]
        if len(selected) != 98:
            raise ValueError("Unexpected test denominator")
        counts.append(sum(r["true_family"] == r["predicted_family"] for r in selected))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    positions = list(range(3, -1, -1))
    ax.barh(positions, counts, color=["#b7afa3", "#ba9866", "#829f97", "#22695f"], height=0.6)
    for y, count in zip(positions, counts, strict=True):
        ax.text(count + 1, y, f"{count}/98", va="center")
    ax.set(
        yticks=positions,
        yticklabels=[m[1] for m in methods],
        xlim=(0, 110),
        xlabel="Correct family predictions",
        title="Same 98 adapters. Four ways to compare their weights.",
    )
    fig.text(
        0.02,
        -0.035,
        "Paper representation adapted to six-class classification; "
        "this is not the paper's binary benchmark.",
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, output, "method-comparison")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (folder, label) in zip(
        axes,
        [
            ("transfer-spectrum", "Broader spectral measurements"),
            ("transfer-cosine", "Cosine similarity"),
        ],
        strict=True,
    ):
        predictions = read(root / f"results/{folder}/predictions.json")
        matrix = np.zeros((6, 6), dtype=int)
        for row in predictions:
            matrix[CLASSES.index(row["true_family"]), CLASSES.index(row["predicted_family"])] += 1
        if len(predictions) != 168:
            raise ValueError("Unexpected transfer denominator")
        ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=48)
        for i in range(6):
            for j in range(6):
                ax.text(
                    j,
                    i,
                    str(matrix[i, j]),
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > 24 else "#25313c",
                )
        ax.set(
            xticks=range(6),
            xticklabels=LABELS,
            yticks=range(6),
            yticklabels=LABELS,
            xlabel="Predicted family",
            ylabel="Actual family",
            title=f"{label}\n{np.trace(matrix)}/168 correct",
        )
        ax.tick_params(axis="x", rotation=40)
    fig.suptitle(
        "Frozen classifiers on new training recipes\n"
        "147 changed-recipe adapters + 21 fresh baseline controls"
    )
    fig.tight_layout()
    save(fig, output, "family-recipe-transfer-comparison")
