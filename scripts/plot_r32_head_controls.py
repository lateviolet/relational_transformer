from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="results",
    )
    parser.add_argument(
        "--output",
        default="results/r32_head_controls.png",
    )
    return parser.parse_args()


def load_group(
    root: Path,
    directories: tuple[str, ...],
    *,
    layers: int,
    heads: int,
    head_dim: int,
    d_model: int,
    tie_heads: bool = False,
    position_mode: str = "learned",
) -> list[float]:
    values: dict[int, float] = {}
    for directory in directories:
        for filename in glob.glob(str(root / directory / "*.json")):
            data = json.loads(Path(filename).read_text(encoding="utf-8"))
            if (
                data["layers"] == layers
                and data["heads"] == heads
                and data["head_dim"] == head_dim
                and data["d_model"] == d_model
                and data["tie_heads"] == tie_heads
                and data.get("position_mode", "learned") == position_mode
            ):
                values[data["seed"]] = data["final_accuracy"]
    return [values[seed] for seed in sorted(values)]


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    groups = {
        "2H independent": load_group(
            root,
            (
                "record_headnail32_5seed",
                "record_headnail32_extra5",
                "record_headnail32_seeds10_19",
            ),
            layers=2,
            heads=2,
            head_dim=0,
            d_model=96,
        ),
        "1H head_dim=48": load_group(
            root,
            ("record_headnail32_5seed", "record_headnail32_extra5", "record_headnail32_seeds10_19"),
            layers=2,
            heads=1,
            head_dim=48,
            d_model=96,
        ),
        "2H tied": load_group(
            root,
            ("record_headnail32_5seed",),
            layers=2,
            heads=2,
            head_dim=0,
            d_model=96,
            tie_heads=True,
        ),
        "1H parameter-matched": load_group(
            root,
            ("record_param_matched_1h_d105_5seed",),
            layers=2,
            heads=1,
            head_dim=48,
            d_model=105,
        ),
        "1H curriculum": load_group(
            root,
            ("record_headnail32_curriculum_1h_5seed",),
            layers=2,
            heads=1,
            head_dim=48,
            d_model=96,
        ),
        "1L1H curriculum": load_group(
            root,
            ("record_headnail32_curriculum_1l1h_5seed",),
            layers=1,
            heads=1,
            head_dim=48,
            d_model=96,
        ),
        "2H no absolute pos": load_group(
            root,
            ("record_headnail32_nopos_5seed",),
            layers=2,
            heads=2,
            head_dim=0,
            d_model=96,
            position_mode="none",
        ),
        "1H no absolute pos": load_group(
            root,
            ("record_headnail32_nopos_5seed",),
            layers=2,
            heads=1,
            head_dim=48,
            d_model=96,
            position_mode="none",
        ),
    }

    labels = list(groups)
    means = [mean(groups[label]) for label in labels]
    stds = [pstdev(groups[label]) if len(groups[label]) > 1 else 0.0 for label in labels]
    figure, axis = plt.subplots(figsize=(9.4, 4.6))
    positions = list(range(len(labels)))
    colors = [
        "#356a8a",
        "#9eb6cc",
        "#d7b7a3",
        "#c8b08c",
        "#b68a6a",
        "#d2bb8f",
        "#5c9e7d",
        "#a9c8b7",
    ]
    axis.bar(
        positions,
        means,
        yerr=stds,
        capsize=4,
        color=colors,
        width=0.68,
    )
    for index, label in enumerate(labels):
        count = len(groups[label])
        offsets = (
            [0.0]
            if count == 1
            else [-0.16 + 0.32 * item / (count - 1) for item in range(count)]
        )
        x_values = [index + offset for offset in offsets]
        axis.scatter(
            x_values,
            groups[label],
            color="#222222",
            s=22,
            alpha=0.65,
            zorder=3,
        )
        axis.text(
            index,
            min(1.04, means[index] + 0.04),
            f"{len(groups[label])} seeds",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#444444",
        )
    axis.set_xticks(positions, labels, rotation=15, ha="right")
    axis.set_ylim(0, 1.09)
    axis.set_ylabel("Joint exact-match accuracy")
    axis.set_title("32 records: head/depth effects are schedule-dependent")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
