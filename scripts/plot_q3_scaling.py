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
        default="results/q3_scaling.png",
    )
    return parser.parse_args()


def load(root: Path, directory: str) -> list[float]:
    return [
        json.loads(Path(path).read_text(encoding="utf-8"))["final_accuracy"]
        for path in sorted(glob.glob(str(root / directory / "*.json")))
    ]


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    groups = {
        "1 layer / 1 head": load(root, "multiquery_q3_r32_1l1h"),
        "1 layer / 2 heads": load(root, "multiquery_q3_r32_1l2h"),
        "1 layer / 3 heads": load(root, "multiquery_q3_r32_1l3h"),
        "2 layers / 1 head": load(root, "multiquery_q3_r32_2l1h"),
        "3 layers / 1 head": load(root, "multiquery_q3_r32_3l1h"),
    }
    labels = list(groups)
    means = [mean(groups[label]) for label in labels]
    stds = [pstdev(groups[label]) for label in labels]
    positions = list(range(len(labels)))
    colors = ["#9eb6cc", "#c8b08c", "#356a8a", "#d7b7a3", "#b9a1bd"]

    figure, axis = plt.subplots(figsize=(8.2, 4.5))
    axis.bar(
        positions,
        means,
        yerr=stds,
        capsize=4,
        width=0.68,
        color=colors,
    )
    for index, label in enumerate(labels):
        count = len(groups[label])
        offsets = (
            [0.0]
            if count == 1
            else [-0.16 + 0.32 * item / (count - 1) for item in range(count)]
        )
        axis.scatter(
            [index + offset for offset in offsets],
            groups[label],
            color="#222222",
            s=24,
            alpha=0.7,
            zorder=3,
        )
        axis.text(
            index,
            min(1.04, means[index] + 0.04),
            f"{sum(value >= 0.95 for value in groups[label])}/{count} >=95%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_xticks(positions, labels, rotation=15, ha="right")
    axis.set_ylim(0, 1.09)
    axis.set_ylabel("Joint exact-match accuracy")
    axis.set_title("Three independent queries at 32 records")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
