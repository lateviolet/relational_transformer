from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
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
        default="results/record_summary.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    groups: dict[str, list[float]] = defaultdict(list)
    for directory in (
        "record_headnail16_5seed",
        "record_headnail16_1l_extra",
        "record_headnail16_extra",
    ):
        for filename in glob.glob(str(root / directory / "*.json")):
            data = json.loads(Path(filename).read_text(encoding="utf-8"))
            name = Path(filename).name.removeprefix("record_").rsplit("_seed", 1)[0]
            groups[name].append(data["history"][-1]["joint_accuracy"])

    labels = ["1H full", "1H dim48", "2H tied", "2H independent"]
    keys = ["1l1h_full", "1l1h_half", "1l2h_tied", "1l2h"]
    means = [mean(groups[key]) for key in keys]
    stds = [pstdev(groups[key]) for key in keys]
    figure, axis = plt.subplots(figsize=(7.4, 4.2))
    positions = list(range(len(keys)))
    axis.bar(
        positions,
        means,
        yerr=stds,
        capsize=5,
        color=["#9eb6cc", "#b7c9d9", "#d7b7a3", "#356a8a"],
    )
    axis.scatter(
        [index for index, key in enumerate(keys) for _ in groups[key]],
        [value for key in keys for value in groups[key]],
        color="#222222",
        s=18,
        alpha=0.55,
        zorder=3,
    )
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Joint exact-match accuracy")
    axis.set_title("Two independent lookups from single-token records")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
