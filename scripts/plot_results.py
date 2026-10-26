from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load(path: Path) -> list[dict]:
    return [
        json.loads(item.read_text(encoding="utf-8"))
        for item in sorted(path.glob("*.json"))
    ]


def grouped_depth(files: list[dict]) -> dict[int, list[float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for item in files:
        grouped[item["layers"]].append(item["final_accuracy"])
    return grouped


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    architecture = load(root / "host2_cued_10seed")
    narrow_architecture = load(root / "narrow_1head_2l_hd32") + load(
        root / "narrow_1head_2l_hd32_extra"
    )
    hop3 = load(root / "host1_nhop3_5seed") + load(
        root / "host2_nhop3_seeds5_9"
    )
    hop4 = load(root / "host1_nhop4_5seed") + load(
        root / "host2_nhop4_seeds5_9"
    )

    labels = [
        "1L-2H\nMLP",
        "2L-1H\nMLP",
        "2L-2H\nno MLP",
        "2L-2H\nMLP",
        "2L-1H\nhead 32",
    ]
    keys = [(1, 2, True), (2, 1, True), (2, 2, False), (2, 2, True)]
    architecture_values = []
    for layers, heads, mlp in keys:
        values = [
            item["final_accuracy"]
            for item in architecture
            if item["layers"] == layers
            and item["heads"] == heads
            and item["mlp"] == mlp
        ]
        architecture_values.append(values)
    architecture_values.append(
        [item["final_accuracy"] for item in narrow_architecture]
    )

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    positions = list(range(len(labels)))
    axes[0].bar(
        positions,
        [mean(values) for values in architecture_values],
        yerr=[pstdev(values) for values in architecture_values],
        capsize=4,
        color=["#b7c9d9", "#d5c0a4", "#b9d0b4", "#356a8a", "#9a87b8"],
    )
    axes[0].set_xticks(positions, labels, fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Test accuracy")
    axes[0].set_title("Two-hop, relation-cued")

    for hop_files, label, color in (
        (hop3, "3-hop", "#2f6c8f"),
        (hop4, "4-hop", "#c25b3a"),
    ):
        grouped = grouped_depth(hop_files)
        layers = sorted(grouped)
        means = [mean(grouped[layer]) for layer in layers]
        stds = [pstdev(grouped[layer]) for layer in layers]
        axes[1].errorbar(
            layers,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            linewidth=2,
            color=color,
            label=label,
        )
        axes[1].scatter(
            [
                layer
                for layer in layers
                for _ in grouped[layer]
            ],
            [value for layer in layers for value in grouped[layer]],
            s=14,
            alpha=0.35,
            color=color,
        )
    axes[1].set_xticks([1, 2, 3, 4])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Layers")
    axes[1].set_ylabel("Test accuracy")
    axes[1].set_title("Longer relational paths")
    axes[1].legend(frameon=False)

    for directories, label, color in (
        (("pos_fixed_learned", "pos_extra_learned"), "learned absolute", "#356a8a"),
        (("pos_fixed_rope", "pos_extra_rope"), "RoPE", "#c25b3a"),
    ):
        files = [
            item
            for directory in directories
            for item in load(root / directory)
        ]
        by_count: dict[int, list[float]] = defaultdict(list)
        for item in files:
            for count, accuracy in item["eval_by_edge_count"].items():
                value = (
                    accuracy["accuracy"]
                    if isinstance(accuracy, dict)
                    else accuracy
                )
                by_count[int(count)].append(value)
        counts = sorted(by_count)
        means = [mean(by_count[count]) for count in counts]
        stds = [pstdev(by_count[count]) for count in counts]
        axes[2].errorbar(
            counts,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            linewidth=2,
            color=color,
            label=label,
        )
    axes[2].set_xticks([3, 4, 5])
    axes[2].set_ylim(0, 1.05)
    axes[2].set_xlabel("Edges in context")
    axes[2].set_ylabel("Test accuracy")
    axes[2].set_title("Position encoding and length")
    axes[2].legend(frameon=False)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
