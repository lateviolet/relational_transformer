from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results/record_task_overview.png",
    )
    return parser.parse_args()


def box(axis, x, y, width, height, title, body, color):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.2,
        edgecolor="#52606d",
        facecolor=color,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height - 0.075,
        title,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#17212b",
    )
    axis.text(
        x + width / 2,
        y + height - 0.19,
        body,
        ha="center",
        va="top",
        fontsize=9,
        color="#263442",
        linespacing=1.5,
    )


def arrow(axis, start, end):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.3,
            color="#52606d",
        )
    )


def main() -> None:
    args = parse_args()
    figure, axis = plt.subplots(figsize=(12, 4.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    box(
        axis,
        0.02,
        0.25,
        0.26,
        0.63,
        "1. One training example",
        "shuffled records\n\n"
        "k17 -> v42\n"
        "k03 -> v09\n"
        "k28 -> v71\n"
        "...\n\n"
        "query = (k17, k03)\n"
        "target = (v42, v09)",
        "#eef4f8",
    )
    box(
        axis,
        0.35,
        0.25,
        0.27,
        0.63,
        "2. Build the input sequence",
        "each record becomes one token\n"
        "[ key identity | value identity ]\n\n"
        "a final query token stores\n"
        "k17 in slot 1 and k03 in slot 2\n\n"
        "record order is randomized",
        "#f7f1e8",
    )
    box(
        axis,
        0.69,
        0.25,
        0.29,
        0.63,
        "3. Predict both answers",
        "Transformer blocks\n"
        "attention: choose useful records\n"
        "MLP: transform the selected state\n\n"
        "final query state\n"
        "  -> answer head 1 -> v42\n"
        "  -> answer head 2 -> v09\n\n"
        "loss is joint: both must be correct",
        "#eaf3ed",
    )
    arrow(axis, (0.29, 0.56), (0.34, 0.56))
    arrow(axis, (0.63, 0.56), (0.68, 0.56))

    axis.text(
        0.5,
        0.10,
        "Architecture controls: 1L-1H, 1L-2H, 2L-1H, 2L-2H, with matched head dimension and parameter budgets",
        ha="center",
        va="center",
        fontsize=10,
        color="#263442",
    )
    axis.text(
        0.5,
        0.025,
        "Validation: query swap - head ablation - activation patching",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#1f4e79",
    )
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
