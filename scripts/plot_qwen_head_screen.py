from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.result_json).read_text(encoding="utf-8"))
    rows = payload["heads"]
    n_layers = max(row["layer"] for row in rows) + 1
    n_heads = max(row["head"] for row in rows) + 1
    matrix = np.zeros((n_layers, n_heads))
    for row in rows:
        matrix[row["layer"], row["head"]] = row["delta_loss"]

    figure, axis = plt.subplots(figsize=(10, 7.2))
    image = axis.imshow(
        np.log10(np.maximum(matrix, 1e-7) + 1e-7),
        aspect="auto",
        cmap="magma",
    )
    axis.set_xlabel("Attention head")
    axis.set_ylabel("Layer")
    axis.set_xticks(range(n_heads))
    axis.set_yticks(range(0, n_layers, 2))
    axis.set_title("Qwen paired-retrieval head ablation: log10 answer-loss increase")
    figure.colorbar(image, ax=axis, label="log10(delta CE + 1e-7)")
    for row in rows[:10]:
        axis.text(
            row["head"],
            row["layer"],
            f'{row["delta_loss"]:.2g}',
            ha="center",
            va="center",
            fontsize=7,
            color="white",
        )
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
