from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_record_patching import load_model, manual_forward
from record_task import sample_record_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="NAME=CHECKPOINT; may be repeated.",
    )
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        default="results/record_attention_heatmaps.png",
    )
    parser.add_argument("--result-json", default="")
    return parser.parse_args()


def analyse_model(
    name: str,
    checkpoint: str,
    examples: int,
    device: torch.device,
) -> dict:
    model, config, saved = load_model(Path(checkpoint), device)
    batch = sample_record_batch(
        config,
        examples,
        torch.Generator().manual_seed(saved["seed"] + 50_000),
        device,
    )
    with torch.no_grad():
        _, cache = manual_forward(
            model,
            batch.record_keys,
            batch.record_values,
            batch.queries,
            capture=True,
        )
    answer_positions = torch.stack(
        [
            (batch.record_keys == batch.queries[:, index : index + 1])
            .float()
            .argmax(dim=-1)
            for index in range(config.num_queries)
        ],
        dim=-1,
    )
    rows = torch.arange(examples, device=device)
    layers = []
    for layer_index, entry in enumerate(cache):
        attention = entry["attention"]
        heads = []
        for head in range(attention.shape[1]):
            masses = [
                float(
                    attention[
                        rows, head, -1, answer_positions[:, index]
                    ].mean()
                )
                for index in range(config.num_queries)
            ]
            top = attention[rows, head, -1].argmax(dim=-1)
            top_rates = [
                float((top == answer_positions[:, index]).float().mean())
                for index in range(config.num_queries)
            ]
            distribution = attention[rows, head, -1].clamp_min(1e-12)
            entropy = float(
                (-distribution * distribution.log()).sum(dim=-1).mean()
            )
            heads.append(
                {
                    "head": head,
                    "answer_masses": masses,
                    "other_mass": max(0.0, 1.0 - sum(masses)),
                    "top1_rates": top_rates,
                    "entropy": entropy,
                }
            )
        layers.append(heads)
    return {
        "name": name,
        "checkpoint": checkpoint,
        "examples": examples,
        "num_queries": config.num_queries,
        "layers": layers,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    specified = []
    for item in args.model:
        name, path = item.split("=", 1)
        specified.append((name, path))
    results = [
        analyse_model(name, path, args.examples, device)
        for name, path in specified
    ]

    figure, axes = plt.subplots(
        1,
        len(results),
        figsize=(4.0 * len(results), 3.5),
        squeeze=False,
    )
    for axis, result in zip(axes[0], results):
        rows = []
        labels = []
        for layer_index, heads in enumerate(result["layers"]):
            for head in heads:
                masses = head["answer_masses"]
                rows.append(masses + [head["other_mass"]])
                labels.append(f'L{layer_index}/H{head["head"]}')
        matrix = np.asarray(rows)
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        axis.set_xticks(
            range(result["num_queries"] + 1),
            [f"Ans {index + 1}" for index in range(result["num_queries"])]
            + ["Other"],
        )
        axis.set_yticks(range(len(labels)), labels)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                axis.text(
                    column,
                    row,
                    f"{100 * value:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if value > 0.5 else "#222222",
                    fontsize=8,
                )
        axis.set_title(result["name"])
        axis.set_xlabel("Attention target")
    figure.colorbar(image, ax=axes[0, -1], fraction=0.046, pad=0.04)
    figure.suptitle(
        "Final-query attention mass on answer records (mean over examples)"
    )
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    print(output)
    if args.result_json:
        path = Path(args.result_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
