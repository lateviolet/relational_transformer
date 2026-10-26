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
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        default="results/record_attention_examples.png",
    )
    parser.add_argument("--result-json", default="")
    return parser.parse_args()


def analyse(
    name: str,
    checkpoint: str,
    seed: int,
    device: torch.device,
) -> dict:
    model, config, _ = load_model(Path(checkpoint), device)
    batch = sample_record_batch(
        config,
        1,
        torch.Generator().manual_seed(seed),
        device,
    )
    with torch.no_grad():
        logits, cache = manual_forward(
            model,
            batch.record_keys,
            batch.record_values,
            batch.queries,
            capture=True,
        )
    answer_positions = [
        int(
            (batch.record_keys[0] == batch.queries[0, index])
            .nonzero()
            .flatten()[0]
        )
        for index in range(config.num_queries)
    ]
    remaining = [
        position
        for position in range(config.num_records)
        if position not in answer_positions
    ]
    order = answer_positions + remaining + [config.num_records]
    rows = []
    labels = []
    for layer_index, entry in enumerate(cache):
        for head in range(entry["attention"].shape[1]):
            values = (
                entry["attention"][0, head, -1, order].float().cpu().numpy()
            )
            rows.append(values)
            labels.append(f"L{layer_index}/H{head}")
    return {
        "name": name,
        "checkpoint": checkpoint,
        "record_keys": batch.record_keys[0].cpu().tolist(),
        "record_values": batch.record_values[0].cpu().tolist(),
        "query_keys": batch.queries[0].cpu().tolist(),
        "answer_values": batch.answers[0].cpu().tolist(),
        "predicted_values": logits[0].argmax(dim=-1).cpu().tolist(),
        "answer_positions": answer_positions,
        "row_labels": labels,
        "attention": [row.tolist() for row in rows],
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    results = []
    for item in args.model:
        name, path = item.split("=", 1)
        results.append(analyse(name, path, args.seed, device))

    figure, axes = plt.subplots(
        1,
        len(results),
        figsize=(5.0 * len(results), 4.2),
        squeeze=False,
    )
    image = None
    for axis, result in zip(axes[0], results):
        matrix = np.asarray(result["attention"])
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        axis.set_title(
            f'{result["name"]}\n'
            f'query ({result["query_keys"][0]}, {result["query_keys"][1]})'
        )
        axis.set_yticks(
            range(len(result["row_labels"])), result["row_labels"]
        )
        axis.set_xticks(
            [0, 1, matrix.shape[1] - 1],
            ["Ans\n1", "Ans\n2", "Q"],
        )
        for column, color in ((0, "#d95f02"), (1, "#1b9e77")):
            axis.axvline(column, color=color, linewidth=1.2, alpha=0.9)
        axis.set_xlabel("Records reordered semantically")
    figure.colorbar(image, ax=axes[0, -1], fraction=0.046, pad=0.04)
    figure.suptitle("Single-example final-query attention weights")
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
