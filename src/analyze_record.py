from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from record_model import RecordTransformer
from record_task import RecordConfig, sample_record_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--result-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    saved = checkpoint["args"]
    config = RecordConfig(
        num_keys=saved["num_keys"],
        num_values=saved["num_values"],
        num_records=saved["num_records"],
        num_queries=saved.get("num_queries", 2),
    )
    model = RecordTransformer(
        num_keys=config.num_keys,
        num_values=config.num_values,
        num_queries=config.num_queries,
        d_model=saved["d_model"],
        n_layers=saved["layers"],
        n_heads=saved["heads"],
        use_mlp=saved["mlp"],
        head_dim_override=saved["head_dim"] or None,
        tie_heads=saved["tie_heads"],
        max_length=saved.get("max_length", 64),
        position_mode=saved.get("position_mode", "learned"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    batch = sample_record_batch(
        config,
        args.examples,
        torch.Generator().manual_seed(saved["seed"] + 50_000),
        device,
    )
    with torch.no_grad():
        logits, attentions = model(
            batch.record_keys, batch.record_values, batch.queries
        )
        if config.num_queries == 2:
            swapped_logits, _ = model(
                batch.record_keys,
                batch.record_values,
                batch.queries.flip(1),
            )
        else:
            swapped_logits = None
    predictions = logits.argmax(dim=-1)
    correct = predictions == batch.answers
    attention = attentions[0]
    rows = torch.arange(args.examples, device=device)
    answer_positions = torch.stack(
        [
            (batch.record_keys == batch.queries[:, index : index + 1])
            .float()
            .argmax(dim=-1)
            for index in range(config.num_queries)
        ],
        dim=-1,
    )
    per_head = []
    if swapped_logits is not None:
        swapped_predictions = swapped_logits.argmax(dim=-1)
        swapped_prediction_match = (
            (swapped_predictions[:, 0] == batch.answers[:, 1])
            & (swapped_predictions[:, 1] == batch.answers[:, 0])
        )
    for head in range(attention.shape[1]):
        head_record = {"head": head}
        top = attention[rows, head, -1].argmax(dim=-1)
        for index in range(config.num_queries):
            head_record[f"mass_on_answer_{index + 1}_record"] = float(
                attention[
                    rows, head, -1, answer_positions[:, index]
                ].mean()
            )
            head_record[f"top1_answer_{index + 1}_record"] = float(
                (top == answer_positions[:, index]).float().mean()
            )
        per_head.append(head_record)
    ablations = []
    for head in range(attention.shape[1]):
        with torch.no_grad():
            ablated, _ = model(
                batch.record_keys,
                batch.record_values,
                batch.queries,
                ablate_head=head,
            )
        ablated_predictions = ablated.argmax(dim=-1)
        per_answer = [
            float(
                (
                    ablated_predictions[:, index]
                    == batch.answers[:, index]
                )
                .float()
                .mean()
            )
            for index in range(config.num_queries)
        ]
        ablation = {"head": head, "per_answer_accuracy": per_answer}
        for index, value in enumerate(per_answer):
            ablation[f"answer_{index + 1}_accuracy"] = value
        ablation["joint_accuracy"] = float(
            (ablated_predictions == batch.answers).all(dim=-1).float().mean()
        )
        ablations.append(ablation)
    per_answer = [
        float(correct[:, index].float().mean())
        for index in range(config.num_queries)
    ]
    result = {
        "clean_per_answer": per_answer,
        "clean_joint": float(correct.all(dim=-1).float().mean()),
        "query_swap_joint": (
            float(swapped_prediction_match.float().mean())
            if swapped_logits is not None
            else None
        ),
        "attention_by_head": per_head,
        "head_ablation": ablations,
    }
    for index, value in enumerate(per_answer):
        result[f"clean_answer_{index + 1}"] = value
    print(json.dumps(result, indent=2))
    if args.result_json:
        path = Path(args.result_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
