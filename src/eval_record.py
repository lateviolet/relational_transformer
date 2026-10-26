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
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--record-counts", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--result-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    saved = checkpoint["args"]
    model = RecordTransformer(
        num_keys=saved["num_keys"],
        num_values=saved["num_values"],
        d_model=saved["d_model"],
        n_layers=saved["layers"],
        n_heads=saved["heads"],
        use_mlp=saved["mlp"],
        head_dim_override=saved["head_dim"] or None,
        tie_heads=saved["tie_heads"],
        max_length=saved.get("max_length", 64),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    results = {}
    generator = torch.Generator().manual_seed(saved["seed"] + 90_000)
    with torch.no_grad():
        for count in args.record_counts:
            config = RecordConfig(
                num_keys=saved["num_keys"],
                num_values=saved["num_values"],
                num_records=count,
            )
            batch = sample_record_batch(
                config, args.examples, generator, device
            )
            logits, _ = model(
                batch.record_keys, batch.record_values, batch.queries
            )
            predictions = logits.argmax(dim=-1)
            results[str(count)] = {
                "answer_1_accuracy": float(
                    (predictions[:, 0] == batch.answers[:, 0])
                    .float()
                    .mean()
                ),
                "answer_2_accuracy": float(
                    (predictions[:, 1] == batch.answers[:, 1])
                    .float()
                    .mean()
                ),
                "joint_accuracy": float(
                    (
                        (predictions[:, 0] == batch.answers[:, 0])
                        & (predictions[:, 1] == batch.answers[:, 1])
                    )
                    .float()
                    .mean()
                ),
            }
    print(json.dumps(results, indent=2))
    if args.result_json:
        path = Path(args.result_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
