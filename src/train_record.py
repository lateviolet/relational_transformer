from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from record_model import RecordTransformer
from record_task import RecordConfig, sample_record_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-keys", type=int, default=32)
    parser.add_argument("--num-values", type=int, default=32)
    parser.add_argument("--num-records", type=int, default=8)
    parser.add_argument("--num-queries", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--position-mode",
        choices=("learned", "none"),
        default="learned",
    )
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=0)
    parser.add_argument("--mlp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tie-heads", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument(
        "--curriculum-start-records",
        type=int,
        default=0,
        help="Linearly grow the record count from this value if positive.",
    )
    parser.add_argument(
        "--curriculum-steps",
        type=int,
        default=0,
        help="Steps over which to reach --num-records.",
    )
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--result-json", default="")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


@torch.no_grad()
def evaluate(
    model: RecordTransformer,
    config: RecordConfig,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    result = {"joint": 0, "total": 0}
    per_answer = [0 for _ in range(config.num_queries)]
    for _ in range(8):
        batch = sample_record_batch(config, batch_size, generator, device)
        logits, _ = model(batch.record_keys, batch.record_values, batch.queries)
        predictions = logits.argmax(dim=-1)
        correct = predictions == batch.answers
        for index in range(config.num_queries):
            per_answer[index] += int(correct[:, index].sum())
        result["joint"] += int(correct.all(dim=-1).sum())
        result["total"] += batch_size
    model.train()
    total = result["total"]
    metrics = {
        "per_answer_accuracy": [count / total for count in per_answer],
        "joint_accuracy": result["joint"] / total,
    }
    for index, accuracy_value in enumerate(metrics["per_answer_accuracy"]):
        metrics[f"answer_{index + 1}_accuracy"] = accuracy_value
    return metrics


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    device = resolve_device(args.device)
    config = RecordConfig(
        num_keys=args.num_keys,
        num_values=args.num_values,
        num_records=args.num_records,
        num_queries=args.num_queries,
    )
    model = RecordTransformer(
        num_keys=config.num_keys,
        num_values=config.num_values,
        num_queries=config.num_queries,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        use_mlp=args.mlp,
        head_dim_override=args.head_dim or None,
        tie_heads=args.tie_heads,
        max_length=args.max_length,
        position_mode=args.position_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    start_time = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        current_count = config.num_records
        if (
            args.curriculum_start_records > 0
            and args.curriculum_start_records < config.num_records
            and args.curriculum_steps > 0
        ):
            progress = min(step / args.curriculum_steps, 1.0)
            current_count = int(
                round(
                    args.curriculum_start_records
                    + (config.num_records - args.curriculum_start_records) * progress
                )
            )
        current_config = RecordConfig(
            num_keys=config.num_keys,
            num_values=config.num_values,
            num_records=current_count,
            num_queries=config.num_queries,
        )
        batch = sample_record_batch(
            current_config, args.batch_size, generator, device
        )
        logits, _ = model(batch.record_keys, batch.record_values, batch.queries)
        loss = torch.stack(
            [
                F.cross_entropy(
                    logits[:, index], batch.answers[:, index]
                )
                for index in range(config.num_queries)
            ]
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(
                model,
                config,
                args.eval_batch_size,
                generator,
                device,
            )
            record = {"step": step, "loss": float(loss), **metrics}
            history.append(record)
            print(json.dumps(record), flush=True)

    result = {
        **vars(args),
        "elapsed_seconds": round(time.time() - start_time, 3),
        "final_accuracy": history[-1]["joint_accuracy"],
        "history": history,
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else ""
        ),
    }
    print(json.dumps(result, indent=2))
    if args.result_json:
        path = Path(args.result_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.checkpoint:
        path = Path(args.checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "args": vars(args)}, path)


if __name__ == "__main__":
    main()
