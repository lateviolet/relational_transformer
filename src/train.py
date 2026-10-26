from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data import TaskConfig, Vocabulary, batch_to, sample_episodes, validate_episode
from model import TinyDecoderTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=[
            "two_hop",
            "cued_two_hop",
            "cued_nhop",
            "dual_retrieval",
            "parallel_copy",
            "parallel_intersection",
        ],
        default="two_hop",
    )
    parser.add_argument("--num-nodes", type=int, default=32)
    parser.add_argument("--edge-count", type=int, default=8)
    parser.add_argument("--num-relations", type=int, default=2)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument(
        "--eval-edge-counts",
        nargs="*",
        type=int,
        default=[],
        help="Optional out-of-distribution distractor counts evaluated after training.",
    )
    parser.add_argument(
        "--train-edge-counts",
        nargs="*",
        type=int,
        default=[],
        help="Sample one of these distractor counts independently at each training step.",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument(
        "--max-length",
        type=int,
        default=0,
        help="Optional fixed positional capacity across compared runs.",
    )
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--mlp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--position-mode", choices=["learned", "rope"], default="learned"
    )
    parser.add_argument("--tie-heads", action="store_true")
    parser.add_argument("--attention-out-scale", type=float, default=1.0)
    parser.add_argument(
        "--head-dim",
        type=int,
        default=0,
        help="Optional attention head dimension; 0 uses d_model / heads.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--result-json", default="")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> dict:
    description = {"device": str(device)}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu_uuid = getattr(properties, "uuid", None)
        if not gpu_uuid:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            gpu_uuid = visible.split(",")[device.index or 0]
        description.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory": properties.total_memory,
                "gpu_uuid": str(gpu_uuid),
            }
        )
    return description


def make_model(
    args: argparse.Namespace,
    vocab: Vocabulary,
    max_length: int,
    device: torch.device,
) -> TinyDecoderTransformer:
    model = TinyDecoderTransformer(
        vocab_size=vocab.size,
        num_nodes=vocab.num_nodes,
        max_length=max_length,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        use_mlp=args.mlp,
        position_mode=args.position_mode,
        num_answer_heads=2
        if args.task in {"dual_retrieval", "parallel_copy"}
        else 1,
        tie_heads=args.tie_heads,
        attention_out_scale=args.attention_out_scale,
        head_dim_override=args.head_dim or None,
    )
    return model.to(device)


@torch.no_grad()
def evaluate(
    model: TinyDecoderTransformer,
    config: TaskConfig,
    vocab: Vocabulary,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, float]:
    model.eval()
    correct_primary = 0
    correct_secondary = 0
    correct_joint = 0
    total = 0
    for _ in range(8):
        batch = sample_episodes(config, batch_size, generator)
        tokens, answers, secondary = batch_to(
            batch.tokens, batch.answers, batch.secondary_answers, device
        )
        outputs = model(tokens)[:, -1]
        if config.task in {"dual_retrieval", "parallel_copy"}:
            predictions = outputs.argmax(dim=-1)
            primary = predictions[:, 0] == answers
            secondary_correct = predictions[:, 1] == secondary
            correct_primary += int(primary.sum())
            correct_secondary += int(secondary_correct.sum())
            correct_joint += int((primary & secondary_correct).sum())
        else:
            primary = outputs.argmax(dim=-1) == answers
            correct_primary += int(primary.sum())
            correct_secondary += int(primary.sum())
            correct_joint += int(primary.sum())
        total += batch_size
    model.train()
    return {
        "accuracy": correct_primary / total,
        "secondary_accuracy": correct_secondary / total,
        "joint_accuracy": correct_joint / total,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    device = resolve_device(args.device)

    config = TaskConfig(
        task=args.task,
        num_nodes=args.num_nodes,
        edge_count=args.edge_count,
        num_relations=args.num_relations,
        hops=args.hops,
    )
    vocab = Vocabulary(config.num_nodes, config.num_relations)
    relation_prefix = (
        2
        if config.task == "dual_retrieval"
        else 1
        if config.task in {"cued_two_hop", "cued_nhop"}
        else 0
    )
    train_edge_counts = sorted(
        set([config.edge_count, *args.train_edge_counts])
    )
    eval_edge_counts = sorted(
        set([*train_edge_counts, *args.eval_edge_counts])
    )
    if config.task == "parallel_intersection":
        max_length = max(11 + 3 * edge_count for edge_count in eval_edge_counts)
    else:
        max_length = max(
            2 + relation_prefix + 3 * edge_count + 3
            for edge_count in eval_edge_counts
        )
    if args.max_length:
        max_length = max(max_length, args.max_length)
    model = make_model(args, vocab, max_length, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    sample = sample_episodes(config, 4, generator)
    for index in range(4):
        validate_episode(config, sample, index)

    history: list[dict] = []
    start_time = time.time()
    for step in range(1, args.steps + 1):
        if len(train_edge_counts) > 1:
            selected_index = int(
                torch.randint(
                    len(train_edge_counts), (1,), generator=generator
                )
            )
            train_config = TaskConfig(
                task=args.task,
                num_nodes=args.num_nodes,
                edge_count=train_edge_counts[selected_index],
                num_relations=args.num_relations,
                hops=args.hops,
            )
        else:
            train_config = config
        batch = sample_episodes(train_config, args.batch_size, generator)
        tokens, answers, secondary = batch_to(
            batch.tokens, batch.answers, batch.secondary_answers, device
        )
        outputs = model(tokens)[:, -1]
        if args.task in {"dual_retrieval", "parallel_copy"}:
            loss = 0.5 * (
                F.cross_entropy(outputs[:, 0], answers)
                + F.cross_entropy(outputs[:, 1], secondary)
            )
        else:
            loss = F.cross_entropy(outputs, answers)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        should_evaluate = step == 1 or step % args.eval_every == 0 or step == args.steps
        if should_evaluate:
            metrics = evaluate(
                model,
                config,
                vocab,
                args.eval_batch_size,
                device,
                generator,
            )
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                **metrics,
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    eval_by_edge_count = {}
    for edge_count in eval_edge_counts:
        eval_config = TaskConfig(
            task=args.task,
            num_nodes=args.num_nodes,
            edge_count=edge_count,
            num_relations=args.num_relations,
            hops=args.hops,
        )
        eval_by_edge_count[str(edge_count)] = evaluate(
            model,
            eval_config,
            vocab,
            args.eval_batch_size,
            device,
            generator,
        )

    result = {
        "task": args.task,
        "num_nodes": args.num_nodes,
        "edge_count": args.edge_count,
        "num_relations": args.num_relations,
        "hops": args.hops,
        "d_model": args.d_model,
        "max_length": max_length,
        "layers": args.layers,
        "heads": args.heads,
        "mlp": args.mlp,
        "position_mode": args.position_mode,
        "tie_heads": args.tie_heads,
        "attention_out_scale": args.attention_out_scale,
        "head_dim": args.head_dim,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "train_edge_counts": train_edge_counts,
        **describe_device(device),
        "elapsed_seconds": round(time.time() - start_time, 3),
        "final_accuracy": history[-1]["accuracy"],
        "eval_by_edge_count": eval_by_edge_count,
        "history": history,
    }

    if args.result_json:
        path = Path(args.result_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if args.checkpoint:
        path = Path(args.checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "args": vars(args),
                "vocabulary": vars(vocab),
            },
            path,
        )

    print(json.dumps({"summary": result["final_accuracy"], **result}, indent=2))


if __name__ == "__main__":
    main()
