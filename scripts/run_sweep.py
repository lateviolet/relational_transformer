from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path


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
    parser.add_argument("--num-nodes", type=int, default=12)
    parser.add_argument("--edge-count", type=int, default=3)
    parser.add_argument("--num-relations", type=int, default=2)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--eval-edge-counts", nargs="*", type=int, default=[])
    parser.add_argument("--train-edge-counts", nargs="*", type=int, default=[])
    parser.add_argument(
        "--position-mode", choices=["learned", "rope"], default="learned"
    )
    parser.add_argument("--tie-heads", action="store_true")
    parser.add_argument("--attention-out-scale", type=float, default=1.0)
    parser.add_argument("--head-dim", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--results-dir", default="results/sweep")
    parser.add_argument(
        "--gpu-uuids",
        nargs="+",
        default=["0"],
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[],
        help="Optional architecture names; defaults to every architecture.",
    )
    return parser.parse_args()


def run_one(
    name: str,
    layers: int,
    heads: int,
    use_mlp: bool,
    args: argparse.Namespace,
    script: Path,
    output_dir: Path,
    gpu_uuid: str,
    seed: int,
) -> dict:
    result_path = output_dir / f"{args.task}_{name}_seed{seed}.json"
    command = [
        sys.executable,
        str(script),
        "--task",
        args.task,
        "--num-nodes",
        str(args.num_nodes),
        "--edge-count",
        str(args.edge_count),
        "--num-relations",
        str(args.num_relations),
        "--hops",
        str(args.hops),
        "--max-length",
        str(args.max_length),
        "--position-mode",
        args.position_mode,
        "--lr",
        str(args.lr),
        "--layers",
        str(layers),
        "--heads",
        str(heads),
        "--steps",
        str(args.steps),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--result-json",
        str(result_path),
    ]
    command.append("--mlp" if use_mlp else "--no-mlp")
    if args.tie_heads:
        command.append("--tie-heads")
    if args.attention_out_scale != 1.0:
        command.extend(["--attention-out-scale", str(args.attention_out_scale)])
    if args.head_dim:
        command.extend(["--head-dim", str(args.head_dim)])
    if args.eval_edge_counts:
        command.extend(["--eval-edge-counts", *map(str, args.eval_edge_counts)])
    if args.train_edge_counts:
        command.extend(["--train-edge-counts", *map(str, args.train_edge_counts)])
    environment = dict(os.environ)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    print(json.dumps({"running": name, "gpu_uuid": gpu_uuid}), flush=True)
    subprocess.run(command, check=True, env=environment)
    return {
        "name": name,
        "seed": seed,
        "gpu_uuid": gpu_uuid,
        "result": str(result_path),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parents[1] / "src" / "train.py"
    configurations = [
        ("1l1h_nomlp", 1, 1, False),
        ("1l2h_nomlp", 1, 2, False),
        ("2l1h_nomlp", 2, 1, False),
        ("2l2h_nomlp", 2, 2, False),
        ("1l1h_mlp", 1, 1, True),
        ("1l2h_mlp", 1, 2, True),
        ("2l1h_mlp", 2, 1, True),
        ("2l2h_mlp", 2, 2, True),
        ("3l2h_mlp", 3, 2, True),
        ("4l2h_mlp", 4, 2, True),
        ("3l1h_mlp", 3, 1, True),
        ("4l1h_mlp", 4, 1, True),
    ]
    if args.configs:
        requested = set(args.configs)
        configurations = [
            configuration
            for configuration in configurations
            if configuration[0] in requested
        ]
        missing = requested - {configuration[0] for configuration in configurations}
        if missing:
            raise ValueError(f"unknown configurations: {sorted(missing)}")

    if args.workers < 1:
        raise ValueError("workers must be positive")
    tasks = [
        (name, layers, heads, use_mlp, seed)
        for name, layers, heads, use_mlp in configurations
        for seed in args.seeds
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for index, (name, layers, heads, use_mlp, seed) in enumerate(tasks):
            gpu_uuid = args.gpu_uuids[index % len(args.gpu_uuids)]
            futures.append(
                pool.submit(
                    run_one,
                    name,
                    layers,
                    heads,
                    use_mlp,
                    args,
                    script,
                    output_dir,
                    gpu_uuid,
                    seed,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            print(json.dumps(future.result()), flush=True)


if __name__ == "__main__":
    main()
