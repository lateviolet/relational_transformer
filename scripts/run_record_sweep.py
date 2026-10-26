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
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--gpu-uuids", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--configs", nargs="+", default=[])
    parser.add_argument("--num-records", type=int, default=8)
    parser.add_argument("--num-keys", type=int, default=32)
    parser.add_argument("--num-values", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--position-mode",
        choices=("learned", "none"),
        default="learned",
    )
    parser.add_argument("--curriculum-start-records", type=int, default=0)
    parser.add_argument("--curriculum-steps", type=int, default=0)
    return parser.parse_args()


def run_one(
    name: str,
    layers: int,
    heads: int,
    head_dim: int,
    tie_heads: bool,
    seed: int,
    gpu_uuid: str,
    args: argparse.Namespace,
) -> dict:
    script = Path(__file__).resolve().parents[1] / "src" / "train_record.py"
    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"record_{name}_seed{seed}.json"
    command = [
        sys.executable,
        str(script),
        "--device",
        "cuda",
        "--num-records",
        str(args.num_records),
        "--num-keys",
        str(args.num_keys),
        "--num-values",
        str(args.num_values),
        "--d-model",
        str(args.d_model),
        "--layers",
        str(layers),
        "--heads",
        str(heads),
        "--steps",
        str(args.steps),
        "--max-length",
        str(args.max_length),
        "--position-mode",
        args.position_mode,
        "--curriculum-start-records",
        str(args.curriculum_start_records),
        "--curriculum-steps",
        str(args.curriculum_steps),
        "--lr",
        str(args.lr),
        "--seed",
        str(seed),
        "--result-json",
        str(result_path),
    ]
    if head_dim:
        command.extend(["--head-dim", str(head_dim)])
    if tie_heads:
        command.append("--tie-heads")
    environment = dict(os.environ)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    print(json.dumps({"running": name, "seed": seed, "gpu": gpu_uuid}), flush=True)
    subprocess.run(command, check=True, env=environment)
    return {"name": name, "seed": seed, "result": str(result_path)}


def main() -> None:
    args = parse_args()
    configurations = [
        ("1l1h_full", 1, 1, 0, False),
        ("1l1h_half", 1, 1, 48, False),
        ("1l2h", 1, 2, 0, False),
        ("1l2h_tied", 1, 2, 0, True),
        ("2l1h_full", 2, 1, 0, False),
        ("2l1h_half", 2, 1, 48, False),
        ("2l2h", 2, 2, 0, False),
        ("2l2h_tied", 2, 2, 0, True),
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
    tasks = [
        (name, layers, heads, head_dim, tie, seed)
        for name, layers, heads, head_dim, tie in configurations
        for seed in args.seeds
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for index, task in enumerate(tasks):
            gpu_uuid = args.gpu_uuids[index % len(args.gpu_uuids)]
            futures.append(pool.submit(run_one, *task, gpu_uuid, args))
        for future in concurrent.futures.as_completed(futures):
            print(json.dumps(future.result()), flush=True)


if __name__ == "__main__":
    main()
