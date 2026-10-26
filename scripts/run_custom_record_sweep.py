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
    parser.add_argument("--num-queries", type=int, required=True)
    parser.add_argument("--num-records", type=int, required=True)
    parser.add_argument("--num-keys", type=int, default=128)
    parser.add_argument("--num-values", type=int, default=128)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--position-mode", choices=("learned", "none"), default="learned")
    parser.add_argument("--curriculum-start-records", type=int, default=0)
    parser.add_argument("--curriculum-steps", type=int, default=0)
    parser.add_argument("--tie-heads", action="store_true")
    return parser.parse_args()


def run_one(
    seed: int,
    gpu_uuid: str,
    args: argparse.Namespace,
) -> dict:
    script = Path(__file__).resolve().parents[1] / "src" / "train_record.py"
    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"q{args.num_queries}_r{args.num_records}_"
        f"l{args.layers}h{args.heads}_hd{args.head_dim}_seed{seed}"
    )
    result_path = output_dir / f"{stem}.json"
    command = [
        sys.executable,
        str(script),
        "--device",
        "cuda",
        "--num-queries",
        str(args.num_queries),
        "--num-records",
        str(args.num_records),
        "--num-keys",
        str(args.num_keys),
        "--num-values",
        str(args.num_values),
        "--d-model",
        str(args.d_model),
        "--layers",
        str(args.layers),
        "--heads",
        str(args.heads),
        "--head-dim",
        str(args.head_dim),
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
    if args.tie_heads:
        command.append("--tie-heads")
    environment = dict(os.environ)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    print(json.dumps({"running": stem, "gpu": gpu_uuid}), flush=True)
    subprocess.run(command, check=True, env=environment)
    return {"seed": seed, "result": str(result_path)}


def main() -> None:
    args = parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                run_one,
                seed,
                args.gpu_uuids[index % len(args.gpu_uuids)],
                args,
            )
            for index, seed in enumerate(args.seeds)
        ]
        for future in concurrent.futures.as_completed(futures):
            print(json.dumps(future.result()), flush=True)


if __name__ == "__main__":
    main()
