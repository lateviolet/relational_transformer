from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+")
    parser.add_argument("--output-csv", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for directory in args.result_dirs:
        for path in sorted(Path(directory).glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if "final_accuracy" not in data:
                continue
            key = (
                data["task"],
                data.get("hops", 2),
                data["layers"],
                data["heads"],
                data["mlp"],
                data.get("position_mode", "learned"),
            )
            grouped[key].append(float(data["final_accuracy"]))

    rows = []
    for (task, hops, layers, heads, mlp, position_mode), values in sorted(
        grouped.items()
    ):
        rows.append(
            {
                "task": task,
                "hops": hops,
                "layers": layers,
                "heads": heads,
                "mlp": mlp,
                "position_mode": position_mode,
                "runs": len(values),
                "mean_accuracy": statistics.mean(values),
                "std_accuracy": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "min_accuracy": min(values),
                "max_accuracy": max(values),
            }
        )

    for row in rows:
        print(
            f"{row['task']:<14} hops={row['hops']} "
            f"L={row['layers']} H={row['heads']} "
            f"MLP={int(row['mlp'])} n={row['runs']} "
            f"pos={row['position_mode']} "
            f"mean={row['mean_accuracy']:.3f} "
            f"std={row['std_accuracy']:.3f} "
            f"range=[{row['min_accuracy']:.3f}, {row['max_accuracy']:.3f}]"
        )

    if args.output_csv:
        path = Path(args.output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
