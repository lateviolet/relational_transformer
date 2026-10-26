from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shards = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in args.inputs
    ]
    heads = [head for shard in shards for head in shard["heads"]]
    heads.sort(key=lambda row: row["delta_loss"], reverse=True)
    result = {
        "adapter": shards[0]["adapter"],
        "data": shards[0]["data"],
        "examples": shards[0]["examples"],
        "baseline_loss": shards[0]["baseline_loss"],
        "baseline_answer_token_accuracy": shards[0][
            "baseline_answer_token_accuracy"
        ],
        "heads": heads,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["heads"][:16], indent=2))


if __name__ == "__main__":
    main()
