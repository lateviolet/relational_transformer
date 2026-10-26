from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nlp_data import write_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="data/nlp_data",
    )
    args = parser.parse_args()
    write_splits(Path(args.output_dir))


if __name__ == "__main__":
    main()
