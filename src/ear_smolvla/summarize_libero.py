"""Print and export LIBERO suite success rates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def checkpoint_order(value: str) -> tuple[int, int | str]:
    suffix = Path(value).name.rsplit("-", 1)[-1]
    return (0, int(suffix)) if suffix.isdigit() else (1, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/summary.csv"))
    args = parser.parse_args()
    rows = []
    for path in args.results:
        result = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "suite": result["suite"],
                "success_rate": result["success_rate"],
                "episodes_per_task": result["episodes_per_task"],
                "checkpoint": result["checkpoint"],
            }
        )
    rows.sort(key=lambda row: (checkpoint_order(row["checkpoint"]), row["suite"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checkpoints = sorted({row["checkpoint"] for row in rows}, key=checkpoint_order)
    for checkpoint in checkpoints:
        selected = [row for row in rows if row["checkpoint"] == checkpoint]
        print(Path(checkpoint).name)
        for row in selected:
            print(f"  {row['suite']:16} {row['success_rate']:7.1%}")
        print(f"  {'average':16} {sum(row['success_rate'] for row in selected) / len(selected):7.1%}")
    print(args.output)


if __name__ == "__main__":
    main()
