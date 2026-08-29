"""Print and export LIBERO suite success rates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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
    rows.sort(key=lambda row: row["suite"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['suite']:16} {row['success_rate']:7.1%}")
    print(f"{'average':16} {sum(row['success_rate'] for row in rows) / len(rows):7.1%}")
    print(args.output)


if __name__ == "__main__":
    main()
