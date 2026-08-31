"""Download the four suites used by the standard VLA LIBERO benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/libero/vla40"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="yifengzhu-hf/LIBERO-datasets",
        repo_type="dataset",
        local_dir=args.output,
        allow_patterns=[f"{suite}/**" for suite in SUITES],
    )
    counts = {suite: len(list((args.output / suite).glob("*.hdf5"))) for suite in SUITES}
    if any(count != 10 for count in counts.values()):
        raise RuntimeError(f"Incomplete LIBERO VLA-40 download: {counts}")
    print({"output": str(args.output.resolve()), "suites": counts, "total_tasks": sum(counts.values())})


if __name__ == "__main__":
    main()
