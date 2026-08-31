"""Plot ground-truth, EAR, and action splines in the EEF XY plane."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch

from .eval_libero import to_device
from .libero import EAR_SPLINE_TARGET, LIBEROBatchProcessor, LIBEROPolicy
from .train_libero import LIBEROHDF5Sampler

TOP_VIEW_STYLES = {
    "Ground-truth spline": {"color": "#000000", "linestyle": "-", "marker": "o"},
    "EAR spline": {"color": "#0072B2", "linestyle": "--", "marker": "s"},
    "Action-head spline": {"color": "#D55E00", "linestyle": "-.", "marker": "^"},
}


def save_top_view(
    path: Path,
    task: str,
    ground_truth: np.ndarray,
    ear: np.ndarray,
    action: np.ndarray,
    fps: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 7))
    series = (
        ("Ground-truth spline", ground_truth),
        ("EAR spline", ear),
        ("Action-head spline", action),
    )
    for label, trajectory in series:
        style = TOP_VIEW_STYLES[label]
        horizon = (len(trajectory) - 1) / fps
        axis.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            label=f"{label} ({horizon:.2f}s)",
            linewidth=2.5,
            markersize=5,
            markevery=max(1, len(trajectory) // 8),
            **style,
        )
    axis.scatter(
        ground_truth[0, 0],
        ground_truth[0, 1],
        color="#009E73",
        marker="*",
        s=180,
        label="Current EEF",
        zorder=5,
    )
    axis.set(title=task, xlabel="EEF x (m)", ylabel="EEF y (m)")
    axis.axis("equal")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--task-contains")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--rotate-images-180", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    device = torch.device(args.device)
    policy = LIBEROPolicy.from_pretrained(args.checkpoint).eval().to(device)
    processor = LIBEROBatchProcessor.load(
        policy.config, args.checkpoint / "processor", tokenizer=policy.model.tokenizer
    )
    sampler = LIBEROHDF5Sampler(
        args.data,
        policy.config.ear_horizon,
        rotate_images_180=args.rotate_images_180,
        seed=args.seed,
    )
    if args.task_contains:
        needle = args.task_contains.lower()
        sampler.entries = [entry for entry in sampler.entries if needle in entry[3].lower()]
        if not sampler.entries:
            raise ValueError(f"No demonstration task contains {args.task_contains!r}")

    torch.manual_seed(args.seed)
    try:
        for index in range(args.samples):
            for _ in range(100):
                raw = sampler.sample_batch(1)
                if int((~raw["observation.state_is_pad"][0]).sum()) == policy.config.ear_horizon:
                    break
            else:
                raise RuntimeError("Could not sample a complete EAR horizon")

            task = str(raw["task"][0])
            batch = to_device(processor(raw, training=True), device)
            with torch.no_grad():
                plan = policy.model.plan(*policy._planning_inputs(batch))
                plan = policy._anchor(plan, batch)
                ground_truth_params = policy.model.ear_spline.fit(batch[EAR_SPLINE_TARGET])
                ground_truth = policy.model.ear_spline.decode(ground_truth_params)
                ear = policy.model.ear_spline.decode(plan.ear.mean)
                action = policy.model.action_spline.decode(plan.params)

            name = re.sub(r"[^a-zA-Z0-9_-]+", "_", task).strip("_")[:80]
            path = args.output / f"sample-{index:02d}-{name}.png"
            save_top_view(
                path,
                task,
                ground_truth[0, :, :3].float().cpu().numpy(),
                ear[0, :, :3].float().cpu().numpy(),
                action[0, :, :3].float().cpu().numpy(),
                policy.config.dataset_fps,
            )
            print(path)
    finally:
        sampler.close()
        policy._planner.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
