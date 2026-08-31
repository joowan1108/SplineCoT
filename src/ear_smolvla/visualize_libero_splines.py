"""Plot ground-truth, EAR, and action splines in the EEF XY plane."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch

from .eval_libero import to_device
from .libero import (
    ACTION_SPLINE_TARGET,
    EAR_SPLINE_TARGET,
    LIBEROBatchProcessor,
    LIBEROPolicy,
)
from .train_libero import LIBEROHDF5Sampler

TOP_VIEW_STYLES = {
    "EAR ground truth": {"color": "#000000", "linestyle": "-", "marker": "o"},
    "EAR prediction": {"color": "#0072B2", "linestyle": "--", "marker": "s"},
    "Action ground truth": {"color": "#009E73", "linestyle": ":", "marker": "D"},
    "Action prediction": {"color": "#D55E00", "linestyle": "-.", "marker": "^"},
}


def save_top_view(
    path: Path,
    task: str,
    ear_ground_truth: np.ndarray,
    action_ground_truth: np.ndarray,
    ear: np.ndarray,
    action: np.ndarray,
    ear_samples: list[np.ndarray],
    confidence: np.ndarray,
    available: np.ndarray,
    fps: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 7))
    for index, trajectory in enumerate(ear_samples):
        axis.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color="#56B4E9",
            alpha=0.25,
            linewidth=1,
            label="Individual MC EAR samples" if index == 0 else None,
        )
    series = (
        ("EAR ground truth", ear_ground_truth),
        ("EAR prediction", ear),
        ("Action ground truth", action_ground_truth),
        ("Action prediction", action),
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
        ear_ground_truth[0, 0],
        ear_ground_truth[0, 1],
        color="#CC79A7",
        marker="*",
        s=180,
        label="Current EEF",
        zorder=5,
    )
    axis.set(title=task, xlabel="EEF x (m)", ylabel="EEF y (m)")
    axis.text(
        0.99,
        0.01,
        f"EAR confidence mean={confidence.mean():.3f}\navailable={available.sum()}/{len(available)}",
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
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
    parser.add_argument("--rotate-images-180", action=argparse.BooleanOptionalAction, default=True)
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
                ear_ground_truth = policy.model.ear_spline.decode(
                    policy.model.ear_spline.fit(batch[EAR_SPLINE_TARGET], constrain_start=True)
                )
                action_ground_truth = policy.model.action_spline.decode(
                    policy.model.action_spline.fit(batch[ACTION_SPLINE_TARGET], constrain_start=True)
                )
                ear = policy.model.ear_spline.decode(plan.ear.mean)
                action = policy.model.action_spline.decode(plan.params)
                ear_samples = [
                    policy.model.ear_spline.decode(sample)[0, :, :3].float().cpu().numpy()
                    for sample in plan.ear.samples
                ]

            name = re.sub(r"[^a-zA-Z0-9_-]+", "_", task).strip("_")[:80]
            path = args.output / f"sample-{index:02d}-{name}.png"
            save_top_view(
                path,
                task,
                ear_ground_truth[0, :, :3].float().cpu().numpy(),
                action_ground_truth[0, :, :3].float().cpu().numpy(),
                ear[0, :, :3].float().cpu().numpy(),
                action[0, :, :3].float().cpu().numpy(),
                ear_samples,
                plan.ear.segment_confidence[0].float().cpu().numpy(),
                plan.ear.segment_available[0].cpu().numpy(),
                policy.config.dataset_fps,
            )
            print(path)
    finally:
        sampler.close()


if __name__ == "__main__":
    main()
