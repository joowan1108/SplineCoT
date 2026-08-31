"""Measure planning and execution latency on one LIBERO task observation."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .eval_libero import policy_input, task_init_states, to_device, xyzw_to_axis_angle
from .libero import LIBEROBatchProcessor, LIBEROPolicy, libero_pose_from_state


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device: torch.device, function):
    synchronize(device)
    start = time.perf_counter()
    result = function()
    synchronize(device)
    return result, (time.perf_counter() - start) * 1_000


def latency_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--suite",
        choices=("libero_spatial", "libero_object", "libero_goal", "libero_10"),
        required=True,
    )
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--execution-steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.settle_steps < 0:
        parser.error("--warmup/--settle-steps must be nonnegative and --repeats must be positive")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    device = torch.device(args.device)
    policy = LIBEROPolicy.from_pretrained(args.checkpoint).eval().to(device)
    processor = LIBEROBatchProcessor.load(
        policy.config, args.checkpoint / "processor", tokenizer=policy.model.tokenizer
    )
    suite = benchmark.get_benchmark_dict()[args.suite]()
    if not 0 <= args.task_id < suite.n_tasks:
        parser.error(f"--task-id must be between 0 and {suite.n_tasks - 1}")
    execution_steps = (
        policy.config.n_action_steps if args.execution_steps is None else args.execution_steps
    )
    if not 1 <= execution_steps <= policy.config.action_horizon:
        parser.error(f"--execution-steps must be between 1 and {policy.config.action_horizon}")

    task = suite.get_task(args.task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    initial_states = task_init_states(task, get_libero_path)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        control_freq=int(policy.config.dataset_fps),
    )

    def reset_observation():
        env.seed(args.seed)
        observation = env.reset()
        observation = env.set_init_state(initial_states[args.seed % len(initial_states)])
        for _ in range(args.settle_steps):
            observation, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        return observation

    observation = reset_observation()
    batch = to_device(
        processor(policy_input(observation, task.language), training=False),
        device,
    )
    images, image_masks, task_tokens, task_mask, state, current_pose = policy._planning_inputs(batch)

    def vlm_plus_ear():
        context, _ = policy.model.make_prefix(images, image_masks, task_tokens, task_mask, state)
        context = context.detached()
        mean, ear = policy.model.sample_ear_distribution(context, current_pose)
        phase = torch.zeros(state.shape[0], device=device)
        guidance = policy.model.ear_spline.select_parameter_guidance(
            mean,
            phase,
            policy.config.action_phase_span,
            ear.segment_confidence,
        ).detached()
        return context, guidance, phase

    def action_head(context, guidance, phase):
        params = policy.model.sample_action(context, guidance, phase, current_pose)
        return policy.model._denormalize_spline(params)

    def execute(params: torch.Tensor, current_observation: dict) -> int:
        completed = 0
        for _ in range(execution_steps):
            state_array = np.concatenate(
                [
                    current_observation["robot0_eef_pos"],
                    xyzw_to_axis_angle(current_observation["robot0_eef_quat"]),
                    current_observation["robot0_gripper_qpos"],
                ]
            ).astype(np.float32)
            raw_state = torch.from_numpy(state_array).unsqueeze(0).to(device)
            pose = libero_pose_from_state(raw_state)
            field, phase, _ = policy.model.action_spline.closest_point_field(
                params[..., : policy.model.pose_dim],
                pose,
                policy.config.field_attraction,
                policy.config.field_progression,
            )
            gripper = policy.model.action_spline.evaluate(params, phase)[
                ..., policy.model.pose_dim : policy.model.pose_dim + 1
            ]
            action = policy.model._field_to_action(
                field,
                pose,
                gripper,
                torch.zeros(pose.shape[0], 1, device=device),
            )
            current_observation, _, done, _ = env.step(
                np.clip(action[0].float().cpu().numpy(), -1, 1)
            )
            completed += 1
            if done:
                break
        return completed

    vlm_ear_times: list[float] = []
    action_times: list[float] = []
    execution_times: list[float] = []
    combined_compute_times: list[float] = []
    completed_steps: list[int] = []
    torch.manual_seed(args.seed)
    try:
        with torch.inference_mode():
            for _ in range(args.warmup):
                context, guidance, phase = vlm_plus_ear()
                params = action_head(context, guidance, phase)
            if args.warmup:
                execute(params, reset_observation())

            for _ in range(args.repeats):
                (context, guidance, phase), vlm_ear_ms = timed(device, vlm_plus_ear)
                params, action_ms = timed(
                    device, lambda: action_head(context, guidance, phase)
                )
                execution_observation = reset_observation()
                completed, execution_ms = timed(
                    device, lambda: execute(params, execution_observation)
                )
                vlm_ear_times.append(vlm_ear_ms)
                action_times.append(action_ms)
                execution_times.append(execution_ms)
                combined_compute_times.append(action_ms + execution_ms)
                completed_steps.append(completed)
    finally:
        env.close()
        policy._planner.shutdown(wait=True, cancel_futures=True)

    nominal_execution_ms = execution_steps / policy.config.dataset_fps * 1_000
    result = {
        "suite": args.suite,
        "task_id": args.task_id,
        "task": task.name,
        "checkpoint": str(args.checkpoint),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "vlm_calls_per_plan": 1,
        "ear_mc_samples": policy.config.mc_samples,
        "execution_steps": execution_steps,
        "control_hz": policy.config.dataset_fps,
        "nominal_execution_ms": nominal_execution_ms,
        "completed_execution_steps": completed_steps,
        "timings_ms": {
            "vlm_plus_ear": latency_stats(vlm_ear_times),
            "action_head": latency_stats(action_times),
            "simulator_execution_compute": latency_stats(execution_times),
            "action_head_plus_simulator_execution_compute": latency_stats(combined_compute_times),
            "action_head_plus_nominal_execution": {
                key: value + nominal_execution_ms
                for key, value in latency_stats(action_times).items()
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
