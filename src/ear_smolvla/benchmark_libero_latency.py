"""Measure planning and execution latency on one LIBERO task observation."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .eval_libero import policy_input, task_init_states, to_device
from .libero import LIBEROBatchProcessor, LIBEROPolicy
from .libero_config import LIBEROConfig


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
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Measure a freshly initialized current-config model instead of a checkpoint",
    )
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
    if args.random_init == (args.checkpoint is not None):
        parser.error("provide exactly one of --checkpoint or --random-init")
    if args.warmup < 0 or args.repeats < 1 or args.settle_steps < 0:
        parser.error("--warmup/--settle-steps must be nonnegative and --repeats must be positive")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    device = torch.device(args.device)
    if args.random_init:
        policy = LIBEROPolicy(LIBEROConfig(device=str(device))).eval().to(device)
        processor = LIBEROBatchProcessor(policy.config, tokenizer=policy.model.tokenizer)
        checkpoint_name = "random-init"
    else:
        policy = LIBEROPolicy.from_pretrained(args.checkpoint).eval().to(device)
        processor = LIBEROBatchProcessor.load(
            policy.config, args.checkpoint / "processor", tokenizer=policy.model.tokenizer
        )
        checkpoint_name = str(args.checkpoint)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    if not 0 <= args.task_id < suite.n_tasks:
        parser.error(f"--task-id must be between 0 and {suite.n_tasks - 1}")
    execution_steps = policy.config.action_horizon if args.execution_steps is None else args.execution_steps
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
        params = policy.model._denormalize_spline(params)
        return policy.model._unnormalize_native_actions(params)

    def execute(params: torch.Tensor, current_observation: dict) -> int:
        completed = 0
        actions = policy.model.action_spline.decode(params)[:, :execution_steps]
        for action in actions[0]:
            current_observation, _, done, _ = env.step(
                np.clip(action.float().cpu().numpy(), -1, 1)
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

    nominal_execution_ms = execution_steps / policy.config.dataset_fps * 1_000
    result = {
        "suite": args.suite,
        "task_id": args.task_id,
        "task": task.name,
        "checkpoint": checkpoint_name,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "vlm_calls_per_plan": 1,
        "ear_layers": policy.config.spline_reasoner_layers,
        "action_expert_layers": policy.config.action_expert_layers,
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
