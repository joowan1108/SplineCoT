"""Evaluate a trained checkpoint in the official LIBERO simulator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .libero import LIBEROBatchProcessor, LIBEROPolicy

SUITE_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def xyzw_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-8)
    if quaternion[3] < 0:
        quaternion = -quaternion
    vector, scalar = quaternion[:3], float(quaternion[3])
    length = float(np.linalg.norm(vector))
    if length < 1e-7:
        return 2 * vector
    return vector / length * (2 * np.arctan2(length, scalar))


def policy_input(observation: dict, instruction: str) -> dict[str, object]:
    state = np.concatenate(
        [
            observation["robot0_eef_pos"],
            xyzw_to_axis_angle(observation["robot0_eef_quat"]),
            observation["robot0_gripper_qpos"],
        ]
    ).astype(np.float32)

    def image(name: str) -> torch.Tensor:
        value = np.ascontiguousarray(observation[name][::-1, ::-1])
        return torch.from_numpy(value).permute(2, 0, 1).float().div(255).unsqueeze(0)

    return {
        "observation.state": torch.from_numpy(state).unsqueeze(0),
        "observation.images.image": image("agentview_image"),
        "observation.images.image2": image("robot0_eye_in_hand_image"),
        "task": [instruction],
    }


def to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def task_init_states(task, get_libero_path) -> np.ndarray:
    path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    return np.asarray(torch.load(path, map_location="cpu", weights_only=False))


def evaluate(args: argparse.Namespace) -> dict:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    device = torch.device(args.device)
    policy = LIBEROPolicy.from_pretrained(args.checkpoint).eval().to(device)
    processor = LIBEROBatchProcessor.load(
        policy.config, args.checkpoint / "processor", tokenizer=policy.model.tokenizer
    )
    suite = benchmark.get_benchmark_dict()[args.suite]()
    episodes = []
    try:
        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            initial_states = task_init_states(task, get_libero_path)
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl),
                camera_heights=256,
                camera_widths=256,
                control_freq=20,
            )
            try:
                for episode in range(args.episodes):
                    env.seed(args.seed + episode)
                    observation = env.reset()
                    observation = env.set_init_state(initial_states[episode % len(initial_states)])
                    for _ in range(args.settle_steps):
                        observation, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                    policy.reset()
                    success = False
                    steps = 0
                    episode_limit = args.max_steps or SUITE_MAX_STEPS[args.suite]
                    for steps in range(1, episode_limit + 1):
                        batch = processor(policy_input(observation, task.language), training=False)
                        action = policy.select_action(to_device(batch, device))[0].float().cpu().numpy()
                        observation, _, done, _ = env.step(np.clip(action, -1, 1))
                        success = bool(env.check_success())
                        if success or done:
                            break
                    record = {
                        "task_id": task_id,
                        "task": task.name,
                        "episode": episode,
                        "success": success,
                        "steps": steps,
                    }
                    episodes.append(record)
                    print(json.dumps(record))
            finally:
                env.close()
    finally:
        policy._planner.shutdown(wait=True, cancel_futures=True)

    tasks = []
    for task_id in range(suite.n_tasks):
        selected = [episode for episode in episodes if episode["task_id"] == task_id]
        tasks.append(
            {
                "task_id": task_id,
                "task": selected[0]["task"],
                "success_rate": sum(item["success"] for item in selected) / len(selected),
            }
        )
    return {
        "suite": args.suite,
        "checkpoint": str(args.checkpoint),
        "episodes_per_task": args.episodes,
        "success_rate": sum(item["success"] for item in episodes) / len(episodes),
        "tasks": tasks,
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", choices=tuple(SUITE_MAX_STEPS), required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("--episodes and --max-steps must be positive")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{args.suite}: {result['success_rate']:.1%} -> {args.output}")


if __name__ == "__main__":
    main()
