"""Train the LIBERO profile directly from official demonstration HDF5 files."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .libero import LIBEROBatchProcessor, LIBEROPolicy
from .libero_config import LIBEROConfig


def _task_text(path: Path, attributes) -> str:
    raw = attributes.get("problem_info")
    if isinstance(raw, bytes):
        raw = raw.decode()
    if raw:
        try:
            instruction = json.loads(raw).get("language_instruction")
            if isinstance(instruction, list):
                instruction = instruction[0]
            if instruction:
                return str(instruction)
        except (json.JSONDecodeError, AttributeError):
            pass
    return path.stem.removesuffix("_demo").replace("_", " ")


class LIBEROHDF5Sampler:
    """Small random-window reader for the official `data/demo_*/` layout."""

    def __init__(self, root: Path, horizon: int, *, rotate_images_180: bool, seed: int):
        import h5py

        self.h5py = h5py
        self.horizon = horizon
        self.rotate_images_180 = rotate_images_180
        self.random = random.Random(seed)
        self.handles = {}
        paths = (
            [root]
            if root.is_file()
            else sorted({*root.rglob("*.hdf5"), *root.rglob("*.h5")})
        )
        self.entries = []
        for path in paths:
            with h5py.File(path, "r") as file:
                if "data" not in file:
                    continue
                task = _task_text(path, file["data"].attrs)
                for demo_name, demo in file["data"].items():
                    length = int(demo.attrs.get("num_samples", demo["actions"].shape[0]))
                    if length:
                        self.entries.append((path, demo_name, length, task))
        if not self.entries:
            raise ValueError(f"No LIBERO demonstrations found under {root}")

    def _file(self, path: Path):
        if path not in self.handles:
            self.handles[path] = self.h5py.File(path, "r")
        return self.handles[path]

    @staticmethod
    def _pad(values: np.ndarray, length: int) -> np.ndarray:
        if len(values) == length:
            return values
        return np.concatenate([values, np.repeat(values[-1:], length - len(values), axis=0)])

    def sample_batch(self, batch_size: int) -> dict[str, object]:
        states, actions, pads, agent_images, wrist_images, tasks = [], [], [], [], [], []
        for _ in range(batch_size):
            # ponytail: uniform episodes; length weighting can follow if sampling bias is measured.
            path, demo_name, length, task = self.random.choice(self.entries)
            start = self.random.randrange(length)
            stop = min(length, start + self.horizon)
            demo = self._file(path)[f"data/{demo_name}"]
            ee = np.asarray(demo["obs/ee_states"][start:stop], dtype=np.float32)
            gripper = np.asarray(demo["obs/gripper_states"][start:stop], dtype=np.float32)
            action = np.asarray(demo["actions"][start:stop], dtype=np.float32)
            if ee.shape[-1] != 6 or gripper.shape[-1] != 2 or action.shape[-1] != 7:
                raise ValueError(f"Unexpected LIBERO dimensions in {path}:{demo_name}")
            valid = stop - start
            states.append(torch.from_numpy(self._pad(np.concatenate([ee, gripper], -1), self.horizon)))
            actions.append(torch.from_numpy(self._pad(action, self.horizon)))
            pads.append(torch.arange(self.horizon) >= valid)

            image = np.asarray(demo["obs/agentview_rgb"][start])
            wrist = np.asarray(demo["obs/eye_in_hand_rgb"][start])
            if image.ndim != 3 or wrist.ndim != 3 or image.shape[-1] != 3 or wrist.shape[-1] != 3:
                raise ValueError(f"LIBERO RGB images must be HWC in {path}:{demo_name}")
            if self.rotate_images_180:
                image, wrist = image[::-1, ::-1].copy(), wrist[::-1, ::-1].copy()
            agent_images.append(torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255)
            wrist_images.append(torch.from_numpy(wrist.copy()).permute(2, 0, 1).float() / 255)
            tasks.append(task)
        return {
            "observation.state": torch.stack(states),
            "observation.state_is_pad": torch.stack(pads),
            "observation.images.image": torch.stack(agent_images),
            "observation.images.image2": torch.stack(wrist_images),
            "action": torch.stack(actions),
            "task": tasks,
        }

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()


def _to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="LIBERO HDF5 root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rotate-images-180",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Correct the official HDF5 camera convention",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("LIBERO training requires a CUDA GPU")

    torch.manual_seed(args.seed)
    config = LIBEROConfig()
    policy = (
        LIBEROPolicy.from_pretrained(args.checkpoint)
        if args.checkpoint
        else LIBEROPolicy(config)
    ).train().to("cuda")
    config = policy.config
    language_trainable = [
        name
        for name, parameter in policy.model.vlm.named_parameters()
        if parameter.requires_grad and ("text_model" in name or "lm_head" in name)
    ]
    if language_trainable:
        raise RuntimeError(f"Language parameters unexpectedly trainable: {language_trainable[:4]}")
    if not all(
        parameter.requires_grad
        for parameter in policy.model._vlm_model().vision_model.parameters()
    ):
        raise RuntimeError("The full LIBERO vision encoder is not trainable")

    if args.checkpoint and not args.stats:
        processor = LIBEROBatchProcessor.load(
            config, args.checkpoint / "processor", tokenizer=policy.model.tokenizer
        )
    else:
        stats = torch.load(args.stats, weights_only=True) if args.stats else None
        processor = LIBEROBatchProcessor(
            config, tokenizer=policy.model.tokenizer, stats=stats
        )
    vision = list(policy.model._vlm_model().vision_model.parameters())
    vision_ids = {id(parameter) for parameter in vision}
    other = [
        parameter
        for parameter in policy.get_optim_params()
        if id(parameter) not in vision_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": vision, "lr": config.vision_encoder_lr},
            {"params": other, "lr": config.optimizer_lr},
        ],
        betas=config.optimizer_betas,
        eps=config.optimizer_eps,
        weight_decay=config.optimizer_weight_decay,
    )
    sampler = LIBEROHDF5Sampler(
        args.data,
        config.ear_horizon,
        rotate_images_180=args.rotate_images_180,
        seed=args.seed,
    )
    device = torch.device("cuda")
    optimizer.zero_grad(set_to_none=True)
    try:
        for step in range(args.steps):
            raw = _to_device(sampler.sample_batch(args.batch_size), device)
            batch = processor(raw, training=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = policy(batch)
                (loss / args.gradient_accumulation).backward()
            if (step + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    policy.get_optim_params(), config.optimizer_grad_clip_norm
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 10 == 0:
                print({"step": step, **metrics})
        if args.steps % args.gradient_accumulation:
            torch.nn.utils.clip_grad_norm_(
                policy.get_optim_params(), config.optimizer_grad_clip_norm
            )
            optimizer.step()
    finally:
        sampler.close()
    policy.save_pretrained(args.output)
    processor.save(args.output / "processor")


if __name__ == "__main__":
    main()
