"""Single-GPU trainer over prebatched RoboCasa `.pt` files."""

from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path

import torch

from .config import EARSmolVLAConfig
from .model import EARSmolVLAPolicy
from .processor import BatchProcessor


def _to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def stage_config(stage: str, config: EARSmolVLAConfig | None = None) -> EARSmolVLAConfig:
    config = config or EARSmolVLAConfig(curriculum_stage="joint_teacher_forced")
    if stage == "vlm_warmup":
        return replace(
            config,
            curriculum_stage=stage,
            train_vlm_objective=True,
            train_spline_reasoner=False,
            train_action_expert=False,
        )
    if stage == "joint_teacher_forced":
        return replace(
            config,
            curriculum_stage=stage,
            train_vlm_objective=True,
            train_spline_reasoner=True,
            train_action_expert=True,
        )
    else:
        raise ValueError(f"Unknown curriculum stage {stage}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=Path, required=True, help="Directory of raw batch-*.pt files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("vlm_warmup", "joint_teacher_forced"),
        default="joint_teacher_forced",
    )
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint from the preceding curriculum stage")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU")
    files = sorted(args.batches.glob("*.pt"))
    if not files:
        raise ValueError(f"No .pt batches found in {args.batches}")
    if not args.checkpoint and not args.stats:
        raise ValueError("The first curriculum stage requires --stats for state/action normalization")

    device = torch.device("cuda")
    policy = (
        EARSmolVLAPolicy.from_pretrained(args.checkpoint)
        if args.checkpoint
        else EARSmolVLAPolicy(stage_config(args.stage))
    )
    config = stage_config(args.stage, policy.config)
    policy.config = policy.model.config = config
    policy.train()
    if args.checkpoint and not args.stats:
        processor = BatchProcessor.load(
            config, args.checkpoint / "processor", tokenizer=policy.model.tokenizer
        )
    else:
        stats = torch.load(args.stats, weights_only=True) if args.stats else None
        processor = BatchProcessor(config, tokenizer=policy.model.tokenizer, stats=stats)
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=config.optimizer_lr,
        betas=config.optimizer_betas,
        eps=config.optimizer_eps,
        weight_decay=config.optimizer_weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    for step, batch_file in zip(range(args.steps), itertools.cycle(files), strict=False):
        raw = _to_device(torch.load(batch_file, weights_only=True), device)
        batch = processor(raw, training=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = policy(batch, guidance_progress=step / max(1, args.steps - 1))
            (loss / args.gradient_accumulation).backward()
        if (step + 1) % args.gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(policy.get_optim_params(), config.optimizer_grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step % 10 == 0:
            print({"step": step, **metrics})
    if args.steps % args.gradient_accumulation:
        torch.nn.utils.clip_grad_norm_(policy.get_optim_params(), config.optimizer_grad_clip_norm)
        optimizer.step()
    policy.save_pretrained(args.output)
    processor.save(args.output / "processor")


if __name__ == "__main__":
    main()
