"""RTX 3090 batch-4 forward/backward memory gate."""

from __future__ import annotations

import argparse
import json
import time

import torch

from .config import EARSmolVLAConfig
from .model import EARSmolVLAPolicy
from .processor import (
    ACTION_CODE_TOKEN_MASK,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    ACTION_SPLINE_TARGET,
    CONTROL_MODE_TARGET,
    EAR_SPLINE_TARGET,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    SPLINE_CURRENT_POSE,
)


def make_batch(policy: EARSmolVLAPolicy, batch_size: int) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    config = policy.config
    task = policy.model.tokenizer(
        ["pick up the mug and place it on the counter"] * batch_size,
        max_length=config.tokenizer_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    source_horizon = config.ear_horizon
    state = torch.zeros(batch_size, source_horizon, config.state_dim, device=device)
    state[..., 3] = state[..., 10] = 1
    fast = torch.zeros(batch_size, config.max_action_tokens, dtype=torch.long, device=device)
    fast[:, :32] = torch.randint(
        policy.model.tokenizer.vocab_size - 384,
        policy.model.tokenizer.vocab_size - 128,
        (batch_size, 32),
        device=device,
    )
    fast_mask = torch.zeros_like(fast, dtype=torch.bool)
    fast_mask[:, :32] = True
    batch = {
        config.state_key: state,
        config.action_key: torch.randn(
            batch_size, config.action_horizon, config.action_dim, device=device
        ),
        OBS_LANGUAGE_TOKENS: task.input_ids.to(device),
        OBS_LANGUAGE_ATTENTION_MASK: task.attention_mask.bool().to(device),
        ACTION_TOKENS: fast,
        ACTION_TOKEN_MASK: fast_mask,
        ACTION_CODE_TOKEN_MASK: fast_mask,
        SPLINE_CURRENT_POSE: state[:, 0, :14],
        EAR_SPLINE_TARGET: torch.cat(
            [
                state[:, : config.ear_horizon, :14],
                torch.randn(batch_size, config.ear_horizon, 1, device=device),
            ],
            dim=-1,
        ),
        ACTION_SPLINE_TARGET: torch.cat(
            [
                state[:, : config.action_horizon, :14],
                torch.randn(batch_size, config.action_horizon, 1, device=device),
            ],
            dim=-1,
        ),
        CONTROL_MODE_TARGET: torch.ones(batch_size, 1, device=device),
    }
    batch.update(
        {
            key: torch.rand(batch_size, 3, *config.resize_imgs_with_padding, device=device)
            for key in config.image_keys
        }
    )
    return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vision-lora-rank", type=int, choices=(8, 16), default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The memory gate requires an NVIDIA CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    if properties.total_memory < 23 * 1024**3:
        raise RuntimeError(f"Expected a 24 GB GPU, found {properties.total_memory / 1024**3:.1f} GiB")

    config = EARSmolVLAConfig(vision_lora_rank=args.vision_lora_rank)
    policy = EARSmolVLAPolicy(config).train()
    batch = make_batch(policy, args.batch_size)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=config.optimizer_lr)
    torch.cuda.reset_peak_memory_stats()
    durations = []
    for _ in range(args.steps):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = policy(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.get_optim_params(), config.optimizer_grad_clip_norm)
        optimizer.step()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
    report = {
        "gpu": properties.name,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "vision_lora_rank": args.vision_lora_rank,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "mean_step_seconds": sum(durations) / len(durations),
        "headroom_gib": (properties.total_memory - torch.cuda.max_memory_reserved()) / 1024**3,
        "last_metrics": metrics,
    }
    print(json.dumps(report, indent=2))
    if report["headroom_gib"] < 1:
        raise RuntimeError("Memory gate failed: less than 1 GiB reserved headroom")


if __name__ == "__main__":
    main()
