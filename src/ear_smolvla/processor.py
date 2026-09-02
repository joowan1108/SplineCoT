"""Minimal RoboCasa preprocessing without a LeRobot runtime dependency."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from .config import EARSmolVLAConfig
from .spline import make_quaternion_signs_continuous, normalize_pose_quaternions

OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"
ACTION_TOKENS = "action.tokens"
ACTION_TOKEN_MASK = "action.token_mask"
ACTION_CODE_TOKEN_MASK = "action.code_token_mask"
SPLINE_CURRENT_POSE = "spline.current_pose"
EAR_SPLINE_TARGET = "spline.ear_target"
ACTION_SPLINE_TARGET = "spline.action_target"
CONTROL_MODE_TARGET = "action.control_mode_target"


def resample_sequence(values: Tensor, mask: Tensor, samples: int) -> Tensor:
    """Resample each valid prefix to a fixed number of spline fitting samples."""
    rows = []
    for row in range(values.shape[0]):
        count = int(mask[row].sum())
        if count < 1:
            raise ValueError("trajectory contains no valid sample")
        valid = values[row, :count].T[None]
        if count == 1:
            rows.append(valid[0, :, :1].T.expand(samples, -1))
        else:
            rows.append(F.interpolate(valid, size=samples, mode="linear", align_corners=True)[0].T)
    return torch.stack(rows)


def select_short_horizon(values: Tensor, mask: Tensor, samples: int) -> Tensor:
    """Keep the next real control ticks and repeat only a missing tail."""
    rows = []
    for row in range(values.shape[0]):
        count = min(int(mask[row].sum()), samples)
        if count < 1:
            raise ValueError("trajectory contains no valid sample")
        selected = values[row, :count]
        if count < samples:
            selected = torch.cat([selected, selected[-1:].expand(samples - count, -1)])
        rows.append(selected)
    return torch.stack(rows)


def build_pose_targets(
    state: Tensor,
    *,
    pad_mask: Tensor | None = None,
    ear_samples: int,
    action_samples: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build current, broad EAR, and short action pose targets from raw state."""
    if state.ndim != 3 or state.shape[-1] < 14:
        raise ValueError(f"state must have shape [B,H,>=14], got {tuple(state.shape)}")
    pose = normalize_pose_quaternions(make_quaternion_signs_continuous(state[..., :14]))
    mask = torch.ones(pose.shape[:2], dtype=torch.bool, device=pose.device)
    if pad_mask is not None:
        mask &= ~pad_mask[:, : pose.shape[1]].bool().to(mask.device)
    current = pose[:, 0]
    ear = normalize_pose_quaternions(resample_sequence(pose, mask, ear_samples))
    action = normalize_pose_quaternions(select_short_horizon(pose, mask, action_samples))
    return current, ear, action, mask


class BatchProcessor:
    """Tokenize FAST targets and build both structured spline targets."""

    def __init__(
        self,
        config: EARSmolVLAConfig,
        *,
        tokenizer: Any | None = None,
        action_tokenizer: Any | None = None,
        stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        from transformers import AutoProcessor, AutoTokenizer

        self.config = config
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            config.vlm_model_name, revision=config.vlm_revision
        )
        self.action_tokenizer = action_tokenizer or AutoProcessor.from_pretrained(
            config.action_tokenizer_name, trust_remote_code=True
        )
        self.stats = stats or {}
        self.reconstruction_rmse: float | None = None

    def _normalize(self, key: str, value: Tensor) -> Tensor:
        stats = self.stats.get(key)
        if not stats:
            return value
        if "scale" in stats and "offset" in stats:
            return value * stats["scale"].to(value) + stats["offset"].to(value)
        return (value - stats["mean"].to(value)) / stats["std"].to(value).clamp_min(1e-6)

    def unnormalize_actions(self, actions: Tensor) -> Tensor:
        stats = self.stats.get(self.config.action_key)
        if not stats:
            return actions
        if "scale" in stats and "offset" in stats:
            return (actions - stats["offset"].to(actions)) / stats["scale"].to(actions)
        return actions * stats["std"].to(actions) + stats["mean"].to(actions)

    def _tokenize_actions(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        rows, masks, code_masks, recovered = [], [], [], []
        vocab_size = self.tokenizer.vocab_size
        special_ids = torch.tensor(self.tokenizer.all_special_ids, device=actions.device)
        prompt = torch.tensor(
            self.tokenizer.encode("Action: ", add_special_tokens=False), device=actions.device
        )
        end = torch.tensor(self.tokenizer.encode("|"), device=actions.device)
        for action in actions:
            codes = torch.as_tensor(
                self.action_tokenizer(action[None].cpu()), device=actions.device
            ).flatten()
            mapped = vocab_size - 1 - self.config.fast_skip_tokens - codes
            if self.config.validate_fast_vocabulary:
                if mapped.numel() and ((mapped < 0).any() or (mapped >= vocab_size).any()):
                    raise ValueError("Mapped FAST token lies outside the SmolVLM vocabulary")
                if mapped.numel() and special_ids.numel() and torch.isin(mapped, special_ids).any():
                    raise ValueError("Mapped FAST code range overlaps a reserved SmolVLM token")
            bos = torch.tensor([self.tokenizer.bos_token_id], device=actions.device)
            row = torch.cat([bos, prompt, mapped, end])
            if len(row) > self.config.max_action_tokens:
                raise ValueError(
                    f"FAST sequence has {len(row)} tokens; max_action_tokens={self.config.max_action_tokens}"
                )
            code_mask = torch.zeros(self.config.max_action_tokens, dtype=torch.bool, device=actions.device)
            code_mask[1 + len(prompt) : 1 + len(prompt) + len(mapped)] = True
            mask = torch.arange(self.config.max_action_tokens, device=actions.device) < len(row)
            rows.append(F.pad(row, (0, self.config.max_action_tokens - len(row))))
            masks.append(mask)
            code_masks.append(code_mask)
            recovered.append(codes.tolist())
        if self.reconstruction_rmse is None:
            decoded = torch.as_tensor(
                self.action_tokenizer.decode(
                    recovered, time_horizon=actions.shape[-2], action_dim=actions.shape[-1]
                ),
                dtype=actions.dtype,
                device=actions.device,
            )
            if decoded.shape != actions.shape or not torch.isfinite(decoded).all():
                raise ValueError("FAST round trip returned an invalid action tensor")
            self.reconstruction_rmse = float((decoded - actions).square().mean().sqrt())
            if self.reconstruction_rmse > self.config.fast_max_reconstruction_rmse:
                raise ValueError(
                    f"FAST reconstruction RMSE {self.reconstruction_rmse:.4f} exceeds "
                    f"{self.config.fast_max_reconstruction_rmse:.4f}"
                )
        return torch.stack(rows), torch.stack(masks), torch.stack(code_masks)

    def __call__(self, source: dict[str, Any], *, training: bool) -> dict[str, Any]:
        batch = dict(source)
        state = batch[self.config.state_key]
        if training and state.ndim != 3:
            raise ValueError("Training requires a future state sequence")
        batch_size = state.shape[0]
        tasks = batch.get("task", [""] * batch_size)
        if not isinstance(tasks, (list, tuple)):
            tasks = [str(tasks)] * batch_size
        task_tokens = self.tokenizer(
            [str(task) for task in tasks],
            max_length=self.config.tokenizer_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        batch[OBS_LANGUAGE_TOKENS] = task_tokens.input_ids.to(state.device)
        batch[OBS_LANGUAGE_ATTENTION_MASK] = task_tokens.attention_mask.bool().to(state.device)

        if state.ndim == 3:
            current, ear_pose, action_pose, mask = build_pose_targets(
                state,
                pad_mask=batch.get(f"{self.config.state_key}_is_pad"),
                ear_samples=self.config.ear_horizon,
                action_samples=self.config.action_horizon,
            )
        else:
            current = normalize_pose_quaternions(state[..., :14])
        batch[SPLINE_CURRENT_POSE] = current
        batch[self.config.state_key] = self._normalize(self.config.state_key, state)
        if not training:
            return batch
        if self.config.action_key not in batch:
            raise ValueError("Training requires action trajectories")

        raw_actions = batch[self.config.action_key]
        if raw_actions.ndim != 3 or raw_actions.shape[1] < 1:
            raise ValueError("actions must have shape [B,H,12]")
        if raw_actions.shape[1] != state.shape[1]:
            raise ValueError("state and action suffixes must share the same time axis")
        common = min(raw_actions.shape[1], mask.shape[1])
        action_mask = mask[:, :common]
        normalized_actions = self._normalize(self.config.action_key, raw_actions)
        gripper = normalized_actions[:, :common, 11:12]
        ear_gripper = resample_sequence(gripper, action_mask, self.config.ear_horizon)
        action_gripper = select_short_horizon(gripper, action_mask, self.config.action_horizon)
        batch[EAR_SPLINE_TARGET] = torch.cat([ear_pose, ear_gripper], dim=-1)
        batch[ACTION_SPLINE_TARGET] = torch.cat([action_pose, action_gripper], dim=-1)
        batch[CONTROL_MODE_TARGET] = raw_actions[:, 0, 4:5]

        fast_source = select_short_horizon(
            normalized_actions[:, :common], action_mask, self.config.action_horizon
        )
        batch[self.config.action_key] = fast_source
        fast, fast_mask, code_mask = self._tokenize_actions(fast_source)
        batch[ACTION_TOKENS] = fast
        batch[ACTION_TOKEN_MASK] = fast_mask
        batch[ACTION_CODE_TOKEN_MASK] = code_mask
        return batch

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.stats, path / "normalization.pt")
        (path / "processor.json").write_text(
            json.dumps({"reconstruction_rmse": self.reconstruction_rmse}, indent=2), encoding="utf-8"
        )
        self.tokenizer.save_pretrained(path / "tokenizer")
        self.action_tokenizer.save_pretrained(path / "action_tokenizer")

    @classmethod
    def load(
        cls, config: EARSmolVLAConfig, directory: str | Path, *, tokenizer: Any | None = None
    ) -> "BatchProcessor":
        from transformers import AutoProcessor, AutoTokenizer

        path = Path(directory)
        processor = cls(
            config,
            tokenizer=tokenizer or AutoTokenizer.from_pretrained(path / "tokenizer"),
            action_tokenizer=AutoProcessor.from_pretrained(path / "action_tokenizer", trust_remote_code=True),
            stats=torch.load(path / "normalization.pt", map_location="cpu", weights_only=True),
        )
        state = json.loads((path / "processor.json").read_text(encoding="utf-8"))
        processor.reconstruction_rmse = state["reconstruction_rmse"]
        return processor
