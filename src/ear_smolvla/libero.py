"""LIBERO data adaptation and policy entry point."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .libero_config import LIBEROConfig
from .model import EARSmolVLAPolicy
from .processor import (
    ACTION_CODE_TOKEN_MASK,
    ACTION_SPLINE_TARGET,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    CONTROL_MODE_TARGET,
    EAR_SPLINE_TARGET,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    SPLINE_CURRENT_POSE,
    BatchProcessor,
    select_short_horizon,
)
from .spline import make_quaternion_signs_continuous, normalize_pose_quaternions

LIBERO_QUATERNION = (slice(3, 7),)


def axis_angle_to_quaternion(axis_angle: Tensor) -> Tensor:
    """Convert axis-angle vectors to normalized wxyz quaternions."""
    angle = axis_angle.norm(dim=-1, keepdim=True)
    scale = torch.where(
        angle > 1e-7,
        torch.sin(angle / 2) / angle.clamp_min(1e-7),
        0.5 - angle.square() / 48,
    )
    return torch.cat([torch.cos(angle / 2), axis_angle * scale], dim=-1)


def libero_pose_from_state(state: Tensor) -> Tensor:
    if state.shape[-1] != 8:
        raise ValueError(f"LIBERO state must end in 8 values, got {state.shape[-1]}")
    pose = torch.cat([state[..., :3], axis_angle_to_quaternion(state[..., 3:6])], dim=-1)
    if state.ndim == 3:
        pose = make_quaternion_signs_continuous(pose, LIBERO_QUATERNION)
    return normalize_pose_quaternions(pose, quaternion_slices=LIBERO_QUATERNION)


class LIBEROBatchProcessor(BatchProcessor):
    """Build fixed-future LIBERO EAR/action targets without skill annotations."""

    config: LIBEROConfig

    def unnormalize_actions(self, actions: Tensor) -> Tensor:
        return actions

    def __call__(self, source: dict[str, Any], *, training: bool) -> dict[str, Any]:
        batch = dict(source)
        state = batch[self.config.state_key]
        if training and state.ndim != 3:
            raise ValueError("LIBERO training requires state suffixes [B,H,8]")
        batch_size = state.shape[0]
        tasks = batch.get("task", [""] * batch_size)
        if not isinstance(tasks, (list, tuple)):
            tasks = [str(tasks)] * batch_size
        encoded = self.tokenizer(
            [str(task) for task in tasks],
            max_length=self.config.tokenizer_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        batch[OBS_LANGUAGE_TOKENS] = encoded.input_ids.to(state.device)
        batch[OBS_LANGUAGE_ATTENTION_MASK] = encoded.attention_mask.bool().to(state.device)

        pose = libero_pose_from_state(state)
        current_pose = pose[:, 0] if pose.ndim == 3 else pose
        batch[SPLINE_CURRENT_POSE] = current_pose
        batch[self.config.state_key] = self._normalize(self.config.state_key, state)
        if not training:
            return batch
        if self.config.action_key not in batch:
            raise ValueError("LIBERO training requires action suffixes")

        raw_actions = batch[self.config.action_key]
        if raw_actions.ndim != 3 or raw_actions.shape[-1] != self.config.action_dim:
            raise ValueError("LIBERO actions must have shape [B,H,7]")
        if raw_actions.shape[:2] != state.shape[:2]:
            raise ValueError("LIBERO state/action suffixes must share a time axis")
        mask = torch.ones(state.shape[:2], dtype=torch.bool, device=state.device)
        pad_mask = batch.get(f"{self.config.state_key}_is_pad")
        if pad_mask is not None:
            mask &= ~pad_mask.bool().to(mask.device)

        ear_pose = select_short_horizon(pose, mask, self.config.ear_horizon)
        action_pose = select_short_horizon(pose, mask, self.config.action_horizon)
        # LIBERO relative actions already use the environment's [-1, 1] convention.
        normalized_actions = raw_actions
        gripper = normalized_actions[..., -1:]
        ear_gripper = select_short_horizon(gripper, mask, self.config.ear_horizon)
        action_gripper = select_short_horizon(gripper, mask, self.config.action_horizon)
        batch[EAR_SPLINE_TARGET] = torch.cat([ear_pose, ear_gripper], dim=-1)
        batch[ACTION_SPLINE_TARGET] = torch.cat([action_pose, action_gripper], dim=-1)
        batch[CONTROL_MODE_TARGET] = torch.zeros(batch_size, 1, device=state.device)

        fast_source = select_short_horizon(
            normalized_actions, mask, self.config.action_horizon
        )
        batch[self.config.action_key] = fast_source
        fast, fast_mask, code_mask = self._tokenize_actions(fast_source)
        batch[ACTION_TOKENS] = fast
        batch[ACTION_TOKEN_MASK] = fast_mask
        batch[ACTION_CODE_TOKEN_MASK] = code_mask
        return batch


class LIBEROPolicy(EARSmolVLAPolicy):
    config_class = LIBEROConfig
    name = "ear_smolvla_libero"
