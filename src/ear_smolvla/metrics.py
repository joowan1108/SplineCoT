"""Small metrics used by EAR-SmolVLA evaluations."""

from __future__ import annotations

import torch
from torch import Tensor


def spline_trajectory_errors(
    trajectory: Tensor, reference: Tensor, quaternion_slices: tuple[slice, ...]
) -> dict[str, Tensor]:
    """Errors for decoded pose splines or native 7D LIBERO action splines."""
    if trajectory.shape != reference.shape or trajectory.ndim != 3:
        raise ValueError("trajectory and reference must share shape [B,H,D]")
    trajectory, reference = trajectory.float(), reference.float()
    if not quaternion_slices:
        if trajectory.shape[-1] != 7:
            raise ValueError("native action metrics require 7D LIBERO actions")
        difference = trajectory - reference
        predicted_gripper = trajectory[..., -1]
        target_gripper = reference[..., -1]
        return {
            "translation_rmse": difference[..., :3].square().mean().sqrt(),
            "rotation_action_rmse": difference[..., 3:6].square().mean().sqrt(),
            "gripper_rmse": (predicted_gripper - target_gripper).square().mean().sqrt(),
            "gripper_accuracy": (
                (predicted_gripper >= 0) == (target_gripper >= 0)
            ).float().mean(),
        }
    translations = [slice(quaternion.start - 3, quaternion.start) for quaternion in quaternion_slices]
    translation_error = torch.cat(
        [trajectory[..., part] - reference[..., part] for part in translations], dim=-1
    )
    rotation_errors = []
    for quaternion in quaternion_slices:
        predicted = torch.nn.functional.normalize(trajectory[..., quaternion], dim=-1)
        target = torch.nn.functional.normalize(reference[..., quaternion], dim=-1)
        dot = (predicted * target).sum(dim=-1).abs().clamp(max=1)
        rotation_errors.append(torch.rad2deg(2 * torch.acos(dot)))
    predicted_gripper = trajectory[..., -1]
    target_gripper = reference[..., -1]
    return {
        "translation_rmse": translation_error.square().mean().sqrt(),
        "rotation_error_deg": torch.stack(rotation_errors, dim=-1).mean(),
        "gripper_rmse": (predicted_gripper - target_gripper).square().mean().sqrt(),
        "gripper_accuracy": ((predicted_gripper >= 0) == (target_gripper >= 0)).float().mean(),
    }


def trajectory_metrics(
    trajectory: Tensor, reference: Tensor | None = None, dt: float = 0.05
) -> dict[str, Tensor]:
    """Return smoothness and optional reference error for `[B, H, D]` paths."""
    if trajectory.ndim != 3 or trajectory.shape[1] < 4:
        raise ValueError("trajectory must have shape [B, H>=4, D]")
    velocity = torch.diff(trajectory, dim=1) / dt
    acceleration = torch.diff(velocity, dim=1) / dt
    jerk = torch.diff(acceleration, dim=1) / dt
    result = {
        "mean_speed": velocity.norm(dim=-1).mean(),
        "mean_acceleration": acceleration.norm(dim=-1).mean(),
        "mean_jerk": jerk.norm(dim=-1).mean(),
        "max_velocity_jump": torch.diff(velocity, dim=1).norm(dim=-1).max(),
    }
    if reference is not None:
        if reference.shape != trajectory.shape:
            raise ValueError("reference and trajectory shapes must match")
        result["reference_rmse"] = (trajectory - reference).square().mean().sqrt()
    return result


def perturbation_recovery(distance: Tensor, threshold: float, dt: float = 0.05) -> dict[str, Tensor]:
    """Measure whether and when each perturbed rollout returns to the spline tube."""
    if distance.ndim != 2:
        raise ValueError("distance must have shape [B, H]")
    recovered = distance <= threshold
    has_recovered = recovered.any(dim=1)
    first = recovered.float().argmax(dim=1)
    recovery_time = torch.where(has_recovered, first.to(distance.dtype) * dt, torch.inf)
    return {
        "recovery_success": has_recovered.float().mean(),
        "recovery_time": recovery_time[has_recovered].mean()
        if has_recovered.any()
        else torch.tensor(torch.inf, device=distance.device),
    }
