"""C1 quadratic splines, local parameter guidance, and closed-loop flow."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

BASE_QUAT = slice(3, 7)
EE_QUAT = slice(10, 14)
POSE_DIM = 14


def _valid_quaternion_slices(dimension: int, quaternion_slices: tuple[slice, ...]) -> tuple[slice, ...]:
    return tuple(item for item in quaternion_slices if item.stop is not None and item.stop <= dimension)


def make_quaternion_signs_continuous(
    poses: Tensor, quaternion_slices: tuple[slice, ...] = (BASE_QUAT, EE_QUAT)
) -> Tensor:
    """Choose equivalent quaternion signs continuously along a trajectory."""
    result = poses.clone()
    for quat_slice in _valid_quaternion_slices(poses.shape[-1], quaternion_slices):
        q = result[..., quat_slice]
        if q.shape[-2] < 2:
            continue
        dots = (q[..., 1:, :] * q[..., :-1, :]).sum(dim=-1)
        flips = torch.where(dots < 0, -torch.ones_like(dots), torch.ones_like(dots))
        signs = torch.cat([torch.ones_like(flips[..., :1]), flips], dim=-1).cumprod(dim=-1)
        result[..., quat_slice] = q * signs[..., None]
    return result


def normalize_pose_quaternions(
    poses: Tensor,
    eps: float = 1e-8,
    quaternion_slices: tuple[slice, ...] = (BASE_QUAT, EE_QUAT),
) -> Tensor:
    result = poses.clone()
    for quat_slice in _valid_quaternion_slices(poses.shape[-1], quaternion_slices):
        result[..., quat_slice] = torch.nn.functional.normalize(result[..., quat_slice], dim=-1, eps=eps)
    return result


def quaternion_component_velocity_to_angular(q: Tensor, q_dot: Tensor) -> Tensor:
    """Convert a wxyz quaternion derivative to angular velocity."""
    q = torch.nn.functional.normalize(q, dim=-1)
    w, xyz = q[..., :1], q[..., 1:]
    dw, dxyz = q_dot[..., :1], q_dot[..., 1:]
    return 2 * (w * dxyz - dw * xyz - torch.linalg.cross(dxyz, xyz, dim=-1))


@dataclass
class ParameterGuidance:
    """Only EAR control parameters whose temporal support overlaps the action window."""

    tokens: Tensor  # [B,G,D]
    support: Tensor  # [B,G,2], global EAR phase
    valid: Tensor  # [B,G], selected token rather than batch padding
    confidence: Tensor  # [B,G]
    role: Tensor  # [B,G], local start/handle/end role

    def detached(self) -> ParameterGuidance:
        return ParameterGuidance(
            self.tokens.detach(),
            self.support.detach(),
            self.valid.detach(),
            self.confidence.detach(),
            self.role.detach(),
        )


class QuadraticSpline(nn.Module):
    """Piecewise quadratic Bezier spline with C1 continuity by construction."""

    basis_version = "quadratic-c1-kplus2-v1"

    def __init__(
        self,
        segments: int,
        samples: int,
        quaternion_slices: tuple[slice, ...] = (BASE_QUAT, EE_QUAT),
    ):
        super().__init__()
        if segments < 1 or samples < 2:
            raise ValueError("segments must be positive and samples must be at least two")
        self.segments = segments
        self.samples = samples
        self.quaternion_slices = quaternion_slices
        basis = self._basis(torch.linspace(0, 1, samples, dtype=torch.float64))
        self.register_buffer("sample_basis", basis.float(), persistent=False)
        self.register_buffer("fit_matrix", torch.linalg.pinv(basis).float(), persistent=False)
        self.register_buffer(
            "start_conditioned_fit_matrix",
            torch.linalg.pinv(basis[:, 1:]).float(),
            persistent=False,
        )

    def _basis(self, phase: Tensor) -> Tensor:
        columns = self.segments + 2
        identity = torch.eye(columns, dtype=phase.dtype, device=phase.device)
        decoded = self._decode_from_params(identity.unsqueeze(-1), phase)
        return decoded.squeeze(-1).transpose(0, 1)

    def segment_controls(self, params: Tensor) -> Tensor:
        """Return the three actual control vectors of every local segment."""
        controls = []
        previous_handle = params[..., 1, :]
        controls.append(torch.stack([params[..., 0, :], previous_handle, params[..., 2, :]], dim=-2))
        for segment in range(1, self.segments):
            start = params[..., segment + 1, :]
            handle = 2 * start - previous_handle
            end = params[..., segment + 2, :]
            controls.append(torch.stack([start, handle, end], dim=-2))
            previous_handle = handle
        return torch.stack(controls, dim=-3)

    # Kept for compatibility with existing analysis notebooks.
    _controls = segment_controls

    def _decode_from_params(self, params: Tensor, phase: Tensor) -> Tensor:
        controls = self.segment_controls(params)
        scaled = phase.clamp(0, 1) * self.segments
        segment = scaled.floor().long().clamp_max(self.segments - 1)
        local = scaled - segment.to(scaled.dtype)
        local = torch.where(phase >= 1, torch.ones_like(local), local)
        selected = controls[..., segment, :, :]
        weights = torch.stack([(1 - local) ** 2, 2 * (1 - local) * local, local**2], dim=-1)
        return torch.einsum("...tpd,tp->...td", selected, weights)

    def fit(self, trajectory: Tensor, *, constrain_start: bool = False) -> Tensor:
        if trajectory.shape[-2] != self.samples:
            raise ValueError(f"expected {self.samples} samples, got {trajectory.shape[-2]}")
        if self.quaternion_slices:
            trajectory = make_quaternion_signs_continuous(trajectory, self.quaternion_slices)
        if constrain_start:
            start = trajectory[..., :1, :]
            basis = self.sample_basis.to(trajectory)
            residual = trajectory - basis[:, :1] * start
            tail = torch.einsum(
                "ph,...hd->...pd",
                self.start_conditioned_fit_matrix.to(trajectory),
                residual,
            )
            return torch.cat([start, tail], dim=-2)
        return torch.einsum("ph,...hd->...pd", self.fit_matrix, trajectory)

    def decode(self, params: Tensor, samples: int | None = None) -> Tensor:
        sample_count = samples or self.samples
        if sample_count == self.samples:
            basis = self.sample_basis.to(dtype=params.dtype, device=params.device)
            decoded = torch.einsum("hp,...pd->...hd", basis, params)
        else:
            phase = torch.linspace(0, 1, sample_count, dtype=params.dtype, device=params.device)
            decoded = self._decode_from_params(params, phase)
        return (
            normalize_pose_quaternions(decoded, quaternion_slices=self.quaternion_slices)
            if self.quaternion_slices
            else decoded
        )

    def evaluate(self, params: Tensor, phase: Tensor) -> Tensor:
        """Evaluate one phase per batch element."""
        controls = self.segment_controls(params)
        scaled = phase.clamp(0, 1) * self.segments
        segment = scaled.floor().long().clamp_max(self.segments - 1)
        local = torch.where(phase >= 1, torch.ones_like(phase), scaled - segment.to(phase.dtype))
        gather = segment[..., None, None].expand(*segment.shape, 1, controls.shape[-2] * controls.shape[-1])
        selected = (
            controls.flatten(-2).gather(-2, gather).squeeze(-2).reshape(*segment.shape, 3, controls.shape[-1])
        )
        weights = torch.stack([(1 - local) ** 2, 2 * (1 - local) * local, local**2], dim=-1)
        value = torch.einsum("...p,...pd->...d", weights, selected)
        return (
            normalize_pose_quaternions(value, quaternion_slices=self.quaternion_slices)
            if self.quaternion_slices
            else value
        )

    def endpoint_derivatives(self, params: Tensor) -> tuple[Tensor, Tensor]:
        controls = self.segment_controls(params)
        return 2 * (controls[..., 0, 1, :] - controls[..., 0, 0, :]), 2 * (
            controls[..., -1, 2, :] - controls[..., -1, 1, :]
        )

    def closest_point_field(
        self, params: Tensor, state: Tensor, attraction: float = 2.0, progression: float = 1.0
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Closest-point attraction plus tangent progression from the latest state."""
        controls = self.segment_controls(params)
        p0, p1, p2 = controls.unbind(dim=-2)
        quadratic = p0 - 2 * p1 + p2
        linear = 2 * (p1 - p0)
        offset = p0 - state.unsqueeze(-2)
        cubic_a = 2 * quadratic.square().sum(dim=-1)
        cubic_b = 3 * (quadratic * linear).sum(dim=-1)
        cubic_c = linear.square().sum(dim=-1) + 2 * (offset * quadratic).sum(dim=-1)
        cubic_d = (offset * linear).sum(dim=-1)

        roots = torch.linspace(0, 1, 9, dtype=params.dtype, device=params.device)
        roots = roots.expand(*cubic_a.shape, -1).clone()
        for _ in range(10):
            value = ((cubic_a[..., None] * roots + cubic_b[..., None]) * roots + cubic_c[..., None]) * roots
            value = value + cubic_d[..., None]
            derivative = (3 * cubic_a[..., None] * roots + 2 * cubic_b[..., None]) * roots
            derivative = derivative + cubic_c[..., None]
            safe = torch.where(derivative.abs() < 1e-8, torch.ones_like(derivative), derivative)
            roots = (roots - value / safe).clamp(0, 1)
        # Endpoints are unconditional candidates; Newton must not move them away.
        roots = torch.cat([roots, torch.zeros_like(roots[..., :1]), torch.ones_like(roots[..., :1])], dim=-1)

        candidates = quadratic[..., None, :] * roots[..., None].square()
        candidates = candidates + linear[..., None, :] * roots[..., None] + p0[..., None, :]
        squared = (candidates - state[..., None, None, :]).square().sum(dim=-1)
        flat = squared.flatten(-2)
        nearest = flat.argmin(dim=-1)
        candidate_flat = candidates.flatten(-3, -2)
        gather = nearest[..., None, None].expand(*nearest.shape, 1, state.shape[-1])
        closest = candidate_flat.gather(-2, gather).squeeze(-2)
        roots_flat = roots.flatten(-2)
        best_root = roots_flat.gather(-1, nearest[..., None]).squeeze(-1)
        segment = torch.div(nearest, roots.shape[-1], rounding_mode="floor")
        segment_gather = segment[..., None, None].expand(*segment.shape, 1, state.shape[-1])
        best_quadratic = quadratic.gather(-2, segment_gather).squeeze(-2)
        best_linear = linear.gather(-2, segment_gather).squeeze(-2)
        tangent = torch.nn.functional.normalize(
            2 * best_quadratic * best_root[..., None] + best_linear, dim=-1, eps=1e-8
        )
        correction = closest - state
        field = progression * tangent + attraction * correction
        phase = (segment.to(state.dtype) + best_root) / self.segments
        distance = flat.gather(-1, nearest[..., None]).sqrt().squeeze(-1)
        return field, phase, distance

    def select_parameter_guidance(
        self,
        params: Tensor,
        phase: Tensor,
        phase_span: float,
        segment_confidence: Tensor | None = None,
    ) -> ParameterGuidance:
        """Pack only overlapping segments, then expose each of their three parameters."""
        batch, _, dimension = params.shape
        controls = self.segment_controls(params)
        starts = torch.arange(self.segments, device=params.device, dtype=params.dtype) / self.segments
        ends = starts + 1 / self.segments
        window_end = (phase + phase_span).clamp_max(1)
        overlaps = (ends[None] > phase[:, None]) & (starts[None] < window_end[:, None])
        counts = overlaps.sum(dim=1).clamp_min(1)
        max_tokens = int(counts.max()) * 3
        tokens = params.new_zeros(batch, max_tokens, dimension)
        support = params.new_zeros(batch, max_tokens, 2)
        valid = torch.zeros(batch, max_tokens, dtype=torch.bool, device=params.device)
        confidence = params.new_ones(batch, max_tokens)
        role = torch.zeros(batch, max_tokens, dtype=torch.long, device=params.device)
        for row in range(batch):
            indices = overlaps[row].nonzero(as_tuple=False).flatten()
            if not len(indices):
                index = min(int(phase[row] * self.segments), self.segments - 1)
                indices = torch.tensor([index], device=params.device)
            count = len(indices) * 3
            tokens[row, :count] = controls[row, indices].reshape(-1, dimension)
            intervals = torch.stack([starts[indices], ends[indices]], dim=-1)
            support[row, :count] = intervals[:, None].expand(-1, 3, -1).reshape(-1, 2)
            valid[row, :count] = True
            role[row, :count] = torch.arange(3, device=params.device).repeat(len(indices))
            if segment_confidence is not None:
                confidence[row, :count] = segment_confidence[row, indices, None].expand(-1, 3).reshape(-1)
        return ParameterGuidance(tokens, support, valid, confidence, role)

    def propagate_parameter_covariance(self, parameter_samples: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Monte Carlo parameter moments and Eq. 11 trajectory marginal variance.

        `parameter_samples` has shape [S,B,P,D]. Cross-dimension covariance is
        unnecessary for the per-time variance used by confidence masking.
        """
        if parameter_samples.ndim != 4 or parameter_samples.shape[0] < 2:
            raise ValueError("parameter_samples must have shape [S>=2,B,P,D]")
        if self.quaternion_slices:
            parameter_samples = parameter_samples.clone()
            for quat_slice in _valid_quaternion_slices(parameter_samples.shape[-1], self.quaternion_slices):
                reference = parameter_samples[:1, ..., quat_slice]
                dot = (parameter_samples[..., quat_slice] * reference).sum(dim=(-1, -2))
                sign = torch.where(dot < 0, -torch.ones_like(dot), torch.ones_like(dot))
                parameter_samples[..., quat_slice] *= sign[..., None, None]
        mean = parameter_samples.mean(dim=0)
        centered = parameter_samples - mean[None]
        covariance = torch.einsum("sbpd,sbqd->bdpq", centered, centered)
        covariance = covariance / (parameter_samples.shape[0] - 1)
        basis = self.sample_basis.to(parameter_samples)
        trajectory_variance = torch.einsum("hp,bdpq,hq->bhd", basis, covariance, basis)
        return mean, covariance, trajectory_variance.clamp_min(0)

    def segment_confidence(
        self, trajectory_variance: Tensor, temperature: float, threshold: float
    ) -> tuple[Tensor, Tensor]:
        """Turn Eq. 11 variance into one conservative confidence per segment."""
        if temperature <= 0 or threshold < 0:
            raise ValueError("temperature must be positive and threshold nonnegative")
        phase = torch.linspace(0, 1, trajectory_variance.shape[1], device=trajectory_variance.device)
        uncertainty = []
        scalar = trajectory_variance.mean(dim=-1)
        for segment in range(self.segments):
            inside = (phase >= segment / self.segments) & (phase <= (segment + 1) / self.segments)
            uncertainty.append(scalar[:, inside].amax(dim=1))
        uncertainty = torch.stack(uncertainty, dim=1)
        confidence = torch.exp(-uncertainty / temperature).clamp(0, 1)
        return confidence, uncertainty <= threshold


def soften_partial_guidance(
    confidence: Tensor,
    valid: Tensor,
    partial_rows: Tensor,
    minimum: float,
    maximum: float,
) -> Tensor:
    """Lower confidence for a contiguous half of the selected EAR segments."""
    if partial_rows.shape != valid.shape[:1]:
        raise ValueError("partial_rows must have shape [B]")
    if not 0 <= minimum <= maximum <= 1:
        raise ValueError("guidance confidence range must lie in [0, 1]")
    result = confidence.clone()
    for row in partial_rows.nonzero(as_tuple=False).flatten().tolist():
        token_count = int(valid[row].sum())
        segment_count = token_count // 3
        if segment_count < 2:
            continue
        length = max(1, segment_count // 2)
        start = int(torch.randint(0, segment_count - length + 1, (), device=valid.device))
        values = torch.empty(length, dtype=result.dtype, device=result.device).uniform_(minimum, maximum)
        result[row, start * 3 : (start + length) * 3] = values.repeat_interleave(3)
    return result


def project_quaternion_velocity(
    field: Tensor,
    current_pose: Tensor,
    quaternion_slices: tuple[slice, ...] = (BASE_QUAT, EE_QUAT),
) -> Tensor:
    """Remove radial quaternion velocity without changing the spline workflow."""
    result = field.clone()
    for quat_slice in _valid_quaternion_slices(field.shape[-1], quaternion_slices):
        quaternion = torch.nn.functional.normalize(current_pose[..., quat_slice], dim=-1)
        derivative = result[..., quat_slice]
        result[..., quat_slice] = derivative - quaternion * (quaternion * derivative).sum(
            dim=-1, keepdim=True
        )
    return result


def field_to_robocasa_action(
    field: Tensor, current_pose: Tensor, gripper: Tensor, control_mode: Tensor
) -> Tensor:
    """Map the pose-space flow to the pinned 12D RoboCasa action order."""
    field = project_quaternion_velocity(field, current_pose)
    base_angular = quaternion_component_velocity_to_angular(
        current_pose[..., BASE_QUAT], field[..., BASE_QUAT]
    )
    ee_angular = quaternion_component_velocity_to_angular(current_pose[..., EE_QUAT], field[..., EE_QUAT])
    base = torch.cat([field[..., :3], base_angular[..., 2:3]], dim=-1)
    return torch.cat([base, control_mode, field[..., 7:10], ee_angular, gripper[..., :1]], dim=-1)


def field_to_libero_action(field: Tensor, current_pose: Tensor, gripper: Tensor) -> Tensor:
    """Map absolute EEF-pose flow to LIBERO relative OSC pose actions."""
    quaternion_slice = slice(3, 7)
    field = project_quaternion_velocity(field, current_pose, (quaternion_slice,))
    angular = quaternion_component_velocity_to_angular(
        current_pose[..., quaternion_slice], field[..., quaternion_slice]
    )
    return torch.cat([field[..., :3], angular, gripper[..., :1]], dim=-1).clamp(-1, 1)
