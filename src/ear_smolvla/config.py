"""Serializable configuration for the standalone dual-spline policy."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EARSmolVLAConfig:
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    vlm_revision: str = "482adb537c021c86670beed01cd58990d01e72e4"
    lerobot_revision: str = "7427f31801f944f5ec9ce53cda02862ce7d1638b"
    spline_policy_revision: str = "7b3893d3de3780fd94be72fc2be3b5898fbb5b47"
    knowledge_insulation_reference: str = "arXiv:2505.23705"
    spline_basis_version: str = "quadratic-c1-kplus2-v1"

    device: str = "cuda"
    load_vlm_weights: bool = True
    quantize_language_base_int8: bool = True
    num_vlm_layers: int = 16
    lm_lora_rank: int = 8
    vision_lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    image_keys: tuple[str, ...] = (
        "observation.images.robot0_agentview_left_image",
        "observation.images.robot0_eye_in_hand_image",
        "observation.images.robot0_agentview_right_image",
    )
    resize_imgs_with_padding: tuple[int, int] = (512, 512)
    state_key: str = "observation.state"
    action_key: str = "action"
    state_dim: int = 16
    action_dim: int = 12
    max_state_dim: int = 32
    spline_dim: int = 15  # base/EEF pose14 + continuous gripper1
    control_mode_values: tuple[float, float] = (-1.0, 1.0)

    # The EAR is broad; the final executable spline is short and precise.
    ear_segments: int = 14
    ear_horizon: int = 64
    action_segments: int = 6
    action_horizon: int = 16
    n_action_steps: int = 4
    dataset_fps: float = 20.0
    num_flow_steps: int = 10

    tokenizer_max_length: int = 96
    max_action_tokens: int = 256
    action_tokenizer_name: str = "lerobot/fast-action-tokenizer"
    fast_skip_tokens: int = 128
    validate_fast_vocabulary: bool = True
    fast_max_reconstruction_rmse: float = 0.25

    spline_reasoner_layers: int = 8
    action_expert_layers: int = 16
    expert_hidden_size: int = 768
    expert_heads: int = 12
    expert_ffn_size: int = 3072
    expert_dropout: float = 0.0
    gradient_checkpointing: bool = True
    training_kv_cache: bool = False
    detach_expert_context: bool = True
    detach_action_guidance: bool = True

    # Final exact / soft-teacher / inferred guidance probabilities. Training
    # starts at 1:1:0 and continuously reaches this mixture.
    guidance_mask_ratios: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    predicted_guidance_start_fraction: float = 0.1
    predicted_guidance_full_fraction: float = 0.5
    partial_guidance_min_confidence: float = 0.1
    partial_guidance_max_confidence: float = 0.7

    # Put translation, quaternion, and gripper channels on comparable scales
    # before flow matching and uncertainty estimation.
    spline_translation_scale: float = 0.25
    spline_rotation_scale: float = 1.0
    spline_gripper_scale: float = 1.0
    trajectory_reconstruction_weight: float = 1.0

    # Conditional Monte Carlo uncertainty shares one VLM representation.
    mc_samples: int = 4
    confidence_temperature: float = 0.01
    confidence_variance_threshold: float = 0.05

    field_attraction: float = 2.0
    field_progression: float = 1.0

    fast_loss_weight: float = 1.0
    spline_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    mode_loss_weight: float = 1.0
    train_vlm_objective: bool = True
    train_spline_reasoner: bool = True
    train_action_expert: bool = True
    curriculum_stage: str = "joint_teacher_forced"

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10.0

    @property
    def ear_parameter_count(self) -> int:
        return self.ear_segments + 2

    @property
    def action_parameter_count(self) -> int:
        return self.action_segments + 2

    @property
    def action_phase_span(self) -> float:
        return min(1.0, (self.action_horizon - 1) / (self.ear_horizon - 1))

    def guidance_probabilities(self, progress: float) -> tuple[float, float, float]:
        if not 0 <= progress <= 1:
            raise ValueError("training progress must lie in [0, 1]")
        if progress <= self.predicted_guidance_start_fraction:
            blend = 0.0
        elif progress >= self.predicted_guidance_full_fraction:
            blend = 1.0
        else:
            blend = (progress - self.predicted_guidance_start_fraction) / (
                self.predicted_guidance_full_fraction - self.predicted_guidance_start_fraction
            )
        initial = (0.5, 0.5, 0.0)
        return tuple(
            (1 - blend) * start + blend * end
            for start, end in zip(initial, self.guidance_mask_ratios, strict=True)
        )

    def __post_init__(self) -> None:
        if self.num_vlm_layers != 16:
            raise ValueError("EAR-SmolVLA uses the first 16 SmolVLM text layers")
        if self.spline_reasoner_layers not in (6, 8):
            raise ValueError("spline_reasoner_layers must be 8 or the documented 6-layer fallback")
        if self.action_expert_layers != 16:
            raise ValueError("The final action spline expert must have 16 layers")
        if self.expert_hidden_size % self.expert_heads:
            raise ValueError("expert_hidden_size must be divisible by expert_heads")
        if (self.state_dim, self.action_dim, self.spline_dim) != (16, 12, 15):
            raise ValueError("RoboCasa dimensions must be state16, action12, structured spline15")
        if len(self.control_mode_values) != 2 or self.control_mode_values[0] >= self.control_mode_values[1]:
            raise ValueError("control_mode_values must contain increasing low/high commands")
        if len(self.image_keys) != 3:
            raise ValueError("EAR-SmolVLA expects exactly three RoboCasa cameras")
        if self.n_action_steps > self.action_horizon:
            raise ValueError("n_action_steps cannot exceed action_horizon")
        if self.ear_horizon <= self.action_horizon:
            raise ValueError("EAR horizon must be broader than the final action horizon")
        if self.mc_samples < 2:
            raise ValueError("At least two Monte Carlo samples are required for variance")
        if (
            any(value < 0 for value in self.guidance_mask_ratios)
            or abs(sum(self.guidance_mask_ratios) - 1) >= 1e-6
        ):
            raise ValueError("guidance_mask_ratios must be nonnegative and sum to one")
        if not 0 <= self.predicted_guidance_start_fraction < self.predicted_guidance_full_fraction <= 1:
            raise ValueError("predicted guidance schedule must satisfy 0 <= start < full <= 1")
        if not 0 <= self.partial_guidance_min_confidence <= self.partial_guidance_max_confidence <= 1:
            raise ValueError("partial guidance confidence range must lie in [0, 1]")
        if (
            min(
                self.spline_translation_scale,
                self.spline_rotation_scale,
                self.spline_gripper_scale,
            )
            <= 0
        ):
            raise ValueError("spline channel scales must be positive")
        if self.trajectory_reconstruction_weight < 0:
            raise ValueError("trajectory_reconstruction_weight must be nonnegative")
        if self.quantize_language_base_int8 and not self.load_vlm_weights:
            raise ValueError("INT8 quantization requires pretrained VLM weights")
        if self.training_kv_cache:
            raise ValueError("Persistent KV caching must remain disabled during training")
        if self.spline_basis_version != "quadratic-c1-kplus2-v1":
            raise ValueError(f"Unsupported spline basis {self.spline_basis_version}")
        if self.curriculum_stage not in {"vlm_warmup", "joint_teacher_forced"}:
            raise ValueError(f"Unknown curriculum stage {self.curriculum_stage}")

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> EARSmolVLAConfig:
        data = json.loads((Path(directory) / "config.json").read_text(encoding="utf-8"))
        # Compatibility with checkpoints saved before conditional MC reused one VLM context.
        data.pop("mc_image_noise_std", None)
        data.pop("mc_state_noise_std", None)
        for key in (
            "image_keys",
            "resize_imgs_with_padding",
            "control_mode_values",
            "optimizer_betas",
            "guidance_mask_ratios",
        ):
            data[key] = tuple(data[key])
        return cls(**data)
