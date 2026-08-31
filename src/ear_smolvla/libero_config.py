"""LIBERO-only settings; RoboCasa defaults remain untouched."""

from __future__ import annotations

from dataclasses import dataclass

from .config import EARSmolVLAConfig


@dataclass
class LIBEROConfig(EARSmolVLAConfig):
    embodiment: str = "libero"
    image_keys: tuple[str, ...] = (
        "observation.images.image",
        "observation.images.image2",
    )
    resize_imgs_with_padding: tuple[int, int] = (256, 256)
    state_dim: int = 8  # EEF position3 + axis-angle3 + gripper qpos2
    action_dim: int = 7  # relative EEF translation3 + rotation3 + gripper1
    spline_dim: int = 8  # absolute EEF position3 + quaternion4 + gripper1
    pose_dim: int = 7
    has_control_mode: bool = False
    initial_gripper_command: float = -1.0

    ear_segments: int = 10
    ear_horizon: int = 64
    action_segments: int = 6
    action_horizon: int = 16
    n_action_steps: int = 4
    dataset_fps: float = 20.0

    # FAST gradients cross the frozen LM and update the full vision encoder.
    use_language_lora: bool = False
    use_vision_lora: bool = False
    train_vision_encoder_full: bool = True
    quantize_language_base_int8: bool = False
    lm_lora_rank: int = 0
    vision_lora_rank: int = 0
    optimizer_lr: float = 1e-4
    vision_encoder_lr: float = 1e-5

    def __post_init__(self) -> None:
        if self.embodiment != "libero":
            raise ValueError("LIBEROConfig embodiment must be libero")
        if (self.state_dim, self.action_dim, self.spline_dim, self.pose_dim) != (8, 7, 8, 7):
            raise ValueError("LIBERO dimensions must be state8, action7, spline8, pose7")
        if len(self.image_keys) != 2:
            raise ValueError("LIBERO requires agent-view and wrist cameras")
        if self.use_language_lora or self.use_vision_lora:
            raise ValueError("LIBERO uses no LoRA adapters")
        if not self.train_vision_encoder_full:
            raise ValueError("LIBERO vision encoder must be fully trainable")
        if self.num_vlm_layers != 16 or self.action_expert_layers != 16:
            raise ValueError("LIBERO keeps the locked 16-layer VLM/EAR/action experts")
        if self.spline_reasoner_layers not in (6, 8, 16):
            raise ValueError("LIBERO EAR spline expert must have 16 layers (6/8 are legacy checkpoints)")
        if self.expert_hidden_size % self.expert_heads:
            raise ValueError("expert_hidden_size must be divisible by expert_heads")
        if self.ear_horizon <= self.action_horizon:
            raise ValueError("EAR horizon must exceed the action horizon")
        if self.n_action_steps > self.action_horizon:
            raise ValueError("n_action_steps cannot exceed action_horizon")
        if self.mc_samples < 2:
            raise ValueError("At least two Monte Carlo samples are required")
        if (
            any(value < 0 for value in self.guidance_mask_ratios)
            or abs(sum(self.guidance_mask_ratios) - 1) >= 1e-6
        ):
            raise ValueError("guidance_mask_ratios must be nonnegative and sum to one")
        if not 0 <= self.predicted_guidance_start_fraction < self.predicted_guidance_full_fraction <= 1:
            raise ValueError("predicted guidance schedule must satisfy 0 <= start < full <= 1")
        if not 0 <= self.partial_guidance_min_confidence <= self.partial_guidance_max_confidence <= 1:
            raise ValueError("partial guidance confidence range must lie in [0, 1]")
        if min(self.spline_translation_scale, self.spline_rotation_scale, self.spline_gripper_scale) <= 0:
            raise ValueError("spline channel scales must be positive")
        if self.trajectory_reconstruction_weight < 0:
            raise ValueError("trajectory_reconstruction_weight must be nonnegative")
        if self.quantize_language_base_int8 and not self.load_vlm_weights:
            raise ValueError("INT8 language loading requires pretrained weights")
        if self.training_kv_cache:
            raise ValueError("Training KV cache must remain disabled")
        if self.spline_basis_version != "quadratic-c1-kplus2-v1":
            raise ValueError(f"Unsupported spline basis {self.spline_basis_version}")
        if self.curriculum_stage not in {"vlm_warmup", "joint_teacher_forced"}:
            raise ValueError(f"Unknown curriculum stage {self.curriculum_stage}")
