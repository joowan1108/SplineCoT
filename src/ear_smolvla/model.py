"""Knowledge-insulated VLM with broad EAR and executable spline flows."""

from __future__ import annotations

import math
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
import torch.utils.checkpoint
from torch import Tensor, nn

from .config import EARSmolVLAConfig
from .metrics import spline_trajectory_errors
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
)
from .spline import (
    POSE_DIM,
    ParameterGuidance,
    QuadraticSpline,
    field_to_libero_action,
    field_to_robocasa_action,
    project_quaternion_velocity,
    random_teacher_guidance_mask,
)

LANGUAGE_INT8_SKIP_MODULES = ["model.vision_model", "model.connector", "lm_head"]


def sample_noise(shape: tuple[int, ...], device: torch.device | str) -> Tensor:
    return torch.randn(shape, device=device, dtype=torch.float32)


def sample_time_beta(batch_size: int, device: torch.device | str) -> Tensor:
    distribution = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
    return (distribution.sample((batch_size,)).to(device) * 0.999 + 0.001).float()


def sinusoidal_embedding(time: Tensor, dimension: int, device: torch.device) -> Tensor:
    if dimension % 2:
        raise ValueError("expert_hidden_size must be even")
    fraction = torch.linspace(0, 1, dimension // 2, dtype=torch.float64, device=device)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    angle = time[..., None] * (2 * math.pi / period)
    return torch.cat([angle.sin(), angle.cos()], dim=-1)


def pad_vector(value: Tensor, dimension: int) -> Tensor:
    return value if value.shape[-1] >= dimension else F.pad(value, (0, dimension - value.shape[-1]))


def resize_with_pad(image: Tensor, size: tuple[int, int]) -> Tensor:
    height, width = size
    current_height, current_width = image.shape[-2:]
    if (current_height, current_width) == size:
        return image
    ratio = max(current_width / width, current_height / height)
    resized = F.interpolate(
        image,
        size=(int(current_height / ratio), int(current_width / ratio)),
        mode="bilinear",
        align_corners=False,
    )
    return F.pad(resized, (width - resized.shape[-1], 0, height - resized.shape[-2], 0))


def detach_tree(value):
    if isinstance(value, Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(detach_tree(item) for item in value)
    if isinstance(value, list):
        return [detach_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: detach_tree(item) for key, item in value.items()}
    return value


@dataclass
class VLMContext:
    final: Tensor
    layers: tuple[Tensor, ...]
    mask: Tensor

    def detached(self) -> VLMContext:
        return VLMContext(detach_tree(self.final), detach_tree(self.layers), self.mask.detach())


@dataclass
class EARPlan:
    mean: Tensor
    covariance: Tensor
    trajectory_variance: Tensor
    segment_confidence: Tensor
    segment_available: Tensor


@dataclass
class ActionPlan:
    params: Tensor
    control_mode: Tensor
    ear: EARPlan
    ear_phase: Tensor


class FlowExpert(nn.Module):
    """Flow transformer with VLM cross-attention and optional EAR-parameter attention."""

    def __init__(
        self,
        *,
        value_dim: int,
        horizon: int,
        layers: int,
        context_dim: int,
        config: EARSmolVLAConfig,
        guidance_dim: int = 0,
    ):
        super().__init__()
        self.hidden_size = config.expert_hidden_size
        self.heads = config.expert_heads
        self.value_in = nn.Linear(value_dim, self.hidden_size)
        self.value_out = nn.Linear(self.hidden_size, value_dim)
        self.value_position = nn.Parameter(torch.empty(1, horizon, self.hidden_size))
        self.time_in = nn.Linear(self.hidden_size, self.hidden_size)
        self.time_out = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.normal_(self.value_position, std=0.02)

        def decoder_layer() -> nn.TransformerDecoderLayer:
            return nn.TransformerDecoderLayer(
                d_model=self.hidden_size,
                nhead=self.heads,
                dim_feedforward=config.expert_ffn_size,
                dropout=config.expert_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )

        self.layers = nn.ModuleList(decoder_layer() for _ in range(layers))
        self.context_projections = nn.ModuleList(
            nn.Linear(context_dim, self.hidden_size) for _ in range(layers)
        )
        self.guidance_in = nn.Linear(guidance_dim, self.hidden_size) if guidance_dim else None
        self.guidance_support = nn.Linear(2, self.hidden_size) if guidance_dim else None
        self.guidance_query_phase = nn.Linear(1, self.hidden_size) if guidance_dim else None
        self.guidance_role = nn.Embedding(3, self.hidden_size) if guidance_dim else None
        self.guidance_null = (
            nn.Parameter(torch.empty(1, 1, self.hidden_size)) if guidance_dim else None
        )
        self.guidance_attention = (
            nn.ModuleList(
                nn.MultiheadAttention(
                    self.hidden_size,
                    self.heads,
                    dropout=config.expert_dropout,
                    batch_first=True,
                )
                for _ in range(layers)
            )
            if guidance_dim
            else None
        )
        self.guidance_norm = (
            nn.ModuleList(nn.LayerNorm(self.hidden_size) for _ in range(layers))
            if guidance_dim
            else None
        )
        if self.guidance_null is not None:
            nn.init.normal_(self.guidance_null, std=0.02)
        segments = horizon - 2
        query_fraction = torch.cat(
            [torch.tensor([0.0, 0.5 / segments]), torch.arange(1, segments + 1) / segments]
        )
        self.register_buffer("query_fraction", query_fraction, persistent=False)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.gradient_checkpointing = config.gradient_checkpointing

    def _guidance_memory(
        self,
        guidance: ParameterGuidance,
        query_count: int,
    ) -> tuple[Tensor, Tensor]:
        assert self.guidance_in is not None
        assert self.guidance_support is not None
        assert self.guidance_role is not None
        assert self.guidance_null is not None
        memory = self.guidance_in(guidance.tokens.to(self.guidance_in.weight.dtype))
        memory = memory + self.guidance_support(guidance.support.to(memory.dtype))
        memory = memory + self.guidance_role(guidance.role).to(memory.dtype)
        memory = torch.where(guidance.available[..., None], memory, self.guidance_null.to(memory))

        confidence = torch.where(
            guidance.available,
            guidance.confidence.clamp_min(1e-6),
            torch.ones_like(guidance.confidence),
        )
        bias = confidence.log()[:, None].expand(-1, query_count, -1)
        bias = bias.masked_fill(~guidance.valid[:, None], -1e4)
        return memory, bias.repeat_interleave(self.heads, dim=0)

    def forward(
        self,
        values: Tensor,
        time: Tensor,
        context: VLMContext,
        guidance: ParameterGuidance | None = None,
        guidance_phase: Tensor | None = None,
        guidance_phase_span: float = 1.0,
    ) -> Tensor:
        hidden = self.value_in(values.to(self.value_in.weight.dtype))
        hidden = hidden + self.value_position[:, : values.shape[1]].to(hidden.dtype)
        time_embedding = sinusoidal_embedding(time, self.hidden_size, values.device).to(hidden.dtype)
        hidden = hidden + self.time_out(F.silu(self.time_in(time_embedding)))[:, None]
        guidance_memory = guidance_bias = None
        if guidance is not None:
            if self.guidance_attention is None:
                raise ValueError("guidance supplied to an expert without guidance attention")
            if guidance_phase is None:
                raise ValueError("guidance_phase is required with EAR guidance")
            assert self.guidance_query_phase is not None
            global_phase = (
                guidance_phase[:, None]
                + self.query_fraction[None, : values.shape[1]].to(guidance_phase)
                * guidance_phase_span
            ).clamp_max(1)
            hidden = hidden + self.guidance_query_phase(global_phase[..., None]).to(hidden.dtype)
            guidance_memory, guidance_bias = self._guidance_memory(
                guidance, values.shape[1]
            )

        source_layers = context.layers or (context.final,)
        memory_padding_mask = ~context.mask.bool()
        for index, (layer, projection) in enumerate(
            zip(self.layers, self.context_projections, strict=True)
        ):
            source_index = min(
                len(source_layers) - 1,
                round(index * (len(source_layers) - 1) / max(1, len(self.layers) - 1)),
            )
            memory = projection(source_layers[source_index]).to(hidden.dtype)

            def run_layer(
                current: Tensor,
                current_layer=layer,
                current_memory=memory,
                current_padding_mask=memory_padding_mask,
            ) -> Tensor:
                return current_layer(
                    current,
                    current_memory,
                    memory_key_padding_mask=current_padding_mask,
                )

            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden = torch.utils.checkpoint.checkpoint(run_layer, hidden, use_reentrant=False)
            else:
                hidden = run_layer(hidden)
            if guidance_memory is not None:
                assert self.guidance_attention is not None and self.guidance_norm is not None
                attended = self.guidance_attention[index](
                    self.guidance_norm[index](hidden),
                    guidance_memory.to(hidden.dtype),
                    guidance_memory.to(hidden.dtype),
                    attn_mask=guidance_bias.to(hidden.dtype),
                    need_weights=False,
                )[0]
                hidden = hidden + attended
        return self.value_out(self.norm(hidden)).float()


class EARSmolVLAModel(nn.Module):
    """FAST-supervised VLM, broad EAR spline flow, and short action spline flow."""

    def __init__(self, config: EARSmolVLAConfig):
        super().__init__()
        from peft import LoraConfig, inject_adapter_in_model, prepare_model_for_kbit_training
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

        self.config = config
        quantization = None
        if config.quantize_language_base_int8:
            quantization = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=LANGUAGE_INT8_SKIP_MODULES,
            )
        if config.load_vlm_weights:
            kwargs: dict[str, Any] = {
                "revision": config.vlm_revision,
                "dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
            }
            if quantization is not None:
                kwargs["quantization_config"] = quantization
                kwargs["device_map"] = (
                    {"": config.device} if str(config.device).startswith("cuda") else "auto"
                )
            self.vlm = AutoModelForImageTextToText.from_pretrained(config.vlm_model_name, **kwargs)
        else:
            base_config = AutoConfig.from_pretrained(
                config.vlm_model_name, revision=config.vlm_revision
            )
            self.vlm = AutoModelForImageTextToText.from_config(base_config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.vlm_model_name, revision=config.vlm_revision
        )
        self._text_model().layers = self._text_model().layers[: config.num_vlm_layers]
        context_dim = self.vlm.config.text_config.hidden_size

        for parameter in self.vlm.parameters():
            parameter.requires_grad = False
        if quantization is not None:
            self.vlm = prepare_model_for_kbit_training(
                self.vlm, use_gradient_checkpointing=config.gradient_checkpointing
            )
        use_language_lora = getattr(config, "use_language_lora", True)
        use_vision_lora = getattr(config, "use_vision_lora", True)
        if use_language_lora or use_vision_lora:
            lora = LoraConfig(
                r=config.lm_lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                rank_pattern={r".*vision_model.*": config.vision_lora_rank},
            )
            self.vlm = inject_adapter_in_model(lora, self.vlm, adapter_name="ear")
        if getattr(config, "train_vision_encoder_full", False):
            for parameter in self._vlm_model().vision_model.parameters():
                parameter.requires_grad = True
        for parameter in self._vlm_model().connector.parameters():
            parameter.requires_grad = True
        if config.gradient_checkpointing:
            for module in (self._text_model(), self._vlm_model().vision_model):
                enable = getattr(module, "gradient_checkpointing_enable", None)
                if enable is not None:
                    enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        self.state_proj = nn.Linear(config.max_state_dim, context_dim)
        self.pose_dim = getattr(config, "pose_dim", POSE_DIM)
        self.quaternion_slices = (
            (slice(3, 7),) if getattr(config, "embodiment", "robocasa") == "libero"
            else (slice(3, 7), slice(10, 14))
        )
        self.ear_spline = QuadraticSpline(
            config.ear_segments, config.ear_horizon, self.quaternion_slices
        )
        self.action_spline = QuadraticSpline(
            config.action_segments, config.action_horizon, self.quaternion_slices
        )
        self.ear_expert = FlowExpert(
            value_dim=config.spline_dim,
            horizon=config.ear_parameter_count,
            layers=config.spline_reasoner_layers,
            context_dim=context_dim,
            config=config,
        )
        self.action_expert = FlowExpert(
            value_dim=config.spline_dim,
            horizon=config.action_parameter_count,
            layers=config.action_expert_layers,
            context_dim=context_dim,
            guidance_dim=config.spline_dim,
            config=config,
        )
        self.mode_head = (
            nn.Linear(context_dim, 1) if getattr(config, "has_control_mode", True) else None
        )
        self.register_buffer("_zero", torch.tensor(0.0), persistent=False)
        self._validate_backbone_partition()
        for parameter in self.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(torch.bfloat16)

    def _vlm_model(self):
        return self.vlm.model

    def _text_model(self):
        return self._vlm_model().text_model

    def _validate_backbone_partition(self) -> None:
        full_vision = getattr(self.config, "train_vision_encoder_full", False)
        unexpected = [
            name
            for name, parameter in self.vlm.named_parameters()
            if parameter.requires_grad
            and "lora_" not in name
            and ".connector." not in name
            and not (full_vision and "vision_model" in name)
        ]
        if unexpected:
            raise RuntimeError(f"Unexpected trainable VLM base parameters: {unexpected[:8]}")
        lora_names = [
            name
            for name, parameter in self.vlm.named_parameters()
            if parameter.requires_grad and "lora_" in name
        ]
        if getattr(self.config, "use_language_lora", True) and not any(
            "text_model" in name for name in lora_names
        ):
            raise RuntimeError("Language attention LoRA matched no parameters")
        if getattr(self.config, "use_vision_lora", True) and not any(
            "vision_model" in name for name in lora_names
        ):
            raise RuntimeError("Vision attention LoRA matched no parameters")
        if not getattr(self.config, "use_language_lora", True) and any(
            parameter.requires_grad for parameter in self._text_model().parameters()
        ):
            raise RuntimeError("LIBERO language model must remain frozen")
        if full_vision and not all(
            parameter.requires_grad for parameter in self._vlm_model().vision_model.parameters()
        ):
            raise RuntimeError("LIBERO vision encoder must be fully trainable")

    def embed_images(
        self, images: list[Tensor], image_masks: list[Tensor]
    ) -> tuple[list[Tensor], list[Tensor]]:
        embeddings, masks = [], []
        for image, image_mask in zip(images, image_masks, strict=True):
            hidden = self._vlm_model().vision_model(
                pixel_values=image.to(dtype=self._vlm_model().vision_model.dtype),
                patch_attention_mask=None,
            ).last_hidden_state
            hidden = self._vlm_model().connector(hidden) * math.sqrt(hidden.shape[-1])
            embeddings.append(hidden)
            masks.append(image_mask[:, None].expand(hidden.shape[:2]))
        return embeddings, masks

    def _embed_tokens(self, tokens: Tensor) -> Tensor:
        hidden = self._text_model().get_input_embeddings()(tokens)
        return hidden * math.sqrt(hidden.shape[-1])

    def _run_vlm(self, embeddings: Tensor, mask: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        output = self._text_model()(
            inputs_embeds=embeddings,
            attention_mask=mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        return output.last_hidden_state, tuple(output.hidden_states[1:])

    def make_prefix(
        self,
        images: list[Tensor],
        image_masks: list[Tensor],
        task_tokens: Tensor,
        task_mask: Tensor,
        state: Tensor,
        fast_tokens: Tensor | None = None,
        fast_mask: Tensor | None = None,
    ) -> tuple[VLMContext, dict[str, Tensor]]:
        image_embeddings, masks = self.embed_images(images, image_masks)
        embeddings = image_embeddings + [self._embed_tokens(task_tokens)]
        masks += [task_mask.bool()]
        state_embedding = self.state_proj(state.to(self.state_proj.weight.dtype))[:, None]
        embeddings.append(state_embedding.to(embeddings[0].dtype))
        masks.append(torch.ones(state.shape[0], 1, dtype=torch.bool, device=state.device))
        prefix_end = sum(item.shape[1] for item in embeddings)
        if fast_tokens is not None:
            embeddings.append(self._embed_tokens(fast_tokens))
            masks.append(fast_mask.bool())
        all_embeddings = torch.cat(embeddings, dim=1)
        all_mask = torch.cat(masks, dim=1)
        final, layers = self._run_vlm(all_embeddings, all_mask)
        context = VLMContext(
            final[:, :prefix_end],
            tuple(layer[:, :prefix_end] for layer in layers),
            all_mask[:, :prefix_end],
        )
        return context, {
            "hidden": final,
            "fast_start": torch.tensor(prefix_end, device=state.device),
        }

    def _fast_loss(
        self, sequence: dict[str, Tensor], tokens: Tensor, code_mask: Tensor
    ) -> Tensor:
        logits = self.vlm.lm_head(sequence["hidden"]).float()
        labels = torch.full(logits.shape[:2], -100, dtype=torch.long, device=logits.device)
        start = int(sequence["fast_start"])
        labels[:, start : start + tokens.shape[1]] = torch.where(code_mask.bool(), tokens, -100)
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )

    @staticmethod
    def _flow_batch(
        target: Tensor, time: Tensor | None = None, noise: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        noise = sample_noise(tuple(target.shape), target.device) if noise is None else noise
        time = sample_time_beta(target.shape[0], target.device) if time is None else time
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * target
        return x_t, noise - target, time

    def forward(
        self,
        *,
        images: list[Tensor],
        image_masks: list[Tensor],
        task_tokens: Tensor,
        task_mask: Tensor,
        fast_tokens: Tensor,
        fast_mask: Tensor,
        fast_code_mask: Tensor,
        state: Tensor,
        ear_target: Tensor,
        action_target: Tensor,
        control_mode_target: Tensor,
    ) -> dict[str, Tensor]:
        with torch.set_grad_enabled(self.config.train_vlm_objective and torch.is_grad_enabled()):
            context, sequence = self.make_prefix(
                images, image_masks, task_tokens, task_mask, state, fast_tokens, fast_mask
            )
            fast_loss = (
                self._fast_loss(sequence, fast_tokens, fast_code_mask)
                if self.config.train_vlm_objective
                else self._zero.detach()
            )
        expert_context = context.detached() if self.config.detach_expert_context else context
        ear_params = self.ear_spline.fit(ear_target)
        action_params = self.action_spline.fit(action_target)

        if self.config.train_spline_reasoner:
            ear_x, ear_velocity, ear_time = self._flow_batch(ear_params)
            ear_prediction = self.ear_expert(ear_x, ear_time, expert_context)
            ear_loss = F.mse_loss(ear_prediction, ear_velocity)
            with torch.no_grad():
                predicted_ear_params = ear_x - ear_time[:, None, None] * ear_prediction
                ear_metrics = spline_trajectory_errors(
                    self.ear_spline.decode(predicted_ear_params),
                    ear_target,
                    self.quaternion_slices,
                )
        else:
            ear_loss = self._zero.detach()
            ear_metrics = {}

        phase = torch.zeros(ear_params.shape[0], device=ear_params.device)
        guidance = self.ear_spline.select_parameter_guidance(
            ear_params, phase, self.config.action_phase_span
        )
        guidance.available = random_teacher_guidance_mask(
            guidance.available, self.config.guidance_mask_ratios
        )
        if self.config.detach_action_guidance:
            guidance = guidance.detached()
        if self.config.train_action_expert:
            action_x, action_velocity, action_time = self._flow_batch(action_params)
            action_prediction = self.action_expert(
                action_x,
                action_time,
                expert_context,
                guidance,
                phase,
                self.config.action_phase_span,
            )
            action_loss = F.mse_loss(action_prediction, action_velocity)
            with torch.no_grad():
                predicted_action_params = (
                    action_x - action_time[:, None, None] * action_prediction
                )
                action_metrics = spline_trajectory_errors(
                    self.action_spline.decode(predicted_action_params),
                    action_target,
                    self.quaternion_slices,
                )
            if self.mode_head is not None:
                mode_midpoint = sum(self.config.control_mode_values) / 2
                mode_loss = F.binary_cross_entropy_with_logits(
                    self.mode_head(expert_context.final[:, -1]).float(),
                    (control_mode_target > mode_midpoint).float(),
                )
            else:
                mode_loss = self._zero.detach()
        else:
            action_loss = mode_loss = self._zero.detach()
            action_metrics = {}
        total = (
            self.config.fast_loss_weight * fast_loss
            + self.config.spline_loss_weight * ear_loss
            + self.config.action_loss_weight * action_loss
            + self.config.mode_loss_weight * mode_loss
        )
        return {
            "loss": total,
            "fast_loss": fast_loss,
            "ear_loss": ear_loss,
            "action_loss": action_loss,
            "mode_loss": mode_loss,
            **{f"ear_{key}": value for key, value in ear_metrics.items()},
            **{f"action_{key}": value for key, value in action_metrics.items()},
        }

    def _integrate(self, velocity_fn, shape: tuple[int, ...], device: torch.device) -> Tensor:
        value = sample_noise(shape, device)
        delta = -1.0 / self.config.num_flow_steps
        for step in range(self.config.num_flow_steps):
            time = torch.full((shape[0],), 1.0 + step * delta, device=device)
            value = value + delta * velocity_fn(value, time)
        return value

    def sample_ear(self, context: VLMContext) -> Tensor:
        shape = (
            context.final.shape[0],
            self.config.ear_parameter_count,
            self.config.spline_dim,
        )
        return self._integrate(
            lambda value, time: self.ear_expert(value, time, context),
            shape,
            context.final.device,
        )

    def sample_action(
        self, context: VLMContext, guidance: ParameterGuidance, phase: Tensor
    ) -> Tensor:
        shape = (
            context.final.shape[0],
            self.config.action_parameter_count,
            self.config.spline_dim,
        )
        return self._integrate(
            lambda value, time: self.action_expert(
                value,
                time,
                context,
                guidance,
                phase,
                self.config.action_phase_span,
            ),
            shape,
            context.final.device,
        )

    @torch.no_grad()
    def plan(
        self,
        images: list[Tensor],
        image_masks: list[Tensor],
        task_tokens: Tensor,
        task_mask: Tensor,
        state: Tensor,
        current_pose: Tensor,
    ) -> ActionPlan:
        context, _ = self.make_prefix(images, image_masks, task_tokens, task_mask, state)
        context = context.detached()
        sample_count = self.config.mc_samples
        noisy_images = [
            torch.cat(
                [
                    (image + torch.randn_like(image) * self.config.mc_image_noise_std).clamp(-1, 1)
                    for _ in range(sample_count)
                ]
            )
            for image in images
        ]
        repeated_masks = [mask.repeat(sample_count) for mask in image_masks]
        noisy_state = torch.cat(
            [
                state + torch.randn_like(state) * self.config.mc_state_noise_std
                for _ in range(sample_count)
            ]
        )
        sample_context, _ = self.make_prefix(
            noisy_images,
            repeated_masks,
            task_tokens.repeat(sample_count, 1),
            task_mask.repeat(sample_count, 1),
            noisy_state,
        )
        flat_samples = self.sample_ear(sample_context.detached())
        sample_tensor = flat_samples.reshape(
            sample_count,
            state.shape[0],
            self.config.ear_parameter_count,
            self.config.spline_dim,
        )
        mean, covariance, variance = self.ear_spline.propagate_parameter_covariance(sample_tensor)
        confidence, available = self.ear_spline.segment_confidence(
            variance,
            self.config.confidence_temperature,
            self.config.confidence_variance_threshold,
        )
        ear = EARPlan(mean, covariance, variance, confidence, available)
        _, phase, _ = self.ear_spline.closest_point_field(
            mean[..., : self.pose_dim], current_pose
        )
        guidance = self.ear_spline.select_parameter_guidance(
            mean,
            phase,
            self.config.action_phase_span,
            confidence,
            available,
        ).detached()
        action_params = self.sample_action(context, guidance, phase)
        if self.mode_head is None:
            mode = torch.zeros(context.final.shape[0], 1, device=context.final.device)
        else:
            mode_logits = self.mode_head(context.final[:, -1]).float()
            low_mode, high_mode = self.config.control_mode_values
            mode = torch.where(
                mode_logits.sigmoid() >= 0.5,
                torch.full_like(mode_logits, high_mode),
                torch.full_like(mode_logits, low_mode),
            )
        return ActionPlan(action_params, mode, ear, phase)


class EARSmolVLAPolicy(nn.Module):
    config_class = EARSmolVLAConfig
    name = "ear_smolvla"
    supports_gradient_checkpointing = True

    def __init__(self, config: EARSmolVLAConfig):
        super().__init__()
        self.config = config
        self.model = EARSmolVLAModel(config)
        self._planner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ear-planner")
        self.reset()

    def reset(self) -> None:
        if getattr(self, "_pending", None) is not None:
            self._pending.cancel()
        self._active: ActionPlan | None = None
        self._pending: Future[ActionPlan] | None = None
        self._steps_on_active = 0
        self._last_velocity: Tensor | None = None
        self._last_gripper: Tensor | None = None
        self.last_planning_error: Exception | None = None

    def supports_text_generation(self) -> bool:
        return False

    def prepare_state(self, batch: dict[str, Tensor]) -> Tensor:
        source = batch[self.config.state_key]
        return pad_vector(source[:, 0] if source.ndim == 3 else source, self.config.max_state_dim)

    def prepare_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        missing = [key for key in self.config.image_keys if key not in batch]
        if missing:
            raise ValueError(f"Training/inference batch is missing cameras: {missing}")
        images, masks = [], []
        for key in self.config.image_keys:
            image = batch[key][:, -1] if batch[key].ndim == 5 else batch[key]
            image = resize_with_pad(image, self.config.resize_imgs_with_padding)
            images.append(image * 2.0 - 1.0)
            masks.append(
                batch.get(
                    f"{key}_padding_mask",
                    torch.ones(image.shape[0], dtype=torch.bool, device=image.device),
                ).bool()
            )
        return images, masks

    def _planning_inputs(self, batch: dict[str, Tensor]):
        images, image_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        current_pose = batch.get(SPLINE_CURRENT_POSE)
        if current_pose is None:
            raw = batch[self.config.state_key]
            current_pose = (raw[:, 0] if raw.ndim == 3 else raw)[:, : self.model.pose_dim]
        return (
            images,
            image_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state,
            current_pose,
        )

    def _anchor(self, plan: ActionPlan, batch: dict[str, Tensor]) -> ActionPlan:
        raw = batch[self.config.state_key]
        raw = raw[:, 0] if raw.ndim == 3 else raw
        current_pose = batch.get(SPLINE_CURRENT_POSE, raw[:, : self.model.pose_dim])
        if self._last_gripper is not None:
            gripper = self._last_gripper.to(current_pose)
        elif getattr(self.config, "embodiment", "robocasa") == "libero":
            gripper = torch.full_like(
                current_pose[:, :1], self.config.initial_gripper_command
            )
        else:
            gripper = (
                raw[:, self.model.pose_dim : self.model.pose_dim + 1]
                if raw.shape[-1] > self.model.pose_dim
                else torch.full_like(
                    current_pose[:, :1],
                    getattr(self.config, "initial_gripper_command", 0.0),
                )
            )
        start = torch.cat([current_pose, gripper], dim=-1).to(plan.params)
        velocity = self._last_velocity
        if velocity is None or velocity.shape != start.shape:
            velocity = torch.zeros_like(start)
        duration = self.config.action_horizon / self.config.dataset_fps
        segment_duration = duration / self.config.action_segments
        plan.params = self.model.action_spline.constrain_start(
            plan.params, start, velocity.to(plan.params), segment_duration
        )
        return plan

    def forward(self, batch: dict[str, Tensor], **kwargs):
        required = (
            ACTION_TOKENS,
            ACTION_TOKEN_MASK,
            ACTION_CODE_TOKEN_MASK,
            EAR_SPLINE_TARGET,
            ACTION_SPLINE_TARGET,
            CONTROL_MODE_TARGET,
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"EAR-SmolVLA training batch is missing: {missing}")
        images, image_masks = self.prepare_images(batch)
        losses = self.model(
            images=images,
            image_masks=image_masks,
            task_tokens=batch[OBS_LANGUAGE_TOKENS],
            task_mask=batch[OBS_LANGUAGE_ATTENTION_MASK],
            fast_tokens=batch[ACTION_TOKENS],
            fast_mask=batch[ACTION_TOKEN_MASK],
            fast_code_mask=batch[ACTION_CODE_TOKEN_MASK],
            state=self.prepare_state(batch),
            ear_target=batch[EAR_SPLINE_TARGET],
            action_target=batch[ACTION_SPLINE_TARGET],
            control_mode_target=batch[CONTROL_MODE_TARGET],
        )
        metrics = {key: float(value.detach()) for key, value in losses.items() if key != "loss"}
        metrics["loss"] = float(losses["loss"].detach())
        return losses["loss"], metrics

    def get_optim_params(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def _field_to_action(
        self, field: Tensor, current_pose: Tensor, gripper: Tensor, control_mode: Tensor
    ) -> Tensor:
        if getattr(self.config, "embodiment", "robocasa") == "libero":
            return field_to_libero_action(field, current_pose, gripper)
        return field_to_robocasa_action(field, current_pose, gripper, control_mode)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        plan = self._anchor(self.model.plan(*self._planning_inputs(batch)), batch)
        trajectory = self.model.action_spline.decode(plan.params)
        pose_dim = self.model.pose_dim
        next_pose = torch.cat(
            [trajectory[:, 1:, :pose_dim], trajectory[:, -1:, :pose_dim]], dim=1
        )
        pose_field = (next_pose - trajectory[..., :pose_dim]) * self.config.dataset_fps
        return self._field_to_action(
            pose_field,
            trajectory[..., :pose_dim],
            trajectory[..., pose_dim : pose_dim + 1],
            plan.control_mode[:, None].expand(-1, trajectory.shape[1], -1),
        )

    def _submit_plan(self, batch: dict[str, Tensor]) -> None:
        inputs = detach_tree(self._planning_inputs(batch))
        self._pending = self._planner.submit(self._plan_in_background, inputs)

    def _plan_in_background(self, inputs) -> ActionPlan:
        plan = self.model.plan(*inputs)
        if plan.params.is_cuda:
            torch.cuda.current_stream(plan.params.device).synchronize()
        return plan

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if self._active is None:
            self._active = self._anchor(self.model.plan(*self._planning_inputs(batch)), batch)
            self._steps_on_active = 0
        if self._pending is None:
            self._submit_plan(batch)
        if self._pending is not None and self._pending.done():
            error = self._pending.exception()
            if error is not None:
                self.last_planning_error = error
                self._pending = None
            elif self._steps_on_active >= self.config.n_action_steps:
                self._active = self._anchor(self._pending.result(), batch)
                self._pending = None
                self._steps_on_active = 0
        raw = batch[self.config.state_key]
        raw = raw[:, 0] if raw.ndim == 3 else raw
        current_pose = batch.get(SPLINE_CURRENT_POSE, raw[:, : self.model.pose_dim])
        progression = self.config.field_progression
        if self._steps_on_active >= self.config.action_horizon:
            progression = 0.0
        field, phase, _ = self.model.action_spline.closest_point_field(
            self._active.params[..., : self.model.pose_dim],
            current_pose,
            self.config.field_attraction,
            progression,
        )
        gripper = self.model.action_spline.evaluate(self._active.params, phase)[
            ..., self.model.pose_dim : self.model.pose_dim + 1
        ]
        action = self._field_to_action(field, current_pose, gripper, self._active.control_mode)
        pose_velocity = project_quaternion_velocity(
            field, current_pose, self.model.quaternion_slices
        )
        self._last_velocity = torch.cat(
            [pose_velocity, torch.zeros_like(gripper)], dim=-1
        ).detach()
        self._last_gripper = gripper.detach()
        self._steps_on_active += 1
        return action

    def save_pretrained(self, directory: str | Path) -> None:
        path = Path(directory)
        self.config.save(path)
        trainable = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        torch.save(
            {
                name: value.detach().cpu()
                for name, value in self.state_dict().items()
                if name in trainable
            },
            path / "trainable_state.pt",
        )

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> EARSmolVLAPolicy:
        policy = cls(cls.config_class.load(directory))
        state = torch.load(
            Path(directory) / "trainable_state.pt", map_location="cpu", weights_only=True
        )
        policy.load_state_dict(state, strict=False)
        return policy
