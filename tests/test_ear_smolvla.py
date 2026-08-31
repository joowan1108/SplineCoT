import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ear_smolvla.benchmark_libero_latency import latency_stats
from ear_smolvla.config import EARSmolVLAConfig
from ear_smolvla.eval_libero import rollout_video_path, xyzw_to_axis_angle
from ear_smolvla.libero import (
    LIBEROBatchProcessor,
    axis_angle_to_quaternion,
)
from ear_smolvla.libero_config import LIBEROConfig
from ear_smolvla.metrics import (
    perturbation_recovery,
    spline_trajectory_errors,
    trajectory_metrics,
)
from ear_smolvla.model import (
    LANGUAGE_INT8_SKIP_MODULES,
    EARSmolVLAModel,
    FlowExpert,
    VLMContext,
)
from ear_smolvla.processor import BatchProcessor, build_pose_targets
from ear_smolvla.spline import (
    QuadraticSpline,
    field_to_libero_action,
    make_quaternion_signs_continuous,
    project_quaternion_velocity,
    soften_partial_guidance,
)
from ear_smolvla.train import stage_config
from ear_smolvla.train_libero import LIBEROHDF5Sampler, checkpoint_steps
from ear_smolvla.visualize_libero_splines import TOP_VIEW_STYLES


def small_config() -> EARSmolVLAConfig:
    return EARSmolVLAConfig(
        device="cpu",
        quantize_language_base_int8=False,
        expert_hidden_size=24,
        expert_heads=4,
        expert_ffn_size=48,
        gradient_checkpointing=False,
    )


def test_config_is_the_locked_dual_spline_architecture():
    config = small_config()
    assert (config.num_vlm_layers, config.spline_reasoner_layers, config.action_expert_layers) == (16, 16, 16)
    assert (config.ear_segments, config.ear_parameter_count, config.ear_horizon) == (14, 16, 64)
    assert (config.action_segments, config.action_parameter_count, config.action_horizon) == (6, 8, 16)
    assert config.n_action_steps == 4 and config.dataset_fps == 20
    assert config.spline_dim == 15 and config.mc_samples >= 2


def test_latency_summary_reports_percentiles_in_milliseconds():
    summary = latency_stats([1.0, 2.0, 3.0, 4.0])
    assert summary["mean"] == 2.5 and summary["median"] == 2.5
    assert summary["min"] == 1.0 and summary["max"] == 4.0
    assert 3.0 < summary["p95"] < 4.0


def test_guidance_curriculum_moves_continuously_from_teacher_to_final_mix():
    config = small_config()
    assert config.guidance_probabilities(0) == (0.5, 0.5, 0.0)
    assert config.guidance_probabilities(config.predicted_guidance_full_fraction) == pytest.approx(
        config.guidance_mask_ratios
    )
    middle = config.guidance_probabilities(
        (config.predicted_guidance_start_fraction + config.predicted_guidance_full_fraction) / 2
    )
    assert 0 < middle[2] < config.guidance_mask_ratios[2]


def test_libero_profile_is_additive_and_trains_only_full_vision():
    config = LIBEROConfig(device="cpu")
    assert (config.state_dim, config.action_dim, config.spline_dim, config.pose_dim) == (8, 7, 8, 7)
    assert len(config.image_keys) == 2 and config.has_control_mode is False
    assert not config.use_language_lora and not config.use_vision_lora
    assert not config.quantize_language_base_int8
    assert config.train_vision_encoder_full
    assert (config.ear_horizon, config.action_horizon, config.dataset_fps) == (64, 16, 20)


def test_libero_xyzw_quaternion_conversion():
    assert np.allclose(xyzw_to_axis_angle(np.array([0, 0, 0, 1])), 0)
    assert np.allclose(xyzw_to_axis_angle(np.array([1, 0, 0, 0])), np.array([np.pi, 0, 0]), atol=1e-6)


def test_rollout_videos_only_cover_first_episode_of_first_three_tasks(tmp_path):
    assert rollout_video_path(tmp_path, "libero_spatial", 2, "task", 0) is not None
    assert rollout_video_path(tmp_path, "libero_spatial", 3, "task", 0) is None
    assert rollout_video_path(tmp_path, "libero_spatial", 0, "task", 1) is None


def test_top_view_splines_have_distinct_styles():
    assert len({style["color"] for style in TOP_VIEW_STYLES.values()}) == 4
    assert len({style["linestyle"] for style in TOP_VIEW_STYLES.values()}) == 4
    assert len({style["marker"] for style in TOP_VIEW_STYLES.values()}) == 4


def test_three_checkpoints_cover_the_full_training_run():
    assert checkpoint_steps(160_000, 16, 3) == (53_328, 106_656, 160_000)


def test_int8_quantization_excludes_the_full_vision_and_connector_trees():
    from transformers.quantizers.quantizers_utils import should_convert_module

    assert not should_convert_module(
        "model.vision_model.encoder.layers.0.self_attn.q_proj", LANGUAGE_INT8_SKIP_MODULES
    )
    assert not should_convert_module("model.connector.modality_projection", LANGUAGE_INT8_SKIP_MODULES)
    assert should_convert_module("model.text_model.layers.0.self_attn.q_proj", LANGUAGE_INT8_SKIP_MODULES)


def test_only_two_training_stages_remain():
    warmup = stage_config("vlm_warmup", small_config())
    assert warmup.train_vlm_objective and not warmup.train_spline_reasoner
    joint = stage_config("joint_teacher_forced", warmup)
    assert joint.train_vlm_objective and joint.train_spline_reasoner and joint.train_action_expert


def test_broad_ear_resamples_but_short_action_keeps_real_ticks():
    state = torch.zeros(1, 40, 16)
    state[..., 0] = torch.arange(40)
    state[..., 3] = state[..., 10] = 1
    current, ear, action, mask = build_pose_targets(state, ear_samples=64, action_samples=16)
    assert current.shape == (1, 14) and ear.shape == (1, 64, 14)
    assert action.shape == (1, 16, 14) and mask.all()
    torch.testing.assert_close(action[0, :, 0], torch.arange(16, dtype=action.dtype))
    torch.testing.assert_close(ear[0, -1, 0], torch.tensor(39.0))


def test_quadratic_spline_is_c1_and_start_is_conditioned_during_fitting():
    spline = QuadraticSpline(segments=3, samples=32)
    params = torch.randn(2, 5, 4)
    controls = spline.segment_controls(params)
    left = 2 * (controls[:, :-1, 2] - controls[:, :-1, 1])
    right = 2 * (controls[:, 1:, 1] - controls[:, 1:, 0])
    torch.testing.assert_close(left, right)

    trajectory = torch.randn(2, 32, 4)
    fitted = spline.fit(trajectory, constrain_start=True)
    torch.testing.assert_close(fitted[:, 0], trajectory[:, 0])
    torch.testing.assert_close(spline.decode(fitted)[:, 0], trajectory[:, 0])


def test_closest_point_always_considers_boundary_and_attracts_back():
    spline = QuadraticSpline(segments=1, samples=16)
    params = torch.tensor([[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]])
    state = torch.tensor([[-0.3, 0.2]])
    field, phase, distance = spline.closest_point_field(params, state)
    torch.testing.assert_close(phase, torch.zeros_like(phase))
    torch.testing.assert_close(distance, torch.tensor([(0.3**2 + 0.2**2) ** 0.5]))
    assert field[0, 1] < 0


def test_overlap_selection_exposes_each_segments_three_real_controls():
    spline = QuadraticSpline(segments=4, samples=20)
    params = torch.arange(6, dtype=torch.float32)[None, :, None]
    guidance = spline.select_parameter_guidance(params, torch.tensor([0.26]), phase_span=0.24)
    # Only segment E1 overlaps [0.26, 0.50); all three local controls are retained.
    assert guidance.tokens.shape == (1, 3, 1)
    torch.testing.assert_close(guidance.tokens[0, :, 0], torch.tensor([2.0, 3.0, 3.0]))
    assert guidance.valid.all() and (guidance.confidence == 1).all()
    assert guidance.role.tolist() == [[0, 1, 2]]


def test_partial_teacher_guidance_softens_whole_segments_without_removing_tokens():
    confidence = torch.ones(3, 6)
    valid = torch.ones(3, 6, dtype=torch.bool)
    softened = soften_partial_guidance(confidence, valid, torch.tensor([False, True, False]), 0.25, 0.25)
    assert (softened[0] == 1).all() and (softened[2] == 1).all()
    assert (softened[1] == 0.25).sum() == 3 and (softened[1] == 1).sum() == 3


def test_zero_confidence_removes_guidance_value_and_residual_strength():
    config = small_config()
    expert = FlowExpert(value_dim=3, horizon=4, layers=1, context_dim=10, guidance_dim=3, config=config)
    guidance = QuadraticSpline(2, 10).select_parameter_guidance(torch.randn(1, 4, 3), torch.zeros(1), 0.5)
    guidance.confidence.zero_()
    _, value, _, strength = expert._guidance_memory(guidance, query_count=4)
    torch.testing.assert_close(value, torch.zeros_like(value))
    torch.testing.assert_close(strength, torch.zeros_like(strength))


def test_flow_condition_keeps_the_current_pose_out_of_noise():
    target = torch.randn(2, 5, 3)
    noise = torch.randn_like(target)
    condition = torch.zeros_like(target, dtype=torch.bool)
    condition[:, 0] = True
    mixed, velocity, _ = EARSmolVLAModel._flow_batch(
        target,
        time=torch.tensor([0.25, 0.75]),
        noise=noise,
        condition_mask=condition,
    )
    torch.testing.assert_close(mixed[:, 0], target[:, 0])
    torch.testing.assert_close(velocity[:, 0], torch.zeros_like(velocity[:, 0]))


def test_mc_parameter_covariance_propagates_through_spline_basis():
    spline = QuadraticSpline(segments=2, samples=17)
    samples = torch.randn(8, 2, 4, 3)
    mean, covariance, trajectory_variance = spline.propagate_parameter_covariance(samples)
    assert mean.shape == (2, 4, 3) and covariance.shape == (2, 3, 4, 4)
    assert trajectory_variance.shape == (2, 17, 3)
    assert (trajectory_variance >= 0).all()
    decoded = torch.stack([spline.decode(sample) for sample in samples])
    torch.testing.assert_close(trajectory_variance, decoded.var(dim=0, unbiased=True), atol=2e-5, rtol=2e-5)


def test_action_parameter_attention_blocks_upstream_gradients():
    config = small_config()
    expert = FlowExpert(value_dim=3, horizon=4, layers=1, context_dim=10, guidance_dim=3, config=config)
    expert.guidance_query_phase.to(torch.bfloat16)
    source = torch.randn(2, 5, 10, requires_grad=True)
    context = VLMContext(source, (source,), torch.ones(2, 5, dtype=torch.bool)).detached()
    ear = (
        QuadraticSpline(2, 10)
        .select_parameter_guidance(torch.randn(2, 4, 3, requires_grad=True), torch.zeros(2), 0.5)
        .detached()
    )
    output = expert(torch.randn(2, 4, 3), torch.rand(2), context, ear, torch.zeros(2), 0.5).sum()
    output.backward()
    assert source.grad is None
    assert any(parameter.grad is not None for parameter in expert.parameters())


def test_quaternion_signs_and_metrics():
    pose = torch.zeros(1, 12, 14)
    pose[..., 3] = pose[..., 10] = 1
    pose[:, 6:, 3:7] *= -1
    continuous = make_quaternion_signs_continuous(pose)
    assert (continuous[:, 1:, 3:7] * continuous[:, :-1, 3:7]).sum(-1).min() >= 0
    velocity = project_quaternion_velocity(torch.randn(1, 14), continuous[:, 0])
    assert abs((velocity[:, 3:7] * continuous[:, 0, 3:7]).sum()) < 1e-6
    trajectory = torch.stack([torch.linspace(0, 1, 10), torch.zeros(10)], dim=-1)[None]
    assert trajectory_metrics(trajectory, trajectory)["reference_rmse"] == 0
    recovery = perturbation_recovery(torch.tensor([[0.4, 0.2, 0.05]]), 0.1, 0.05)
    assert recovery["recovery_success"] == 1


def test_decoded_spline_metrics_separate_pose_and_gripper_errors():
    reference = torch.zeros(1, 4, 8)
    reference[..., 3] = 1
    reference[..., 7] = -1
    prediction = reference.clone()
    prediction[..., :3] += 1
    prediction[..., 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    prediction[..., 7] = 1
    errors = spline_trajectory_errors(prediction, reference, (slice(3, 7),))
    assert torch.isclose(errors["translation_rmse"], torch.tensor(1.0))
    assert torch.isclose(errors["rotation_error_deg"], torch.tensor(180.0))
    assert torch.isclose(errors["gripper_rmse"], torch.tensor(2.0))
    assert errors["gripper_accuracy"] == 0


def test_fast_tokenizer_round_trip_contract():
    class Encoding(dict):
        __getattr__ = dict.__getitem__

    class TextTokenizer:
        vocab_size = 1000
        bos_token_id = 1
        all_special_ids = [0, 1, 2]

        def encode(self, text, add_special_tokens=False):
            return [10]

        def __call__(self, text, **kwargs):
            return Encoding(
                input_ids=torch.ones(len(text), 2, dtype=torch.long),
                attention_mask=torch.ones(len(text), 2),
            )

    class ActionTokenizer:
        def __init__(self):
            self.actions = []

        def __call__(self, action):
            self.actions.append(action.numpy())
            return [[5, 6]]

        def decode(self, tokens, *, time_horizon, action_dim):
            return np.concatenate(self.actions, axis=0)

    processor = BatchProcessor(small_config(), tokenizer=TextTokenizer(), action_tokenizer=ActionTokenizer())
    tokens, mask, code_mask = processor._tokenize_actions(torch.randn(2, 4, 3))
    assert tokens.shape == mask.shape == code_mask.shape == (2, 256)
    assert processor.reconstruction_rmse == 0


def test_libero_processor_builds_fixed_future_pose_splines():
    class Encoding(dict):
        __getattr__ = dict.__getitem__

    class TextTokenizer:
        vocab_size = 1000
        bos_token_id = 1
        all_special_ids = [0, 1, 2]

        def encode(self, text, add_special_tokens=False):
            return [10]

        def __call__(self, text, **kwargs):
            return Encoding(
                input_ids=torch.ones(len(text), 2, dtype=torch.long),
                attention_mask=torch.ones(len(text), 2),
            )

    class ActionTokenizer:
        def __init__(self):
            self.actions = []

        def __call__(self, action):
            self.actions.append(action.numpy())
            return [[5, 6]]

        def decode(self, tokens, *, time_horizon, action_dim):
            return np.concatenate(self.actions, axis=0)

    config = LIBEROConfig(device="cpu", quantize_language_base_int8=False)
    processor = LIBEROBatchProcessor(config, tokenizer=TextTokenizer(), action_tokenizer=ActionTokenizer())
    state = torch.zeros(1, 64, 8)
    state[..., 0] = torch.arange(64)
    actions = torch.zeros(1, 64, 7)
    batch = processor(
        {config.state_key: state, config.action_key: actions, "task": ["pick up mug"]},
        training=True,
    )
    assert batch["spline.ear_target"].shape == (1, 64, 8)
    assert batch["spline.action_target"].shape == (1, 16, 8)
    torch.testing.assert_close(batch["spline.ear_target"][0, -1, 0], torch.tensor(63.0))
    torch.testing.assert_close(batch["spline.action_target"][0, -1, 0], torch.tensor(15.0))


def test_libero_rotation_and_action_mapping():
    identity = axis_angle_to_quaternion(torch.zeros(2, 3))
    torch.testing.assert_close(identity[:, 0], torch.ones(2))
    torch.testing.assert_close(identity[:, 1:], torch.zeros(2, 3))
    pose = torch.zeros(2, 7)
    pose[:, 3] = 1
    action = field_to_libero_action(torch.randn(2, 7) * 10, pose, torch.ones(2, 1))
    assert action.shape == (2, 7)
    assert action.abs().max() <= 1


def test_libero_hdf5_sampler_reads_official_layout(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "pick_up_mug_demo.hdf5"
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        data.attrs["problem_info"] = json.dumps({"language_instruction": "pick up mug"})
        demo = data.create_group("demo_0")
        demo.attrs["num_samples"] = 3
        obs = demo.create_group("obs")
        obs.create_dataset("ee_states", data=np.zeros((3, 6), dtype=np.float32))
        obs.create_dataset("gripper_states", data=np.zeros((3, 2), dtype=np.float32))
        obs.create_dataset("agentview_rgb", data=np.zeros((3, 4, 4, 3), dtype=np.uint8))
        obs.create_dataset("eye_in_hand_rgb", data=np.zeros((3, 4, 4, 3), dtype=np.uint8))
        demo.create_dataset("actions", data=np.zeros((3, 7), dtype=np.float32))
    sampler = LIBEROHDF5Sampler(path, 5, rotate_images_180=False, seed=0)
    try:
        batch = sampler.sample_batch(2)
    finally:
        sampler.close()
    assert batch["observation.state"].shape == (2, 5, 8)
    assert batch["action"].shape == (2, 5, 7)
    assert batch["observation.images.image"].shape == (2, 3, 4, 4)
    assert batch["task"] == ["pick up mug", "pick up mug"]


def test_source_has_no_subtask_or_lerobot_runtime_dependency():
    source = "\n".join(path.read_text() for path in Path("src/ear_smolvla").glob("*.py"))
    assert "subtask" not in source.lower()
    assert "from lerobot" not in source and "import lerobot" not in source
