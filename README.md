# Dual-Spline EAR-SmolVLA

This repository implements the architecture in `hypothesis.md` without a
LeRobot runtime dependency. LeRobot and Spline Policy commits are pinned only
for provenance and tensor conventions.

## Final architecture

- SmolVLM2-2.2B: frozen INT8 language base, rank-8 language LoRA, rank-16
  vision LoRA, trainable connector, first 16 text layers.
- VLM objective: FAST action-token CE only. There is no subtask decoder or
  subtask CE.
- EAR: 16-layer flow expert producing 16 free parameters for a 14-segment C1
  quadratic spline over a 64-tick broad horizon.
- Action expert: 16-layer flow expert producing 8 free parameters for a
  6-segment C1 quadratic spline over the next 16 ticks.
- Execution: four ticks are normally consumed before handoff; the action
  spline flow field is recomputed from the latest state at 20 Hz.

Both splines use 15D structured values: base pose 7, end-effector pose 7, and
continuous gripper 1. Control mode is predicted separately. The external 12D
action order is base motion 4, control mode 1, end-effector translation 3,
end-effector rotation 3, and gripper 1.
`control_mode_values` defaults to `(-1, 1)` and should be changed if a dataset
uses another pair of categorical command values.

## Batch contract

Each prebatched `.pt` file contains the three configured camera tensors,
`observation.state` with shape `[B,H,16]`, `action` with shape `[B,H,12]`, and
`task` as a list of full instructions. An optional
`observation.state_is_pad` marks invalid suffixes. No subtask annotation is
read.

The EAR target resamples the whole available demonstration suffix to 64
points. The action target keeps the next 16 real control ticks, repeating the
last tick only when the episode suffix is shorter. Both targets are fitted by
the fixed quadratic-spline pseudoinverse.

## Remote RTX 3090 commands

```bash
uv sync --extra test
uv run pytest -q
uv run ear-memory-smoke --batch-size 4 --steps 3
```

```bash
uv run ear-train \
  --batches data/robocasa_batches \
  --stats data/normalization.pt \
  --stage joint_teacher_forced \
  --steps 30000 \
  --output outputs/ear_smolvla_joint
```

Use `vlm_warmup` first only when FAST representation pretraining is needed;
otherwise train `joint_teacher_forced`. Joint training begins with exact and
soft-confidence teacher EAR guidance at 1:1. Stop-gradient inferred MC-EAR
guidance ramps from zero after 10% of training to a final exact/soft/inferred
1:1:1 mixture at 50%. Low-confidence segments remain valid parameters but
contribute less to attention; there is no learned NULL guidance.

For inference, preprocess with `BatchProcessor(..., training=False)` and call
`policy.select_action(batch)` every control tick. The policy executes all 16
ticks of the Active action spline, then synchronously builds its replacement
from the first post-horizon observation. The new spline starts at that measured
pose, providing C0 plan continuity without cross-plan C1 constraints.

## LIBERO profile

The additive LIBERO profile leaves the RoboCasa configuration unchanged. It
uses two cameras, state 8, action 7, and an 8D `EEF pose7 + gripper1` spline.
The BF16 language model and LM head are fully frozen with no language LoRA;
FAST CE trains the complete vision encoder, connector, and state projection.

Download only the four suites used by the standard VLA benchmark (Spatial,
Object, Goal, and LIBERO-10), then train on their 40 tasks. LIBERO-90 is not
downloaded and is also excluded by the trainer unless `--include-libero-90`
is explicitly passed:

```bash
uv sync --extra test --extra libero
uv run ear-download-libero --output data/libero/vla40
uv run ear-train-libero \
  --data data/libero/vla40 \
  --steps 160000 \
  --batch-size 4 \
  --gradient-accumulation 16 \
  --num-checkpoints 3 \
  --sample-metrics-every 10000 \
  --output outputs/libero_vla40_160k
```

The reader consumes the official `data/demo_*/` HDF5 layout directly, so it
does not install LeRobot. It samples arbitrary episode times, keeps the next
64 real ticks for EAR and the next 16 for the action spline, and pads only at
episode ends. Raw HDF5 images are rotated 180 degrees by default; pass
`--no-rotate-images-180` if the local dataset was already corrected.
LIBERO actions remain in their native `[-1,1]` convention; an optional stats
file is applied only to the VLM state input.
The trainer additionally reports full 10-step inference-sampling metrics on
one example every `--sample-metrics-every` microsteps; pass `0` to disable it.

Simulator evaluation is an optional install and does not add LeRobot:

```bash
uv sync --extra eval
MUJOCO_GL=egl uv run ear-eval-libero \
  --checkpoint outputs/libero_spatial/checkpoint-160000 \
  --suite libero_spatial \
  --episodes 10 \
  --output results/libero_spatial.json
uv run ear-summarize-libero results/*.json --output results/summary.csv
```

Measure one-observation planning latency and the deployed action interval on a
single task. Conditional EAR uncertainty uses four flow samples from one shared
VLM representation:

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 uv run ear-benchmark-libero-latency \
  --checkpoint outputs/libero_spatial/checkpoint-160000 \
  --suite libero_spatial \
  --task-id 0 \
  --warmup 5 \
  --repeats 30 \
  --output results/latency/libero_spatial-task-0.json
```

The default execution interval is the complete 16-tick action-spline horizon
at 20 Hz. The JSON separates measured simulator compute time from the fixed
physical control duration.
