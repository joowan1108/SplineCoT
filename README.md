# Dual-Spline EAR-SmolVLA

This repository implements the architecture in `hypothesis.md` without a
LeRobot runtime dependency. LeRobot and Spline Policy commits are pinned only
for provenance and tensor conventions.

## Final architecture

- SmolVLM2-2.2B: frozen INT8 language base, rank-8 language LoRA, rank-16
  vision LoRA, trainable connector, first 16 text layers.
- VLM objective: FAST action-token CE only. There is no subtask decoder or
  subtask CE.
- EAR: 8-layer flow expert producing 16 free parameters for a 14-segment C1
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
otherwise train `joint_teacher_forced`. During joint training, exact fitted
EAR parameters are always supplied to the action expert with per-sample
no-mask, partial-mask, and all-NULL cases sampled at 1:1:1.

For inference, preprocess with `BatchProcessor(..., training=False)` and call
`policy.select_action(batch)` every control tick. The active action spline is
cheaply evaluated on the control path while a one-worker background planner
builds the pending MC-EAR/action-spline plan.

## LIBERO profile

The additive LIBERO profile leaves the RoboCasa configuration unchanged. It
uses two cameras, state 8, action 7, and an 8D `EEF pose7 + gripper1` spline.
The BF16 language model and LM head are fully frozen with no language LoRA;
FAST CE trains the complete vision encoder, connector, and state projection.

Install only the small HDF5 reader extra and point the trainer at the official
LIBERO demonstration directory:

```bash
uv sync --extra test --extra libero
uv run ear-train-libero \
  --data /path/to/LIBERO/libero/datasets/libero_spatial \
  --steps 30000 \
  --batch-size 1 \
  --gradient-accumulation 16 \
  --output outputs/libero_spatial
```

The reader consumes the official `data/demo_*/` HDF5 layout directly, so it
does not install LeRobot. It samples arbitrary episode times, keeps the next
64 real ticks for EAR and the next 16 for the action spline, and pads only at
episode ends. Raw HDF5 images are rotated 180 degrees by default; pass
`--no-rotate-images-180` if the local dataset was already corrected.
LIBERO actions remain in their native `[-1,1]` convention; an optional stats
file is applied only to the VLM state input.
