# Implementation and validation plan

1. Prepare RoboCasa batches with three images, full instruction, state/action
   suffixes, padding masks, and normalization statistics.
2. Optionally train `vlm_warmup` with FAST CE; otherwise begin with the joint
   stage.
3. Train `joint_teacher_forced`: exact EAR target guidance, 1:1:1 learned-NULL
   masking, detached VLM context, and detached EAR guidance.
4. Run unit tests for fixed horizons, C1 decoding and handoff, endpoint
   projection, overlap token selection, NULL modes, covariance propagation,
   and gradient ownership.
5. Run the RTX 3090 memory smoke test remotely with batch size four; reduce
   gradient accumulation microbatch before changing the locked architecture.
6. Calibrate observation-noise magnitude and segment variance threshold on a
   held-out perturbation set.
7. Measure planner p50/p95/p99 latency and set the execution/buffer interval so
   p99 completes before Active exhaustion.
8. Evaluate ungated, gated, all-NULL, and decoded-guidance ablations before
   claiming uncertainty or spline-guidance improvements.
