# Implementation and validation plan

1. Prepare RoboCasa batches with three images, full instruction, state/action
   suffixes, padding masks, and normalization statistics.
2. Optionally train `vlm_warmup` with FAST CE; otherwise begin with the joint
   stage.
3. Train `joint_teacher_forced` with a continuous curriculum: exact and
   soft-confidence teacher guidance begin at 1:1, inferred MC-EAR rises from
   zero after 10% to one third at 50%, and the final mixture is 1:1:1.
4. Run unit tests for fixed horizons, C1 decoding, conditioned starts,
   endpoint projection, overlap token selection, soft confidence, covariance
   propagation, and gradient ownership.
5. Run the RTX 3090 memory smoke test remotely with batch size four; reduce
   gradient accumulation microbatch before changing the locked architecture.
6. Calibrate observation-noise magnitude and segment variance threshold on a
   held-out perturbation set.
7. Measure planner p50/p95/p99 latency and set the execution/buffer interval so
   p99 completes before Active exhaustion.
8. Evaluate ungated, gated, inferred-guidance, and decoded-guidance ablations before
   claiming uncertainty or spline-guidance improvements.
