# Uncertainty-Gated Dual-Spline EAR-SmolVLA

## Research hypothesis

A long-horizon spline EAR can provide action-chain-of-thought guidance without
textual subskill generation, while a shorter executable spline expert can
convert only the temporally relevant and reliable part of that guidance into
precise closed-loop control. Representing both branches with compatible C1
quadratic splines should make parameter-level comparison direct, reduce
handoff discontinuity, and preserve Spline Policy's local perturbation
recovery in the action actually executed by the robot.

The central novelty is uncertainty-gated parameter attention. The action
expert receives only EAR segments whose temporal support overlaps its current
short horizon. Each selected segment remains three separate control-parameter
vectors, and Monte Carlo trajectory uncertainty determines whether those
vectors are visible as guidance or replaced by a learned NULL representation.

## Backbone and gradient ownership

The backbone is SmolVLM2-2.2B in a SmolVLA-style system. The pretrained
language base is frozen and loaded in INT8. Rank-8 language LoRA and rank-16
vision LoRA parameters are trained in bfloat16; the vision connector and state
projection are also trainable. The first 16 language layers provide the VLM
context.

The VLM receives three current camera views, the full task instruction, and
the current robot state. It does not generate a textual subtask. During
training, the demonstrated short action sequence is encoded with FAST and the
VLM is optimized only by FAST token cross-entropy.

VLM hidden states are stop-gradient inputs to both continuous experts. The
action expert also receives a stop-gradient copy of EAR guidance. Therefore:

- FAST CE updates VLM adapters, connector, and state projection.
- EAR flow loss updates only the EAR expert.
- Action spline flow loss updates only the action expert and its guidance
  attention.
- Control-mode loss updates its separate prediction head.

## Common spline representation

Both continuous branches use piecewise quadratic Bezier splines. A spline
with K segments is generated from K+2 free parameter vectors. The decoder
reflects each next handle around the shared segment boundary, which enforces
C1 continuity at every internal boundary by construction.

Each free parameter has 15 channels:

- mobile-base position 3 and quaternion 4;
- end-effector position 3 and quaternion 4;
- continuous gripper reference 1.

The categorical control mode is predicted separately. Quaternion signs are
made continuous before fitting, decoded quaternions are normalized, and
quaternion derivatives are projected to the quaternion tangent space before
conversion to angular velocity. This is a practical component-space spline
with tangent-projected execution, not an exact Riemannian SO(3) spline.

## EAR spline

The EAR is an 8-layer flow-matching transformer. It predicts 16 free vectors,
which decode to a 14-segment C1 spline over a broad 64-tick representation of
the available atomic-skill trajectory. At 20 Hz this nominal representation
contains up to 3.2 seconds of guidance; longer or shorter demonstrations are
mapped to the fixed representation by valid-prefix linear resampling.

The EAR is a predictive reference and is not evaluated as the control field
on every robot tick. Its job is to express broad motion intent and expose
which future intervals are reliable enough to guide the action expert.

## Action spline expert

The action expert is a 16-layer flow-matching transformer. It predicts eight
free vectors, which decode to a six-segment C1 spline covering the next 16
real control ticks, or 0.8 seconds at 20 Hz. Normally four actions, or 0.2
seconds, are executed before a ready pending plan may replace it.

Each action parameter query first self-attends with the other action
parameters and cross-attends to the detached VLM context. A separate
parameter-attention operation then attends only to control vectors from EAR
segments overlapping the current action time window. EAR start, handle, and
end vectors are separate keys and values with temporal-support and local-role
embeddings; they are not pooled into a segment token and are not decoded into
action points before attention.

## Teacher-forced stabilization

During training, guidance always comes from the spline fitted to the exact
demonstration trajectory. Predicted EAR output is never substituted into the
action branch during the initial model training. Each batch item independently
uses one of three equally likely guidance conditions:

1. all overlapping EAR parameters visible;
2. a random contiguous span replaced by learned NULL guidance;
3. all overlapping EAR parameters replaced by learned NULL guidance.

The three conditions are mixed 1:1:1. NULL is a learned embedding rather than
a numeric zero, so the action expert can distinguish missing guidance from a
valid parameter whose value happens to be zero.

## Monte Carlo uncertainty gating

Uncertainty gating is used only during inference planning. Several perturbed
versions of the current images and robot state are batched together and passed
through the EAR with independent flow initial noise. Their predicted spline
parameters form Monte Carlo samples. The sample mean is the candidate EAR
spline and the unbiased sample covariance is propagated through the fixed
spline basis to obtain trajectory variance at the 64 query times.

Each EAR segment receives the maximum mean channel variance within its
temporal support. A configurable variance threshold creates the hard
confidence mask; an exponential confidence value also biases attention among
the surviving parameters. An uncertain selected segment is retained in the
temporal layout but its three parameter values are replaced by learned NULL
guidance. If every overlapping segment is uncertain, the action expert runs
with all-NULL EAR guidance rather than using an unreliable reference.

Because both observation perturbation and independent flow initialization are
sampled, the reported variance is total predictive sensitivity rather than a
pure decomposition of epistemic and aleatoric uncertainty. Separate ablations
may hold either source fixed.

## Closed-loop execution and asynchronous planning

The control path owns an Active action spline. At every 20 Hz tick it projects
the latest measured pose to the closest point on that spline and evaluates a
field combining forward tangent motion with attraction back toward the spline.
This field recomputation is the source of local correction and perturbation
recovery; it does not require a VLM forward pass.

At the same time, one background planning worker consumes the latest submitted
observation, computes the VLM context, batched Monte Carlo EAR distribution,
overlapping confidence-gated guidance, and a Pending action spline. After at
least four active ticks, a completed Pending spline can be committed. If it is
not ready, the Active spline continues; after its nominal horizon, tangent
progression is disabled so the field safely attracts toward the endpoint.

The action spline's internal boundaries are C1 by construction. At each
Active-to-Pending handoff, its first free parameter is replaced by the latest
measured 15D state and its second is chosen from the current 15D velocity and
first segment duration. This hard anchor makes position and first derivative
continuous at the actual commit time. Discrete control-mode changes and
gripper semantics remain outside the geometric C1 guarantee.

## Data and fixed target construction

Training uses RoboCasa365-style trajectories and the pinned 12D action order:
mobile-base motion 4, categorical control mode 1, end-effector translation 3,
end-effector rotation 3, and gripper 1. Only full task text is required.

For every valid trajectory suffix:

- the raw 14D pose is quaternion-sign corrected before spline fitting;
- the whole valid suffix is resampled to 64 points for the broad EAR target;
- the next 16 original ticks form the action target, with endpoint repetition
  only for a shorter valid suffix;
- the continuous normalized gripper channel is appended;
- a fixed least-squares pseudoinverse maps each target trajectory to its exact
  K+2 free spline parameters;
- flow-matching noise is added to these parameters for the corresponding
  expert loss.

## Evaluation

The main comparisons are: no EAR guidance, ungated EAR parameter attention,
uncertainty-gated EAR attention, decoded-point guidance, and a non-spline
action expert. Measurements include task success, contact-stage success,
perturbation recovery and recovery time, false progression after failed
grasps, active-to-pending velocity jump, trajectory jerk, uncertainty
calibration, fraction of NULL guidance, planning p50/p95/p99 latency, control
tick latency, peak VRAM, and training throughput.

The asynchronous design removes VLM inference from the hard 20 Hz control
path, but it does not make planning free. The deployment buffer and four-tick
commit interval must be tuned from measured p99 planning latency. Effectiveness,
skill-duration coverage, confidence threshold, and the advantage over world
models or subgoal-image guidance remain empirical claims to validate.
