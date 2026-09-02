# Uncertainty-Gated Dual-Spline EAR-SmolVLA

## Research hypothesis

A long-horizon spline EAR can provide action-chain-of-thought guidance without
textual subskill generation, while a shorter executable spline expert can
convert only the temporally relevant and reliable part of that guidance into
precise closed-loop control. Representing both branches with compatible C1
quadratic splines should make parameter-level comparison direct, reduce
handoff discontinuity, and preserve Spline Policy's local perturbation
recovery in the action actually executed by the robot.

The central novelty is uncertainty-weighted parameter attention. The action
expert receives only EAR segments whose temporal support overlaps its current
short horizon. Each selected segment remains three separate control-parameter
vectors, and Monte Carlo trajectory uncertainty continuously scales how
strongly those vectors guide the action expert.

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
- Decoded EAR trajectory loss updates only the EAR expert.
- Decoded action-spline trajectory loss updates only the action expert and its guidance
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

The EAR is a 16-layer flow-matching transformer. It predicts 16 free vectors,
which decode to a 14-segment C1 spline over a broad 32-tick representation of
the available atomic-skill trajectory. At 20 Hz this nominal representation
contains up to 1.6 seconds of guidance; longer or shorter demonstrations are
mapped to the fixed representation by valid-prefix linear resampling.

The EAR is a predictive reference and is not evaluated as the control field
on every robot tick. Its job is to express broad motion intent and expose
which future intervals are reliable enough to guide the action expert.

## Action spline expert

The action expert is a 16-layer flow-matching transformer. It predicts eight
free vectors, which decode to a six-segment C1 spline covering the next 16
real control ticks, or 0.75 seconds between endpoints at 20 Hz. All 16 actions
are executed before the next observation is used to construct a new plan.

Each action parameter query first self-attends with the other action
parameters and cross-attends to the detached VLM context. A separate
parameter-attention operation then attends only to control vectors from EAR
segments overlapping the current action time window. EAR start, handle, and
end vectors are separate keys and values with temporal-support and local-role
embeddings; they are not pooled into a segment token and are not decoded into
action points before attention.

## Continuous teacher-forced stabilization

Training begins with a 1:1 mixture of exact EAR guidance and exact EAR guidance
whose confidence is reduced over a contiguous half of the selected segments.
No spline parameter is deleted or modified. From 10% to 50% of the training
run, the inferred-EAR probability increases linearly from zero to one third;
the final mixture is exact, soft teacher, and inferred EAR at 1:1:1.

The inferred branch uses the same four-sample Monte Carlo mean and trajectory
uncertainty as rollout inference and remains stop-gradient. Confidence adds a
relative attention-logit bias, scales each EAR value, and gates the total
guidance residual. There is no learned NULL representation or hard masking of
selected EAR parameters; only batch padding is excluded.

## Monte Carlo uncertainty gating

The images, instruction, and current robot state are passed through the VLM
once. Its representation is shared across several EAR samples, each generated
from independent flow initial noise. Their predicted spline parameters form
conditional Monte Carlo samples. The sample mean is the candidate EAR spline
and the unbiased sample covariance is propagated through the fixed spline
basis to obtain trajectory variance at the 64 query times.

Each EAR segment receives the maximum mean channel variance within its
temporal support. An exponential confidence value biases attention toward
reliable parameters and scales both their values and the resulting guidance
residual. If every overlapping segment has low confidence, EAR influence
smoothly approaches zero without deleting or changing the spline parameters.
A configurable variance threshold is retained only as a diagnostic metric.

The reported variance measures the spread of the learned conditional EAR
distribution for one fixed observation. It does not measure sensitivity to
image or state perturbations; calibration must therefore compare this variance
against held-out spline error.

## Closed-loop execution and synchronous plan replacement

The control path owns an Active action spline. At every 20 Hz tick it projects
the latest measured pose to the closest point on that spline and evaluates a
field combining forward tangent motion with attraction back toward the spline.
This field recomputation is the source of local correction and perturbation
recovery; it does not require a VLM forward pass.

After all 16 ticks have executed, the first subsequent observation is passed
synchronously through one VLM call, the conditional Monte Carlo EAR,
confidence-gated overlap attention, and the action expert. The completed plan
then replaces the Active plan before another action is issued. No Pending plan
is generated from an observation captured during the previous action horizon.

The action spline's internal boundaries are C1 by construction. Its first
free parameter is conditioned on the measured pose from the post-horizon
observation, so consecutive executed plans are C0 at the actual robot pose.
No derivative constraint is imposed across EAR or action-plan boundaries;
internal spline C1 remains unchanged.

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
- a start-conditioned least-squares fit fixes the first spline parameter to
  the measured pose and solves the remaining K+1 parameters;
- translation, quaternion, and gripper channels are scaled before flow noise;
- the flow experts predict clean spline samples and are supervised only by
  decoded dense-trajectory loss;
- the measured start pose remains inpainted throughout training and inference.

## Evaluation

The main comparisons are: no EAR guidance, ungated EAR parameter attention,
uncertainty-gated EAR attention, decoded-point guidance, and a non-spline
action expert. Measurements include task success, contact-stage success,
perturbation recovery and recovery time, false progression after failed
grasps, plan-boundary velocity jump, trajectory jerk, uncertainty
calibration, effective guidance strength, planning p50/p95/p99 latency, control
tick latency, peak VRAM, and training throughput.

Synchronous replanning intentionally permits a hold between action horizons.
Its duration must be reported with synchronized CUDA timing alongside task
performance. Effectiveness, skill-duration coverage, confidence calibration,
and the advantage over world models or subgoal-image guidance remain empirical
claims to validate.


 ### 1. Knot 배정 기준

  모든 채널은 먼저 median/MAD 기반으로 정규화합니다.

  #### Action-only

  6개 action 채널의 시점별 fitting error를 하나로 합칩니다.

  $
  e_a(t)=
  \sqrt{\frac{1}{6}\sum_{d=0}^{5}
  (\hat a_{t,d}-a_{t,d})^2}
  $

  현재 spline으로 fitting한 뒤, 허용된 후보 중 (e_a(t))가 가장 큰 지점에 knot을 하나 추가하고 다시 fitting합니다. 전체
  RMSE가 1.0 이하가 될 때까지 반복합니다.

  #### FT-guided action fitting

  Action error와 FT 변화 점수를 합칩니다.

  $
  P(t)=
  \frac{e_a(t)}{1.0}
  +
  \frac{
  \sqrt{s_F(t)^2+s_\tau(t)^2}
  }{
  q_{0.95}
  }
  $

  여기서

  $
  s_F(t)=\sqrt{\frac{1}{3}\sum_{i=1}^{3}(\Delta F_i(t))^2}
  $

  $
  s_\tau(t)=\sqrt{\frac{1}{3}\sum_{i=1}^{3}(\Delta \tau_i(t))^2}
  $

  입니다. Force와 torque weight는 현재 모두 1.0입니다. (P(t))가 가장 큰 허용 지점에 knot을 추가합니다.

  FT 변화량의 상위 5%, 즉 local window의 95th percentile 이상인 지점은 FT event로 취급합니다. Action RMSE가 1.0 이하이고
  모든 FT event가 knot의 ±2 step 안에 포함되면 종료합니다.

  Event 1388에서는 다음과 같이 나왔습니다.

  - Action-only knots: 1352, 1384, 1392, 1403
  - FT-guided knots: 1352, 1359, 1372, 1387, 1403

  ### 2. Low/high 기준

  Low/high 배경은 전체 episode 0에서 계산합니다.

  먼저 정규화된 변화량을 계산하고 5-step moving average를 적용합니다.

  $
  A(t)=\operatorname{MA}_5\left(\lVert\Delta a_t\rVert_2\right)
  $

  $
  W(t)=\operatorname{MA}_5
  \left(
  \sqrt{
  \lVert\Delta F_t\rVert_2^2+
  \lVert\Delta\tau_t\rVert_2^2
  }
  \right)
  $

  전체 episode의 median이 임계값입니다.

  - Action threshold: 1.1919235289
  - FT threshold: 0.3632404146

  분류는 다음과 같습니다.

   구분                     조건
  ━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Low action / Low FT      (A(t)\le1.1919,\ W(t)\le0.3632)
  ───────────────────────  ─────────────────────────────────
   Low action / High FT     (A(t)\le1.1919,\ W(t)>0.3632)
  ───────────────────────  ─────────────────────────────────
   High action / Low FT     (A(t)>1.1919,\ W(t)\le0.3632)
  ───────────────────────  ─────────────────────────────────
   High action / High FT    (A(t)>1.1919,\ W(t)>0.3632)

  따라서 low/high는 절대적인 물리 단위가 아니라 episode 0 내부의 상대적인 분류입니다.

  ### 3. Max knots 기준

  설정은 다음과 같습니다.

  - B-spline degree: 3, cubic
  - Maximum interior knots: 30
  - Minimum knot gap: 2 steps
  - Local fitting window: event 기준 ±40 steps, 총 81 samples
  - RMSE tolerance: 1.0

  max knots=30은 endpoint knots를 제외한 interior knot의 최대 개수입니다. Cubic B-spline은 양 끝 knot가 각각 4번 반복되
  므로, interior knots가 30개라면 전체 knot-vector 길이는 최대 38입니다.

  다만 30개를 항상 사용하는 것은 아닙니다. Event 1388에서는 종료 조건을 먼저 만족했기 때문에:

  - Action-only: interior knots 4개
  - FT-guided: interior knots 5개
  - Action+wrench: interior knots 8개

  만 생성되었습니다. 즉 30은 knot 개수를 맞추는 목표가 아니라 과도한 knot 삽입을 막는 상한입니다.
