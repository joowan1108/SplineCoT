# Specification audit

| Requirement | Implementation | Status |
|---|---|---|
| No textual subskill prediction or CE | Processor/model accept full task only; FAST is the only token loss | Implemented |
| Stop-gradient VLM context | `VLMContext.detached()` before both experts | Implemented |
| Stop-gradient EAR guidance | Exact, partial-exact, and inferred EAR parameters are detached before action attention | Implemented |
| Long EAR and short final spline | 14 segments/64 points and 6 segments/16 ticks | Implemented |
| Parameter-space overlap attention | Selected EAR segment control vectors remain independent K/V tokens | Implemented |
| Continuous guidance curriculum | exact/soft teacher starts 1:1; inferred MC-EAR ramps to a final 1:1:1 mixture | Implemented |
| Start-conditioned spline flow | current pose is fixed during target fitting, flow corruption, and integration | Implemented |
| Channel-scaled flow | translation, quaternion, and gripper use explicit scales | Implemented |
| Dense trajectory supervision | decoded trajectory loss supplements parameter-flow loss | Implemented |
| Inference-equivalent metrics | periodic full 10-step EAR/action sampling metrics | Implemented |
| MC uncertainty propagation | one shared VLM context with independent EAR flow samples; basis-propagated covariance | Implemented |
| Soft confidence before action expert | confidence biases logits, scales values, and gates the guidance residual; no learned NULL | Implemented |
| Action flow realization | closest-point tangent plus attraction recomputed from latest pose | Implemented |
| Internal C1 action spline | quadratic control construction enforces equal boundary derivatives | Implemented |
| Non-destructive plan handoff | no post-sampling handle overwrite; first pose is conditioned before sampling | Implemented |
| Endpoint-safe projection | endpoints remain unconditional closest-point candidates | Implemented |
| Asynchronous Active/Pending execution | one background planning worker; active field stays on control path | Implemented |
| LeRobot-independent package | no LeRobot imports or runtime dependency | Implemented |
| Additive LIBERO profile | separate state8/action7/spline8 config and HDF5 trainer | Implemented |
| LIBERO language freeze | no LM LoRA; text model and LM head excluded from optimizer | Implemented |
| LIBERO full vision training | every vision encoder parameter enabled with a separate LR | Implemented |

Hardware OOM margin, 20 Hz deadline compliance, confidence calibration, and
task-performance claims require the planned remote RTX 3090 and RoboCasa
experiments; source inspection cannot establish them.
