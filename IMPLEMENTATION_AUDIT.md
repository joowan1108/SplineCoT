# Specification audit

| Requirement | Implementation | Status |
|---|---|---|
| No textual subskill prediction or CE | Processor/model accept full task only; FAST is the only token loss | Implemented |
| Stop-gradient VLM context | `VLMContext.detached()` before both experts | Implemented |
| Stop-gradient EAR guidance | Exact, partial-exact, and inferred EAR parameters are detached before action attention | Implemented |
| Long EAR and short final spline | 14 segments/32 points and 6 segments/16 ticks | Implemented |
| Parameter-space overlap attention | Selected EAR segment control vectors remain independent K/V tokens | Implemented |
| Continuous guidance curriculum | exact/soft teacher starts 1:1; inferred MC-EAR ramps to a final 1:1:1 mixture | Implemented |
| Start-conditioned RoboCasa spline flow | current pose is fixed during target fitting, flow corruption, and integration | Implemented |
| Spline Policy LIBERO targets | limit-normalized action chunks are downsampled to flow seeds; clean parameter predictions are supervised only after dense decode | Implemented |
| Channel-scaled RoboCasa flow | translation, quaternion, and gripper use explicit scales | Implemented |
| Dense trajectory supervision | decoded trajectory loss is the sole EAR/action spline objective, matching `prediction_type: sample` | Implemented |
| Inference-equivalent metrics | periodic full 10-step EAR/action sampling metrics | Implemented |
| MC uncertainty propagation | one shared VLM context with independent EAR flow samples; basis-propagated covariance | Implemented |
| Soft confidence before action expert | confidence biases logits, scales values, and gates the guidance residual; no learned NULL | Implemented |
| RoboCasa action flow realization | closest-point tangent plus attraction recomputed from latest pose | Implemented |
| LIBERO native-action execution | decoded 7D action chunks are sent directly to the environment | Implemented |
| Internal C1 action spline | quadratic control construction enforces equal boundary derivatives | Implemented |
| RoboCasa C0 plan handoff | after 16 ticks, the next plan starts from the post-horizon measured pose | Implemented |
| Endpoint-safe projection | endpoints remain unconditional closest-point candidates | Implemented |
| Synchronous full-horizon replacement | no Pending plan; replan only from the observation after all 16 ticks | Implemented |
| LeRobot-independent package | no LeRobot imports or runtime dependency | Implemented |
| Additive LIBERO profile | separate state8/action7/spline7 config and HDF5 trainer | Implemented |
| LIBERO language freeze | no LM LoRA; text model and LM head excluded from optimizer | Implemented |
| LIBERO full vision training | every vision encoder parameter enabled with a separate LR | Implemented |

Hardware OOM margin, 20 Hz deadline compliance, confidence calibration, and
task-performance claims require the planned remote RTX 3090 and RoboCasa
experiments; source inspection cannot establish them.
