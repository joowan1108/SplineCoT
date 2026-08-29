# Specification audit

| Requirement | Implementation | Status |
|---|---|---|
| No textual subskill prediction or CE | Processor/model accept full task only; FAST is the only token loss | Implemented |
| Stop-gradient VLM context | `VLMContext.detached()` before both experts | Implemented |
| Stop-gradient EAR guidance | Exact fitted EAR parameters are detached before action attention | Implemented |
| Long EAR and short final spline | 14 segments/64 points and 6 segments/16 ticks | Implemented |
| Parameter-space overlap attention | Selected EAR segment control vectors remain independent K/V tokens | Implemented |
| 1:1:1 teacher masking | full, contiguous partial, all-NULL per sample | Implemented |
| MC uncertainty propagation | batched perturbed observations and flow samples; basis-propagated covariance | Implemented |
| Confidence gating before action expert | uncertain selected controls become learned NULL | Implemented |
| Action flow realization | closest-point tangent plus attraction recomputed from latest pose | Implemented |
| Internal C1 action spline | quadratic control construction enforces equal boundary derivatives | Implemented |
| C1 plan handoff | latest state and velocity hard-set first two free parameters at commit | Implemented |
| Endpoint-safe projection | endpoints remain unconditional closest-point candidates | Implemented |
| Asynchronous Active/Pending execution | one background planning worker; active field stays on control path | Implemented |
| LeRobot-independent package | no LeRobot imports or runtime dependency | Implemented |
| Additive LIBERO profile | separate state8/action7/spline8 config and HDF5 trainer | Implemented |
| LIBERO language freeze | no LM LoRA; text model and LM head excluded from optimizer | Implemented |
| LIBERO full vision training | every vision encoder parameter enabled with a separate LR | Implemented |

Hardware OOM margin, 20 Hz deadline compliance, confidence calibration, and
task-performance claims require the planned remote RTX 3090 and RoboCasa
experiments; source inspection cannot establish them.
