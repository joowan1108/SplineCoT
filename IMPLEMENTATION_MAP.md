# Implementation map

## Training

```text
three images + full instruction + current state
                    |
                 SmolVLM
                 /     \
            FAST CE    detached layer context
                           /             \
                 EAR spline flow      action spline flow
                       |                    ^
       exact / soft teacher / inferred EAR  |
                       +-- overlap selection + stop-gradient guidance
```

- `processor.py`: builds full-instruction FAST tokens, 64-point EAR targets,
  and next-16-tick action targets.
- `spline.py`: fixed C1 basis, least-squares fitting, overlap selection,
  confidence conversion, covariance propagation, closest-point field, and
  12D action mapping.
- `model.py / FlowExpert`: action-parameter self-attention, layerwise VLM
  cross-attention, and per-layer EAR parameter cross-attention.
- `model.py / EARSmolVLAModel.forward`: applies the two stop-gradient
  boundaries and combines FAST, EAR flow, action flow, and mode losses.

## Parameter attention

```text
EAR segment Ej = [start, handle, end]
                    |
current EAR phase + 16-tick action window
                    |
keep overlapping segments only
                    |
[E1.start, E1.handle, E1.end, E2.start, E2.handle, E2.end]
                    |
confidence softly scales each selected control; padding -> excluded
                    |
8 phase-encoded action free-parameter queries cross-attend to all selected controls
```

The attention dimensions are hidden-space projections of 15D spline control
vectors. Support endpoints and the three local parameter roles are embedded
before learned multi-head Q/K/V attention. Confidence contributes a log-score
bias, scales each value, and gates the total guidance residual.

## Inference

```text
control worker, 20 Hz                 background planning worker

latest pose                           submitted observation
    |                                      |
closest point on Active spline             VLM (one call)
    |                                      |
tangent + attraction field                 shared-context MC EAR samples
    |                                      |
12D robot action                           mean/covariance -> time variance
                                           |
                                overlap + soft-confidence weighting
                                           |
                                   Pending action spline

             Pending ready after >=4 ticks -> Active
```

The control path does not call the VLM. It only evaluates the small quadratic
spline field from the latest measured state. The current pose is conditioned
as the first spline parameter during both training and inference; no generated
handle is overwritten after sampling.
