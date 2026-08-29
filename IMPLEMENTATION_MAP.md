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
         exact fitted EAR target            |
                       +-- overlap selection + random learned-NULL mask
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
uncertain -> learned NULL; padding -> excluded
                    |
8 phase-encoded action free-parameter queries cross-attend to all selected controls
```

The attention dimensions are hidden-space projections of 15D spline control
vectors. Support endpoints and the three local parameter roles are embedded
before learned multi-head Q/K/V attention. Confidence contributes a log-score
bias only for available guidance.

## Inference

```text
control worker, 20 Hz                 background planning worker

latest pose                           submitted observation
    |                                      |
closest point on Active spline             VLM
    |                                      |
tangent + attraction field                 batched MC EAR samples
    |                                      |
12D robot action                           mean/covariance -> time variance
                                           |
                                overlap + confidence-to-NULL gating
                                           |
                                   Pending action spline

             Pending ready after >=4 ticks -> hard C1 anchor -> Active
```

The control path does not call the VLM. It only evaluates the small quadratic
spline field from the latest measured state.
