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

- `processor.py`: builds full-instruction FAST tokens, 32-point EAR targets,
  and next-16-tick action targets.
- `spline.py`: fixed C1 basis, RoboCasa least-squares fitting, Spline Policy
  action downsampling, overlap selection, confidence conversion, covariance
  propagation, closest-point field, and 12D RoboCasa action mapping.
- `model.py / FlowExpert`: action-parameter self-attention, layerwise VLM
  cross-attention, and per-layer EAR parameter cross-attention.
- `model.py / EARSmolVLAModel.forward`: applies the two stop-gradient
  boundaries and combines FAST, decoded EAR/action trajectory, and mode losses.

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

The attention dimensions are hidden-space projections of 15D RoboCasa pose
controls or 7D LIBERO action controls. Support endpoints and the three local
parameter roles are embedded before learned multi-head Q/K/V attention.
Confidence contributes a log-score bias, scales each value, and gates the
total guidance residual.

## Inference

```text
Active action spline
        |
closest-point field from latest pose
        |
execute all 16 ticks
        |
post-horizon observation
        |
VLM once -> shared-context MC EAR -> confidence-weighted action spline
        |
first parameter fixed to measured pose -> replace Active
```

The control path does not call the VLM. It only evaluates the small quadratic
spline field from the latest measured state during an action horizon. Planning
is synchronous only after that horizon. The post-horizon measured pose is
conditioned as the first spline parameter; no generated handle is overwritten.

LIBERO uses the same EAR overlap, confidence, and attention path, but follows
Spline Policy's native-action representation:

```text
32 normalized actions -> 12 EAR flow seeds -> overlapping EAR controls
                                             -> action attention
16 normalized actions -> 8 action flow seeds -> C1 decode -> 16 native actions
```

The 16 decoded actions are executed in order and then replaced from the first
post-horizon observation. LIBERO does not start-condition action parameters or
run a pose-space closest-point field.
