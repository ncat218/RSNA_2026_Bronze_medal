# RSNA 13/14 audit and next controlled experiment — 2026-10-06

## Evidence

- `results_13.zip`: `VISIBLE_OUTPUT_PASS` on the three studies in the saved
  0.943 notebook Output. This does **not** verify the hidden scoring rerun.
- `results_14.zip`: `COMPLETE`, folds 0–4, 4,349 weak studies, 58 Gold58 audit
  studies, 327,497 encoded images, no empty studies, 25.46 minutes.
- `results_09.zip` B2 control: `COMPLETE`, same exact weak StudyInstanceUIDs,
  fold assignments, encoder SHA256, and image count; 12.71 minutes.
- Paired against L1 weak labels thresholded at 0.5, macro OOF AUC proxy is
  **0.858970 B2 versus 0.855159 plane-aware**, delta **−0.003811**.
  Validation BCE is worse for plane-aware on **all five** folds.
- Gold58 audit-only macro AUC is **0.888217 B2 versus 0.877670 plane-aware**,
  delta **−0.010547**. Gold58 is small and not independent of several public
  checkpoint-selection paths; it must not determine ensemble weights.
- Plane-aware improves weak OOF MCL by +0.0125 and PF OA by +0.0067 but worsens
  10/12 other targets. These isolated gains do not justify a target-specific
  submission blend without full 0.943 OOF.

## Decision

Do **not** submit or blend notebook 14 into the protected 0.943 baseline. A
hierarchical plane-attention head on frozen RadImageNet features did not improve
the controlled B2 validation. This does not disprove cross-plane modeling with
other backbones or more complete image inputs.

Next test: `15_B2_RadImageNet_SampledNeighbor25D_FiveFold.ipynb`. Keep the
frozen RadImageNet encoder, exact MIL head, folds, labels and stopping rule of
notebook 09. Change only the encoder's three input channels from one slice
repeated three times to previous/current/next **sampled cache** slices within
the same MRI slot. This is not guaranteed to be immediate DICOM adjacency.
Compare notebook 15 OOF with notebook 09 by exact StudyInstanceUID and fold,
report weak-label proxy and per-target effects, and keep Gold58 audit-only.
No 0.943 blend or submission should be prepared merely from a `COMPLETE` run.

Reproduce the paired 09/14 metrics locally with `analyze_results_13_14.py`.
