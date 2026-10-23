# Three remaining slots — fixed plan, 2026-10-22

Only THREE slots remain according to the user. Create/use at most44A and44B now; reserve the last slot. Do not restart a broad Public weight/crop sweep.

## Hypotheses

| Probe | Reference | Only disabled option | Target |
|---|---|---|---|
|44A|40B|`--fluid-stacker`|Effusion|
|44B|40B|`--state-stacker`|Baker's|

Each removes one optional deployed R6 postprocessor, not the model head or image features. Original MRI models, crop, physical spacing, mirrorTTA, side classifier, S6 and40B fusion remain unchanged. The33-derived prediction is modified by this explicit ablation; it is not exact33 any more. The worker already supports absent stacker arguments. Volume features are still requested if the remaining stacker needs them.

Motivation is limited: diagnostic30 on58reused gold showed macro changes+0.000104 and+0.000075 versus its29B proxy, with paired intervals containing zero. Those are not reliable predicted Public gains. These trials ask whether optional postprocessing adds value outside that reused diagnostic, not whether gold fit is proven. The stackers' exact fitting provenance is not established.

## Run

- Use the exact same FIVE inputs/versions as successful40B/39, not the reduced three-input standalone43 setup.
- T4x2, Internet Off, Save & Run All, separate Kaggle notebooks. No prior Output attachment required.
- Inspect `rsna_39/39_run_audit.json`: experiment44A/44B, COMPLETE, ablation flag matches the notebook; inspect warnings.
- Inspect `44_postprocessor_audit.json`: one option/path removed and the other stacker retained. Only then submit once. Send both Public scores and Output ZIPs.
- Artifact names containing39 remain intentional for compatibility; experiment metadata identifies each run.

Check the competition's live remaining time and actual GPU queue before launching. The inherited eight-hour cap is not a runtime estimate. Actual GPU/hidden execution is not yet verified.40A's hidden exception remains unresolved; do not assume these optional-flag edits fix it. Earlier43 failures were missing anatomy input, subsequently corrected.

## Final slot

Keep it unused until44 results are available. A future combined ablation is not authorized as an automatic next step and is not already proven useful: only consider it if the two individual changes help and there is time for a reviewed, successful rerun. Otherwise preserve the slot and existing0.958 candidates rather than forcing another experiment. No changes to final selection are made by these files.

This is a narrow postprocessing diagnostic and may have a tiny effect. It does not promise0.96+ or PrivateTop10. If new public research cannot identify independently trained, compatible and accessible model evidence, acknowledge that limitation instead of manufacturing a breakthrough claim.
