# Notebook 38 — S6 diagnostic

Import `38_S6_Complementarity_Gold_Diagnostic.ipynb` into a **new** Kaggle notebook. Leave successful notebook33 unchanged.

Attach exactly:

1. RSNA Knee competition data, including `train_series`.
2. https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights/versions/5 — Version **5**.
3. https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004/versions/1 — Version **1**, same anatomy/code input as notebook33.

No Output33, R6 bundle, Raptor checkpoint or study cache needed. Metadata contains exact version IDs; verify actual inputs after importing because existing draft attachments can override metadata.

## Fix after the first failed run

The supplied log failed before inference: no mount matching the anatomy dataset and `anatomy.pt` was found. That alone cannot distinguish a missing attachment from an unexpected mount name/layout. The current notebook also searches renamed/nested non-competition inputs by marker, keeps all original content hashes, and writes attached-directory inventory plus `worker_outputs/38_input_resolution.json` on resolution failure. It cannot manufacture a missing input. Confirm the anatomy/code dataset above is actually attached, not just the S6 weights dataset or an old notebook Output. Import this regenerated notebook; do not rerun the previous file unchanged.

Settings: **GPU T4x2, Internet Off, Save & Run All**. This is inference, not training. The two-hour diagnostic budget is a safety ceiling, not a measured ETA. Local CPU mocks do not guarantee Kaggle GPU execution, no OOM or hidden-run success.

This notebook deliberately produces **no submission.csv**. Do not submit it to the competition. It scores up to three visible test studies and 58 gold studies with S6 only, checking strict checkpoint/worker/anatomy identity. No Raptor fallback is allowed. It does not read raw reports for analysis or transmit them externally.

Download and send the complete Output ZIP. Main evidence:

- `rsna_38_s6/38_diagnostic_report.json` (status COMPLETE).
- `38_s6_gold_probabilities.csv`, `38_per_target_metrics.csv`, `38_discordant_cases.csv`.
- `worker_outputs/visible/` and `worker_outputs/gold/` logs and receipts.

If it fails, send the report and worker log; do not bypass hashes or keep rerunning unchanged.

Interpretation: stored R6/Raptor references are reused diagnostics; proxy29B is not the exact 0.958 anchor33. No weights will be chosen by maximizing gold AUC. After reviewing real S6 outputs, decide whether a single fixed small-weight anchor33+S6 submission is warranted.
