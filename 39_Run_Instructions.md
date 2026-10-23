# Notebook39 — fixed S6 addition to protected33

File: `39_Anchor0958_Plus_S6_Fixed10_Rank_Submit.ipynb`.

Create a new Kaggle notebook. Attach exactly the same four inputs as successful33, plus S6:

1. RSNA Knee competition test data.
2. Nartaa publication SWA weights **Version1**, containing both Raptor384 and224.
3. **The exact R6 frozen w1 bundle attached to successful33** (`rsna-knee-r6-frozen-fe055406-w1`). Do not replace with latest/R7. Metadata retains33's datasetVersion20761842.
4. [Heliosli OrthoFoundation code/anatomy Version1](https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004/versions/1).
5. [Nartaa S6 all-data weights Version5](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights/versions/5).

No Output38/33 or study cache needed. GPU **T4x2**, **Internet Off**, **Save & Run All**. Inspect actual attachments; importing metadata may not replace existing draft inputs.

Unlike38, **39 creates submission.csv and is intended for one Public submission**. Require `rsna_39/39_run_audit.json` status COMPLETE; inspect warnings. Send Public Score and full Output ZIP afterwards. If it fails, send logs/audit instead of rerunning unchanged.

Fixed fusion: 90% of unchanged33's rank-blended output + 10% S6 percentile ranks. All original33 per-target weights stay unchanged. This is **not** the probability-scale blend tested on reused gold. 0.939505 was diagnostic gold AUC, not Public Score; our protected33 remains Public0.958. There is no promise this new probe improves it.

S6 and33 run serially in separate processes. Eight-hour global cap is a timeout, not a measured ETA. No training is done. CPU source/fusion/controller tests are available in `39_Local_Validation.json`; Kaggle GPU/hidden-run success is unverified. Keep33 unchanged as the anchor.
