# Reproducing notebook 40B

## Kaggle setup

Import [final_submission_40B.ipynb](../notebooks/final_submission_40B.ipynb) into Kaggle. Use **T4 ×2 GPUs**, **Internet off**, and the exact inputs used by the successful 40B/39 run:

| Input | Version / identity |
|---|---|
| RSNA Knee competition | Competition data; access through Kaggle |
| Nartaa Raptor publication SWA weights | `rsna-knee-publication-swa-weights-20261007`, Version 1; both 384 and 224 checkpoints |
| R6 frozen bundle | `rsna-knee-r6-frozen-fe055406-w1`, original v1/w1 bundle; do not substitute R7/latest |
| Heliosli OrthoFoundation assets | `rsna-orthofoundation-fold0-epoch7-20261004`, Version 1 |
| Nartaa S6 weights | `rsna-knee-s6fall-weights`, Version 5 |

Original Kaggle dataset-version IDs are preserved in [submission.json](../configs/submission.json). Inspect the actual notebook attachments: importing notebook metadata may not attach or select every required dataset automatically. Obtain each input from its original provider under its access conditions. No saved Output39 dataset is required.

## Run and inspect

1. Save a new version and run all cells. The original notebook introduction reflects its pre-submission experimental status; current result documentation is in the README.
2. Require `40B COMPLETE` and audit `experiment = 40B_anchor33_S6_probability_0.1` with `status = COMPLETE`.
3. Inspect `warnings`, `s6_failed_uids`, the anchor receipt and stage logs. COMPLETE can include explicitly reported fallbacks.
4. Check the final `/kaggle/working/submission.csv` has one row per test study, the exact IDs and 12 expected finding columns, and finite predictions.

The inherited `rsna_39/` output paths are intentional: preserving them avoids edits to the tested stage code. The audit is at `/kaggle/working/rsna_39/39_run_audit.json`; S6 probabilities, receipts and worker logs are stored alongside it. The anchor export is `/kaggle/working/39_anchor33.csv`.

Full inference requires the Kaggle/Linux GPU environment and external inputs. `requirements.txt` only covers the small extracted fusion tests, not the complete GPU environment. There is no claim that a local Windows run reproduces the submission.

## Notebook identity

Use the SHA-256 in `configs/submission.json` to check the archived notebook before editing it. The curated copy preserves its Git-archived bytes. Historical validation evidence is kept separately in `docs/original_40B_validation.json`; its earlier file hash is not treated as a match to this archived copy.
