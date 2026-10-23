# R6 public notebook: reproduction and isolated experiments

Date: 2026-10-15. Source: user-supplied `Kaggle public/0-9576-latest-public-dataset-score-in-one-click.ipynb`. The user reports a new public notebook around 0.957. The file title claims 0.9576; its saved output contains only three visible studies and does not independently establish that Public score on our account.

## What is actually different

The reference uses frozen R6 bundle `fe055406fad6a12f` (`heliosli/rsna-knee-r6-frozen-fe055406`), not our R7 bundle `8c7b666e27dc0e8f`. The embedded public worker modules match those of 25A byte-for-byte. Thus the model/auxiliary checkpoint bundle and mirror configuration change, rather than a new worker architecture.

Its R6 uses the existing anatomical mirror TTA: correct sagittal/other-plane transforms, medial/lateral output permutation, and MCL from the plain pass. Its two Raptor checkpoints are the same 384 and 224 releases already studied. Effective final rank weights are R6 50%, Raptor-384 32.5%, Raptor-224 17.5%. Our failed-to-improve 25B used plain R7 in this slot, so it does not test the new R6/TTA combination.

The reference source's seven scoring cells are preserved exactly in 29A. Added preflight and audit cells check attachments, bundle ID, mirror status, schema, and scored/fallback counts. GPU metadata is corrected from the supplied file's `isGpuEnabled: false` to T4 enabled. Source and cell hashes are recorded in `29_probe_manifest.json`. All original acknowledgements are retained.

## Three independent submissions

| Notebook | Compact model | Raptor 384 | Raptor 224 | Question |
|---|---|---:|---:|---|
| 29A | R6 + mirror TTA, 50% | 32.5% | 17.5% | Reproduce the supplied public recipe |
| 29B | R6 + mirror TTA, 50% | 50% | 0% | Does 224 add useful signal to this compact model? |
| 29C | R6 plain, 50% | 32.5% | 17.5% | Does R6 mirror TTA help this combination? |

All targets use the same weights. 29B omits the 224 model stage entirely; its blend is the original two-engine 50/50 formula. 29C changes only the R6 TTA flag and verifies that workers actually ran without TTA. This source's frozen R6 manifest predates its mirror key; an unexpected manifest-level mirror override stops the plain probe at preflight.

Each notebook creates submission.csv. They do not depend on one another's Output or on notebook 28. They may run independently; parallel execution depends on account session availability. Submit each once after its saved audit reports COMPLETE. Three runs use three submission slots; retain the other two for a concrete next hypothesis based on these results rather than a sweep of arbitrary weights.

## Inputs for all three

1. RSNA Knee Abnormality Detection competition.
2. [nartaa/rsna-knee-publication-swa-weights-20261007](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007).
3. [heliosli/rsna-knee-r6-frozen-fe055406 — VERSION 1 ONLY](https://www.kaggle.com/datasets/heliosli/rsna-knee-r6-frozen-fe055406/versions/1).
4. [heliosli/rsna-orthofoundation-fold0-epoch7-20261004](https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004).

Version incident, verified via Kaggle's public dataset-view API on 2026-10-15: the owner replaced R6 with R7 in version 2 on 2026-10-14, retaining the dataset slug and file names. Select version 1 explicitly in Kaggle Input; do not assume the dataset name or imported notebook metadata guarantees the correct version. Version 1 manifest SHA256 is `6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648`; version 2 is `c4bb0f7399ada9a020e0c15cae6252033cf0c4335e999961793a176063074ec6`. All three preflights now record the observed bundle and full manifest hash before rejecting an incompatible version. R7 weights are not a substitute. GPU T4x2, Internet Off, Save & Run All. No study cache, training, 26/27 Output or local model download is required.

Send each Public score and Output ZIP, especially its 29A/B/C_run_audit.json, blend_receipt.json, submission_receipt.json and submission.csv. The saved reference took 444.5 seconds for the three visible studies. That is not a hidden-set runtime estimate; mirrored R6 adds an image pass and the inherited internal deadline includes all stages.

## Decisions after the three results

29A establishes the new reproducible anchor if it improves the confirmed 0.956. The B comparison isolates the 224 contribution; the C comparison isolates TTA. Equal displayed scores remain inconclusive below leaderboard precision. If the compact model is the main gain, notebook 28's R7 mirror diagnostic provides a separate comparison of another checkpoint bundle. A later R6/R7 blend needs evidence of complementary errors and a runtime budget; neither the release number nor the three visible-study correlations establish that evidence.

## Questions worth asking the authors / competition Discussion

1. Are R6 and R7 trained on identical study groups and folds? Can they share aligned OOF probabilities, fold assignments and target provenance for the two frozen bundles?
2. What changes separate the R6 and R7 checkpoints: training labels, state categories, OA extent targets, synovitis decomposition, or training schedule? Which changes have paired validation evidence?
3. Were each bundle's Effusion and Baker's stacker coefficients fitted on plain or mirror-averaged outputs? Does applying the original stacker to mirror-averaged logits require refitting, and were all stackers cross-fitted?
4. What is the per-target plain-versus-mirror result on data excluded from checkpoint/stacker selection, particularly Synovitis, Lateral OA and PF OA? Are the mirrored state-category and binary permutations exactly symmetric for both bundles?
5. Is there an independently held-out, image-labelled evaluation set or a scanner/site-held-out test that can compare the bundles without reusing the 58 gold studies? The gold selection history is important when interpreting small gains.
6. What is the actual hidden scoring runtime and fallback rate of R6 TTA + two Raptors? A saved three-study smoke test does not answer that.

These questions target missing evidence, not a request for hidden-test labels or patient predictions.

## Verification and limitations

All generated code cells compile. The original scoring code of 29A is unchanged, and each ablation is checked against its intended source modifications. Local tests execute the actual blend/audit cells on 58-row prediction fixtures, verify rank formulas and UID order, and exercise both-model failure fallback. GPU/DICOM model inference has not been executed locally; Kaggle runs provide that verification. Checkpoint identity uses the reference's existing hashes and manifest checks, without inventing replacement hashes.
