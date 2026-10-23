# S6 source audit — 2026-10-19

## Decision

Keep notebook 33 (user-confirmed Public 0.958) unchanged. Prioritize a separate S6 complementarity diagnostic over further speculative pooling changes or immediate retraining. No new S6 notebook has been executed locally or on Kaggle by this audit; no improvement is established.

## Primary sources inspected

- Author discussion: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749359
- Public inference release: https://www.kaggle.com/code/nartaa/rsna-knee-s6fall-solo
- Dataset, current version 5 at inspection: https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights/versions/5
- Via the public Kaggle file API, read `training_code/README.md`, `training_code/labels/build_blendjev.py`, `training_code/s6/make_data.py`, and `E_prior_s42_fall.run_config.json` in full. Listed the dataset files and read dataset version metadata. Downloading one nested file through a path-form endpoint failed; query-form `fileName=` succeeded. Did not download the 200 MB checkpoint or execute the downloaded training/inference code.

## What is now concrete

The version-5 release includes the S6 trainer, training input hashes, frozen membership, target tables, label reconstruction code and CoAtNet training code. This is materially better evidence than discussion snippets alone. Byte identity to the original training run remains the publisher's assertion until independently checked against the published sums.

The supplied run configuration specifies ConvNeXt-small, views=v4, cache size 320, crop 288, three-slice windows at 3.5 mm, 64 training windows, spatial attention, no extent head, no EMA, 16 epochs, seed 42, batch 2, encoder LR 1e-4 and head LR 1e-3. ImageNet initialization is pinned by SHA256. The trainer hash in that configuration is `ae18d2e11de9a16202934787a9428ecdcefc0543d7944a75182926fb53965967`.

The README says epoch 16 was used without Gold-58 selection. Treat this as an author statement, not independently proven training history. It says 4,263 training studies = 4,349 weak studies minus 86 studies deleted locally to save disk space. Those 86 were NOT removed by the hosts and were NOT a stated quality filter.

`build_blendjev.py` defines each target as `(blend + JEV)/2` where JEV is finite, otherwise retains blend, aligned by UID and exported float32 over 4,349 rows. `make_data.py` writes these values into the `|prior` target columns, leaves `|strict` blank and blanks all gold targets. The README describes arm E_prior as unit weights with weight-normalised BCE. These inspected files do not establish a new empirical class-prior imputation algorithm; the meaning of silent-report values must be traced into the upstream label tables and loss implementation before porting it.

The author reports S6-only Public 0.950, without OAI training data for this checkpoint. Its published inference pipeline can fall back to the OAI-trained Raptor; therefore the complete pipeline is not automatically OAI-free. Anatomy dependencies require their own provenance checks.

## Corrections to the external research memo

1. R6 already uses v4 views; it is not merely a single-local-crop baseline. S6's useful diversity hypothesis concerns supervision, resolution, heads and training recipe, not discovery of local/global views absent from R6.
2. Gold exclusion from gradients does not establish independent validation. Release history, epoch/recipe selection and repeated use of gold must be distinguished. The new README's no-gold-selection claim is encouraging but remains attributed.
3. A saved run of a few visible test studies is not an estimate of hidden-test incremental runtime. Measure S6 preprocessing and inference separately; do not promise a 2m38s addition.
4. S6 code is declared Apache-2.0; learned S6 weights are declared CC BY-NC 4.0. Do not generalize code licensing to weights or assume existing Raptor weights are Apache-2.0. Winner eligibility requires the official terms, not a conclusion from this audit.
5. Do not fine-tune R6 on alleged hard-zero silent labels until its actual target/loss contract has been inspected. The memo has not established that defect.
6. Gold must not be used to fit per-target blend weights. A 58-study diagnostic is neither a trustworthy leaderboard forecast nor an independent holdout for the entire anchor.

## Next experiment (built after this audit)

`38_S6_Complementarity_Gold_Diagnostic.ipynb` now packages the inspected public component-B controller with exact source identity and embedded results26/30 references. See `38_Run_Instructions.md` and `38_Local_Validation.json`. Kaggle GPU inference remains unverified. No new S6 score exists in this project yet.

**38: pinned S6 complementarity and runtime diagnostic.**

- Pin dataset version 5, checkpoint identity and every loaded worker/anatomy dependency. Record actual loaded files and model/fallback identity per study. Inspect the worker before running it.
- First reproduce the published S6 visible-study predictions if a matching reference is available; otherwise label reproduction unverified rather than inventing a tolerance check.
- Score the same 58 gold studies already available for R6/Raptor, with failed S6 predictions explicitly flagged. Never count Raptor fallback as S6 evidence.
- Report per-target AUC with paired uncertainty, error/rank correlation, discordant cases, actual runtime and memory. Existing gold references may represent 29B rather than the exact 33 mixture (Raptor224 gold predictions are missing); label that limitation instead of calling it full-anchor validation.
- If needed, use a small deterministic weak-study sample only for structure and throughput. These studies may have been used for training; no performance/generalisation claim from them.
- If the diagnostic shows useful diversity with an acceptable measured cost, predeclare one fixed probability-blend probe retaining all 33 target weights: `0.90 * anchor33 + 0.10 * S6`. This is a proposed weight, not an optimized or validated setting. Run it separately and keep 33 protected. Do not silently introduce rank blending into an anchor that uses a different fusion rule.
- Stop for mismatched identity, unsafe hidden-runtime projection, unexplained fallback or no credible complementary signal. Do not turn minor gold gains into a claim of Private improvement.

## Outstanding user results

The user subsequently provided a Kaggle screenshot confirming notebook36 Public 0.957 (below anchor33 0.958), notebook35 Version2 and notebook37 Version1 failed during scoring reruns. Their new tracebacks have not been provided: do not assume the prior NumPy failure explains these reruns, and do not ask for another unchanged run. These failures are not scientific results for the phase/crop hypotheses. Keep them separate from new S6 experiments.
