# 29A/B: confirmed 0.957 anchor and next experiments

## Verified output evidence

Both scores are user-confirmed Public 0.957. Both output ZIPs pass their audits: exact R6 bundle `fe055406fad6a12f`, anatomical mirror TTA on both workers, 3/3 visible studies scored, zero fallback. Workers use four ConvNeXt-small fold models at crop 352, with spatial/extent/synovitis heads, laterality/anchor logic, Effusion fluid stacker and Baker's state stacker. These are not newly discovered components; they are already active in the successful base.

| Configuration | R6 TTA | Raptor384 | Raptor224 | Visible run seconds | Public |
|---|---:|---:|---:|---:|---:|
| 29A | 50% | 32.5% | 17.5% | 432.1 | 0.957 |
| 29B | 50% | 50% | 0% | 455.8 | 0.957 |

Weights combine model-specific percentile ranks. R6 CSVs are byte-identical across A/B. Raptor384 raw probabilities differ by at most 0.00006333; their visible rank outputs do not change. Adding Raptor224 changes 10/36 final values: two each for ACL, MCL, Synovitis, Contusion and Fracture, with maximum absolute change ~0.05833335. Seven other visible target columns are identical. This is a real but small-sample model difference, not evidence of superiority or uselessness of 224 on hidden data. Three-study correlations must not determine weights.

29A's 224 stage costs 24.4 seconds on the three visible studies. 29B happens to take longer overall because R6 decode/setup/compile timings vary; this does not imply that adding 224 speeds inference. Hidden scoring time and fallback counts are not present in these saved output ZIPs. Public success does not establish Private performance.

The gain from the earlier 0.956 configuration to these 0.957 configurations changed BOTH compact checkpoint family (R7 -> R6) and mirror activation. Pending 29C separates the mirror effect within R6. It is premature to attribute all gain to mirror alone.

## Concrete unresolved weakness

Both visible R6 runs report missing Effusion fluid volume on 2/3 studies. Their stacker explicitly has a missing-feature coefficient, so this is NOT proof of a decoder bug or incorrect inference. However, the worker already exports `fluid_stacker.csv` and `state_stacker.csv` containing model probabilities before and after postprocessing. The 29 output packaging did not retain these tables. That is an observability gap: we cannot tell whether the fixed postprocessor helps on the current TTA distribution, or which studies are affected.

The original worker comments describe the fluid stacker as fitted to the deployed earlier logit mixture and the mirror option as a later feature. This motivates measuring plain/TTA/stacker compatibility; it does not prove the coefficients were fitted on plain-only data or that they are wrong. Ask the author for that provenance.

## Executable next step: 30 diagnostic

`30_R6_TTA_Error_Stacker_Gold_Diagnostic.ipynb` preserves the successful 29 R6 worker source and all checkpoint/module hashes. It first reproduces the exact visible 29B R6 predictions, then runs 58 gold studies once with R6 TTA and saves existing worker diagnostic tables.

Predeclared comparisons, no fitted weights or stacker coefficients:

1. Current 29B: 50% Raptor384 ranks + 50% R6 TTA ranks.
2. Same formula, replace only Effusion with its pre-fluid-stacker probability.
3. Same formula, replace only Baker's with its pre-state-stacker probability.
4. Fixed family-diversity hypothesis: 50% Raptor384 + 40% R6 TTA + 10% R7 plain. R7 plain references come from experiment 26; this is NOT R7 TTA.

The stored 26 Raptor384/R7-plain predictions and gold labels are embedded. No Raptor inference, new training, large cache or saved-output attachment is required. No root `submission.csv` is produced: DO NOT SUBMIT 30. Attach competition including training MRI, R6 dataset VERSION 1, and the same OrthoFoundation code/anatomy dataset as 29. T4x2, Internet Off. Run duration has not been measured; the three-hour timeout is only a stop limit.

The gold set was reused in upstream development and selection. Results and bootstrap intervals are exploratory, NOT independent CV. A positive result can motivate one fixed prospective Public probe; do not search arbitrary target weights across 58 studies. This analysis is anchored to 29B, not full 29A with 224.

## What could actually break the plateau

- Short term: finish 29C; use 30 to identify a concrete postprocessing regression or complementary R6/R7 errors. Package only a justified fixed change, retaining the 0.957 anchors unchanged. A regression in one target can matter because the competition averages twelve target AUCs.
- Medium term: obtain aligned R6/R7 OOF probabilities and fold/provenance records. That allows cross-fitted family blending and target-specialist selection, rather than tuning to Public scores or gold reused for selection. Different bundle versions do not guarantee independent errors.
- Larger change: train a genuinely complementary model, with series/plane-aware bags and targeted ligament/meniscus regions, retaining a separate study context stream. Match training and inference preprocessing; the rejected experiment 26 showed that simply deleting cross-series windows at serving time did not improve the existing model. This branch requires grouped splits, report-label provenance, complete cache inventory and a short memory/throughput gate before a full run. It is not yet packaged or promised to beat 0.957.

The public search in this pass did not establish an independently documented new >0.957 architecture. Existing public materials and their claims are leads, not proof of a reproducible Private gain. The [public experiment repository](https://github.com/Beiciccc/rsna-knee-abnormality-detection) documents baseline reproductions and emphasizes provenance; it does not supply evidence that our specific next ablation will improve.

## Questions to take to the authors / Discussion

1. Were R6 Effusion/Baker's stackers fitted using plain logits, mirrored-average logits, or both? Were fitting predictions out of fold?
2. What fraction of validation studies lacks valid fluid volume, and what is per-target AUC for present versus missing volume?
3. Can authors publish aligned R6/R7 OOF predictions, fold IDs, target-state definitions and training-set differences?
4. What evidence compares R6/R7 on held-out sites/scanners or a set not reused for checkpoint/stacker selection?
5. What are hidden scoring runtime and fallback counts for mirrored R6 + both Raptor resolutions?

## Verification boundary

Archive audits and prediction comparisons were executed locally. Generated notebook cells compile; actual analysis is tested on 58-row identity fixtures, including worker-dump mismatch rejection. Public worker sources remain unchanged. GPU/DICOM inference for 30 has not been executed locally. No new score is guaranteed.
