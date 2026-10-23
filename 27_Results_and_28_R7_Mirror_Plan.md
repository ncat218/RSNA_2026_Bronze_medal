# Results 27 and R7 mirror TTA diagnostic

Date: 2026-10-15. User-reported Public result for 27: 0.956. Incumbent 25A and probe 25B also display 0.956. These rounded scores do not establish equal underlying AUC or Private performance.

## Verified output 27

The supplied archive reports COMPLETE, three visible studies, zero Raptor/R7 fallback, and 396.3 seconds total. Its OA Raptor weights are all 0.35, correctly giving R7 0.65. Direct comparison of submission.csv with 27_base25A_reference.csv confirms zero changed non-OA values and five changed OA values among the visible three studies. Therefore this is an executed OA probe with no displayed Public improvement, rather than a missing configuration change. Workers are PASS, not timed out, and have no flip_tta receipt entry.

Retain 25A as the simpler incumbent. There is no new evidence to keep adjusting the OA weight on the same development data.

## Next image change already supported by the pinned worker

The embedded compact_infer.py is public R7 worker v4c. Its existing --flip-tta option implements the trainer's mirror contract:

- Reverse sagittal slice channels and negate their relative position.
- Reflect coronal and axial windows in width; exchange coronal bilateral view IDs 2 and 3.
- Re-index the mirrored output's Medial/Lateral Meniscus and Medial/Lateral OA targets before averaging logits per fold.
- Preserve the plain pass for MCL, because mirrored training batches excluded MCL.
- Read the state heads from the corresponding forwards and apply the existing state/fluid stackers.

This differs from an arbitrary width flip of every plane. The option is already implemented in the exact pinned worker but is absent from the scored 25A/27 path. State-spec category symmetry and binary-head compatibility are necessary; the worker checks them. The new notebook checks category symmetry before decoding and requires all workers to report active mirror TTA.

## Notebook 28

28_R7_Anatomical_Mirror_Gold_Diagnostic.ipynb first reruns plain R7 on the three visible studies and compares its raw probabilities with results 27. A maximum discrepancy above 0.001 stops the experiment. The new stage then scores the 58 gold images using the same R7 code and --flip-tta. Exact R7 and Raptor gold reference predictions from results 26 are embedded, so no saved output needs attachment and no Raptor inference is repeated.

The analysis compares the original 25A 50/50 blend with the same blend using mirrored R7. It uses float32 output scores, handles exact rank ties, reports per-target AUC and Spearman correlation, and produces paired resampling intervals. It reports MCL's probability difference as a check of the plain-only exception. Zero scored-study fallback is required. No training, weight fitting, test-score tuning, or root submission.csv is involved.

Attach three inputs: the RSNA competition including train_series; heliosli/rsna-knee-compact-r7cnxfsss; heliosli/rsna-orthofoundation-fold0-epoch7-20261004. GPU T4x2, Internet Off, Save & Run All. Do not competition-submit notebook 28. The three-hour stop limit is not an ETA; this new TTA duration has not been measured. Its additional mirrored model pass increases the R7 GPU work, while decoding remains once per gold study for that stage. Hidden-test runtime would need consideration before a later submission probe.

Send the Output ZIP with 28_diagnostic_report.json, 28_per_target_auc.csv, 28_preflight.json and 28_r7_tta_gold_receipt.json. If the new image pass improves the combined diagnostic and its target pattern is acceptable, package one prospective hidden-test probe with that exact implementation. If it is harmful or fails the transform contract, stop this serving experiment and return to independent training and report/image supervision work.

## Correction to prior provenance notes

The Raptor author's current account explicitly says training used 4,349 weak competition studies and excluded the 58 gold studies from gradient training, while reusing them for selection. Earlier project wording treated the all-data checkpoint name as evidence that gold images entered gradient training. That inference was too strong. These gold diagnostics still are not independent validation because of upstream development and selection, but the disclosed reason should be stated accurately.

Sources: public worker source embedded and hash-checked in 25A; results 26.zip and results 27.zip; [Raptor author's training account](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792); [host's clinical task and target definitions](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343).
