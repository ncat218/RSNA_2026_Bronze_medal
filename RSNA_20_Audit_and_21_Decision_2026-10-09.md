# RSNA Knee — notebook 20 audit and next probe, 2026-10-09

## Verified result

The newly delivered `results 10.zip` is notebook **20** Output (not the older
notebook 10 output). `density_ablation_report.json` says `COMPLETE`, all 15
fold/arm checkpoints exist, 4,349 weak and 58 Gold studies were processed,
and all five density-80 fold AUCs match notebook 09 exactly. Total runtime:
14.91 minutes, including 629.10 seconds of shared image encoding. No RAM or
GPU guard triggered. The maximum absolute weak-OOF probability difference
between notebook 20 density-80 and notebook 09 is about 5.9e-7.

| Validated weak-label OOF macro AUC | 80 slices | 32 slices | 16 slices |
| --- | ---: | ---: | ---: |
| Five-fold OOF | 0.858970 | 0.846273 | 0.821821 |
| Gold58 audit (not independent selection) | 0.888217 | 0.863451 | 0.836134 |

Density 32 loses 0.01270 macro AUC against density 80. A 250-draw bootstrap
resampling `report_group` gives an approximate 95% interval for the paired
delta of [-0.01522, -0.01035]. Density 16 loses 0.03715, interval
[-0.04023, -0.03359]. The 32-token arm loses on all 12 targets; especially
Lateral Meniscus (-0.04572), ACL (-0.02469), and Baker's (-0.02237). Thus
notebook 19's 16-slice trainability test mainly answers a sparse-input
question; it does not justify fine-tuning all five folds at 16 slices.

## Small complementary signal, not a score claim

An exploratory, fixed 80%/20% **probability** pool of the density-80 and
density-32 heads reaches 0.860584 weak-OOF macro AUC, up 0.001614 versus
density 80 alone. Its group-bootstrap interval is about [+0.00115,+0.00209]
on the same weak labels. The 50%/50% pool improves only 0.000688 and its
interval crosses zero. Since these pools were inspected after seeing the
ablation, the intervals do **not** establish a prospective gain. Gold58 is
not used to select a weight. There is no evidence yet of a Public or Private
LB gain or of complementarity to the protected 0.943 ensemble.

## Action

Notebook 21 is a cheap, standalone Public-LB probe of this single 80/20
candidate. It requires notebook 20 saved Output and the public RadImageNet
ResNet50 weights, not the 05A cache. It does not change the 0.943 notebook.
The unchanged 80-slice branch is validated against the three visible test
predictions of notebook 10 before `submission.csv` is written. Run once,
inspect `b2_density_blend_receipt.json`, submit once, and compare against the
old B2 standalone Public 0.914. Do not optimize the weight on Public LB.

If notebook 21 fails to beat 0.914 materially, stop density-blend work. If it
does, a single fixed-weight anchor blend may be worth testing, but do not
assume its increase transfers to the protected 0.943 model. The next larger
owned-model hypothesis is full-density, anatomically conditioned
plane/sequence fusion under the same grouped folds, with a frozen-80 control
and Gold58 audit only. That is consistent with published multi-plane knee MRI
attention work, but its performance must be measured here; the paper's
reported AUCs are not transferable to this competition.

Sources: local notebook 20 ZIP and notebooks 09/10/19; Kaggle competition
metric/rules; Kunitomi et al., *Clinically Prioritized Attention-Based Fusion
of Multi-Plane Knee MRI for Robust Injury Detection* (2025).
