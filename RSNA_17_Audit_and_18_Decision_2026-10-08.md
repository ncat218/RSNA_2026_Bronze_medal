# RSNA notebook 17 audit and next model gate — 2026-10-08

## Public-LB result and output integrity

The user-provided Kaggle screenshot shows G1 = **0.942**, G2 = **0.942**,
G3 = **0.943**, against the protected 0.943 anchor and the all-target B2
probe 16 = **0.942**. Because Kaggle displays three decimals, do not infer
exact per-group AUC deltas by subtracting the rounded values. G1/G2 are below
the anchor at displayed precision; G3 ties but has not shown an improvement.

Read-only inspection of `results_17 G1.zip`, `results 17 G2.zip`, and
`results_17 G3.zip` found `anchor_b2_group_probe_receipt.json` status
`COMPLETE` for the intended group in each archive. Each visible run had 3
studies, 20 DINO members, 5 A5 folds, 4 Raptor views, 4 CoAt members, all
five B2 folds, zero empty studies, and no logged DICOM decode failures.
Submission/anchor/B2 SHA256 values match bytes in each archive. The four
selected target columns match the predeclared rank formula exactly and the
other eight match the anchor exactly. On this three-study sample all 12 final
values happen to equal the anchor due to coarse rank resolution; this does
not imply the hidden rerun was unchanged. The ZIPs contain no hidden test
labels or full hidden predictions, so no per-target Public or Private AUC is
recoverable from them.

## Why the next experiment is not another blend sweep

The partial DINO+E13 weak OOF proxy had suggested adding B2, but that proxy
omits A5, Raptor and all CoAt branches of the 0.943 anchor. The controlled
Public probes now show no improvement from B2 frozen features. Repeatedly
changing B2 weights against Public LB would risk fitting the displayed
leaderboard rather than Private robustness.

On the exact same 4,349 weak-label UIDs and original report-group folds,
the owned B3 frozen CoAtNet model from `results_12.zip` had macro AUC
**0.778793**, far below B2 RadImageNet frozen **0.858970** from
`results_09.zip`. Gold58 audit macro AUC was **0.740407** versus **0.888217**;
Gold58 is small and has selection-bias caveats for public checkpoints. B3
frozen is not a candidate blend.

The next controlled direction is to adapt the RadImageNet encoder rather
than keep it frozen. This is a hypothesis, not a score promise. The original
[RadImageNet paper](https://pubs.rsna.org/radiology/doi/full/10.1148/ryai.210315)
reports transfer benefits on knee MRI tasks, and the
[PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial)
distinguishes fixed feature extraction from fine-tuning. Neither source
establishes that fine-tuning will improve this competition's Private score.

## Notebook 18 resource preflight

`18_B2_RadImageNet_Layer4_Finetune_Preflight.ipynb` inherits notebook 09's
asset/fold contract. It requires the competition input, notebook 02 silver
label Output, complete 05A study-cache Dataset, and public RadImageNet
ResNet50 checkpoint; it does **not** require notebook 09 heads. It uses one
T4 of a T4×2 session with Internet Off. It runs only 64 fold-0-training and
16 fold-0-validation weak studies for two epochs, selecting at most 16
uniform cached slices per study. Only ResNet50 layer4 and a new MIL head
receive gradients. Gold58 never enters the training dataset. A 15-minute
work cap and RAM/VRAM guards write `preflight_18.json`.

This is **not** a submission and its 16-study validation BCE is not an AUC
claim. Build full grouped-CV training only after the receipt confirms safe
memory and a plausible five-fold time budget. Compare any eventual new model
with B2 on identical UIDs/folds; keep Gold58 audit-only and ensure the
original 0.943 submission remains selected rather than the 0.942 notebook 16
until a better candidate has real evidence.
