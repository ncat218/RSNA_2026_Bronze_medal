# RSNA Knee — public-code and domain research, 2026-10-08

## Current evidence

- Protected anchor: Public 0.943. The frozen B2 five-fold standalone scored
  0.914; all-target B2 blend 0.942; three target-group probes scored
  0.942/0.942/0.943. None has demonstrated a displayed improvement.
- Notebook 18 `results 18.zip` is a successful resource smoke test only:
  64 train + 16 validation studies, two epochs, 16 cached slices/study,
  16.2 s total, peak allocated GPU 0.454 GiB, process RSS 1.757 GiB.
  Validation BCE 0.5573 → 0.5453 is not an AUC or LB result. The reported
  2.11 h for five folds × four epochs excludes validation, I/O scale and
  checkpoint overhead.

## Relevant public observations, with evidence strength

1. In a [Kaggle discussion on single-model scores](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304), a participant reports 0.942 raw / 0.943 TTA at 288 px, and says their own labels were not measurably better than public ones. This is self-reported and not a controlled comparison to our model.
2. In a [Kaggle discussion on model/slice choices](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743148), participants report 0.94-range results using roughly 224–288 px and emphasize slice quality, augmentation and regularization. These are useful hypotheses, not verified causal effects; they do not justify changing resolution, slice selection and augmentation simultaneously.
3. A [2026 knee-MRI multi-view research abstract](https://www.ors.org/wp-content/uploads/AM26/1183.pdf) reports multi-view gains for ACL and meniscus tasks and uses spatial constraints/attention. Its datasets, labels and evaluation are different from this competition, so its AUC numbers cannot be transferred to our leaderboard.
4. The [official PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) distinguishes frozen feature extraction from fine-tuning. It supports a paired ablation of encoder trainability, not a prediction that our score will increase.

## Controlled next step

Notebook 19 trains two arms on the same existing fold-0 train and validation
UIDs, same input slices and head initialization: frozen RadImageNet versus
unfrozen ResNet50 layer4. It selects checkpoints by weak validation BCE,
reports weak per-target AUC and Gold58 audit after selection, and creates no
submission. This isolates trainability; the old notebook09 model used up to
80 slices/study and is context, not the matched control.

If layer4 does not beat the matched frozen arm on held-out weak data, do not
spend five-fold GPU time on that exact configuration. If it does, first check
whether gains are broad across labels and stable rather than driven by one
target or the small Gold58 audit. Only then extend the fixed configuration to
five folds, generate test predictions with exactly the same 16-token
preprocessing, and test standalone / protected-anchor blend without using
Public LB to tune many weights.

Separate future hypothesis: improve **slice informativeness** while keeping
the model and fold fixed, e.g. compare uniform 16 tokens with 16 tokens
selected by a predeclared non-label anatomical/quality rule. This addresses
the public-code observation without confounding it with encoder fine-tuning.
Do not use hidden-test labels, reports at inference, or Gold58 to optimize
that choice.
