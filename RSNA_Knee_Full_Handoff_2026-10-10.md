# HANDOFF — RSNA 2026 Knee Abnormality Detection
## Research, experiments, failure analysis, and roadmap toward a Private-LB-robust Top-10 solution

**Last updated:** 2026-10-10  
**Competition deadline:** 2026-10-22  
**Primary objective:** maximize **Private Leaderboard** robustness; aspirational target **~0.96 Private / Top 10**.  
**Important:** 0.96 is a strategic target, not a guaranteed outcome.

---

# 0. Executive handoff

The project has reached the following state:

- The **report/supervision pipeline is already reasonably mature**.
- The current bottleneck is much more likely to be **how the MRI study is represented and learned**, not the basic pseudo-label construction.
- Baseline `04` is scientifically clean but visually weak:
  - frozen DINOv2-Small
  - sparse 8 positions/slot
  - CLS-only features
  - Public LB = **0.748**
  - Gold-58 macro AUC ≈ **0.734**
- `05A` was created as a **dense-slice trainable MIL branch**:
  - ResNet18 from scratch
  - 16 slices/slot
  - 224×224
  - 5 MRI slots
  - 12 pathology queries
  - end-to-end training on 4,349 soft-labeled studies
- The first full `05A` run did **not fail scientifically**; it failed because the Kaggle kernel was killed by **host RAM OOM** after completing fold 0 and entering fold 1.
- The current next action is therefore:
  1. **patch 05A RAM/data-loader behavior without changing the scientific hypothesis**;
  2. run only 2 folds in debug/development mode;
  3. obtain trustworthy Gold-58 audit metrics;
  4. decide whether a full 5-fold run is worth the compute.
- Parallel strategic roadmap:
  - `05B` backbone/pretraining ladder
  - `05C` image representation/sampling/resolution ablations
  - then `06` MRI↔multilingual-report contrastive learning
  - `07` true CoPAS-lite cross-plane model
  - `08` global+local anatomical evidence
  - finally a **diverse ensemble of strong families**

The central project philosophy is:

> **Do not build one increasingly complicated monolithic model. Build a small number of strong, scientifically distinct model families, keep the 58 gold studies locked, and ensemble only models that are both strong and complementary.**

---

# 1. Competition framing

## 1.1 What one study actually is

A competition sample is not one 2D image.

Conceptually:

```text
StudyInstanceUID
│
├── Sagittal MRI series
│   ├── slice 1.dcm
│   ├── slice 2.dcm
│   └── ...
│
├── Coronal MRI series
│   └── ...
│
├── Axial MRI series
│   └── ...
│
├── possibly multiple sequences / multiple series per plane
│
└── train-time radiology report
```

EDA findings already established:

- **4,407 train studies**
- **24,371 train series**
- mean ≈ **5.53 series/study**
- every study has sagittal/coronal/axial coverage
- typical series length ≈ 30 slices, but the distribution has a long tail
- scanner, pixel spacing, slice thickness, series count, and FOV vary substantially

## 1.2 The 12 targets

The task is multi-label prediction of:

1. ACL
2. MCL
3. Medial Meniscus
4. Lateral Meniscus
5. Medial OA
6. Lateral OA
7. PF OA
8. Effusion
9. Synovitis
10. Baker's
11. Contusion
12. Fracture

Multiple labels may be positive simultaneously.

Metric:

```text
macro ROC-AUC across the 12 targets
```

Therefore each target contributes equally.

## 1.3 Gold labels versus probabilities

The official structured gold labels are **binary 0/1**, not probabilities.

Only **58 studies** have the full official 12-label vector.

The other **4,349 studies** have:

```text
MRI + radiology report
but no complete official 12-label vector
```

At test time the model must output probabilities.

---

# 2. Core data/supervision strategy

The competition is best understood as **two coupled problems**.

## 2.1 Supervision problem

Most train studies do not have official structured labels.

Therefore:

```text
radiology report
    ↓
weak / silver supervision
    ↓
MRI model training target
```

## 2.2 MRI representation problem

MRI is:

```text
multi-plane
+
multi-series / multi-sequence
+
multi-slice
+
3D anatomy
```

The model must learn a study representation without losing localized pathology.

These two problems finally meet at:

```text
MRI
→ model
→ 12 probabilities
```

At inference, **MRI must be sufficient**.

---

# 3. Hard project principles

1. **Gold-58 is locked audit only.**
2. Gold studies do not enter optimizer, weak train, weak validation, early stopping, model selection, or ensemble weight tuning.
3. Main training set = **4,349 weak studies**.
4. Main supervision = **continuous L1 soft labels**.
5. Raw report may be used for train-time representation learning later, but not at inference.
6. DICOM slice order must come from **IOP + IPP geometry**, not filename.
7. Crop must respect **physical millimeter scale** before resize.
8. Do not drop studies simply because some series slots are missing; use masks.
9. Duplicate normalized reports must stay in the same CV group.
10. No horizontal flip by default because multiple targets are medial/lateral-specific.
11. Public LB is an occasional confirmation signal, **not a hyperparameter optimizer**.
12. Final inference:
   - Internet OFF
   - notebook-only
   - runtime < 9h
   - preferred internal safety budget ≤ 6–7h
   - all weights/preprocessing available offline

---

# 4. Notebook history and current project state

## 4.1 Notebook 00 — EDA

Key findings:

### DICOM ordering

Lexicographic filename sorting was anatomically wrong in the sampled audit.

Correct approach:

```python
normal = cross(IOP[:3], IOP[3:])
position = dot(IPP, normal)
sort slices by position
```

Fallback to `InstanceNumber` only when geometry is unavailable, and log fallback usage.

### Physical scale

Observed in-plane resolution varies strongly.

Therefore:

```text
raw image
→ determine pixels corresponding to physical crop
→ crop fixed-mm FOV
→ resize
```

rather than naïvely resizing the entire matrix.

### Metadata quality

Raw text metadata can be dirty. Baseline routing uses cleaner competition fields such as:

```text
Anatomical_Plane
Fluid_Sensitive
```

rather than unreliable raw text descriptions.

---

## 4.2 Notebook 01 — Label audit

Eight pseudo-label sources were audited.

Important result:

- `S07` reproduced all 58×12 gold cells exactly
- therefore it was classified as effectively gold-contaminated / leaky
- `S07` was excluded from normal teacher construction

Several other sources were not independent:

- S02 ≈ S03
- S01 ≈ S06
- S04/S05 related

This led to family-level aggregation rather than naïvely voting eight files.

---

## 4.3 Notebook 02 — Silver label construction

Independent teacher families:

```text
GPT56SOL
HYBRID_BLEND
STEVEN_CORE
YUNUS
```

Locked Baseline V1 supervision:

```text
L1_family_mean_RECOMMENDED.csv
```

Properties:

- 4,349 weak studies
- 12 continuous probabilities
- no gold rows
- no NaNs
- no hard thresholding for training

Important conclusion:

> Report supervision is already substantially better engineered than a simple hard regex or one-LLM pseudo-label pipeline.

---

## 4.4 Notebook 03 — Series selection / baseline preparation

Baseline slots:

```text
SAG_FLUID
SAG_NONFLUID
COR_FLUID
COR_NONFLUID
AX_FLUID
```

`AX_NONFLUID` was deferred because availability is low.

Missing slots are masked.

Physical crop:

```text
140 mm
```

The deterministic series manifest was produced here.

---

## 4.5 Notebook 04 — Frozen-DINO baseline

Data flow:

```text
MRI study
  ↓
5 deterministic slots
  ↓
correct geometry sorting
  ↓
8 positions / available slot
  ↓
(prev, center, next)
  ↓
2.5D 3-channel image
  ↓
140 mm physical crop
  ↓
256 × 256
  ↓
DINOv2-Small FROZEN
  ↓
384-D CLS token
  ↓
slot embedding + normalized position
  ↓
12 learnable pathology queries
  ↓
masked label-specific attention
  ↓
12 logits
  ↓
BCEWithLogitsLoss with L1 soft targets
```

Raw reports are **not** model inputs.

Reports are only used through:

```text
L1 soft labels
+
report_group for CV grouping
```

Gold-58 is audit only.

### Results

```text
Public LB ≈ 0.748
Gold-58 macro AUC ≈ 0.7338
runtime ≈ 3h35
```

Gold per-target pattern from the completed run:

- Effusion ≈ 0.872
- Medial OA ≈ 0.845
- Lateral OA ≈ 0.770
- Baker’s ≈ 0.770
- Medial Meniscus ≈ 0.757
- Synovitis ≈ 0.750
- Contusion ≈ 0.722
- Lateral Meniscus ≈ 0.709
- PF OA ≈ 0.708
- ACL ≈ 0.701
- Fracture ≈ 0.611
- MCL ≈ 0.590

Interpretation:

- diffuse/global findings perform relatively better
- localized/small-structure findings are weaker
- this is compatible with the hypothesis that sparse CLS-only representation loses spatial/local information
- but Gold-58 is too small to claim causality

---

# 5. Main research conclusion after studying public Kaggle solutions

The largest likely bottleneck is currently **vision capacity / MRI representation**, not incremental report-label engineering.

Public analysis suggested:

- improving report label source gave relatively small gains
- strong visual backbones / denser image representation gave much larger gains
- strong single-model families can already reach ~0.92–0.94 Public territory
- ensemble gives additional gain, but cannot turn a weak ~0.75 model into a ~0.95 model by itself

Therefore the priority order became:

```text
strong visual model first
→ multimodal/report alignment later
→ plane/sequence modeling
→ diverse ensemble
```

---

# 6. Three research sources and what each contributes

## 6.1 Source A — Public Kaggle solutions

Use this source to answer:

> What actually works on this exact competition?

Main lessons:

- dense or better-chosen slice coverage matters
- adjacent 2.5D windows are strong
- physical crop ~130–140 mm is common in strong pipelines
- higher resolution around ~336–384 is plausible
- trainable backbone matters
- disease-specific pooling/attention is useful
- multiple strong checkpoints help
- rank/probability blending can help
- Public-LB-probed weighting is dangerous and should not be copied blindly

Strong public family characteristics seen during research included:

```text
~48–64 image windows / study
2.5D adjacent slices
physical crop around 130–140 mm
336–384 resolution
trainable CoAtNet-like / DINO / medical CNN families
disease-specific pooling
soft report labels
```

Important caution:

Many 0.93–0.94 notebooks are related forks / descendants of the same public lineage. Do not count them as independent scientific evidence.

---

## 6.2 Source B — CoPAS / knee MRI literature

Use this source to answer:

> How should a multi-plane, multi-sequence MRI study be represented?

Useful CoPAS concepts:

- preserve depth
- plane-specific branches
- cross-plane attention
- cross-sequence interaction
- auxiliary plane supervision

Do not copy immediately:

- full heavy CoPAS
- many 3D encoders
- strict requirement for all sequences
- 11-volume construction

Target is **CoPAS-lite**, not full CoPAS.

---

## 6.3 Source C — MRI ↔ Report / medical VLM research

Use this source to answer:

> How can the report teach the MRI encoder while keeping test inference MRI-only?

Preferred formulation:

```text
MRI → image encoder → z_image
                       ↕
                  contrastive / distillation
                       ↕
Report → text encoder → z_text
```

Meanwhile:

```text
MRI tokens
→ 12 disease-aware queries
→ 12 logits
→ soft BCE
```

At inference:

```text
MRI only
→ vision branch
→ 12 probabilities
```

Do **not** prioritize:

```text
MRI → generated report → classifier
```

The reports are multilingual, so a future text encoder must be explicitly multilingual or preceded by a controlled translation/normalization pipeline.

Candidates discussed:

- BGE-M3
- XLM-R-derived models
- multilingual sentence encoders
- translation to English + medical English encoder

BGE-M3 is a candidate, **not an assumed winner**.

---

# 7. Disease-specific modeling conclusion

Do not build 12 complete backbones.

Preferred design:

```text
shared MRI anatomy representation
        ↓
12 disease-specific queries / attention heads
```

Each disease may learn different:

- plane importance
- sequence importance
- slice importance
- spatial region importance

Baseline 04 already has disease-specific queries, but they receive relatively poor inputs:

```text
sparse DINO CLS features
```

Future improvement should focus on giving them **richer MRI tokens**, not multiplying complete backbones.

---

# 8. Current notebook 05A — architecture

Existing notebooks:

```text
05A_RSNA_Knee_DenseSlice_MIL_Train.ipynb
05A_RSNA_Knee_DenseSlice_MIL_Submit.ipynb
```

## 8.1 Scientific hypothesis

Question:

> Does denser MRI coverage plus a trainable lightweight image encoder beat sparse frozen DINO features?

Important caveat:

05A changes several factors at once:

```text
frozen DINO
→ trainable ResNet18

sparse 2.5D positions
→ denser single-slice representation
```

Therefore a gain cannot be attributed to slice density alone.

This is why a later `05B` backbone/pretraining ladder is needed.

## 8.2 Current config

```text
Experiment:
05A_RSNA_Knee_MultiPlane_DenseSlice_MIL

IMAGE_SIZE = 224
PHYSICAL_CROP_MM = 140
N_SLICES_PER_SLOT = 16
SLICE_DROPOUT = 0.10

Slots:
SAG_FLUID
SAG_NONFLUID
COR_FLUID
COR_NONFLUID
AX_FLUID

Backbone:
ResNet18
weights=None

Token dim = 256
Attention heads = 4

Encoder LR = 1e-4
Head LR = 3e-4
Weight decay = 1e-4
Max epochs = 15
Patience = 4

Study batch size = 2
```

Architecture:

```text
single grayscale slice
→ repeated to 3 channels
→ ResNet18
→ 512-D feature
→ projection to 256-D
→ slot embedding
→ normalized slice-position embedding
→ 12 pathology queries
→ masked multi-head MIL attention
→ 12 outputs
```

Training:

- 4,349 weak studies
- L1 soft labels
- soft BCE
- GroupKFold using `report_group`
- gold completely excluded from fold construction
- early stop based on weak validation soft BCE
- gold audit after checkpoint selection

Submit notebook:

- adaptive slice budget `16 → 12 → 8`
- benchmark before full hidden inference
- deadline-aware reduction if needed

---

# 9. First 05A full run — failure analysis

## 9.1 What happened

Kaggle reported:

```text
Your notebook tried to allocate more memory than is available.
```

Later:

```text
Kernel died while waiting for execute reply.
DeadKernelError: Kernel died.
```

This happened around **7,338 s**.

The run had already completed:

```text
fold 0: all 15 epochs
fold 1: started and completed at least epoch 2
```

Therefore this was not an immediate syntax/model failure.

## 9.2 Training behavior before crash

Fold 0 improved normally:

```text
epoch 1   val BCE ~0.5792   diag AUC ~0.6358
epoch 8   val BCE ~0.5452   diag AUC ~0.7139
epoch 12  val BCE ~0.5355   diag AUC ~0.7310
epoch 13  val BCE ~0.5334   diag AUC ~0.7330
epoch 14  val BCE ~0.5348   diag AUC ~0.7336
epoch 15  val BCE ~0.5365   diag AUC ~0.7288
```

Interpretation:

- training numerically stable
- validation improved
- no obvious NaN/divergence
- model was learning something
- crash was engineering/OOM, not evidence that the hypothesis failed

## 9.3 Cache facts

```text
cache GiB ≈ 11.895
average valid slices/study ≈ 74.30
reused attached cached studies = 4407
decode failures = 0
trainable parameters ≈ 11.65M
```

The ~11.9 GiB cache is **disk cache**, not automatically 11.9 GiB of resident RAM.

---

# 10. Likely RAM root cause

Cached images are efficient:

```text
uint8 grayscale
[T, H, W]
```

But the CPU collate pipeline expands them to:

```text
float32 RGB
[T, 3, H, W]
```

before GPU transfer.

Per-pixel amplification:

```text
1 byte
→ 3 channels × 4-byte float32
≈ 12×
```

With roughly 74 slices/study and batch size 2, one CPU image batch alone can be ~90 MB before accounting for:

- padding
- worker copies
- prefetch
- pinning
- numpy/Python overhead
- multiple loaders
- workers surviving across folds

Existing loader behavior:

```text
num_workers = 2
persistent_workers = True
prefetch_factor = 2
pin_memory = True
```

This is consistent with gradual RAM accumulation and host OOM.

---

# 11. 05A RAMFIX plan

Scientific hypothesis must remain unchanged.

## 11.1 CPU representation

Keep:

```text
[B, T, H, W]
uint8
```

Do not convert to float32 or RGB on CPU.

After H2D:

```text
GPU:
uint8
→ float / 255
→ add channel
→ expand to 3 channels
→ flatten [B*T,3,H,W]
→ ResNet18
```

Prefer `expand` over `repeat` where safe.

## 11.2 Conservative loader

Use:

```text
NUM_WORKERS = 1
PERSISTENT_WORKERS = False
PREFETCH_FACTOR = 1
PIN_MEMORY = False
```

## 11.3 Cleanup between folds

Explicitly delete:

- model
- optimizer
- scheduler
- train/val/gold loaders
- large arrays/tensors

Then:

```python
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()  # when safe
```

## 11.4 Memory monitoring

Track and save:

- process RSS GiB
- system used/available RAM
- CUDA allocated GiB
- CUDA reserved GiB
- CUDA max allocated GiB

Save:

```text
memory_history.csv
```

---

# 12. Development mode after OOM

Add:

```text
DEV_NUM_FOLDS = 2
```

For RAMFIX validation:

- reuse existing fold assignments
- train only first 2 folds
- audit both on gold
- do not regenerate splits
- do not train final full-data model

When `DEV_NUM_FOLDS < 5`, print:

```text
FINAL FULL-DATA TRAINING SKIPPED BECAUSE DEV_NUM_FOLDS < 5
```

Only if 05A demonstrates useful Gold/public signal should a full 5-fold run be considered.

---

# 13. 05A augmentation patch

Current 05A has almost no image-level augmentation beyond slice dropout.

Because ResNet18 is trained from scratch, add mild augmentation:

```text
rotation approximately ±7–10°
brightness jitter ~0.10
gamma ~0.90–1.10
optional tiny scale/translation
```

Rules:

- training only
- same transform parameters across slices of same series/study when practical
- no horizontal flip
- no vertical flip
- no aggressive elastic transforms
- preserve 140-mm physical crop contract

---

# 14. Fold ensemble issue

Original 05A deployment uses one final full-data model trained for the median best fold epoch.

A 5-fold end-to-end CNN ensemble is **not free**.

Benchmark later:

```text
1 model
2 models
3 models
5 models
```

Evaluate:

```text
AUC gain / inference hour
```

Do not automatically deploy all five folds.

---

# 15. Scanner/site leakage audit

Current split protects duplicate reports:

```text
GroupKFold(report_group)
```

But it may not protect against acquisition-domain shortcuts.

Audit by fold:

- Manufacturer
- MagneticFieldStrength
- pixel spacing
- slice thickness
- rows/columns
- FOV
- other reliable acquisition proxies

Save:

```text
scanner_fold_audit.csv
```

Do not immediately change the split or feed scanner metadata to the model.

---

# 16. 05B — backbone / initialization ladder

Important distinction:

```text
"Do not increase DINO size blindly"
≠
"Do not study backbone family"
```

Keep input/sampling/folds/head as fixed as possible.

Compare:

### B0
```text
ResNet18 scratch
```

### B1
```text
ResNet18 ImageNet pretrained
```

### B2
```text
RadImageNet-pretrained CNN
```

### B3
```text
CoAtNet-like backbone
```

Goal:

> Determine whether the strong public gains associated with stronger backbones reproduce under our label pipeline and leakage-safe evaluation.

This should happen before major MRI-report experiments.

---

# 17. 05C — image representation / sampling ablations

After choosing a strong backbone, test:

## Slice strategy
- uniform sparse
- dense uniform
- adjacent local windows
- center-focused windows
- coarse-to-fine

## Input type
- 2D single slice
- 2.5D adjacent three-slice

## Resolution
- 224
- 336
- 384 if runtime allows

## Physical crop
- 130 mm
- 140 mm
- 150 mm

Change one major variable at a time where possible.

---

# 18. Milestone — strong visual baseline

Do not move aggressively into multimodal architecture until at least one image model is clearly strong.

Desired signs:

- large improvement over 04
- Gold-58 does not collapse
- Public confirms direction without repeated probing
- runtime is safe
- MCL / fracture / meniscus improve, not only diffuse findings

---

# 19. Notebook 06 — MRI ↔ multilingual report

## 19.1 Contrastive batch problem

Current study batch ≈ 2.

Standard CLIP-style in-batch contrastive with batch 2 is a poor test because each anchor has essentially one negative.

Therefore solve this first.

## 19.2 06A — cached feasibility probe

Preferred first experiment:

```text
strong 05 visual encoder
→ freeze
→ cache study embeddings

multilingual text encoder
→ cache report embeddings

image projection
↔
text projection

large effective batch:
128 / 256 / 512
```

Use a **global study embedding** for `z_image` initially.

## 19.3 06B — online contrastive training

Only if 06A is positive.

Possible approaches:

- MoCo-style queue / memory bank
- teacher embedding regression
- cosine distillation
- SigLIP-like objectives
- losses that do not rely on giant in-batch negatives

## 19.4 Multilingual requirement

Candidate routes:

### Route A
```text
raw multilingual report
→ multilingual encoder
```

### Route B
```text
report
→ offline controlled translation / normalization
→ English medical encoder
```

BGE-M3 is only a candidate, not the assumed best model.

---

# 20. Notebook 07 — true CoPAS-lite

Do not call a model CoPAS-lite if it merely adds slot embeddings then performs one flat MIL pooling.

A true CoPAS-lite experiment should have separate branches:

```text
Sag tokens → Sag branch ─┐
Cor tokens → Cor branch ─┼→ cross-plane attention
Ax tokens  → Ax branch ──┘
                          ↓
                   fused representation
                          ↓
                  12 disease queries
```

Example:

```text
Sag' =
Sag
+ Attn(Q=Sag, K/V=Cor)
+ Attn(Q=Sag, K/V=Ax)
```

Question:

> Does structured cross-plane communication beat flat MIL pooling?

---

# 21. Sequence-information caveat

Do not pretend competition metadata gives clean PDW/T1W/T2W labels.

Current reliable concepts are closer to:

```text
plane
fluid-sensitive flag
```

Before sequence-specific modeling, audit:

- DICOM sequence metadata
- TE/TR reliability
- textual sequence descriptors
- consistency of Fluid_Sensitive

If exact sequence type is not trustworthy, keep terminology:

```text
Fluid
Non-fluid
```

rather than falsely mapping to T1/T2/PD.

---

# 22. Notebook 08 — global + local anatomical evidence

Only after a strong global model.

Potential local regions:

- meniscus
- ACL/intercondylar notch
- MCL/medial side
- OA compartments / patellofemoral region
- bone/fracture evidence

Do not build 12 full backbones.

Prefer a few anatomically meaningful shared regions.

---

# 23. External datasets / self-supervised branches

Use external data only for a defined capability.

Possible later sources:

### MRNet
- knee anatomy
- ACL
- meniscus

### fastMRI+
- pathology localization
- lesion-aware learning

### SKM-TEA
- anatomy
- segmentation
- ROI/localization

Before use:

1. verify license
2. verify competition legality
3. verify protocol relevance
4. run controlled experiment
5. keep only if robust gain

---

# 24. Ensemble strategy

Target strong/complementary families:

```text
A. Strong dense 2.5D CoAtNet-like model
B. DINO/ViT partially fine-tuned model
C. MRI↔report-aligned model
D. CoPAS-lite cross-plane model
E. Global+local anatomy model
```

Measure diversity with:

- OOF prediction correlation
- residual correlation
- per-target strengths
- Gold-58 behavior

Blend candidates:

- probability mean
- rank mean
- target-specific weights

Do not:

- Optuna-fit weights on tiny Gold-58
- repeatedly probe Public LB
- ensemble many near-identical checkpoints

---

# 25. Items deliberately excluded

Keep these exclusions:

1. MRI → generated report → disease
2. full CoPAS immediately
3. training/fine-tuning on Gold-58
4. DINO Small→Base/Large only because larger
5. YOLO ROI too early
6. external MRI too early
7. Optuna ensemble weights on Public LB
8. large ensembles of near-identical models
9. horizontal flip
10. repeated Public-LB tuning

Again:

```text
avoid blind scaling
≠
avoid backbone-family research
```

---

# 26. Revised roadmap toward a Private-LB-robust Top-10 attempt

```text
04 — DONE
Frozen sparse DINO
Public = 0.748
        ↓
05A — CURRENT
Dense-slice ResNet18-scratch MIL
        ↓
05A-RAMFIX
Fix CPU RAM path + 2-fold dev audit
        ↓
05B
Backbone / pretraining ladder
        ↓
05C
Slice strategy / 2.5D / resolution / crop
        ↓
================================
STRONG VISUAL BASELINE
================================
        ↓
06A
Cached MRI↔multilingual-report contrastive probe
        ↓ if positive
06B
Online image-report training with queue/distillation
        ↓
07
True CoPAS-lite cross-plane model
        ↓
08
Global + Local anatomical model
        ↓
09+
MRI SSL / justified external-data families
        ↓
DIVERSE ENSEMBLE
        ↓
FINAL PRIVATE-LB-ROBUST SUBMISSION
```

---

# 27. Immediate next action

Use the prepared Codex patch specification to create:

```text
05A_RSNA_Knee_DenseSlice_MIL_Train_RAMFIX.ipynb
```

Required patch:

- keep cached images as uint8 grayscale on CPU
- RGB expansion only on GPU
- conservative DataLoader
- no persistent workers
- explicit GC between folds
- process/GPU RAM logging
- `DEV_NUM_FOLDS = 2`
- skip final full-data model in dev mode
- mild augmentation
- no horizontal flip
- scanner/site audit
- reuse the existing 11.9-GiB cache
- do not change architecture / labels / folds / target order

Then collect:

```text
fold best epochs
validation soft BCE
diagnostic weak AUC
Gold-58 macro AUC/AP
per-target Gold AUC
peak RSS
GPU memory
runtime / epoch
projected 5-fold runtime
```

Decision:

### If Gold signal is poor
Stop 05A and move to `05B`.

### If Gold signal is clearly stronger than 04
Run full 05A if runtime permits and preserve it as a distinct ensemble family.

---

# 28. Experiment tracking table

| ID | Model family | Main purpose | Gold AUC | Public | Status |
|---|---|---|---:|---:|---|
| 04 | Frozen DINO | clean baseline | ~0.734 | 0.748 | DONE |
| 04A | Frozen DINO | longer head sanity check | ? | ? | optional |
| 05A | Dense ResNet18 MIL | dense trainable image family | incomplete | — | RAM OOM |
| 05A-RAMFIX | same as 05A | engineering stabilization | ? | — | NEXT |
| 05B | backbone ladder | backbone/pretraining isolation | ? | ? | planned |
| 05C | visual representation | sampling/resolution/FOV | ? | ? | planned |
| 06A | cached MRI↔report | multimodal feasibility | ? | ? | later |
| 06B | online MRI↔report | encoder learns report semantics | ? | ? | later |
| 07 | CoPAS-lite | structured plane fusion | ? | ? | later |
| 08 | global+local | local anatomy evidence | ? | ? | later |

For every experiment save:

- exact config
- notebook/version
- fold assignments
- OOF predictions
- gold predictions
- per-target metrics
- training history
- runtime
- peak memory
- checkpoint size
- submission predictions
- correlation with previous model families

---

# 29. Decision rules

## Keep standalone model if

- internal metric improves consistently
- Gold-58 does not collapse
- Public confirms direction without repeated probing
- runtime is safe
- offline inference is reproducible

## Keep weaker model for ensemble only if

- predictions are meaningfully decorrelated
- it is strong on targets where others are weak
- internal blend evidence supports its contribution

## Drop model if

- improvement appears only after Public tuning
- runtime threatens cap
- complexity rises sharply without robust gain
- predictions are almost identical to a better model
- weak CV improves but Gold/Public repeatedly disagree

---

# 30. Score strategy

Aspirational objective:

```text
Private LB ≈ 0.96
Top 10
```

Operational plan:

```text
Stage 1
Build a genuinely strong single visual model.

Stage 2
Build 3–5 strong model families with different inductive biases.

Stage 3
Blend only models with complementary errors.

Stage 4
Use limited Public submissions as sanity checks,
not as the optimizer.

Stage 5
Choose final submissions for Private robustness.
```

Priority:

```text
Private robustness
>
Public peak
```

---

# 31. Key unresolved questions

1. Is 05A's dense coverage useful once RAM is fixed?
2. How much gain comes from pretrained initialization alone?
3. Does CoAtNet reproduce public competition gains under our own split/labels?
4. What is the best slice strategy?
5. Is 336 enough or is 384 worth the runtime?
6. Is 2.5D significantly better than single-slice input?
7. Does partial DINO fine-tuning provide a competitive ViT family?
8. Does MRI-report alignment add information beyond L1?
9. Which multilingual text encoder works best?
10. Can scanner/site leakage explain inflated CV?
11. Does laterality normalization matter after horizontal flip is disabled?
12. Does explicit cross-plane attention beat flat MIL pooling?
13. Which model pairs produce the lowest useful error correlation?
14. Can all retained final branches stay under a safe 6–7h inference budget?
15. Which external dataset provides a justified capability rather than just more volume?

---

# 32. Files/artifacts currently relevant

Existing / discussed:

```text
00_RSNA_Knee_EDA_v2.ipynb
01_RSNA_Knee_Label_Audit_v2.ipynb
02_RSNA_Knee_Silver_Label_Construction_v1.ipynb
03_RSNA_Knee_Baseline_V1_Prep.ipynb
04_RSNA_Knee_Baseline_V1_Train.ipynb

05A_RSNA_Knee_DenseSlice_MIL_Train.ipynb
05A_RSNA_Knee_DenseSlice_MIL_Submit.ipynb
```

Important generated data/artifacts:

```text
L1_family_mean_RECOMMENDED.csv
baseline_series_manifest.csv
fold_assignments.csv
study_cache/
```

Planned next file:

```text
05A_RSNA_Knee_DenseSlice_MIL_Train_RAMFIX.ipynb
```

Earlier roadmap documents:

```text
RSNA_Knee_Research_Roadmap_2026-09-27.md
RSNA_Knee_Private096_Top10_Roadmap.md
```

This handoff supersedes them as the most complete state summary.

---

# 33. Handoff instruction for a future chat / agent

If continuing this project elsewhere:

1. Read this entire handoff first.
2. Do not restart research from scratch.
3. Preserve:
   - Gold-58 audit only
   - L1 soft labels
   - no Public-LB overfitting
   - correct DICOM geometry
   - physical crop
   - offline submit
4. Continue from:
   ```text
   05A RAMFIX → 2-fold development run
   ```
5. Ask for:
   - `RESULTS_TO_SEND_BACK.txt`
   - `training_history.csv`
   - `gold_fold_metrics.csv`
   - `gold_fold_per_target_metrics.csv`
   - `memory_history.csv`
   - `scanner_fold_audit.csv`
6. Based on those results decide:
   - full 05A?
   - or move immediately to 05B backbone ladder?

Do not jump to 06/07 before this decision unless new evidence strongly changes the roadmap.

---

# 34. Final project statement

> **The report-derived supervision is already usable. The biggest immediate opportunity is to make the model genuinely good at reading knee MRI. Stabilize and evaluate 05A first; then isolate backbone/pretraining and input representation; only after a strong visual model exists should MRI–report alignment and CoPAS-style cross-plane modeling be added as complementary model families. The final Top-10 attempt should come from a robust ensemble of strong, decorrelated models—not from overfitting Public LB or piling complexity into one pipeline.**
