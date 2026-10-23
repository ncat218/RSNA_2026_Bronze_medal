# RSNA Knee Abnormality Detection — Codex Project Context

> **Purpose:** Persistent technical context for Codex before implementing the first image-model baseline.
>
> **Project objective:** Build a robust solution for the **RSNA Knee Abnormality Detection** Kaggle competition, with the working goal of a strong **Private Leaderboard** result. Avoid Public-LB overfitting. Experiments must be reproducible, controlled, and compatible with Kaggle notebook constraints.
>
> **Current stage:** Data understanding and weak-label construction are complete. The next notebook is **04 — Baseline V1 Train**.

---

# 1. Notebook numbering and project state

Use this numbering from now on. Do not rename older stages or overwrite their logic.

| No. | Notebook | Status | Main purpose |
|---|---|---|---|
| `00` | `00_RSNA_Knee_EDA_v2.ipynb` | DONE | Full dataset / report / DICOM EDA |
| `01` | `01_RSNA_Knee_Label_Audit_v2.ipynb` | DONE | Audit report-derived pseudo-label sources against the 58 structured gold studies |
| `02` | `02_RSNA_Knee_Silver_Label_Construction_v1.ipynb` | DONE | Remove leakage/dependency and build family-balanced soft silver labels |
| `03` | `03_RSNA_Knee_Baseline_V1_Prep.ipynb` | DONE | Deterministic MRI series selection + geometry/input audit |
| `04` | `04_RSNA_Knee_Baseline_V1_Train.ipynb` | NEXT | First reproducible MRI image baseline |
| `05+` | controlled experiments | FUTURE | Change one factor at a time |

Original generated names corresponding to the numbered notebooks were:

- `RSNA_Knee_EDA_v2.ipynb`
- `RSNA_Knee_Label_Audit_v2_FIXED.ipynb`
- `RSNA_Knee_Silver_Label_Construction_v1.ipynb`
- `RSNA_Knee_Baseline_V1_Prep.ipynb`

Do **not** modify notebooks `00–03` when implementing notebook `04`.

---

# 2. Competition definition

## 2.1 Task

One **knee MRI study** must be mapped to probabilities for 12 clinically important abnormalities:

1. `ACL`
2. `MCL`
3. `Medial Meniscus`
4. `Lateral Meniscus`
5. `Medial OA`
6. `Lateral OA`
7. `PF OA`
8. `Effusion`
9. `Synovitis`
10. `Baker's`
11. `Contusion`
12. `Fracture`

This is a **study-level multi-label classification** problem.

The competition data pairs MRI studies with their original radiology reports in train, but the final inference system must work on test MRI without a test report.

## 2.2 Metric

Main score:

```text
macro ROC-AUC across the 12 targets
```

Each target AUC contributes equally to the final score.

Consequences:

- ranking quality matters more than a fixed 0.5 classification threshold;
- do not optimize a single accuracy threshold as the main objective;
- weak-label scores should remain **soft** where possible;
- per-target behavior matters because macro averaging gives rare targets the same final weight as common targets.

## 2.3 Submission schema

The submission must contain:

```text
StudyInstanceUID,
ACL,
MCL,
Medial Meniscus,
Lateral Meniscus,
Medial OA,
Lateral OA,
PF OA,
Effusion,
Synovitis,
Baker's,
Contusion,
Fracture
```

The final submission file must be named:

```text
submission.csv
```

## 2.4 Important competition constraints

From the official competition overview/rules captured for this project:

- Kaggle **Notebook-only** submission.
- CPU notebook runtime: **<= 9 hours**.
- GPU notebook runtime: **<= 9 hours**.
- Internet disabled during submission.
- Freely/publicly available external data and pretrained models are allowed subject to the competition rules and the source's own license/terms.
- Maximum team size: **5**.
- Maximum submissions: **5/day**.
- Up to **2 final submissions** may be selected.
- Private Leaderboard determines final standing.
- Final submission deadline in the captured competition overview: **2026-10-22, 23:59 UTC**.
- Winners must provide reproducible solution materials/code/weights and other requested documentation.

Engineering implication: every serious experiment must be reproducible offline and ultimately fit notebook inference time limits.

---

# 3. Competition data structure

## 3.1 `train.csv`

Train has one row per MRI study and includes:

- `StudyInstanceUID`
- `Report`
- the 12 structured target columns

Observed training-label structure:

```text
Total train studies                = 4407
Fully structured gold-labeled      = 58
Partially structured labeled       = 0
No structured labels               = 4349
Gold fraction                      = 1.3161%
```

Therefore this is **not a conventional fully-supervised 4407-study training set**.

Practical framing:

```text
4349 weakly supervised studies
+ 58 image-derived structured gold studies
```

The reports are useful weak supervision, but the 58 structured labels are the independent image-level audit set.

## 3.2 `train_series.csv`

Important columns:

- `StudyInstanceUID`
- `SeriesInstanceUID`
- `Fluid_Sensitive`
- `Fat_Suppression`
- `Anatomical_Plane`

Planes:

- `Sagittal`
- `Coronal`
- `Axial`

DICOMs are organized by study and series.

## 3.3 Test

Test requires study-level image prediction.

Do not build a system that needs `Report` during inference.

---

# 4. EDA — high-level dataset structure

Results from notebook `00`.

## 4.1 Number of studies and series

```text
Train studies       = 4407
Train series        = 24371

Series/study:
mean                = 5.5301
median              = 5
min                 = 3
max                 = 14
```

Series counts by plane:

```text
Sagittal            = 9864
Coronal             = 8609
Axial               = 5898
```

Every one of the 4407 studies has at least:

```text
>= 1 Sagittal series
>= 1 Coronal series
>= 1 Axial series
```

Therefore all three anatomical planes are globally available, but individual acquisition types within each plane are not always present.

---

# 5. EDA — structured gold labels

Only 58 studies have all 12 official structured labels.

Observed positive counts:

| Target | Positive / 58 | Prevalence |
|---|---:|---:|
| ACL | 24 | 0.4138 |
| MCL | 9 | 0.1552 |
| Medial Meniscus | 26 | 0.4483 |
| Lateral Meniscus | 23 | 0.3966 |
| Medial OA | 15 | 0.2586 |
| Lateral OA | 11 | 0.1897 |
| PF OA | 21 | 0.3621 |
| Effusion | 35 | 0.6034 |
| Synovitis | 27 | 0.4655 |
| Baker's | 12 | 0.2069 |
| Contusion | 19 | 0.3276 |
| Fracture | 18 | 0.3103 |

Important limitation:

- 58 is very small.
- MCL has only 9 positives.
- Lateral OA has only 11 positives.
- Per-label AUC estimates have substantial uncertainty.

Therefore:

> The 58 gold studies are a **referee/audit set**, not a large validation set to repeatedly optimize against.

For notebook `04`, **do not put these 58 studies into the optimizer and do not use their metrics for early stopping/model selection**.

---

# 6. EDA — report duplication and leakage risk

Report statistics:

```text
Raw unique reports                 = 4276
Normalized unique reports          = 4247
Duplicate normalized-report groups = 51
Studies inside duplicate groups    = 211
Largest duplicate-report group     = 37
Gold studies in duplicate groups   = 1
```

Why this matters:

Silver labels are derived from reports. If identical reports appear in both weak train and weak validation folds, validation can be artificially easy.

Requirement:

> Any weak-label cross-validation must keep identical normalized reports in the **same fold**.

Notebook `03` already contains a `report_group` hash in the study manifest. Reuse it.

Never randomly split studies without respecting `report_group`.

---

# 7. EDA — acquisition metadata

A key finding:

```text
Fluid_Sensitive != Fat_Suppression : 0 cases in current train
```

Thus in observed train data these two flags are redundant.

Do not create two independent branches based only on these two columns as though they carry independent information.

For baseline routing, `Fluid_Sensitive` is sufficient together with `Anatomical_Plane`.

This finding is for observed train; do not encode an unnecessary hard assertion that future/test metadata can never differ unless verified.

---

# 8. EDA — slice counts

A 3000-series sample showed:

```text
mean slices          = 33.913
median               = 30
min                  = 11
p95                  = 45
p99                  = 160
max                  = 320
```

Interpretation:

- typical knee MRI series is around 30 slices;
- there is a very long slice-count tail;
- reading every slice of every series is computationally wasteful;
- fixed anatomical-position sampling or attention over a bounded number of slices is preferable.

From notebook `03`, selected `AX_FLUID` has especially long stacks:

```text
AX_FLUID selected:
median slices = 32
p95 slices    = 128
max           = 192
```

So the baseline must not simply load all axial slices.

---

# 9. Critical DICOM finding — never sort by filename

Slice-order audit:

```text
Audited series                              = 150
Filename lexicographic order anatomically
monotonic fraction                          = 0.0000
```

That is:

```text
0 / 150
```

Filename/SOP UID lexical order is not anatomical order.

### Required ordering

For each DICOM slice:

```python
row_dir = IOP[:3]
col_dir = IOP[3:]
normal = cross(row_dir, col_dir)
position = dot(IPP, normal)
```

Sort by `position`.

Where:

- `IOP` = `ImageOrientationPatient`
- `IPP` = `ImagePositionPatient`

Notebook `03` re-audited 250 selected series:

```text
250 / 250 had complete usable geometry metadata for all slices
```

Therefore geometry-based ordering is a hard baseline requirement.

If notebook `04` implements a fallback for rare test failures:

1. use IOP+IPP first;
2. fallback to `InstanceNumber` only if geometry is missing;
3. log/count every fallback;
4. never silently use lexical filename order.

---

# 10. DICOM decoding / header quality

EDA:

```text
DICOM header sample        = 6000
Header read errors         = 0

Pixel decode sample        = 50
Successful pixel decodes   = 50 / 50
```

This is encouraging but loaders must still use robust `try/except` and log corrupt/unreadable cases.

Do not silently replace failed studies with unlogged constants.

---

# 11. Strong scanner/protocol domain shift

DICOM header sample size: 6000 series.

Approximate manufacturer-family mix:

```text
Siemens family     ~40.5%
Philips family     ~31.3%
GE family          ~22.0%
Canon/Toshiba       ~5.7%
Other               ~0.6%
```

Field strength:

```text
1.5 T              3447 / 6000
3.0 T              2226 / 6000
missing             293
1.16 T               33
1.0 T                 1
```

This indicates significant acquisition-domain heterogeneity.

Do not assume a single scanner/vendor protocol.

For later robustness experiments consider vendor/field-strength stratified analysis, but do not add scanner metadata to Baseline V1 without a controlled experiment.

---

# 12. Pixel spacing, physical scale, and FOV

DICOM header sample:

### Pixel spacing

```text
median             ~0.3125 mm/pixel
p05                ~0.1666
p95                ~0.5625
observed range     ~0.0728 to 1.1719 mm/pixel
```

### Slice thickness

```text
median             = 3.0 mm
p05                = 2.0 mm
p95                = 4.0 mm
observed range     ~0.6 to 6.0 mm
```

The large in-plane resolution variation means:

> Naively resizing the full pixel matrix gives the same tensor size but **not the same anatomical physical scale**.

Baseline decision:

```text
physical center crop = 140 mm
```

Then resize that physical crop to a fixed network resolution.

Notebook `03` showed FOV >= 140 mm for most selected series:

| Slot | FOV >= 140 mm |
|---|---:|
| SAG_FLUID | 94.99% |
| SAG_NONFLUID | 98.99% |
| COR_FLUID | 91.45% |
| COR_NONFLUID | 93.66% |
| AX_FLUID | 92.22% |
| AX_NONFLUID | 98.02% |

For cases with FOV smaller than 140 mm, crop to available image bounds and handle deterministically; do not crash.

Future controlled experiment:

```text
130 mm vs 140 mm vs 150 mm
```

Do not change this inside Baseline V1.

---

# 13. Raw DICOM text metadata is unreliable

From the 6000-header sample:

```text
SeriesDescription == "DummySeriesDesc!"     = 714
SeriesDescription missing                    = 389
ProtocolName missing                         = 6000 / 6000
Laterality missing                           = 3123 / 6000
ImageLaterality missing                      = 6000 / 6000
```

Therefore baseline sequence routing must rely primarily on:

- official `Anatomical_Plane`
- official `Fluid_Sensitive`
- DICOM geometry

Do not base Baseline V1 routing on free-text `SeriesDescription`, `ProtocolName`, or laterality.

---

# 14. Notebook 01 — pseudo-label audit

Because 4349/4407 studies have no official structured labels, several public report-derived label sources were audited against the 58 gold studies.

Eight direct TRAIN label files were found. Each covered all 4407 studies and 12 targets:

```text
S01 hybrid/report_labels_v4hybrid.csv
S02 GPT56Sol/labels_llm_gpt56sol.csv
S03 GPT56Sol/report_labels_gpt56sol.csv
S04 Steven/llm_labels_full.csv
S05 Steven/llm_labels_v2.csv
S06 Steven/llm_labels_v4_blend.csv
S07 4-source-merged/report_labels_v5.csv
S08 Yunus/yunus_llm_labels.csv
```

Initial macro AUC against the 58 gold studies:

| Source | Macro AUC | Macro AP |
|---|---:|---:|
| S07 | 1.0000 | 1.0000 |
| S01 | 0.8991 | 0.8273 |
| S06 | 0.8927 | 0.8277 |
| S05 | 0.8873 | 0.8216 |
| S08 | 0.8798 | 0.7719 |
| S04 | 0.8780 | 0.8129 |
| S02 | 0.8352 | 0.6672 |
| S03 | 0.8352 | 0.6672 |

However S07 was not genuinely perfect.

---

# 15. Critical label finding — S07 gold contamination

S07 had:

```text
exact gold cells matched = 696 / 696
exact gold match rate    = 1.0
```

And:

```text
58 studies × 12 targets = 696 cells
```

Its overall binary fraction was also:

```text
0.01316088
```

which exactly corresponds to the 58/4407 gold-study fraction when applied across all 12 labels.

Conclusion:

> `S07 / report_labels_v5.csv` contains the official gold labels for the 58 gold studies and therefore cannot be evaluated on those same 58 as an independent pseudo-label source.

S07 is marked:

```text
REFERENCE_LEAKY
```

It must **not** be used to determine label-source reliability from the 58 gold studies.

---

# 16. Pseudo-label source dependency

The eight files are not eight independent voters.

Observed dependencies:

```text
S02 vs S03:
mean absolute difference = 0
hard disagreement        = 0
```

They are effectively duplicates.

```text
S01 vs S06:
mean absolute difference ~0.008985
hard disagreement        ~0.000321
```

They are nearly the same label family.

```text
S04 vs S05:
mean absolute difference ~0.012025
hard disagreement        ~0.022880
```

They are another closely related family.

Therefore labels were grouped into independent families.

---

# 17. Notebook 02 — final silver-label families

After excluding S07:

```text
GPT56SOL       = S02 + S03
HYBRID_BLEND   = S01 + S06
STEVEN_CORE    = S04 + S05
YUNUS          = S08
```

Family macro performance on 58 gold studies:

| Family | Macro AUC | Macro AP |
|---|---:|---:|
| HYBRID_BLEND | 0.894675 | 0.829227 |
| STEVEN_CORE | 0.887349 | 0.821612 |
| YUNUS | 0.879751 | 0.771891 |
| GPT56SOL | 0.835194 | 0.667153 |

Candidate silver-label strategies:

| Candidate | Uses gold to construct? | Macro AUC | Macro AP |
|---|---|---:|---:|
| L3 Gold-guided | YES | 0.895920 | 0.841568 |
| **L1 Family Mean** | **NO** | **0.895760** | **0.841357** |
| L0 Hybrid only | NO | 0.894675 | 0.829227 |
| L2 Family Median | NO | 0.893993 | 0.831613 |

L3's apparent gain is tiny and its weights are derived from the same 58 gold studies, creating meta-overfit risk.

### Locked supervision for Baseline V1

Use:

```text
L1_family_mean_RECOMMENDED.csv
```

Properties:

```text
rows                 = 4349
targets              = 12
labels               = soft probabilities
58 gold studies      = excluded
```

Do not threshold these labels at 0.5.

---

# 18. Silver-label agreement / difficult targets

Mean family-agreement weights among the 4349 weak studies:

| Target | Mean agreement |
|---|---:|
| ACL | 0.8961 |
| MCL | 0.8796 |
| Lateral Meniscus | 0.8737 |
| Medial Meniscus | 0.8573 |
| Medial OA | 0.8114 |
| PF OA | 0.8018 |
| Fracture | 0.7998 |
| Lateral OA | 0.7995 |
| Contusion | 0.7978 |
| Baker's | 0.7598 |
| Effusion | 0.7280 |
| Synovitis | 0.6553 |

Interpretation:

- report-derived supervision is relatively stable for ligament/meniscus targets;
- `Synovitis` is the most uncertain report-derived target;
- `Effusion` is also relatively disagreement-heavy.

Baseline V1 must **not** add agreement weighting yet.

Keep:

```text
L1_agreement_weights.csv
```

for a later controlled experiment.

---

# 19. Notebook 03 — deterministic MRI series selection

Candidate slots:

```text
SAG_FLUID
SAG_NONFLUID
COR_FLUID
COR_NONFLUID
AX_FLUID
AX_NONFLUID
```

Selected-series availability:

```text
SAG_FLUID       4150 / 4407
SAG_NONFLUID    4266 / 4407
COR_FLUID       4248 / 4407
COR_NONFLUID    3406 / 4407
AX_FLUID        4407 / 4407
AX_NONFLUID      857 / 4407
```

Original five-slot set:

```text
SAG_FLUID
SAG_NONFLUID
COR_FLUID
COR_NONFLUID
AX_FLUID
```

All five exist together in only:

```text
3082 / 4407 = 69.93%
```

Therefore **never drop a study simply because COR_NONFLUID is missing**.

Four high-coverage core slots:

```text
SAG_FLUID
SAG_NONFLUID
COR_FLUID
AX_FLUID
```

All four are present in:

```text
3850 / 4407 = 87.36%
```

Still, Baseline V1 should support a missing slot rather than dropping the remaining ~12.6%.

### Locked Baseline V1 slot design

Use five modeled slots:

```text
SAG_FLUID
SAG_NONFLUID
COR_FLUID
AX_FLUID
COR_NONFLUID   # optional
```

with an explicit availability/token mask.

`AX_NONFLUID` is not part of Baseline V1 and can be tested later.

---

# 20. Notebook 03 — selected-series geometry

Selected slot summary:

| Slot | selected | median slices | median spacing | FOV >= 140mm |
|---|---:|---:|---:|---:|
| SAG_FLUID | 4150 | 30 | 0.3320 mm | 94.99% |
| SAG_NONFLUID | 4266 | 30 | 0.3320 mm | 98.99% |
| COR_FLUID | 4248 | 30 | 0.3206 mm | 91.45% |
| COR_NONFLUID | 3406 | 30 | 0.3036 mm | 93.66% |
| AX_FLUID | 4407 | 32 | 0.3143 mm | 92.22% |
| AX_NONFLUID | 857 | 34 | 0.2273 mm | 98.02% |

The deterministic selection rule in notebook `03` prefers, within each study × slot:

1. sensible slice count;
2. FOV large enough for the physical crop;
3. higher in-plane resolution;
4. non-extreme slice count;
5. deterministic SeriesInstanceUID tie-break.

Notebook `04` should reuse:

```text
baseline_series_manifest.csv
```

for train series selection rather than implementing a new train selection policy.

For test, construct the equivalent manifest using the **same rule**.

---

# 21. Locked Baseline V1 principles

Baseline V1 is intentionally clean, not a maximal ensemble.

## 21.1 Supervision

```text
Weak training studies             = 4349
Targets                           = L1 family-mean soft labels
Gold studies in optimizer         = 0
Gold studies used for early stop  = NO
Gold studies used for final audit = YES
```

## 21.2 MRI routing

Five slots with masks:

```text
SAG_FLUID
SAG_NONFLUID
COR_FLUID
AX_FLUID
COR_NONFLUID (optional)
```

No study is dropped merely because a slot is absent.

## 21.3 Slice sorting

Always geometry-based:

```text
IOP + IPP -> slice normal -> projected position -> sort
```

## 21.4 Physical preprocessing

Locked first baseline:

```text
physical crop       = 140 mm
network resolution  = 256 x 256
input               = 2.5D adjacent slices
```

Resolution 336 and FOV 130/150 mm are future experiments, not Baseline V1.

## 21.5 Encoder

First baseline direction:

```text
DINOv2-Small
```

Use a **local/offline attached model**. Internet must not be required.

If the DINOv2 checkpoint/model directory is missing, fail clearly; do not silently replace it with random initialization.

A public high-scoring inference notebook attached during research loads DINOv2 through:

```python
from transformers import AutoModel
bb = AutoModel.from_pretrained(local_dinov2_path)
```

and discovers a local directory containing `config.json` with `dinov2` in the path.

This is a useful offline-loading pattern.

## 21.6 Baseline pooling

The desired first study model:

```text
MRI slot
  -> sampled slice positions
  -> 2.5D triplets
  -> frozen DINOv2-S feature vectors
  -> add slot identity / normalized slice position
  -> label-specific attention
  -> 12 logits
```

Use a separate attention query / pooling behavior for each pathology.

Why:

- ACL signal is not located like PF OA;
- meniscus evidence differs from effusion;
- learned label-specific attention is preferable to manually hard-coding fixed disease weights.

Missing tokens/slots must be masked in attention.

## 21.7 Loss

Baseline:

```text
BCEWithLogitsLoss with soft targets
```

No hard pseudo-label threshold.

No agreement weighting in notebook `04`.

No class weighting unless separately justified and recorded; keep the first baseline simple.

---

# 22. Validation design for notebook 04

The 58 gold studies are too small to drive model selection.

Recommended Baseline V1 design:

### Weak-label CV

Use:

```text
5-fold GroupKFold
groups = report_group
```

over the 4349 weak studies.

Selection/early stopping:

```text
weak validation macro ROC-AUC
```

This metric is imperfect because the targets are silver, but it can control head training without contaminating the independent gold audit.

### Gold audit

For every fold checkpoint:

- predict all 58 gold studies;
- store predictions;
- never choose epoch based on gold;
- after fold training is finished, average fold predictions;
- report per-target and macro gold AUC/AP with clear small-sample caution.

Gold is the **audit**, not the tuning loop.

---

# 23. Suggested efficient Baseline V1 implementation

To fit Kaggle runtime and enable later ablations:

## Stage A — feature extraction

Use the frozen DINOv2-S backbone once to create reusable feature caches.

For each study/slot:

1. load selected series;
2. sort by DICOM geometry;
3. sample a bounded number of normalized anatomical positions;
4. construct 2.5D `[previous, center, next]`;
5. crop 140 mm physical FOV;
6. resize to 256;
7. normalize for DINO;
8. extract CLS feature;
9. save float16 feature + slot ID + normalized position + mask.

A clean first configuration is:

```text
8 sampled center positions per available slot
```

This is deliberately simple.

Later notebook numbers can vary slice count.

Do not sample all slices.

## Stage B — small trainable study head

The study head can be trained cheaply on cached features.

Suggested components:

- LayerNorm
- learnable slot embedding
- optional small projection MLP
- 12 learnable label queries
- masked label-specific attention across all valid slot/slice tokens
- 12 independent logits

This provides a strong, interpretable baseline while preserving compute for future controlled experiments.

---

# 24. Robust preprocessing requirements

Notebook `04` should explicitly implement and log:

### Pixel pipeline

For each raw DICOM slice:

1. convert pixel array to float32;
2. apply rescale slope/intercept if present;
3. handle MONOCHROME1 inversion if encountered;
4. robust intensity clipping (e.g. 1st–99th percentile);
5. normalize to `[0,1]`;
6. create 2.5D triplet;
7. physical center crop;
8. resize;
9. apply ImageNet normalization expected by the pretrained encoder.

Do not add CLAHE/N4/Nyúl in Baseline V1.

Those are later experiments.

### Physical crop

Given spacing:

```text
crop_h_pixels = round(140 / row_spacing)
crop_w_pixels = round(140 / col_spacing)
```

center crop to image bounds.

### Missing sequence

Do not invent an image.

Represent the entire missing slot through:

```text
mask = 0
```

and attention masking.

---

# 25. No-report-at-inference rule

Reports are used for:

- deriving weak labels;
- grouping duplicate reports for CV.

Reports must **not** be input features of the final image model.

The inference path must remain:

```text
MRI -> predictions
```

not:

```text
MRI + report -> predictions
```

because test does not provide the training report supervision in the same way.

---

# 26. Baseline V1 outputs that must be saved

Notebook `04` should save:

```text
config.json
feature_manifest_train.csv
feature_manifest_gold.csv
feature_manifest_test.csv
fold_assignments.csv

fold_0_head.pt
...
fold_4_head.pt

weak_cv_metrics.csv
gold_fold_metrics.csv
gold_ensemble_metrics.csv
gold_predictions.csv

submission.csv

RESULTS_TO_SEND_BACK.txt
RSNA_Knee_Baseline_V1_Artifacts.zip
```

If feature caches are large, they do not need to be packed inside the small diagnostics ZIP; save their paths and hashes/config instead.

`RESULTS_TO_SEND_BACK.txt` should include at minimum:

- runtime
- exact input datasets/model paths
- exact seed
- DINO model identifier/config
- number of usable studies/features
- slot coverage
- DICOM fallback counts
- feature extraction errors
- fold weak macro AUC
- each fold gold macro AUC
- ensemble gold macro AUC
- per-target ensemble gold AUC
- submission row count
- NaN/nonfinite count
- min/max probabilities

---

# 27. Required integrity assertions in notebook 04

Before training:

```text
len(weak labels) == 4349
len(gold) == 58
intersection(weak IDs, gold IDs) == 0
12 target columns exist
no duplicated StudyInstanceUID in silver labels
```

For every CV fold:

```text
report_group(train) ∩ report_group(val) == empty
gold IDs not in optimizer
```

Before submission:

```text
submission StudyInstanceUID order == test.csv order
submission rows == test rows
submission has exactly 13 columns
all predictions finite
all predictions in [0,1]
```

For DICOM loading:

```text
filename lexical order is never the primary sort method
```

---

# 28. Domain knowledge — useful later, NOT hard-coded into Baseline V1

These observations motivate future experiments but should not become fixed manual weights in notebook `04`.

| Finding | Strong planes / regions |
|---|---|
| ACL | sagittal primary; coronal/axial support; intercondylar notch |
| MCL | coronal primary; axial support; medial knee |
| Medial/Lateral meniscus | sagittal + coronal; joint line / horns / body |
| Medial/Lateral OA | coronal + sagittal; tibiofemoral compartments |
| PF OA | axial primary + sagittal; patella/trochlea |
| Effusion | fluid-sensitive, global recess/suprapatellar context |
| Synovitis | multi-plane; difficult without contrast; noisy report labels |
| Baker's cyst | posteromedial; axial/sagittal |
| Contusion | fluid-sensitive marrow signal; broad bone context |
| Fracture | cortex/subchondral bone + marrow; morphology + fluid-sensitive context |

Later prefer:

- learned task-specific attention;
- global + local ROI branches;
- coarse-to-fine localization;
- multi-resolution processing;

over fixed manually assigned pixel weights.

---

# 29. External data / pretrained-model research status

Potential resources researched for later experiments:

- DINOv2 — generic self-supervised image encoder.
- RadImageNet weights — generic medical-image pretrained encoder.
- MRNet — knee MRI, especially ACL/meniscus.
- OAI — OA/cartilage/meniscus.
- MOST — OA-focused knee MRI.
- SKM-TEA — anatomy/pathology localization and segmentation.
- fastMRI / fastMRI+ — knee MRI and pathology annotation/localization.
- KneeMRI Croatia — ACL-focused sagittal MRI.

Important:

> Do not automatically download/use external datasets merely because they are public.

Every external source must pass:

```text
license / competition compatibility
-> MRI protocol similarity
-> pathology overlap
-> controlled experiment gain
```

Baseline V1 should use **no additional external raw MRI dataset**.

---

# 30. What was learned from strong public solutions

A public ~0.94 inference pipeline was inspected for architectural ideas, not copied as the baseline.

Useful observations:

- multi-plane / sequence-specific routing;
- physical crop around 130–140 mm;
- DINOv2 and RadImageNet backbones;
- label-specific attention pooling;
- multi-resolution / slice-density diversity;
- learned local evidence plus global context;
- final target-specific ensemble blending.

The high public score came from a complex ensemble, not a single magic model.

Therefore notebook `04` should stay simple enough to become a clean experimental reference.

---

# 31. Controlled-experiment roadmap after Baseline V1

Do not change multiple factors at once.

Candidate sequence:

```text
05 — resolution: 256 -> 336
06 — physical FOV: 140 -> 130 / 150 mm
07 — number / strategy of sampled slices
08 — add AX_NONFLUID
09 — mean/max pooling vs label-specific attention ablation
10 — DINO frozen vs unfreeze last blocks
11 — DINOv2 vs RadImageNet
12 — global + meniscus local branch
13 — ACL/MCL local branch
14 — OA compartment / PF local branch
15 — family-agreement weighted loss
16 — alternate silver-label strategy
17 — medical/knee external pretraining
18+ — robust ensemble / target-specific blending
```

The exact numbering can change, but the principle must remain:

> one major variable per experiment.

Record:

- base notebook / checkpoint;
- changed variable;
- seed;
- weak CV;
- gold audit;
- Public LB if submitted;
- runtime;
- conclusion.

---

# 32. Current locked decisions

Codex should treat these as the current baseline contract unless explicitly instructed otherwise:

```yaml
task:
  type: study_level_multilabel_knee_mri
  n_targets: 12
  metric: macro_roc_auc

supervision:
  weak_studies: 4349
  gold_audit_studies: 58
  silver_file: L1_family_mean_RECOMMENDED.csv
  labels: soft
  gold_in_optimizer: false

cv:
  folds: 5
  grouping: report_group
  gold_for_early_stopping: false

input:
  slots:
    - SAG_FLUID
    - SAG_NONFLUID
    - COR_FLUID
    - AX_FLUID
    - COR_NONFLUID
  missing_slot_mask: true
  ax_nonfluid_baseline: false

dicom:
  slice_sort: IOP_IPP_geometry
  physical_crop_mm: 140
  img_size: 256
  representation: 2.5D
  sampled_positions_per_slot: 8

encoder:
  family: DINOv2
  size: Small
  baseline_trainability: frozen
  offline_local_weights_only: true

study_head:
  pooling: label_specific_attention
  output_logits: 12

loss:
  type: BCEWithLogitsLoss
  soft_targets: true
  agreement_weighting: false

inference:
  report_input: false
  fold_ensemble: mean_probability
```

---

# 33. Central project rule

Whenever a future idea appears — ROI weighting, medical pretraining, larger backbone, more slices, sequence specialists, pseudo-label weighting, ensembling — do **not** assume it is better.

Implement it as a controlled experiment against the locked baseline, keeping all unrelated factors unchanged.

The goal is not to build the most complicated pipeline first.

The goal is to build a trustworthy baseline whose improvements can be attributed to specific decisions.
