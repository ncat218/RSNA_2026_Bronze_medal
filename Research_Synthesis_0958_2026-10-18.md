# RSNA Knee: evidence synthesis and original improvement plan

Updated 2026-10-18. Confirmed Public anchor: **33 = 0.958**, reported by the user. No Private result is known. Preserve notebook 33 unchanged.

## 1. What actually produced the current score

33 uses the same Raptor384 and Raptor224 inference code as successful 29A, byte-identical in those stages, plus the same R6 bundle `fe055406fad6a12f` with anatomical mirror TTA. Its R6 mount points to a w1 copy rather than the mutable original dataset. The scientific difference from 29A is target-specific family weighting:

| Target | Raptor family | R6 |
|---|---:|---:|
| Synovitis | 0.80 | 0.20 |
| Lateral OA | 0.20 | 0.80 |
| PF OA | 0.60 | 0.40 |
| Baker's | 0.35 | 0.65 |
| All others | 0.50 | 0.50 |

Within the Raptor family: 65% ranks of 384 + 35% ranks of 224. The Public increase is a source-recipe reproduction, not an independently validated new model. Target weights' selection provenance is not known. No results 33 ZIP is available locally at this audit, so hidden success/score is user-reported, not inferred from three saved rows.

Source comment correction: 33 inherits a comment claiming Raptor trained on every training study. The author's current description instead states 4,349 weak studies were used for gradient training and 58 gold excluded from gradients but reused for selection. We do not silently edit the successful anchor; its gold-selection limitation remains.

## 2. Consolidated measured history

- Original public ensemble: 0.943. Own frozen RadImageNet/B2 branch: 0.914 standalone; density blend 0.913; combinations failed to beat 0.943.
- Public-source upgrades: 23A/B 0.954; 25A/B and OA probe 27 0.956; 29A/B/C 0.957; current 33 0.958.
- 26: removing Raptor cross-series windows worsened the gold blend; do not transfer that serving-only change.
- 30: R6 TTA scores 58/58, no fallback, exact visible reproduction. Removing fluid/state stackers changes macro by only +0.000104/+0.000075; adding R7 plain at 10% total weight changes it by -0.000294. Not convincing evidence of improvement.
- Synovitis/Effusion on 58 gold: Raptor Pearson 0.844, Spearman 0.780; R6 both ~0.736; gold correlation 0.403. These do not prove clinical disentanglement or compare to older checkpoints fairly across datasets.
- 32: 5 Synovitis-only, 13 Effusion-only. Synovitis AUC within this selected 18-case group is 0.538 R6, 0.585 Raptor, 0.554 in the 29B proxy. No manual report annotations yet. This is exploratory, not independent CV or an estimate of hidden subgroup prevalence.
- Training branch 24: only shard 0 completion was verified here; no complete competition-only trained replacement is established.
- 31 refers to two locally documented recipes over time; results31's multi-reader pipeline and current 33's nine-cell R6 recipe are not interchangeable evidence. Do not claim results31 is a results33 receipt.

## 3. Research reviewed and implications

### Labels and supervision

The [Nartaa training account](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792) states mixed report labels/JEV, external OAI supervision for PF OA/Lateral OA/Synovitis, and masking unavailable labels. This is not uncertainty-band masking. Compute increased with OAI, so the stated progression is not an isolated compute-matched ablation. Eligibility and release licence are separate questions; author permission alone is not a host ruling.

[ibrahimdodo's source](https://github.com/ibrahimdodo/knee-mri-raptor/tree/6cdbcbe4a5e3c04497c68ca021185e9d6162b893) actually implements frozen per-window features, uncertainty-band [0.3,0.5] masking on all targets and a 2x2 masking/pos-weight experiment. It uses an older v10 checkpoint and 62 windows, not our 96-slice/K94 Nartaa pipeline. Its validation masks differ by recipe, and its split is study-random rather than grouped. Its failed fresh-head result is a warning, not proof all head training fails. Do not copy its label thresholds/features into 33.

### Image evidence

The original [MRNet study](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699) supports multi-plane MRI modeling but does not establish an improvement for our specific pretrained models. A [competition participant's controlled geometry experiments](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737597) report adjacent/geometry sampling matters and small structures remain difficult. Those are the author's weaker-pipeline observations, not a direct effect estimate for 33.

The new search did NOT verify a public >0.958 architecture or an independent OOF asset that establishes a better next model. Generic larger backbones, ROI pooling or mirror TTA are not novel additions here: multiple such components already exist in the anchor.

## 4. New actionable source finding: physical sampling phase

R6 manifest already has **eval_stride=1, phys_mm=3.5**, so decreasing eval_stride is a no-op. However, trainer code uses:

`phys_step = max(1, round(3.5 / rendered_slice_spacing))`

Evaluation window centres start at phase 1 and advance by `eval_stride * phys_step`. Thin stacks with physical step >1 therefore skip some centres. Training's cover sampler randomizes phase. A phase-0 evaluation grid retains the SAME trained 3.5mm neighbour spacing, while potentially exposing alternate centres. Ordinary physical-step-1 stacks remain byte-identical in their window lists.

Important extra source check: the builder already averages stacks thinner than 1.75mm into ~3.5mm slabs, so these mostly become physical-step 1. The opportunity is narrower than raw thin-slice metadata suggests, typically rendered spacing ~1.75–2.33mm, plus exceptional short series. 34 therefore first reads gold DICOM headers and estimates the actual slab/spacing contract. If no series is potentially eligible it finishes successfully without loading inference models. Series selection still needs the worker's definitive added/removed-window logs; no result is presumed.

**Hypothesis, not result:** a training-supported alternate sampling phase can reveal focal evidence missed by the single evaluation grid, without changing labels, weights, backbone, ROI or neighbour geometry. It can also harm pooling/stackers; only measurement can decide. This is distinct from adding an already active TTA or changing the family blend.

Implemented: `34_R6_Physical_Sampling_Phase_Gold_Probe.ipynb`.

- New inference worker only; trainer/model/code-dataset hashes remain checked. New worker hash/provenance is recorded.
- Three visible studies first; unchanged bags must reproduce the original probabilities within the existing 0.001 serving tolerance before gold.
- Then 58 gold once. Logs enumerate added/removed windows for every study.
- One loader worker and chunk96 reduce buffering; no long model training.
- Reports R6 change and 29B 50/50 Raptor384 proxy, explicitly NOT full 33 (gold Raptor224 predictions unavailable).
- No submission. No weight fitting. Two-hour timeout is a cap, not an ETA; actual GPU run is unmeasured locally.
- If no bags change, close direction. If unaffected bags drift, diagnose implementation. If focal-target regressions dominate or resources exceed budget, reject. A small reused-gold gain alone is insufficient to select final weights.
- If the direction is supported, next prospective probe keeps ALL 33 weights/checkpoints fixed and changes only the R6 grid; later assess whether original+alternate phases complement each other. Do not launch/submit automatically.

## 5. Parallel priorities, without demanding manual work to make progress

1. Track label blend provenance and Synovitis definitions; CPU source-table audit is useful but still needs pinned original source tables and the actual Nartaa combination recipe. Do not send reports to external services or invent label semantics.
2. Inspect existing case scores and collect optional human report/image annotations. Absence of annotations should not force repeated score-only notebooks or block the image-evidence experiment.
3. Seek aligned OOF for a strong distinct family or train a specialist with genuinely grouped validation, only after throughput/label gates. A frozen weak feature head is not automatically complementary.

## 6. Verification and practical limits

34 code compiles. The actual public baseline/candidate samplers pass 30 synthetic geometry cases, including physical-step-1 identity and in-bounds unique window rows. GPU/DICOM inference is not tested locally. Notebook 33 file hash is recorded before the new work and must remain unchanged. No claim is made of reaching 0.96 or Private Top10; we are implementing a falsifiable new image-evidence experiment rather than promising a result.
