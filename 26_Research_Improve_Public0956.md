# Research to improve the verified 0.956 base

Date: 2026-10-14. Incumbent: notebook 25A, Public 0.956 as reported by the user. Notebook 25B also scored 0.956.

## Findings that change the next experiment

1. The current base already uses the published Raptor 384 checkpoint with a native 320 crop, 94 windows, and anatomical left/right mirror probability averaging. Recommending these as new improvements would duplicate the base. The author's measured history reports a change from 0.948 to 0.949 after this mirror recipe, but that step is already present in 25A.
2. The source of the core Raptor bag sampler treats the concatenated five-series stack as one continuous sequence. Its full-mask 96-slice bag creates 94 triplets. Eight triplets cross selected-series boundaries: centres 25, 26, 47, 48, 65, 66, 77, 78 (zero-based). Two of these boundaries separate series within a plane; two separate planes. Their slices are not neighbouring slices from one acquisition.
3. The window sampler uses the minimum and maximum valid slice indices. It does not require all three slices of a triplet to be valid, so a missing middle slot or failed decoded slice can still contribute a window. The attention head has no separate validity mask. This is an implementation observation, not proof that removing these windows improves the trained model; its training may contain the same artefact.
4. Raptor pools each target using attention over window features, with per-window LayerNorm and no cross-window transformer. Consequently a single backbone feature extraction can support a controlled pooling comparison. Filtering features and reapplying the head is the same mathematical operation as running the head on that subset, subject to ordinary numerical precision. No retraining or extra backbone pass is needed for the candidates.
5. The 25A/25B output ZIPs contain only the three visible test studies. They cannot localize the errors responsible for the 0.956 score. The 58 gold studies can provide target-level diagnostics, but the Raptor checkpoint trained on all training studies and was developed/selected using those gold studies. Neither their AUC nor a bootstrap interval constitutes independent CV.

## What current public evidence supports

- The Raptor author's release documents the 320 crop/mirror recipe and several failed changes, including smaller crops, reduced windows, and a freshly crop-trained model that did not beat the incumbent. This argues for holding crop, checkpoint and mirror fixed in the next test. [Author discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792).
- MRNet processes individual MRI series and later combines plane-specific outputs. This supports preserving series identity when constructing imaging inputs. It does **not** establish that removing cross-series windows from this Raptor checkpoint will help. [Original MRNet paper, Methods](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699).
- Competition participants report that label work, image preprocessing, representation and model diversity can all matter. Their comments are participant observations, not controlled evidence for our base. [Ensemble-improvement discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742926), [single-model discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304).
- No independently verified newer public architecture above our 0.956 base was established by the searches in this research pass. Titles or forks claiming 0.957 do not establish a new method; the previously audited 0.956/0.957 files share their scoring code. Search indexing may lag actual Kaggle updates.

## Executable next step: notebook 26

`26_Raptor_SeriesBoundary_Gold_Diagnostic.ipynb` runs three predeclared policies on the same Raptor features:

| Policy | Change | Windows if all 96 slices are valid |
|---|---|---:|
| baseline | Exact 25A window positions and pooling | 94 |
| same_series | Remove triplets spanning a selected-series boundary | 86 |
| same_series_valid | Additionally require all three slice masks to be valid | 86 |

The backbone is evaluated only twice per study: plain and anatomical mirror, as in 25A. Each head is applied to the same frozen features. R7 runs once on the same 58 studies. The fixed 50/50 rank blend is evaluated for each policy; the notebook does not fit blend weights or choose targets individually.

Before scoring gold, the baseline must reproduce saved 25A probabilities on the three visible studies with maximum absolute error at most 1e-4. A failed reproduction stops the diagnostic. The model/decoder source prefix and the R7 controller code are retained from 25A. The proxy contains UID-only test tables and a temporary symlink to competition training images; no reports or gold labels enter the image workers. The proxy lives under `/tmp`, outside exported notebook Output.

Outputs include per-target AUC, each candidate's removed attention mass, retained-window counts, predictions, exact input provenance fields, run receipts and paired resampling intervals. Zero fallback is required for the final comparison. No root `submission.csv` is written. The 58 gold studies are explicitly labelled reused development data throughout.

Attach the same inputs as 25A:

- RSNA Knee Abnormality Detection competition, including `train_series`.
- `nartaa/rsna-knee-publication-swa-weights-20261007`.
- `heliosli/rsna-knee-compact-r7cnxfsss`.
- `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`.

Use GPU T4×2, Internet Off, Save & Run All. No 05A or 24C cache and no saved 25A/25B output needs attachment; the three reference probability rows are embedded. Runtime has not been measured on Kaggle. The combined deadline is three hours; Raptor has a separate approximately two-hour subprocess timeout. This deadline is a stop limit, not an expected duration.

## Decision after the diagnostic

- Check the baseline reproduction and all 58 R7/Raptor rows first. If incomplete, inspect the concrete error rather than interpreting AUC.
- Check whether excluded windows receive material attention and whether their removal changes ranks. A visible code artefact can have negligible practical effect.
- If a fixed boundary policy improves the blended diagnostic without a large target regression, package one prospective hidden-test probe using that exact policy and the original 50/50 R7 blend. Keep the recorded diagnostic limitations; a later Public gain still does not establish a Private gain.
- If it is harmful or immaterial, reject this serving-only change. Next research priority is genuinely complementary training: series-aware windows plus supervised report labels in the competition-only branch. That requires independent group folds and training the model with the new window policy, not assuming a serving change fixes a representation it was trained to use.
- The pending 24C branch remains a separate training project. The available shard-0 report covers 1,088 of 4,349 weak studies; the other shards and integrity checks are still required for a full-data experiment. No claim is made that a newly trained competition-only model will reach or exceed 0.956.

## Verification performed locally

All notebook code cells parse and compile. Synthetic checks confirm 94 baseline/86 same-series windows, the exact eight boundary centres, removal of windows around an invalid interior slice, handling of an absent 22-slice slot, and rejection of an all-invalid study at inference. The 25A model/decoder prefix and R7 controller source are preserved. The Kaggle GPU models, DICOMs and packages are unavailable locally, so the embedded visible-study reproduction check is the required numerical gate on Kaggle.
