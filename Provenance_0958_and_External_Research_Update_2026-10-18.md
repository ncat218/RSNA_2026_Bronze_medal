# Base 0.958: provenance and external-research decisions

Checked 2026-10-18. Source anchor is notebook 33, unchanged. This is an evidence inventory, NOT an eligibility ruling or legal opinion.

## Component matrix

| Component actually used | Identity/version | Known source | Unresolved |
|---|---|---|---|
| Raptor384 | `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`; nartaa publication dataset v1, datasetVersion 20297367; SHA256 `7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef` | Author states RSNA report labels plus 2,399 baseline OAI knees; external supervision for PF OA/Lateral OA/Synovitis; other missing cells masked | Host eligibility for public OAI-trained weights; exact target blend and OAI mapping |
| Raptor224 | `raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt`; same dataset/version; SHA256 `3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6` | Same OAI cohort according to release description; NOT an OAI-free replacement | Same host question |
| R6 four-fold model | Original publisher `heliosli/rsna-knee-r6-frozen-fe055406` v1, datasetVersion 20611930; bundle `fe055406fad6a12f`; manifest SHA256 `6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648` | Manifest says report-derived training, 4,344 pool studies, 58 gold not used by trainer, ImageNet ConvNeXt initialization, Opus-named target file | Full target creation and upstream provenance, gold reuse outside trainer, complete external-image history and licences |
| R6 mount in 33 | slug `rsna-knee-r6-frozen-fe055406-w1`, datasetVersion 20761842 | Copy is intended to preserve original v1; runtime asserts bundle/model hashes | Reupload owner not established by this audit. Do not confuse uploader with original model author |
| R6 side classifier | `side_cls.pt`; manifest SHA256 `ab623e2b619db4917639220fed15202e8c8aedd07c8e298a66a2afc0f153650a` | Image-based laterality correction is active | Its own training dataset/labels and eligibility not independently documented here |
| R6 Effusion/Baker's stackers | `fluid_stacker.json`, `state_stacker.json`; pinned within same bundle | Effusion uses fluid features + logits; Baker's uses report-state readouts | Fit sample identities, cross-fitting, plain-versus-mirror training inputs |
| R6 anatomy dependency | `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` v1, datasetVersion 20440357; active `anatomy.pt` SHA256 `5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e` | Publisher description explicitly mentions SKM-TEA/KneeXNet auxiliaries and retained upstream licensing; worker uses SKM-TEA anchors | Exact upstream licence/data terms and host eligibility |
| Source code and weights merely present in the anatomy bundle | Same mount | Presence in Input is NOT proof a backbone is used; 33's R6 is `convnext_small`, not `ofl_l` | Do not attribute the entire bundle's contents to active inference |

Links: [Raptor release](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007), [author recipe](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792), [original R6 v1](https://www.kaggle.com/datasets/heliosli/rsna-knee-r6-frozen-fe055406/versions/1), [code/anatomy bundle](https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004).

Dataset descriptions are publisher claims. Bundle hash verifies identity, not correctness, absence of leakage, authorship of every part or competition eligibility. Gold excluded from gradients does not establish independent evaluation when reused for selection.

## Research corrections

### Evidence levels and active-dependency audit (update 2026-10-18)

Use three labels per claim, not a blanket "confirmed" row: **verified locally** (file/hash/source inspection or supplied run log), **publisher states** (training/provenance claims), **unknown** (not established). Source inspection proves a load path exists; it is not a trace of a specific hidden run. No results33 ZIP was supplied for this audit.

| Component | Evidence of active use | Evidence level for provenance | Can it be removed with identical outputs? |
|---|---|---|---|
| Competition data | Scripts read test.csv/test_series and MRI; earlier successful receipts cover visible data | Read path/logs verified; competition terms require official Rules | No; required input. Do not redistribute raw reports/MRI |
| Raptor384/224 checkpoints | 33 source loads exact files and checks hashes; same Raptor source as successful 29 | Identity paths verified; OAI history is publisher-stated | Not from unchanged 33. Dropping either alters ensemble/coverage; no identity ablation established |
| R6 fold weights | Source loads four manifest models; earlier worker logs confirm four | Identity/source verified; report-derived/ImageNet training history publisher-stated | No; would change model/ensemble |
| R6 side_cls.pt | Source builds side net and checks/re-renders conditionally | Active path verified; training data/rights unknown | Not established; may affect selected studies even if some visible rows stay unchanged |
| anatomy.pt | Source loads anchor weights for views v4; earlier logs record SKM-TEA anchor methods | Active path verified; precise upstream data/rights need confirmation | No identical-output proof. Disabling trained anchor geometry risks train–inference shift |
| Effusion/Baker's stackers | Worker logs and experiment30 before/after dumps | Use verified; fitting set/cross-fit/selection history unknown | No; experiment30 shows changed predictions |
| OrthoFoundation model files merely present | R6 uses convnext_small; ofl_l init branch is not selected | Presence/name does not establish training lineage of every file | Individual unused files are not required by the inspected path; the entire dataset CANNOT be detached because it also supplies code/wheels/anatomy.pt |
| ConvNeXt initialization | R6 manifest names ImageNet-in22k-ft-in1k safetensors; full fold state dicts loaded at inference | Init provenance publisher-stated; exact upstream weight licence not audited | Separate original init file is not needed by this full-checkpoint path, but deleting it does not remove learned provenance |
| CoAtNet/timm initialization | Architecture identified in Raptor source; learned checkpoint used | Architecture path verified; exact pretraining-weight provenance/terms need auditing independently of timm code licence | No independent init download observed; training provenance remains embedded in learned weights |
| Offline libraries/source | Setup extracts cloud_code and pinned wheels, installs/imports them, verifies listed source hashes | Load path verified; per-package licences not exhaustively audited | Do not detach bundle or prune dependencies without a clean-environment reproduction and module/import inventory |

Reupload availability is separate from identity: earlier successful access does not guarantee future access. Retain the original immutable v1 reference as a known source option, but do not rehost third-party weights without permission. Test the exact selected saved notebook's inputs before the deadline; do not change the protected anchor in place.

Gold may have been used for stacker fit, epoch selection, threshold selection or other upstream development. `gold_in_training=0` does not answer those questions. Experiments30/34 remain non-independent even if gradient exclusion is true.

### Phase measurement clarification

Record per-target Spearman and absolute rank changes on the changed-bag subset and all studies, alongside sample size and ties. Fold disagreement and plain/mirror disagreement are useful **comparators**, not numerical-repeat noise and not estimates of prediction error. Disagreement smaller than leave-one-fold-out does NOT imply an upper bound equivalent to adding one fold. Compare complete postprocessing contracts if auxiliary stackers are nonlinear; averaging raw fold outputs is not automatically the deployed output.

34 currently exports final candidate probabilities and changed-window inventory; it does NOT export per-fold/plain/mirror logits. Those comparisons require an additional instrumented run. Do not claim them as already measured, and do not change a running experiment's acceptance rules after inspecting its results. More phases generally add GPU work but may share decoding/compilation; measure marginal time instead of multiplying total end-to-end time by phase count.

1. R6 publisher is identified as Heliosli from the original dataset, not goodpjw2008's three-fold reader. The private copy used by 33 does not change that original release provenance.
2. We have inspected actual embedded trainer code: the `train_sampling='cover'` branch randomizes phase; evaluation fixes phase 1. The builder slab-averages very thin stacks at inference. Exact invocation/provenance for the completed training run remains an author question, not something source code alone proves.
3. Alternate-phase disagreement on unlabelled studies measures sensitivity, NOT correctness or proof that averaging reduces error. Correlated bias can remain after averaging; labels/prospective evaluation are still needed.
4. Removing both Raptor stages does not make the pipeline competition-only: R6 uses external anatomy auxiliaries. Removing Raptor is an OAI-exposure reduction candidate only, not a certified compliant fallback.
5. The claim of a host ruling against KneeCoT in thread 743416 was supplied by external research. We could not read the full host thread or Rules body in this pass, so it is not independently verified here. No direct host approval/rejection of this exact public OAI-weights scenario was established. Search absence is not proof of no ruling.

## Decisions

- Keep 33 protected and continue 34 as an explicitly exploratory experiment. Its header gate prevents a phase no-op from triggering full inference.
- Do not launch a new training branch or automatic final-selection change from this document.
- Seek an explicit host answer covering both public pretrained weights and auxiliary anatomy/laterality weights. Do not treat use of public downloads as automatic eligibility.
- Build a fallback only after its active model and auxiliary provenance are audited. R6-only might remove the known Nartaa OAI component, but still needs its own checks and an actual score; no number is assumed.
- Compare image-evidence changes on fixed controls. Do not revive frozen uncertainty-band head training merely because no new OOF was found.

## Questions to send (user action; not posted automatically)

Host:
> May a competition inference submission use publicly available weights trained with OAI images or OAI-derived labels, even when the entrant only downloads the released weights and never accesses OAI records? Does the same interpretation cover public SKM-TEA/KneeXNet anatomical-localization or laterality auxiliaries? Please identify the applicable rule and distinguish each case if necessary.

Heliosli:
> For R6 bundle fe055406fad6a12f, which training datasets and label-generation recipes produced the four fold models, side_cls.pt and stackers? Was random-phase cover sampling and thin-stack slab averaging used in the scored training run? Have alternate phase or multi-phase inference ablations been run, and are aligned OOF predictions available?

Nartaa:
> What is the exact Synovitis blendjev target recipe, including whether v4_blend contributes through both the blend and JEV branches? How are silent reports and OAI Hoffa-synovitis mapped, and which unavailable external labels are masked? Can you provide a host clarification that covers the released weights?
