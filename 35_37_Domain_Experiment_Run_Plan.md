# Original domain-inspired probes of anchor 33 (Public 0.958)

Prepared 2026-10-18. User authorized independent submission experiments; no Kaggle jobs or submissions have been launched from this workspace. None of these new notebooks has a known score. All inherit the anchor's unresolved external-weight eligibility questions; they are not compliance fallbacks.

## Import repair after the user's notebook35 run

The supplied log shows the original R6 pass scored 3/3 with zero fallback, then the parent fusion cell raised `NameError: np is not defined` at its probability check. This is not an OOM or dataset version failure. The shared fusion source in 35 and 37 now explicitly imports numpy/pandas and its standard-library dependencies before launching any R6 pass. Imports inside Raptor subprocesses never populated the parent notebook. 36 does not use this cell and its file hash is unchanged.

The previous fusion unit test executed only the cell tail and injected np into its globals, hiding this bug. The replacement test executes the COMPLETE cell in a clean setup scope with mocked controller outputs; removing the new numpy import must reproduce the exact NameError. Worker files are written as UTF-8 bytes rather than platform-translated text, so their pinned hashes match on both Windows and Linux. GPU inference has not been rerun locally. Re-import the updated local 35 and 37 files; inputs and scientific experiment formulas are unchanged.

## Three independent notebooks

| Notebook | Domain-informed hypothesis | What changes | Main risk |
|---|---|---|---|
| 35 R6 Phase Ensemble | Fixed physical sampling grid may miss focal image evidence on eligible stacks | Equal probability average of full original R6 pass and phase0 pass, after each pass's own stackers | Many stacks have no new centres after slab averaging; extra inference time; pooling/state distribution shift |
| 36 Raptor Series Density MIL | Different numbers of sampled windows need not imply more independent anatomical evidence | For Raptor384 only: attention logits minus log(number of windows in its slot); equal probability average with original pooling, same features | Learned attention may already encode the correct sequence prior. This untrained pooling adjustment can regress |
| 37 R6 Same-Size Shifted Crop | Slight positioning differences may truncate focal evidence near a crop edge | Equal average of original offset16 and offset24 crops, both 352×352 from native384, after full per-pass stackers | Position/extent calibration may shift; exact training augmentation of this checkpoint is not verified; extra runtime |

All retain 33's per-target family weights and both Raptor checkpoints. There is no arbitrary score correction, disease-threshold rule, residual subtraction, relabeling, new backbone or fitting on the 58 gold. These are hypotheses derived from MRI sampling/anatomical coverage, NOT established clinical improvements.

Relevant source grounding: the embedded R6 trainer supports random phase in its covering sampler and random crop positions; exact completed-run choices still require author confirmation. Raptor's attention is a softmax over all windows; the count prior is a new mathematical adjustment, not a recovered training component. [Attention MIL paper](https://proceedings.mlr.press/v80/ilse18a.html) describes learned instance weighting but does not validate this adjustment on knee MRI. [MRNet](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699) supports plane-specific MRI evidence, not a guaranteed gain for any probe here.

## Inputs and settings

Exactly the same four inputs and versions as successful 33:

1. RSNA Knee competition.
2. Nartaa publication SWA weights (both 384 and 224).
3. Exact R6 w1 copy used by 33; do not substitute R7/latest. The same source33 w1 slug is retained.
4. OrthoFoundation code/anatomy dataset used by 33, including active anatomy.pt and offline wheels.

T4x2, Internet Off, Save & Run All. Metadata may not override old draft inputs. No training cache, notebook34 Output or additional model dataset is required.

## Suggested order

36 first: no second backbone encoding or second R6 pass; only a cheap extra attention-head evaluation. Actual hidden runtime is still unmeasured.

35 next if 34 shows eligible changed bags. If 34 reports no eligible series/no new evidence, do not spend a slot merely to submit an effectively unchanged physical-phase model. There is no dependency on 34 files; this is an efficiency decision.

37 is independent. 35/37 each run R6 twice; total model-stage deadline is eight hours from NB_T0 including Raptor stages. This is a cap, not an ETA or a proof of meeting Kaggle's hidden runtime. They use one loader worker and encoder chunk96. No promise of zero OOM is made.

You may run independently if session/GPU capacity allows. Submit each once if its <number>_run_audit.json is COMPLETE. If FALLBACKS_PRESENT, inspect receipts before submitting: missing passes are not silently counted as successful scientific reproduction. A failure in both R6 passes falls back through the original 33 Raptor route; single-pass failure uses the other pass and is logged. Do not change weights after viewing scores or merge three changes at once.

Return score and full Output ZIP for each. Saved output contains three visible test studies; visible differences/correlations are not hidden performance evidence. Exact equality of visible predictions does not imply hidden equality, and rounded equal scores do not establish equivalence.

## Verification performed / remaining

- Every notebook code cell compiles; model/checkpoint/source setup and final 33 target weights are preserved except declared changes.
- 36's actual modified head is tested for identity with the option off and correct inverse-count behavior under synthetic slot duplication.
- 35/37's actual probability-fusion and original blend are tested on aligned 58-row fixtures, including valid-UID unions, single-pass failure and both-pass failure. Synthetic fixtures are NOT model accuracy results.
- Protected 33's file hash remains unchanged.
- GPU/DICOM/Kaggle hidden inference has NOT been run locally. Public/Private improvement is unknown.

Keep 33 selected/protected until a deliberate decision. These probes use up to three submission slots, not an unlimited weight sweep. If all fail to improve, record the negative results and return to supervision/independent-validation work rather than repeating the same probes with nearby constants.
