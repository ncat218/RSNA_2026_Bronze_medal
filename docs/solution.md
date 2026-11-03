# Solution: R6/Raptor anchor with an S6 probability blend

## 1. Task and result

The task is study-level prediction of 12 knee MRI findings: ACL, MCL, medial and lateral meniscus, medial and lateral OA, PF OA, effusion, synovitis, Baker's cyst, contusion and fracture. Each study can contain multiple image series.

The competition result was **459th of 6,300 teams, Bronze Medal**. The author identifies notebook **40B** as the final ranked submission. The certificate establishes the placement; it does not list a notebook ID or the private leaderboard score.

## 2. Final pipeline

40B orchestrates two inference stages in separate processes. It resolves required datasets and verifies checkpoint identities before model execution. S6 runs first, followed by the anchor. Both produce study-aligned predictions; a final fusion step validates the outputs and writes the submission.

### R6/Raptor anchor

The anchor comes from notebook 33 and combines Raptor models at 384 and 224 resolution with the pinned R6 pipeline, including its mirror TTA and existing per-target blend. 40B preserves these internal choices. The anchor is already a blended score, rather than a single model's calibrated probability.

### S6 branch

The embedded S6 configuration specifies a ConvNeXt-Small backbone, size 320, crop 288, v4 views, three slices and physical spacing 3.5 mm. It uses the pinned `E_prior_s42_fall.pt` checkpoint and the associated code/anatomy assets. These are inherited inference settings and community resources, not models newly trained by notebook 40B.

The full original implementation stays embedded in the submission notebook. This writeup describes the orchestration and final fusion; source metadata and embedded code remain the authority for detailed preprocessing behavior.

## 3. Fusion

For each study and finding successfully scored by S6:

```text
final = 0.90 × anchor33_score + 0.10 × S6_probability
```

Notebook 39 used S6 percentile ranks at the same weight. 40B changes only that final representation to raw S6 probabilities, preserving the model-stage sources and anchor weights.

The hypothesis was that raw S6 confidence could contribute information lost when converting S6 outputs to ranks. This mixes two score representations, so calibration mismatch is a real tradeoff. The final placement does not prove the change improved performance over notebook 39; an independent comparison would be needed for that claim.

If an allowed S6 row fails, its final prediction remains the anchor output. Excessive S6 failures or invalid outputs terminate the run. This fallback is reported rather than treated as successful S6 inference.

## 4. Reliability and reproducibility

The controller checks:

- Required input mounts and pinned checkpoint/code hashes.
- Unique study IDs, exact ID coverage, expected column names and ordering.
- Finite predictions within [0, 1], with explicit handling of failed S6 rows.
- Stage exit status, time budgets, receipts and warning flags.
- A final CSV read-back before replacing the final submission file.

A failed run removes the partial final submission and records a FAILED audit. Successful runs retain stage timings, script hashes, warnings and the final submission checksum. The eight-hour controller deadline is a safety cap, not a measured runtime estimate.

The extracted fusion function is [src/fusion.py](../src/fusion.py). Its tests check score calculations, ID alignment, fallback behavior and rejection of invalid inputs. They do not execute DICOM processing or GPU inference.

## 5. Evidence and limitations

Historical notes report rounded Public LB 0.958 for several late candidates. Saved visible outputs contain only a few example studies; they establish implementation behavior on those examples, not hidden-set AUC or a general runtime estimate.

The 58 gold studies were examined repeatedly during development. Their diagnostic results are not an untouched holdout for selecting the final ensemble. Public leaderboard feedback also influenced experimentation. This is a competition inference solution, without demonstrated clinical validation.

The preserved historical validation report predates archiving. Its notebook checksum differs from the Git-archived file checksum recorded in `configs/submission.json`; the original pre-archive file is not available here for a byte-level comparison. The curated notebook is copied unchanged from the archive's Git blob, and all its code cells are syntax-checked during repository preparation. It has not been rerun on Kaggle as part of this presentation update.

## Credits and scope

Community inference code, model weights and anatomy resources underpin the final ensemble, including Nartaa's Raptor/S6 resources, Heliosli's OrthoFoundation assets and the pinned R6 bundle. Dataset metadata, original comments and source hashes are preserved in the notebook. Third-party licenses and access conditions still apply.

The contribution presented in this repository is the assembly and evaluation of the ensemble, controlled fusion experiments, provenance checks, failure handling and submission packaging. Earlier training and encoder experiments are documented in the archive, and are not claimed as the training recipe for the final community checkpoints.
