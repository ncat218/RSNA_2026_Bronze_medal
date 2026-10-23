# Results 26 and the next OA-group probe

The supplied results 26 ZIP completed successfully on all 58 gold studies, with zero R7 fallback and no Raptor errors. Visible-study reproduction differs from saved 25A probabilities by at most 0.0000484, inside the 0.0001 gate. Total elapsed time was 984.84 seconds (16 minutes 25 seconds); the paired Raptor diagnostic itself took 176.76 seconds and reported peak CUDA allocation of 2.966 GiB.

## Decision about cross-series windows

| Policy | Blend gold macro AUC | Difference from base |
|---|---:|---:|
| Original 25A | 0.935508 | — |
| Remove cross-series windows | 0.934948 | -0.000561 |
| Also remove incomplete triplets | 0.934802 | -0.000707 |

The paired resampling intervals include zero. These small decreases do not prove a general deterioration, but provide no positive reason to replace the incumbent window policy. The eight cross-series windows form 8.51% of the 94-window bag, yet their mean attention mass ranges from 0.30% (ACL) to 1.32% (Baker's) across targets. The trained attention already downweights them strongly. Removing them most decreases Baker's gold AUC in the blend (-0.004529).

Retain the original Raptor window policy for notebook 27. No filtered-window Public submission is proposed from this result.

## More useful target-level finding

| Target | R7 gold AUC | Raptor gold AUC | Base blend gold AUC |
|---|---:|---:|---:|
| Medial OA | 0.990698 | 0.981395 | 0.992248 |
| Lateral OA | 0.883946 | 0.775629 | 0.834623 |
| PF OA | 0.903475 | 0.859717 | 0.889961 |

R7's dedicated OA extent heads give a model-based reason to test increasing its influence for the OA group. The especially large Lateral OA difference also shows how the uniform 50/50 blend can reduce the stronger member's ranking on this development set. The other weak blended target is Synovitis (0.832139), where the two model scores are both weaker and the blend already beats each one. Merely shifting its blend weight is not supported by this diagnostic.

## One prospective hypothesis: notebook 27

`27_RSNA_0956_OA_Group_R7_65_Probe.ipynb` uses 65% R7 rank + 35% Raptor rank for the three OA targets. The other nine targets retain the exact original 50/50 blend. The same single shared OA weight is used for all three classes. No weight sweep was performed.

This hypothesis was chosen after inspecting results 26. It is therefore explicitly development-guided, not an independent-CV result. The gold set has already been used by the Raptor author's model development and includes training-image exposure for the all-data Raptor checkpoint. The reported R7 manifest excludes gold from gradient training; that alone does not make the two-model diagnostic independent.

Recalculation from the actual stored predictions, using the float32 output format of the serving notebook, gives gold macro AUC 0.937200, versus the base 0.935401 (+0.001799). Lateral OA increases from 0.834623 to 0.851064, PF OA from 0.889318 to 0.894466, and Medial OA remains 0.992248. Notebook 26 used float64 scores for its diagnostic metric; rounding rank sums to float32 restores some exact ties and slightly changes AUC. The prospective 27 calculation is checked against execution of the actual blend cell, including its float32 conversion. Leaving out each one of the 58 studies in turn while keeping the prediction scores fixed leaves a positive macro difference in all 58 checks (range +0.001150 to +0.002046). This is a sensitivity check on the same data, not a separate validation sample or guarantee of a Public/Private improvement.

Attach the same four inputs as 25A: competition; `nartaa/rsna-knee-publication-swa-weights-20261007`; `heliosli/rsna-knee-compact-r7cnxfsss`; `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`. Use GPU T4x2, Internet Off, Save & Run All. It needs neither the 26 output nor the 224 checkpoint. Model inference is identical to 25A, so there is no additional model pass for this probe. The three-row saved runtime does not establish hidden-test runtime.

After the saved run is complete and `27_run_audit.json` reports COMPLETE, submit once. Compare its Public score with 25A's confirmed 0.956. Do not revise 65% to chase a single Public score. Send the resulting Output ZIP and score. The notebook exports the original blend on the same test rows as `27_base25A_reference.csv` and checks that all nine non-OA output columns match it exactly as float32 values.

The generator validates notebook syntax and preserves the model/decoder and R7 execution source cells from 25A. A local integration test executes the actual new blend cell against all 58 stored predictions, checks the protected nine columns, verifies the OA formula, and exercises the inherited both-models-failed fallback. Kaggle GPU inference remains the user's runtime check.

Source evidence: `results 26.zip`, particularly `26_diagnostic_report.json`, `26_per_target_auc.csv`, `26_attention_summary.csv`, `26_preflight.json` and the raw Raptor/R7 predictions. Machine-readable reanalysis: `26_result_analysis_and_27_hypothesis.json`.
