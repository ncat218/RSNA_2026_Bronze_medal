# 39 result and two controlled probes — 2026-10-20

User reports Public0.958 and rank1062→404 for39. This is encouraging, but the ZIP exposes only three visible studies. Full-precision Public scores are unavailable. Rank movement can reflect unshown digits, tie rules or concurrent leaderboard changes; it does not establish better Private generalization.

## Verified39 evidence

- COMPLETE, no warnings or per-study S6 failures in the saved visible run.
- 671.27 seconds total; S6 stage184.02s, original33 stage470.06s. These are visible-run timings, not hidden-run measurements.
- S6 matches38 visible predictions within 1.11e-16.
- Checkpoint/controller source identities match; final CSV hash matches audit.
- Recomputing `0.9 * unchanged33 + 0.1 * rank(S6)` reproduces the saved submission.

## Independent next probes

| Notebook | Formula | Only changed factor vs39 |
|---|---|---|
| 40A | 80% unchanged33 +20% rank(S6) | S6 dose10%→20% |
| 40B | 90% unchanged33 +10% probability(S6) | S6 rank→confidence |

Both create `submission.csv` and may be submitted independently. Neither changes33's internal per-target weights, model recipe or checkpoint. 40B deliberately mixes raw confidence and anchor ranks: the calibration mismatch may help or harm; it is an experiment, not a mathematically superior fusion rule.

Use **the exact same five input datasets/versions as successful39**. No Output39/cache required. **T4x2, Internet Off, Save & Run All.** Use separate Kaggle notebooks. Metadata pins the same datasets, but check actual draft attachments.

Artifact paths intentionally remain `rsna_39/39_run_audit.json`, because the already-tested model stage sources retain their exact bytes. Check `experiment` starts with40A/40B, status COMPLETE, and inspect warnings before submitting. Send both Public scores and each full Output ZIP. Do not confuse repeated filenames between their separate ZIPs.

CPU tests verify fusion, model source preservation, clean-parent success and cleanup after injected failure. GPU and hidden-run success are not yet verified for40A/B. The inherited eight-hour limit is a timeout, not a measured ETA or OOM guarantee.

## Decision rule

Protect33 and39. These two probes are fixed now, not a broad weight sweep selected by repeated gold fitting. Compare scores at the same time and use full precision if Kaggle exposes it. If neither provides a credible improvement over39, stop increasing S6 weight; seek genuinely different model/supervision evidence rather than consuming more submissions on tiny increments. Public success alone cannot establish a Private0.96+ result.
