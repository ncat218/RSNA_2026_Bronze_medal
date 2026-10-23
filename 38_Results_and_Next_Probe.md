# Notebook38 results — S6 complementarity audit

Verified output: all 58 gold and 3 visible studies scored; no Raptor fallback, retry or worker timeout. Model and anatomy identities match the pinned release.

Total diagnostic: 565.61s; gold inference: 352.2s. This is not a hidden-test runtime guarantee.

## Descriptive gold results

| Model | Macro AUC |
|---|---:|
| r6 | 0.936579 |
| raptor384 | 0.922943 |
| proxy29B | 0.938274 |
| s6 | 0.931376 |
| proxy29B_90_s6_10 | 0.939505 |

Fixed 90% proxy29B + 10% S6 changes macro AUC by +0.001231. Paired bootstrap 95% interval: [-0.000933, +0.003619], includes zero. Gold is reused, so these intervals do not remove selection bias.

## Per-target fixed-blend changes

| Target | Delta AUC | S6–R6 Spearman |
|---|---:|---:|
| ACL | +0.000000 | 0.938 |
| MCL | +0.006803 | 0.865 |
| Medial Meniscus | -0.003606 | 0.935 |
| Lateral Meniscus | +0.001863 | 0.894 |
| Medial OA | -0.003101 | 0.941 |
| Lateral OA | +0.007737 | 0.908 |
| PF OA | -0.001931 | 0.902 |
| Effusion | +0.001242 | 0.792 |
| Synovitis | +0.001195 | 0.751 |
| Baker's | +0.001812 | 0.856 |
| Contusion | +0.000675 | 0.935 |
| Fracture | +0.002083 | 0.834 |

## Decision

S6 is weaker than R6 alone on this reused set, but the fixed blend has a small positive exploratory signal. A single 90% exact-anchor33 + 10% S6 probability-blend probe is justified as a controlled Public experiment, not an expected breakthrough. Keep every internal 33 target weight unchanged and protect 33. Do not tune per-target S6 weights on these results.

The reference here is proxy29B, not exact33: Raptor224 gold is unavailable. A Public result may therefore differ or regress. No new submission notebook has been created by this analysis. All new inference must run on actual hidden test rows; never reuse the three visible predictions as a hidden submission.

The slab_prior anchor method occurred in five gold studies. This is an internal anatomy fallback, not a Raptor model fallback. Do not describe every study as having a successful learned anatomy localization.
