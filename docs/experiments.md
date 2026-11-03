# Selected experiments and lessons

The complete experiments and their original notes are retained on the [archive branch](https://github.com/ncat218/RSNA_2026_Bronze_medal/tree/archive/competition-snapshot-2026-11-03).

| Experiment | Change | Recorded evidence |
|---|---|---|
| 33 | R6/Raptor anchor | Historical user-reported rounded Public LB 0.958 |
| 39 | 90% anchor + 10% S6 percentile rank | Historical user-reported rounded Public LB 0.958 |
| 40A | Raise S6 rank weight to 20% | Controlled weight probe; no verified final ranking recorded here |
| **40B** | **90% anchor + 10% raw S6 probability** | **Author-confirmed final ranked submission, 459/6,300**; historical Public LB 0.958 |
| 43A / 43B | S6 standalone and crop TTA | Historical user-reported rounded Public LB 0.950 |
| 44A | Remove the effusion fluid stacker | Saved output verified the intended single-target change; historical Public LB remained 0.958 at displayed precision |

## What these experiments support

- 39 versus 40B isolates the final representation of S6 at a fixed 10% weight.
- Stage hashes, receipts and CSV comparisons help distinguish an intended experiment from an accidental pipeline change.
- Equal rounded Public LB scores cannot establish equal full-precision performance or rank candidates on the private set.
- A standalone score is not sufficient to predict the value of a component in an ensemble; complementary errors matter.
- Output checks on visible studies are useful regression evidence, but they do not establish hidden-set performance.

The archive also includes DenseSlice MIL, RadImageNet, CoAtNet, ConvNeXt, sampling, mirroring and crop experiments. They provide research history, while the main branch concentrates on explaining and running 40B. No ablation gain or private AUC is invented for this writeup.
