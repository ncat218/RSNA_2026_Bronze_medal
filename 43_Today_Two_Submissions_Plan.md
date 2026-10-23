# Final plan for today's two slots — 2026-10-20

**Use43A and43B, not41/42**, for today's two-submission experiment. This supersedes the earlier10%-blend recommendation.

| Notebook | Standalone S6 evidence | Purpose |
|---|---|---|
|43A|Original centre crop16/16|Own control on current code/dataset; no anchor dilution|
|43B|Same-size crops8/8 and24/24, equal study-logit average|Test symmetric crop TTA, everything else matched|

Both output standalone S6 percentile ranks and create `submission.csv`. They do NOT target0.958 immediately; a lower standalone score is not by itself a failure. Compare A vs B to isolate the crop hypothesis. The author's old standalone0.950 is useful context but not a substitute for our matched control because runtime/fallback/source details can differ.

## Attach to each new notebook

1. RSNA competition test DICOMs.
2. https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights/versions/5
3. https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004/versions/1

**No Raptor/SWA weights, R6 bundle, study cache or Output attachment needed.** Set T4x2 and Internet Off. Save & Run All. Check GPU quota before starting both. They can run independently; need not await A's Public score to start B because both hypotheses/formulas are already fixed.

## Outputs and reliability

-43A:`rsna_43A/43A_run_audit.json`.
-43B:`rsna_43B/43B_run_audit.json`.

Require COMPLETE, then submit each once. Send Public scores and full Output ZIPs. The two scripts share the model/renderer/checkpoint settings, but B uses two serial forwards per decoded study. No genuine study still missing after retry is silently filled; either experiment fails rather than pretending another model's output is S6. This strict policy makes scientific interpretation cleaner but does not guarantee hidden-run completion.40A's hidden exception remains unresolved.

Eight-hour global and up-to-seven-hour worker budgets are safety ceilings, NOT ETAs. Local CPU mocks and1500-row synthetic output tests do not establish GPU memory or DICOM/hidden-run robustness. No GPU test was run locally. Successful39's S6-only stage and38's58gold are relevant evidence, not a hidden-run guarantee for43B.

## Interpret before seeing the score

- If B materially improves A at full precision with clean runs, consider a later fixed-anchor probe replacing S6 with crop-TTA. Do NOT immediately tune per-target weights.
- If the displayed scores tie, call it inconclusive without full precision; rounded equality is not proof of zero effect.
- If B worsens A, reject this exact TTA recipe rather than trying many offsets against Public.
- This Public A/B test does not validate Private generalization or Top10. Keep selected final candidates33/39 protected.

Optional external research prompt: `Research_Prompt_43_English.md`; no need to wait for its answer to run this pair.
