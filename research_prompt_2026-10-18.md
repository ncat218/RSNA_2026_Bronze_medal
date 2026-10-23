# Research prompt — attach report_research_2026-10-18.md

```text
Act as an evidence-driven musculoskeletal MRI researcher and experienced Kaggle competitor. Read the attached report_research_2026-10-18.md as the CURRENT project state. If report.md is also attached, treat it only as historical background; it predates our confirmed Public 0.958 anchor and experiments 34–37.

Our goal is a genuinely useful improvement beyond reproducing public solutions, ultimately targeting Private Top 10. Do not promise a score. We use Kaggle T4×2 and approximately five submissions/day, but GPU time and reliable hidden inference are limiting resources.

RESEARCH THREE PRIORITIES ONLY:

1. Evidence for our CURRENT experiments:
   - R6 randomized sampling phase, physical_step/slab averaging, alternate or multi-phase inference with neighbour spacing unchanged.
   - Same-size crop translations for a spatial/extent-head MRI model.
   - Attention pooling that accounts for unequal series/window sampling counts.
   Prefer the exact model authors, source code and controlled ablations. Distinguish findings on different pipelines. R6 already uses eval_stride=1 and physical spacing; do not suggest those as new. No nearby blend-weight sweeps.

2. ONE strong complementary model or specialist we could actually implement:
   Look for available weights, exact preprocessing, code, licences and aligned OOF predictions with UID/fold/label provenance. Consider compartment-specific OA or Synovitis only if supporting evidence exists. Low prediction correlation alone is not sufficient. Clearly distinguish code/weights available now from papers requiring new training. Do not present a notebook title or claimed leaderboard score as verified performance.

3. Training-label and final-use provenance:
   - Heliosli R6 bundle fe055406fad6a12f: target creation, training phase/crop configuration, side classifier and anatomy-weight sources, gold selection/stacker fit, cross-fitting and plain-versus-mirror stacker inputs.
   - Nartaa publication SWA weights: exact blendjev/Synovitis/OAI mapping, silent-report handling and JEV source.
   - Direct official Rules/host statements covering publicly released OAI-trained weights, SKM-TEA/KneeXNet auxiliaries and winner deliverables/licensing. Do not infer permission or prohibition from silence, successful scoring, public availability, or a dataset title containing “Private”. If the exact official wording cannot be read, say so.

EVIDENCE REQUIREMENTS:
- Browse primary sources and provide direct links, commits/versions and dates where available.
- Label every material claim: code/log verified; publisher-reported; or hypothesis/unknown.
- If only search snippets or README text are accessible, explicitly state that limitation.
- Do not confuse a file present in Input with a model actually loaded by inference.
- Do not assume independent validation: our 58 gold studies were reused upstream, and saved test outputs contain only three visible studies.
- A held-out set already seen by the backbone is not independent model validation.
- Phase/fold disagreement measures sensitivity/diversity, not correctness or proof that averaging improves accuracy.
- Do not suggest external report uploads, model-driven relabelling, fitting weights on gold, or deleting trained anatomy dependencies as an automatically safe fallback.
- Do not contact authors, post questions, submit jobs, or change competition selections on my behalf. Draft questions only.

OUTPUT:
A concise memo with at most THREE actionable findings. For each:
1. Original source and exact evidence.
2. What is genuinely new relative to the protected 0.958 baseline.
3. Mechanism and risks, including training–inference consistency.
4. Smallest reproducible experiment, control and stopping criteria.
5. Required files/assets and links; clearly mark missing dependencies.
6. Resource estimates, explicitly unmeasured unless supported by a comparable benchmark.

End with a prioritized checklist of what I should obtain or ask the authors. If no promising evidence is found, report that honestly. Do not replace missing evidence with generic “use a bigger model”, “add TTA”, or “ensemble more models” advice. Evaluate results from our ongoing probes separately rather than declaring them successful before scores arrive.
```
