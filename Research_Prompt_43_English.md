# Optional external research prompt — 2026-10-20

This research is NOT a prerequisite for running our fixed43A/B controlled experiment.

## Copy/paste prompt

Act as an evidence-first research reviewer for the 2026 Kaggle RSNA Knee Abnormality Detection competition. Use live web research and primary sources only: actual released source, checkpoints/manifests, author posts and official competition material. If browsing or full text is unavailable, state that precisely; do not treat snippets or titles as proof. Do not invent links, hashes, scores or runtime measurements.

Our situation:
- Protected33 is a user-submitted Public0.958 ensemble of Nartaa Raptor384/224 and Heliosli R6 mirror TTA. Original R6 bundle identity fe055406fad6a12f; do not substitute R7 under a renamed dataset.
- Adding10% rank predictions from Nartaa S6 all-data checkpoint E_prior_s42_fall.pt (SHA2560c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12) gave notebook39 displayed Public0.958 and a user-reported improvement to rank404. We do not have full-precision Public or Private scores. No claim of better generalization is justified.
- S6 source is published in nartaa/rsna-knee-s6fall-weights Version5. Author describes4263 competition weak studies, blendjev labels, ImageNet init, crop288/cache320, v4 views,3slice windows,3.5mm,64training windows. The released augmentation code uses uniform crop offsets0..32 and explicitly avoids flips. The original compact_infer.py SHA256 is34f433fe72b732543813165c3397ec80cef01b0d49b25b7b22668714f0b0ef4a.
- We are running two fixed standalone S6 submissions:43A=center crop16/16;43B=equal study-logit average of same-size crops8/8 and24/24. Same checkpoint, other preprocessing, fp16, output rank transform and strict no-model-fallback policy. No gold-based weight fitting. These tests ask whether crop averaging improves standalone S6, not whether10% blending hides a gain.
- Reused Gold58: S6 AUC0.931376; R6 TTA0.936579; proxy29B0.938274;90%proxy+10%raw-S6 gives0.939505, paired bootstrap interval for the delta includes zero. This is NOT independent validation or exact33, and not a Public score.
-40A20%rank blend failed only during hidden rerun;40Bconfidence blend outcome is not confirmed. Their successful visible runs have identical base model outputs. The hidden failure cause is unknown. Please do not diagnose OOM from that alone.

Research only these actionable questions:
1. Has the S6 author or Heliosli published same-checkpoint crop-TTA ablations, exact inference crop policies, or a supported implementation? Distinguish study-logit averaging from token/window pooling, probability averaging, and changing physical FOV.
2. Is there a documented reason that paired diagonal translations8/8 and24/24 could mishandle anatomical views/anchors despite being inside the training crop range? Trace actual input/meta/head contracts; do not assume horizontal MRI flips are label-preserving.
3. Are genuinely distinct, public, reproducible checkpoint families with aligned OOF and provenance available that could complement R6/Raptor/S6? Do not propose ten reuploads of the same weights as architectural diversity. Supply dataset/version, exact inference contract, licences, which gold was used for gradient/selection, and access evidence.
4. Are there author-supported reliability fixes for hidden S6 reruns: loader lifecycle, missing DICOMs, deadlines, per-study failures, CPU/GPU memory? Separate confirmed code defects from speculation; do not remove identity checks, hide failures or silently score with another model.

Return:
- An evidence table: claim, exact source/link/version, what you actually read, confidence and unresolved caveats.
- At most TWO next experiments with one changing factor, fixed decision rule, actual required inputs and estimated compute explicitly labelled if unmeasured.
- Specific questions we can ask the checkpoint author if public code cannot answer them.
- Negative findings: where no evidence supports further inference tweaking.

Do not promise PrivateTop10/0.96+, optimize on58reused gold, propose a large Public weight sweep, or claim code licences automatically cover weights. Do not send radiology reports or participant data to external APIs.
