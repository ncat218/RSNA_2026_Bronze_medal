# S6 standalone crop-TTA result audit — 2026-10-21

## Outcome

User screenshots confirm Public43A=0.950 and43B=0.950. This audit checks saved visible outputs, not hidden-run logs.

| Check |43A centre-only|43B two-crop|
|---|---:|---:|
|Checkpoint|same pinned S6|same pinned S6|
|Crop offsets|(16,16)|(8,8)+(24,24)|
|Saved studies scored|3/3|3/3|
|Model fallback|0|0|
|Saved total time|200.84s|198.47s|
|Sum worker GPU timings|32.40s|37.20s|

## Important distinction

TTA changed 36/36 visible probabilities. Maximum absolute change=0.056453; mean=0.008550.

**Nevertheless, all12 target orderings on these three studies stayed unchanged, so the two final rank-based CSVs are byte-identical.** This does not mean the worker patch failed; the source/receipt records verify the different crop forwards. It also does not prove the hidden submissions were identical: three-study ordering is not the ordering over the hidden test population.

## Decision

No demonstrated Public gain at displayed precision. Stop testing further crop offsets: current evidence does not justify more Public tuning on this narrow hypothesis. Exact score equality and Private behavior remain unknown. Do not swap out successful0.958 final candidates for these standalone controls.

Runtime comparison is inconclusive for hidden deployment.43B has two actual crop forwards but its saved total is slightly smaller; startup, decode and scheduling variation dominate the tiny sample. Do not market it as faster or extrapolate from this to the hidden set.

Archive source hashes, checkpoint/anatomy receipt identities, strict study coverage and final CSV derivations were verified. One visible study uses the built-in slab_prior anatomy method in both conditions; this is not a Raptor prediction fallback.

The earlier Version1 logs established missing anatomy input before inference. The new successful outputs do not support attributing those earlier failures to cropTTA, OOM or the strict missing-study guard.
