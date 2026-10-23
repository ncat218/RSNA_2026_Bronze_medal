"""One input-level probe: fixed two-crop S6 logit TTA, otherwise exact39."""
import ast, copy, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
source_path=ROOT/'research_38_s6/compact_infer_s6_original.py'
original=source_path.read_text(encoding='utf-8')
ORIGINAL_SHA='34f433fe72b732543813165c3397ec80cef01b0d49b25b7b22668714f0b0ef4a'
assert hashlib.sha256(original.encode()).hexdigest()==ORIGINAL_SHA
tree=ast.parse(original)
score_node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='score')
old_score=ast.get_source_segment(original,score_node)
new_score='''def score(images, meta):
        # Fixed same-size translations inside the random-crop range used in S6 training.
        if (a.size, a.crop) != (320, 288) or shared_trunk or models2:
            raise RuntimeError('41 crop TTA requires the pinned single S6 group at 320/288')
        full = images.to(dev, non_blocking=True)
        meta = meta.to(dev)[None]
        mask = torch.ones((1, full.shape[0]), dtype=torch.bool, device=dev)
        crop_offsets = ((16, 16), (24, 24))
        per_crop = []
        for oy, ox in crop_offsets:
            img = full[None, :, :, oy:oy + a.crop, ox:ox + a.crop]
            zs = []
            for m in models:
                z = run_model(m, img, meta, mask, amp)
                if not torch.isfinite(z).all():
                    z = run_model(m, img, meta, mask, 'fp32')
                zs.append(z[0])
            per_crop.append(torch.stack(zs))
        # Average complete study logits AFTER each crop's own spatial/MIL pooling.
        zs = torch.stack(per_crop).mean(0)
        return zs.mean(0), zs'''
assert original.count(old_score)==1
patched=original.replace(old_score,new_score)
receipt_line='    (out / "receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")'
assert patched.count(receipt_line)==1
patched=patched.replace(receipt_line,"    receipt['crop_tta'] = {'offsets': [[16,16],[24,24]], 'crop':288, 'cache_size':320, 'combine':'equal study-logit mean', 'passes':2}\n"+receipt_line)
compile(patched,'<41 crop worker>','exec')
PATCHED_SHA=hashlib.sha256(patched.encode()).hexdigest()
base_path=ROOT/'39_Anchor0958_Plus_S6_Fixed10_Rank_Submit.ipynb'
base=json.loads(base_path.read_text(encoding='utf-8'))
parent=''.join(base['cells'][1]['source'])
t=ast.parse(parent)
node=next(n for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='S6_COMPONENT' for x in n.targets))
old_module=ast.literal_eval(node.value)
extension='\nCROP_TTA_PATCH = '+repr(patched)+'\nCROP_TTA_PATCH_SHA = '+repr(PATCHED_SHA)+'\n'
extension+=r'''
_41_original_prepare_inputs = prepare_inputs

def prepare_inputs():
    # First verify the entire ORIGINAL staged code against the pinned public manifest.
    I = _41_original_prepare_inputs()
    path = I.src/'compact_infer.py'
    old_sha = sha256_file(path)
    if old_sha != '34f433fe72b732543813165c3397ec80cef01b0d49b25b7b22668714f0b0ef4a':
        raise K1Error('Original S6 worker identity differs before41 patch')
    path.write_bytes(CROP_TTA_PATCH.encode('utf-8'))
    if sha256_file(path) != CROP_TTA_PATCH_SHA:
        raise K1Error('41 crop worker write identity mismatch')
    # Explicit new experiment identity, not a bypass of the original release guard.
    SOURCE_SHA['compact_infer.py'] = CROP_TTA_PATCH_SHA
    _RECEIPT['inputs']['experimental_worker'] = {
        'original_sha256':old_sha,'patched_sha256':CROP_TTA_PATCH_SHA,
        'crop_tta_offsets':[[16,16],[24,24]],'passes':2,'combine':'equal study-logit mean'}
    save_json(OUT/'41_worker_patch_audit.json',_RECEIPT['inputs']['experimental_worker'])
    return I

_41_original_collect_part = collect_part

def collect_part(pdir, part, I):
    z,fb,rec = _41_original_collect_part(pdir,part,I)
    expected={'offsets':[[16,16],[24,24]],'crop':288,'cache_size':320,
              'combine':'equal study-logit mean','passes':2}
    if rec.get('crop_tta') != expected:
        raise K1Error('41 worker did not declare the prescribed crop TTA')
    return z,fb,rec
'''
new_module=old_module+extension
lines=parent.splitlines(keepends=True)
lines[node.lineno-1:node.end_lineno]=['S6_COMPONENT = '+repr(new_module)+'\n']
parent=''.join(lines)
parent=parent.replace("'experiment':'39_anchor33_90_S6rank_10'","'experiment':'41_S6_same_size_crop_TTA_fixed10_rank'")
parent=parent.replace("'note':'Different from the probability blend measured on reused gold; improvement unproven.'",
                      "'note':'Only S6 crop TTA changes versus39; two training-range translations; Public/Private gain unverified.'")
line="if not np.isfinite(delta) or delta>2e-4:raise RuntimeError('S6 visible predictions drift from successful38; inspect before submitting')"
assert parent.count(line)==1
parent=parent.replace(line,"if not np.isfinite(delta):raise RuntimeError('Nonfinite crop-TTA visible predictions')\n        REPORT['visible_change_expected']='41 changes S6 crop evidence; delta is not a parity test'")
parent=parent.replace("print('39 COMPLETE'","print('41 COMPLETE'")
compile(parent,'<41 parent>','exec')

intro='''# 41 — S6 same-size crop TTA, fixed 10% rank submission probe

This is a NEW, unscored experiment. It **creates submission.csv**. Baseline39 is user-confirmed Public0.958. Only S6 image evidence changes: score the 288×288 centre crop (offset16,16) and a shifted crop (24,24) from the same320px windows, average their complete study logits, then apply the unchanged39 fusion (90% exact33 +10% S6 percentile ranks). No resizing, slice-sampling change, horizontal flip, label permutation, gold fitting or training.

Rationale: published S6 training code samples crop offsets uniformly in0..32 and explicitly avoids flips. These two crops are within that training augmentation range. This is a hypothesis, not proof that crop averaging helps. It is NOT the R6 crop experiment37. All33 stages/checkpoints, internal per-target weights and S6 model weights are unchanged. Each study is decoded once for S6, but scored twice; extra GPU cost and hidden runtime are unmeasured.

**Attach exactly the same five input datasets/versions as successful39. No saved Output/cache needed. T4x2, Internet Off, Save & Run All.** This file can run independently of40A/B. Use a separate Kaggle notebook, preserving33/39.

The original S6 code/weights are checked first, then the explicit worker patch is applied and pinned to a new hash. Worker receipts must declare both crop offsets and two passes. This is an experimental source, not a claim that the unchanged public worker already supported crop TTA. Visible differences versus38 are expected; no parity claim is made. No gold diagnostic has been run for this new crop recipe.

Artifact directory remains `rsna_39` to preserve inherited stage code. Inspect `rsna_39/39_run_audit.json`: experiment must start41, status COMPLETE; inspect warnings. `rsna_39/s6_outputs/41_worker_patch_audit.json` and each worker receipt record the modification. Then submit ONCE and send score plus Output ZIP. Do not rerun a failure unchanged.

Eight-hour global timeout and inherited S6 stage cap are safety limits, NOT an ETA. GPU memory and hidden-run completion are unverified.40A's hidden exception is unresolved, so the inherited pipeline remains a known reliability risk even though39 submitted successfully. Local CPU tests are not a guarantee. No need to spend all remaining submissions: evaluate this one input-level hypothesis, not a broad weight sweep.

Credits/licence/provenance conditions from Nartaa, Heliosli and original33 remain applicable. Apache code and CC BY-NC weights are distinct; this experiment does not resolve eligibility.
'''
nb=copy.deepcopy(base)
nb['cells'][0]['source']=intro.splitlines(keepends=True)
nb['cells'][0]['id']='rsna41-intro'
nb['cells'][1]['source']=parent.splitlines(keepends=True)
nb['cells'][1]['id']='rsna41-controller'
target=ROOT/'41_Anchor0958_S6_TwoCrop_Fixed10_Rank_Submit.ipynb'
target.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
manifest={'notebook_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
          'original_s6_worker_sha256':ORIGINAL_SHA,'patched_s6_worker_sha256':PATCHED_SHA,
          'crop_offsets':[[16,16],[24,24]],'training_crop_offset_range':[0,32],
          'training_recipe_claim':'Verified random crop implementation; no flips',
          'gold_evaluation_performed':False,'gpu_verified':False,
          'known_risk':'40A hidden rerun exception unresolved; this inherits39 infrastructure.'}
(ROOT/'41_Build_Manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(target.name, target.stat().st_size,'bytes; compile PASS; inference not executed')
