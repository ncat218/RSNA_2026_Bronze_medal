"""One preregistered geometry control for41; same two-pass cost and39 fusion."""
import ast, copy, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
base=json.loads((ROOT/'41_Anchor0958_S6_TwoCrop_Fixed10_Rank_Submit.ipynb').read_text(encoding='utf-8'))
parent=''.join(base['cells'][1]['source'])
def assignment(text,name):
    return next(n for n in ast.parse(text).body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
component_node=assignment(parent,'S6_COMPONENT')
component=ast.literal_eval(component_node.value)
patch_node=assignment(component,'CROP_TTA_PATCH')
patch=ast.literal_eval(patch_node.value)
old_sha=hashlib.sha256(patch.encode()).hexdigest()
assert patch.count('crop_offsets = ((16, 16), (24, 24))')==1
patch=patch.replace('crop_offsets = ((16, 16), (24, 24))','crop_offsets = ((8, 8), (24, 24))')
assert patch.count("'offsets': [[16,16],[24,24]]")==1
patch=patch.replace("'offsets': [[16,16],[24,24]]","'offsets': [[8,8],[24,24]]")
new_sha=hashlib.sha256(patch.encode()).hexdigest()
lines=component.splitlines(keepends=True)
lines[patch_node.lineno-1:patch_node.end_lineno]=['CROP_TTA_PATCH = '+repr(patch)+'\n']
component=''.join(lines)
assert component.count(old_sha)==1
component=component.replace(old_sha,new_sha)
# Receipt declarations and audit names outside the nested patch string.
component=component.replace('[[16,16],[24,24]]','[[8,8],[24,24]]')
component=component.replace('41_worker_patch_audit.json','42_worker_patch_audit.json')
lines=parent.splitlines(keepends=True)
lines[component_node.lineno-1:component_node.end_lineno]=['S6_COMPONENT = '+repr(component)+'\n']
parent=''.join(lines)
assert '41_S6_same_size_crop_TTA_fixed10_rank' in parent
parent=parent.replace('41_S6_same_size_crop_TTA_fixed10_rank','42_S6_symmetric_crop_TTA_fixed10_rank')
parent=parent.replace('Only S6 crop TTA changes versus39; two training-range translations; Public/Private gain unverified.',
                      'Symmetric offsets8/24 control versus41 offsets16/24; fixed before either score; gain unverified.')
parent=parent.replace("print('41 COMPLETE'","print('42 COMPLETE'")
compile(parent,'<42 parent>','exec')
compile(component,'<42 component>','exec')
compile(patch,'<42 crop worker>','exec')
intro='''# 42 — symmetric S6 crop-TTA control, fixed10% rank submission

**Creates submission.csv. New unscored experiment.** Baseline39 is user-confirmed Public0.958. This is a geometry control for41, not a new model or independent source of evidence.

S6 uses two288×288 crops from its320px windows: **(8,8) and(24,24)**, i.e. −8/+8 pixels around the original centre offset16. Each crop receives its own spatial/MIL pooling; study logits are averaged equally. These offsets are inside the random-crop training range0..32. No flip, resize, changed labels/slices or training. Symmetry keeps the mean crop centre unchanged, but does not guarantee statistical superiority.

Comparison:39=centre only;41=centre+positive shift;42=negative+positive shift.41 and42 have the same two-forward cost; only one crop position differs. They are fixed before either new Public result, not selected by gold. Both retain **90% exact33 +10% rank(S6)** and every internal33 target weight.

Attach **the same five inputs/versions as successful39**. T4x2, Internet Off, Save & Run All. No Output/cache attachment needed. Use a separate Kaggle notebook. This can run independently of41, but the hypotheses are closely related.

Output paths intentionally retain `rsna_39/39_run_audit.json` for inherited worker compatibility. Require experiment beginning42, status COMPLETE, inspect warnings. Patch audit:`rsna_39/s6_outputs/42_worker_patch_audit.json`; worker receipts must list offsets8/24 and two passes. Then submit once and return Public Score plus full Output ZIP.

No GPU/hidden run is verified. Additional S6 forward cost may affect runtime. The eight-hour global and inherited S6 stage caps are timeouts, not ETAs.40A's hidden exception remains unresolved; these controls are not a fix for it. Preserve successful33/39 final candidates. Model code/weights, attribution and third-party licence/provenance obligations are inherited from39/41.
'''
nb=copy.deepcopy(base)
nb['cells'][0]['source']=intro.splitlines(keepends=True)
nb['cells'][0]['id']='rsna42-intro'
nb['cells'][1]['source']=parent.splitlines(keepends=True)
nb['cells'][1]['id']='rsna42-controller'
target=ROOT/'42_Anchor0958_S6_SymmetricCrop_Fixed10_Rank_Submit.ipynb'
target.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
manifest={'notebook_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
          'patched_s6_worker_sha256':new_sha,'crop_offsets':[[8,8],[24,24]],
          'gpu_verified':False,'gold_evaluation_performed':False,'hypothesis_family':'same-size crop TTA; correlated with41'}
(ROOT/'42_Build_Manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(target.name,target.stat().st_size,'bytes; compile PASS; no inference run')
