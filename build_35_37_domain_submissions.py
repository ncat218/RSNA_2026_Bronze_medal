"""Three independent, predeclared domain-inspired probes of protected anchor 33."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
anchor_path=ROOT/'33_RSNA_0958_R6_Raptor_Submit.ipynb'
anchor=json.loads(anchor_path.read_text(encoding='utf-8'))
src=lambda i:''.join(anchor['cells'][i]['source'])
def assignment(text,name):
    return next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
def value(text,name): return ast.literal_eval(assignment(text,name).value)
base_sources=value(src(6),'SOURCES')
worker=base_sources['compact_infer.py']
phase_nb=json.loads((ROOT/'34_R6_Physical_Sampling_Phase_Gold_Probe.ipynb').read_text(encoding='utf-8'))
phase_setup=next(''.join(c['source']) for c in phase_nb['cells'] if 'SOURCES =' in ''.join(c['source']))
phase_worker=value(phase_setup,'SOURCES')['compact_infer.py']
crop_worker=worker.replace('    a = ap.parse_args()', '''    ap.add_argument('--crop-offset', type=int, default=None,
                    help='Experimental same-size crop offset for both axes; default preserves centre crop')
    a = ap.parse_args()
    if a.crop_offset is not None and not 0 <= a.crop_offset <= a.size-a.crop:
        raise ValueError('crop offset outside native image')''')
assert crop_worker!=worker
old_off='    off = (a.size - a.crop) // 2'
assert crop_worker.count(old_off)==1
crop_worker=crop_worker.replace(old_off, '    off = (a.size - a.crop) // 2 if a.crop_offset is None else a.crop_offset')
assert 'off2 = (a.size - crop2) // 2' in crop_worker
# There is only one R6 model group in this experiment; forbid a second group.
crop_worker=crop_worker.replace('    a = ap.parse_args()', '''    a = ap.parse_args()
    if a.crop_offset is not None and a.models2:
        raise ValueError('Experimental offset applies to one model group only')''')
receipt_line='    (out / "receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")'
assert crop_worker.count(receipt_line)==1
crop_worker=crop_worker.replace(receipt_line,'''    receipt['experimental_crop_offset'] = a.crop_offset
    print('CROP_OFFSET_AUDIT', json.dumps({'offset_row':off,'offset_col':off,'crop':a.crop,'size':a.size}),flush=True)
'''+receipt_line)
for code in (phase_worker,crop_worker): compile(code,'<new inference worker>','exec')

FUSION=r'''
# Independent R6 passes; combine POST-stacker probabilities, never raw state logits.
# This preserves each pass's Effusion/Baker's postprocessing contract.
# Imports must live in the parent kernel: imports in Raptor subprocesses do not carry over.
import os, sys, json, time, hashlib, shutil, subprocess
import numpy as np
import pandas as pd
EXPERIMENT = __TAG__
EXTRA_WORKER_NAME = __WORKER_NAME__
EXTRA_WORKER_SOURCE = __WORKER_SOURCE__
EXTRA_ARGUMENTS = __EXTRA_ARGS__
(SRC/EXTRA_WORKER_NAME).write_bytes(EXTRA_WORKER_SOURCE.encode('utf-8'))
if hashlib.sha256((SRC/EXTRA_WORKER_NAME).read_bytes()).hexdigest()!=__WORKER_SHA__:
    raise RuntimeError('Experimental worker file identity changed')

def run_r6_pass(tag,worker_name,extra):
    output=TMP/(EXPERIMENT+'_'+tag)
    deadline=NB_T0+8.0*3600
    remaining=deadline-time.time()
    if remaining<120: raise RuntimeError('Global eight-hour inference budget exhausted before '+tag)
    env=dict(os.environ,PYTHONPATH=str(CODE)+os.pathsep+os.environ.get('PYTHONPATH',''),PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,'-u',str(SRC/'probe_dual.py'),'--data',str(DATA),'--split','test',
             '--code',str(CODE),'--bundle',str(BUNDLE),'--arm',ARM,'--output',str(output),
             '--amp','auto','--loader-workers','1','--deadline',str(deadline),
             '--worker',worker_name,'--worker-args',json.dumps(WORKER_ARGS+extra+['--chunk','96']),
             '--max-fallback-fraction','1.0']
    started=time.time()
    rc=subprocess.run(command,env=env,timeout=remaining).returncode
    for log in output.glob('worker*/log.txt'):
        shutil.copyfile(log,WORK/(EXPERIMENT+'_'+tag+'_'+log.parent.name+'_log.txt'))
    path=output/'submission_receipt.json'
    if path.is_file(): shutil.copyfile(path,WORK/(EXPERIMENT+'_'+tag+'_receipt.json'))
    if rc!=0 or not path.is_file(): raise RuntimeError('R6 '+tag+' failed; inspect saved logs/receipt')
    receipt=json.loads(path.read_text())
    frame=pd.read_csv(output/'submission.csv',dtype={'StudyInstanceUID':str})
    ids=pd.read_csv(DATA/'test.csv',dtype={'StudyInstanceUID':str}).StudyInstanceUID.tolist()
    if not frame.StudyInstanceUID.is_unique or set(frame.StudyInstanceUID)!=set(ids):
        raise RuntimeError('R6 pass UID mismatch')
    frame=frame.set_index('StudyInstanceUID').loc[ids]
    if not np.isfinite(frame.values).all() or not ((frame.values>=0)&(frame.values<=1)).all():
        raise RuntimeError('Invalid R6 pass probabilities')
    scored=set()
    for p in output.glob('worker*/predictions.csv'):
        scored.update(pd.read_csv(p,dtype={'StudyInstanceUID':str}).StudyInstanceUID.tolist())
    if len(scored)!=receipt.get('scored'): raise RuntimeError('Receipt/scored UID mismatch')
    frame.to_csv(WORK/(EXPERIMENT+'_'+tag+'_probabilities.csv'),index_label='StudyInstanceUID')
    return frame,scored,receipt,time.time()-started

import pandas as pd
base,base_valid,base_receipt,base_seconds=run_r6_pass('original',WORKER,[])
candidate,candidate_valid,candidate_receipt,candidate_seconds=run_r6_pass('candidate',EXTRA_WORKER_NAME,EXTRA_ARGUMENTS)
if list(base.columns)!=list(candidate.columns): raise RuntimeError('R6 label mismatch')
out=base.copy(); out.iloc[:,:]=.5
both=base_valid&candidate_valid
only_base=base_valid-candidate_valid
only_candidate=candidate_valid-base_valid
for uid in both: out.loc[uid]=.5*base.loc[uid]+.5*candidate.loc[uid]
for uid in only_base: out.loc[uid]=base.loc[uid]
for uid in only_candidate: out.loc[uid]=candidate.loc[uid]
valid=base_valid|candidate_valid
out.to_csv(WORK/'ours_sub.csv',index_label='StudyInstanceUID')
# The original 33 blend detects scored UIDs from `pred`; publish the union,
# not a fabricated all-scored list or the first pass's list.
pred=TMP/(EXPERIMENT+'_fused')
(pred/'worker0').mkdir(parents=True,exist_ok=True)
out.loc[[u for u in out.index if u in valid]].to_csv(pred/'worker0'/'predictions.csv',index_label='StudyInstanceUID')
common=[u for u in base.index if u in both]
change=np.abs(base.loc[common].values-candidate.loc[common].values) if common else np.zeros((0,len(base.columns)))
receipt={'status':'PASS','experiment':EXPERIMENT,'bundle_id':BUNDLE_ID,'rows':len(out),
         'scored':len(valid),'fallback_studies':len(out)-len(valid),
         'pass_fallbacks':{'original':len(out)-len(base_valid),'candidate':len(out)-len(candidate_valid)},
         'probability_fusion':'equal mean after each complete pass/stacker; sole valid pass used on failure',
         'common_scored_rows':len(common),'changed_probability_cells':int((change>1e-7).sum()),
         'max_abs_probability_delta':float(change.max()) if change.size else None,
         'original_seconds':round(base_seconds,1),'candidate_seconds':round(candidate_seconds,1),
         'seconds':round(base_seconds+candidate_seconds,1),
         'seconds_including_notebook_setup':round(time.time()-NB_T0,1)}
(WORK/'submission_receipt.json').write_text(json.dumps(receipt,indent=2))
(WORK/(EXPERIMENT+'_fusion_receipt.json')).write_text(json.dumps(receipt,indent=2))
print('R6_FUSION_COMPLETE',json.dumps(receipt),flush=True)
'''

def fusion(tag,filename,code,args):
    replacements={'__TAG__':repr(tag),'__WORKER_NAME__':repr(filename),'__WORKER_SOURCE__':repr(code),
                  '__EXTRA_ARGS__':repr(args),'__WORKER_SHA__':repr(hashlib.sha256(code.encode()).hexdigest())}
    result=FUSION
    for a,b in replacements.items(): result=result.replace(a,b)
    return result

raptor=src(2)
old_head='''        a = self.att(h)
        a = torch.softmax(a, dim=1)'''
assert raptor.count(old_head)==1
density=raptor.replace(old_head,'''        a = self.att(h)
        if getattr(self, '_apply_series_prior', False):
            a = a.float() - self._series_log_count.view(1, -1, 1)
        a = torch.softmax(a, dim=1)''')
old_logits='''        logits = model(x).float()
        assert torch.isfinite(logits).all(), "Nonfinite fp16 logits"
        o = torch.sigmoid(logits)'''
assert density.count(old_logits)==1
density=density.replace(old_logits,'''        feats = model.encode(x)
        model._apply_series_prior = False
        original = model.head(feats).float()
        try:
            model._apply_series_prior = True
            normalized = model.head(feats).float()
        finally:
            model._apply_series_prior = False
        if not torch.isfinite(original).all() or not torch.isfinite(normalized).all():
            raise RuntimeError('Nonfinite series-normalized logits')
        o = .5 * torch.sigmoid(original) + .5 * torch.sigmoid(normalized)''')
old_call='                    arm_probs[a][i] = infer_pair_probs(model, xw, xm, d)'
assert density.count(old_call)==1
density=density.replace(old_call,'''                    centres = np.asarray([max(1,min(c,len(vol)-2)) for c in _eval_centers(mask,len(vol),K_EVAL)])
                    slots,_ = _slot_index(len(vol))
                    groups = slots[centres]
                    counts = np.bincount(groups,minlength=len(SLOT_SIZES))
                    # Full inverse-count attention prior; mix original/candidate probabilities 50/50.
                    model._series_log_count = torch.as_tensor(np.log(counts[groups]),device=d,dtype=torch.float32)
                    arm_probs[a][i] = infer_pair_probs(model, xw, xm, d)''')

audit=r'''
import hashlib,json
EXPERIMENT=__TAG__
receipt=json.loads((WORK/'blend_receipt.json').read_text())
frame=pd.read_csv(WORK/'submission.csv',dtype={'StudyInstanceUID':str})
ids=pd.read_csv(DATA/'test.csv',dtype={'StudyInstanceUID':str}).StudyInstanceUID.tolist()
expected_columns=list(pd.read_csv(DATA/'sample_submission.csv',nrows=1).columns)
if frame.StudyInstanceUID.tolist()!=ids or list(frame.columns)!=expected_columns:
    raise RuntimeError('Final submission schema/UID mismatch')
values=frame.iloc[:,1:].to_numpy(float)
if not np.isfinite(values).all() or not ((values>=0)&(values<=1)).all():
    raise RuntimeError('Invalid final prediction values')
expected={'Medial OA':.5,'Lateral OA':.2,'PF OA':.6,'Synovitis':.8,"Baker's":.35}
if receipt['weights']['raptor_per_class']!=expected or receipt['weights']['a224']!=.35:
    raise RuntimeError('Protected 33 family weights changed')
failures={k:receipt.get(k,0) for k in ['raptor384_failed','raptor224_failed','ours_failed','both_failed']}
pass_fallbacks={}
fused=WORK/(EXPERIMENT+'_fusion_receipt.json')
if fused.is_file(): pass_fallbacks=json.loads(fused.read_text()).get('pass_fallbacks',{})
status='COMPLETE' if not any(failures.values()) and not any(pass_fallbacks.values()) else 'FALLBACKS_PRESENT'
result={'experiment':EXPERIMENT,'status':status,'visible_rows':len(frame),'failures':failures,
        'pass_fallbacks':pass_fallbacks,'weights':receipt['weights'],
        'total_seconds':receipt.get('total_seconds'),
        'submission_sha256':hashlib.sha256((WORK/'submission.csv').read_bytes()).hexdigest(),
        'note':'Independent prospective probe of 33. Saved visible rows do not establish hidden score/runtime. No labels/weights fitted.'}
(WORK/(EXPERIMENT+'_run_audit.json')).write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
'''

specs=[
 ('35','R6_Phase_Ensemble',fusion('35','compact_phase0.py',phase_worker,[]),None,
  'Same anatomical neighbour spacing, two physical sampling phases. Average each fully postprocessed R6 probability vector equally. No stride reduction: baseline already uses eval_stride=1. Thin-stack slab averaging means many studies may be unchanged. Extra R6 pass can increase hidden runtime; this is unmeasured.'),
 ('36','Raptor_Series_Density_MIL',None,density,
  'Anatomical sequence coverage: Raptor has unequal window counts across its five MRI slots. Subtract log(slot-window-count) from attention logits, then average the original and adjusted probabilities equally. All windows, pixels, crop and mirror stay unchanged. Hypothesis: reduce sampling-count dominance; learned attention may already compensate, so this can harm. This is NOT a trained hierarchical model and NOT a clinical rule. No extra backbone pass.'),
 ('37','R6_SameSize_ShiftedCrop',fusion('37','compact_shiftcrop.py',crop_worker,['--crop-offset','24']),None,
  'Same 352px crop, mildly translated: native384 offset24 on both axes versus original centred offset16. No resizing, larger FOV or reduced resolution. Average complete original/shifted R6 probabilities after their own stackers equally. Training source supports random crop locations, but exact run augmentation provenance is not established; spatial calibration can regress. Extra R6 pass has unmeasured hidden runtime.'),
]
manifest={'anchor':anchor_path.name,'anchor_sha256':hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
          'score_user_reported':.958,'gpu_tested_locally':False,'submissions':[]}
for tag,suffix,new_stage,new_raptor,description in specs:
    nb=copy.deepcopy(anchor)
    introduction=f'''# {tag} — {suffix}: independent domain-inspired submit probe

Anchor: notebook 33, user-confirmed Public 0.958. **This notebook has no known score.**

{description}

Only this hypothesis changes; ALL target-specific 33 weights and checkpoints remain unchanged. No training, hidden labels, gold-based weight fitting or external report API. Keep 33 protected. Do not combine these modifications until each has been evaluated separately. These notebooks are independent and each produces submission.csv.

Attach exactly the same FOUR inputs as successful 33: competition, Nartaa publication weights, R6 exact w1/v1 bundle, OrthoFoundation code/anatomy. GPU T4x2, Internet Off. Metadata may not override existing draft inputs; check actual mounts. Do not delete anatomy.pt or silently substitute R7/latest. Source33's same w1 slug is retained.

Run Save & Run All. Inspect {tag}_run_audit.json; submit once if COMPLETE. FALLBACKS_PRESENT means inspect before using a competition slot. Hidden rerun may be larger; there is no guarantee of runtime, no OOM or score improvement. 35/37 use an eight-hour total deadline including Raptor stages and one loader worker/chunk96; they never publish partial extra-pass predictions as complete. Per-study failed-pass fusion is explicitly logged. Send score and full Output ZIP.
'''
    nb['cells'][0]['source']=introduction.splitlines(keepends=True)
    if new_stage: nb['cells'][7]['source']=new_stage.splitlines(keepends=True)
    if new_raptor: nb['cells'][2]['source']=new_raptor.splitlines(keepends=True)
    # Preserve inference but correct inherited misleading provenance wording in comments.
    b= ''.join(nb['cells'][8]['source']).replace(
        '# Raptor trained on every training study and was selected on the 58 gold, so no clean labelled set exists to fit them.',
        '# Author states gold excluded from gradients but reused for selection; gold is not independent CV.')
    nb['cells'][8]['source']=b.splitlines(keepends=True)
    final_audit=audit.replace('__TAG__',repr(tag))
    nb['cells'].append({'cell_type':'code','metadata':{},'source':final_audit.splitlines(keepends=True),
                        'execution_count':None,'outputs':[]})
    for i,c in enumerate(nb['cells']):
        c['id']=f'rsna-{tag}-{i}'
        if c['cell_type']=='code':
            c.update(execution_count=None,outputs=[])
            code=''.join(c['source'])
            compile(code.split('\n',1)[1] if code.startswith('%%writefile') else code,f'<{tag} cell {i}>','exec')
    target=ROOT/f'{tag}_{suffix}_Submit.ipynb'
    # These experiment paths were created by this generator; never overwrite anchor 33.
    assert target!=anchor_path
    payload=json.dumps(nb,ensure_ascii=False,indent=1)
    if not target.exists() or target.read_text(encoding='utf-8')!=payload:
        target.write_text(payload,encoding='utf-8')
    manifest['submissions'].append({'file':target.name,'hypothesis':description,
        'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'code_compile':True,
        'same_33_blend_weights':True,'known_score':None})
(ROOT/'35_37_Domain_Probe_Manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
assert hashlib.sha256(anchor_path.read_bytes()).hexdigest()==manifest['anchor_sha256']
print(json.dumps(manifest['submissions'],indent=2))
