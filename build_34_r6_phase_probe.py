"""Test a new physical-sampling phase, preserving the successful R6 model contract."""
import ast
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
base = json.loads((ROOT/'30_R6_TTA_Error_Stacker_Gold_Diagnostic.ipynb').read_text(encoding='utf-8'))
anchor = json.loads((ROOT/'33_RSNA_0958_R6_Raptor_Submit.ipynb').read_text(encoding='utf-8'))
text = lambda nb,i: ''.join(nb['cells'][i]['source'])
def literal_assignment(source, name):
    return next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
pre = text(base,1)
ref_node = literal_assignment(pre,'REF')
ref = ast.literal_eval(ref_node.value)
with zipfile.ZipFile(ROOT/'results 30.zip') as z:
    table = list(csv.DictReader(io.StringIO(z.read('30_r6_tta_gold.csv').decode())))
    lookup = {r['StudyInstanceUID']:r for r in table}
    ref['r6'] = [[float(lookup[u][c]) for c in ref['labels']] for u in ref['uids']]
lines = pre.splitlines(keepends=True)
lines[ref_node.lineno-1:ref_node.end_lineno] = ['REF = '+repr(ref)+'\n']
pre = ''.join(lines).replace('30_', '34_')
pre = pre.replace("r7 = require_mount('rsna-knee-r6-frozen-fe055406', 'manifest.json')", '''bundle_roots = []
for slug in ('rsna-knee-r6-frozen-fe055406-w1', 'rsna-knee-r6-frozen-fe055406'):
    for p in mount_candidates(slug):
        if (p / 'manifest.json').is_file():
            digest = hashlib.sha256((p / 'manifest.json').read_bytes()).hexdigest()
            if digest == '6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648':
                bundle_roots.append(p)
if len(bundle_roots) != 1:
    raise RuntimeError('Attach exactly one exact R6 v1 bundle: same input as successful 33, or original R6 Version 1')
r7 = bundle_roots[0]''')
pre = 'import hashlib\n'+pre

setup = text(anchor,6).replace('DEADLINE_HOURS = 8.3','DEADLINE_HOURS = 2.0')
node = literal_assignment(setup,'SOURCES')
sources = ast.literal_eval(node.value)
trainer = sources['compact_train.py']
original_window = next(ast.get_source_segment(trainer,n) for n in ast.parse(trainer).body
                       if isinstance(n,ast.FunctionDef) and n.name=='window_rows')
assert original_window.count('covering_centers(n, 1, eval_stride * ps)') == 1
phase_window = original_window.replace('covering_centers(n, 1, eval_stride * ps)',
                                       'covering_centers(n, 0, eval_stride * ps)')
original_worker = sources['compact_infer.py']
old_line = "        self.window_rows = compact_train.window_rows          # the trainer's own window list (eval mode)"
assert original_worker.count(old_line)==1
worker = original_worker.replace(old_line, old_line+'''\n        self.baseline_window_rows = self.window_rows
        phase_scope = {"math": math, "covering_centers": compact_train.covering_centers,
                       "phys_step": compact_train.phys_step}
        exec(compile(PHASE_WINDOW_SOURCE, '<34 phase0 window sampler>', 'exec'), phase_scope)
        self.window_rows = phase_scope['window_rows']''')
worker = 'PHASE_WINDOW_SOURCE = '+repr(phase_window)+'\n'+worker
old_cand = '            if not cand:\n                raise ValueError("no usable window")'
assert worker.count(old_cand)==1
worker = worker.replace(old_cand, '''            baseline = self.baseline_window_rows(blocks, lambda f, r: meta[r], "eval",
                                                 self.slices, self.eval_stride, phys_mm=self.phys_mm)
            baseline_set, candidate_set = set(baseline), set(cand)
            print("PHASE_AUDIT", json.dumps({"uid": uid, "baseline_windows": len(baseline),
                  "candidate_windows": len(cand), "added_windows": len(candidate_set-baseline_set),
                  "removed_windows": len(baseline_set-candidate_set), "phase": 0}), flush=True)
            if not cand:
                raise ValueError("no usable window")''')
compile(worker,'<34 modified inference worker>','exec')
sources['compact_infer.py']=worker
lines=setup.splitlines(keepends=True)
lines[node.lineno-1:node.end_lineno]=['SOURCES = '+repr(sources)+'\n']
setup=''.join(lines)
sha_node=literal_assignment(setup,'SOURCE_SHA')
hashes=ast.literal_eval(sha_node.value)
hashes['compact_infer.py']=hashlib.sha256(worker.encode()).hexdigest()
lines=setup.splitlines(keepends=True)
lines[sha_node.lineno-1:sha_node.end_lineno]=['SOURCE_SHA = '+repr(hashes)+'\n']
setup=''.join(lines).replace("M_SRC = find_mount(C_SLUG, 'manifest.json')", 'M_SRC = r7')
setup += '''
if manifest.get('eval_stride') != 1 or manifest.get('phys_mm') != 3.5:
    raise RuntimeError('Unexpected R6 physical window contract')
print('34_PHASE_READY: phase 0 instead of phase 1; same 3.5mm neighbours, model, crop, TTA and stackers')
'''

run = text(base,3).replace('30_','34_')
# A new candidate is expected to differ; first check exact predicted coverage, not numerical identity.
cut=run.index('plain_visible, plain_receipt =')
run=run[:cut]+'''
phase_visible, phase_visible_receipt = run_r7_diagnostic(competition, 'visible_phase0', WORKER_ARGS)
print('34_VISIBLE_MICROBENCH_COMPLETE', json.dumps({'seconds':phase_visible_receipt.get('seconds'),
      'scored':phase_visible_receipt.get('scored'), 'fallback':phase_visible_receipt.get('fallback_studies')}))
visible_audits=[]
for log in sorted(WORK.glob('34_visible_phase0_worker*_log.txt')):
    for line in log.read_text(errors='replace').splitlines():
        if line.startswith('PHASE_AUDIT '):
            visible_audits.append(json.loads(line[len('PHASE_AUDIT '):]))
if len(visible_audits)!=len(REF['visible_uids']):
    raise RuntimeError('Visible sampler logs incomplete; stop before gold')
reference_visible=pd.DataFrame(REF['visible_r7'],index=REF['visible_uids'],columns=REF['labels'])
for item in visible_audits:
    if not item['added_windows'] and not item['removed_windows']:
        error=float(np.abs(phase_visible.loc[item['uid']].values-reference_visible.loc[item['uid']].values).max())
        if error>1e-3:
            raise RuntimeError('Unchanged visible window bag differs from original output; stop before gold')
if NB_T0 + 2*3600-time.time()<600:
    raise RuntimeError('Insufficient remaining diagnostic budget; stop before gold')
'''
# Limit loader buffering and encoder batches, not model geometry. Preserve compile mode for numerical consistency.
run=run.replace("'--loader-workers', '2'", "'--loader-workers', '1'")
run=run.replace('deadline = NB_T0 + 3 * 3600','deadline = NB_T0 + 2 * 3600')
run=run.replace('json.dumps(worker_arguments)', "json.dumps(worker_arguments + ['--chunk', '96'])")
gold=text(base,4).replace('rsna30','rsna34').replace('30_','34_')
gold=gold.replace("tta_gold, tta_receipt = run_r7_diagnostic", "tta_gold, tta_receipt = run_r7_diagnostic")
analysis=r'''
from sklearn.metrics import roc_auc_score
import re
uids, labels = REF['uids'], REF['labels']
y = np.asarray(REF['truth'])
baseline = pd.DataFrame(REF['r6'], index=uids, columns=labels)
candidate = tta_gold.loc[uids, labels]
raptor = pd.DataFrame(REF['raptor'], index=uids, columns=labels)
rank = lambda frame: frame.rank(method='average', pct=True)
base_blend = (.5*rank(baseline)+.5*rank(raptor)).astype(np.float32)
candidate_blend = (.5*rank(candidate)+.5*rank(raptor)).astype(np.float32)
rows=[]
for j,label in enumerate(labels):
    a=roc_auc_score(y[:,j],base_blend[label]); b=roc_auc_score(y[:,j],candidate_blend[label])
    rows.append({'target':label,'positives':int(y[:,j].sum()),
                 'r6_base_auc':roc_auc_score(y[:,j],baseline[label]),
                 'r6_phase0_auc':roc_auc_score(y[:,j],candidate[label]),
                 '29B_proxy_base_auc':a,'29B_proxy_phase0_auc':b,'delta':b-a,
                 'mean_abs_probability_change':float(np.abs(candidate[label]-baseline[label]).mean())})
metrics=pd.DataFrame(rows)
metrics.to_csv(WORK/'34_per_target_auc.csv',index=False)
audits=[]
for log in sorted(WORK.glob('34_r6_tta_gold_worker*_log.txt')):
    for line in log.read_text(errors='replace').splitlines():
        if line.startswith('PHASE_AUDIT '):
            audits.append(json.loads(line[len('PHASE_AUDIT '):]))
inventory=pd.DataFrame(audits)
if len(inventory)!=len(uids) or not inventory.uid.is_unique or set(inventory.uid)!=set(uids):
    raise RuntimeError('Missing physical sampling audits; do not interpret the candidate')
inventory=inventory.set_index('uid').loc[uids]
changed=(inventory.added_windows+inventory.removed_windows)>0
if changed.any():
    changed_delta=float(np.abs(candidate.values[changed.values]-baseline.values[changed.values]).max())
else:
    changed_delta=0.0
unchanged_error=float(np.abs(candidate.values[~changed.values]-baseline.values[~changed.values]).max()) if (~changed).any() else None
if unchanged_error is not None and unchanged_error>1e-3:
    raise RuntimeError('Unchanged-window studies do not reproduce baseline within 0.001; investigate serving drift')
inventory.to_csv(WORK/'34_sampling_inventory.csv',index_label='StudyInstanceUID')
candidate.to_csv(WORK/'34_r6_phase0_gold.csv',index_label='StudyInstanceUID')
candidate_blend.to_csv(WORK/'34_29B_proxy_phase0_gold.csv',index_label='StudyInstanceUID')
result={'experiment':'34_R6_physical_sampling_phase0','status':'COMPLETE' if changed.any() else 'NO_NEW_IMAGE_EVIDENCE',
        'gold_studies':len(uids),'changed_window_studies':int(changed.sum()),
        'added_windows_total':int(inventory.added_windows.sum()),
        'removed_windows_total':int(inventory.removed_windows.sum()),
        'unchanged_window_max_abs_error':unchanged_error,'changed_window_max_abs_delta':changed_delta,
        'baseline_proxy_macro_auc':float(metrics['29B_proxy_base_auc'].mean()),
        'candidate_proxy_macro_auc':float(metrics['29B_proxy_phase0_auc'].mean()),
        'delta_proxy_macro_auc':float(metrics.delta.mean()),
        'elapsed_seconds':round(time.time()-NB_T0,2),'submission_created':False,
        'gold_is_independent_validation':False,
        'warning':'29B 50/50 Raptor384 proxy only, NOT full 33 with Raptor224 and per-target weights. Phase0 is not a proven improvement. No weights fitted.'}
(WORK/'34_diagnostic_report.json').write_text(json.dumps(result,indent=2))
assert not (WORK/'submission.csv').exists()
display(metrics.round(5)); print(json.dumps(result,indent=2))
'''

header_gate=r'''
# Read headers only. Thin 3-D stacks are already slab-averaged by the actual
# builder; most therefore have physical step=1 and no phase difference.
try:
    import pydicom
except ImportError:
    import subprocess, sys, tempfile, zipfile
    wheel=None
    if (code/'dependencies.zip').is_file():
        with zipfile.ZipFile(code/'dependencies.zip') as z:
            names=[n for n in z.namelist() if Path(n).name.startswith('pydicom-') and n.endswith('.whl')]
            if len(names)==1:
                wheel=Path(tempfile.gettempdir())/Path(names[0]).name
                wheel.write_bytes(z.read(names[0]))
    else:
        candidates=list(code.rglob('pydicom-*.whl'))
        if len(candidates)==1: wheel=candidates[0]
    if wheel is None:
        raise RuntimeError('pydicom unavailable and pinned offline wheel missing from attached code dataset')
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',str(wheel)],check=True)
    import pydicom
header_rows=[]
for uid in REF['uids']:
    study_root=competition/'train_series'/uid
    if not study_root.is_dir():
        raise RuntimeError('Cannot inventory gold DICOM directory: '+uid)
    for series_dir in sorted(p for p in study_root.iterdir() if p.is_dir()):
        positions=[]; normal=None; failed=0
        for file in series_dir.iterdir():
            if not file.is_file(): continue
            try:
                ds=pydicom.dcmread(file,stop_before_pixels=True,
                                  specific_tags=['ImagePositionPatient','ImageOrientationPatient'])
                ipp=np.asarray(ds.ImagePositionPatient,float)
                iop=np.asarray(ds.ImageOrientationPatient,float)
                if normal is None: normal=np.cross(iop[:3],iop[3:])
                positions.append(float(np.dot(ipp,normal)))
            except Exception:
                failed+=1
        positions=np.sort(np.asarray(positions,float))
        spacing=float(np.median(np.diff(positions))) if len(positions)>1 else 0.0
        rendered_spacing=spacing
        if 0<spacing<1.75:
            g=max(1,int(round(3.5/spacing)))
            n_new=len(positions)//g+int(len(positions)%g>=max(1,g//2))
            if n_new>=2:
                slab=np.asarray([positions[j*g:min(len(positions),(j+1)*g)].mean() for j in range(n_new)])
                rendered_spacing=float(np.median(np.diff(slab)))
        step=max(1,int(round(3.5/max(rendered_spacing,0.1)))) if rendered_spacing>0 else None
        header_rows.append({'StudyInstanceUID':uid,'series':series_dir.name,
                            'header_count':len(positions),'header_failures':failed,
                            'native_spacing':spacing,'rendered_spacing_estimate':rendered_spacing,
                            'physical_step_estimate':step,'potential_new_centres':bool(step and step>1)})
inventory=pd.DataFrame(header_rows)
if inventory.empty:
    raise RuntimeError('No usable series headers; stop before GPU model loading')
inventory.to_csv(WORK/'34_header_phase_gate.csv',index=False)
eligible=inventory.loc[inventory.potential_new_centres,'StudyInstanceUID'].unique().tolist()
HEADER_GATE={'potentially_eligible_studies':len(eligible),
             'potentially_eligible_series':int(inventory.potential_new_centres.sum()),
             'header_failures':int(inventory.header_failures.sum()),
             'unknown_spacing_series':int(inventory.physical_step_estimate.isna().sum()),
             'note':'Conservative header estimate over all series, not proof that the model selects these series. Worker audits are definitive.'}
DO_GPU_PROBE=bool(eligible)
if not DO_GPU_PROBE:
    result={'experiment':'34_R6_physical_sampling_phase0','status':'NO_ELIGIBLE_HEADER_EVIDENCE',
            'header_gate':HEADER_GATE,'gpu_inference_started':False,'submission_created':False,
            'note':'No physical-step>1 series found after estimated slab averaging. Stop this direction without model inference; unknown headers are not proof of absence.'}
    (WORK/'34_diagnostic_report.json').write_text(json.dumps(result,indent=2))
print('34_HEADER_GATE',json.dumps(HEADER_GATE),'GPU_PROBE',DO_GPU_PROBE,flush=True)
'''

intro='''# 34 — R6 physical-sampling phase probe (original experiment)

DIAGNOSTIC ONLY. No training and no submission. Keep confirmed Public 0.958 notebook 33 unchanged.

Source audit found R6 already uses eval_stride=1, but physical step round(3.5mm / rendered spacing) can exceed one on thin stacks. The current sampler starts at phase=1. Training's cover sampler uses random phases. This experiment changes ONLY the evaluation sampling phase to 0, retaining the trained neighbour separation, crop, anchors, laterality, model heads, mirror TTA and stackers. Ordinary stacks with physical step=1 are identical; thin stacks can expose different centres. It does NOT halve neighbour spacing or add arbitrary flips. A new inference worker SHA is recorded; trainer/model hashes are unchanged.

Hypothesis: different trained-compatible sampling phases may expose focal evidence missed by the fixed evaluation grid. This is a hypothesis, not a known accuracy gain. Existing attention and spatial metadata can still make it harmful. The sampler logs added/removed windows for every study; zero changed bags closes the direction.

Attach competition (including train_series), exact R6 v1 bundle (same w1 copy as successful 33 OR original version 1, not both), and same OrthoFoundation code/anatomy as 33. No Raptor weights or saved outputs needed. GPU T4x2, Internet Off, Save & Run All. Metadata is a convenience, not proof of actual mount version.

First read gold DICOM HEADERS ONLY and account for the builder's existing averaging of very thin stacks into ~3.5mm slabs. Only remaining physical-step>1 series can benefit; roughly 1.75–2.33mm rendered spacing is a typical eligible range. If none are found, skip all model inference and finish with a no-evidence report. This prevents spending GPU on an experiment that would be a no-op. If eligible series exist, score three visible studies as a runtime/resource microbenchmark, then 58 gold once. One loader worker and chunk96 limit buffering. A two-hour total timeout is NOT an ETA; duration is unmeasured. Stop on any model fallback or input drift. Return the Output ZIP, especially 34_diagnostic_report.json, 34_header_phase_gate.csv, 34_sampling_inventory.csv and worker logs.

Gold was reused upstream; comparisons are exploratory. Raptor384 gold predictions from 26 and original R6 TTA predictions from 30 are embedded. Blend comparison is the 29B proxy, NOT full 33 (no gold Raptor224 predictions available). Do not fit per-target weights here. No new submission notebook is authorized for automatic launch or submission by this diagnostic.
'''
intro+='\n'+text(anchor,1)
def cell(s,md=False):
    if not md: compile(s,'<34 generated cell>','exec')
    c={'cell_type':'markdown' if md else 'code','metadata':{},'source':s.splitlines(keepends=True)}
    if not md: c.update(execution_count=None,outputs=[])
    return c
nb=copy.deepcopy(anchor)
guard=lambda s:'if DO_GPU_PROBE:\n'+''.join('    '+line+'\n' for line in s.splitlines())
nb['cells']=[cell(intro,True),cell(pre),cell(header_gate)]+[cell(guard(s)) for s in (setup,run,gold,analysis)]
nb['metadata']['kaggle']['dataSources']=[s for s in nb['metadata']['kaggle']['dataSources'] if s.get('sourceId')!=20297367]
target=ROOT/'34_R6_Physical_Sampling_Phase_Gold_Probe.ipynb'
# This generator owns its newly created experiment file; do not point it at the anchor.
assert target.name != '33_RSNA_0958_R6_Raptor_Submit.ipynb'
target.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
(ROOT/'34_Probe_Provenance.json').write_text(json.dumps({
    'anchor':anchor['metadata'].get('kaggle'),'anchor_file_sha256':hashlib.sha256((ROOT/'33_RSNA_0958_R6_Raptor_Submit.ipynb').read_bytes()).hexdigest(),
    'original_worker_sha256':hashlib.sha256(original_worker.encode()).hexdigest(),
    'phase0_worker_sha256':hashes['compact_infer.py'],
    'change':'window_rows eval phase 1 -> 0 only; trainer unchanged; extra logs',
    'training_started':False,'gpu_tested_locally':False},indent=2),encoding='utf-8')

# Run the actual public sampler and candidate sampler on synthetic geometry.
functions={n.name:ast.get_source_segment(trainer,n) for n in ast.parse(trainer).body
           if isinstance(n,ast.FunctionDef) and n.name in ('phys_step','covering_centers')}
import math
scope={'math':math}
for s in functions.values(): exec(s,scope)
exec(original_window,scope); baseline_fn=scope['window_rows']
exec(phase_window,scope); candidate_fn=scope['window_rows']
tested=0
for n in (1,2,3,9,32,101):
    for spacing in (.4,.8,1.75,3.5,5.0):
        blocks=[{'n_slices':n,'views':[(0,0,0)]}]
        meta=lambda f,r:[0,0,0,0,0,math.log1p(spacing)]
        old=baseline_fn(blocks,meta,'eval',3,1,phys_mm=3.5)
        new=candidate_fn(blocks,meta,'eval',3,1,phys_mm=3.5)
        assert all(0<=x<n for row in new for x in row[1:])
        if scope['phys_step'](spacing,3.5)==1: assert old==new
        assert len(new)==len(set(new))
        tested+=1
print(json.dumps({'notebook':target.name,'code_compile':True,'sampler_geometry_tests':tested,
                  'gpu_tested':False,'submission_created':False},indent=2))
