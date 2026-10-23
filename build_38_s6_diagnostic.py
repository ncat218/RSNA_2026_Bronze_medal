"""Package inspected S6 source and existing gold references; no network or GPU use."""
import ast
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PUBLIC_SHA = '6e87350d5ae092b7a889546ca13517303523bc469f7e611ffe945913434d22fe'
public = (ROOT/'research_38_s6/public_s6_source.py').read_text(encoding='utf-8')
assert hashlib.sha256(public.encode()).hexdigest() == PUBLIC_SHA
tree = ast.parse(public)
# Keep original component-B functions unchanged. Drop all component-A execution and
# the original entrypoint; docstrings describing the original rank blend are not our recipe.
drop_functions = {'main', 'run_a', 'a_cmd', 'a_done', 'run_parity', 'blend_and_write'}
parts = []
for node in tree.body:
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        continue
    if isinstance(node, ast.If):
        assert '__name__' in ast.get_source_segment(public, node.test)
        continue
    if isinstance(node, ast.FunctionDef) and node.name in drop_functions:
        continue
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'A_SOURCE_B64', 'A_B64'} for t in node.targets):
        continue
    parts.append(ast.get_source_segment(public, node))
# Resolve reuploaded/renamed/nested Kaggle dataset mounts by marker, not title.
# Do not recurse through competition DICOMs or cached images. Identity checks in
# prepare_inputs remain mandatory; a successful lookup is not permission to substitute weights.
parts.append(r'''
_published_find_mount = find_mount

def mount_inventory():
    entries = []
    for base in (INPUT, INPUT/'datasets', INPUT/'models'):
        if not base.is_dir():
            continue
        for p in sorted(base.iterdir()):
            if p.is_dir():
                entries.append(str(p))
                if base != INPUT:
                    entries.extend(str(q) for q in sorted(p.iterdir()) if q.is_dir())
    return entries[:150]

def find_mount(slug, marker):
    try:
        return _published_find_mount(slug, marker)
    except K1Error:
        pass
    found = []
    blocked = {'competitions', 'train_series', 'test_series', 'study_cache', '__pycache__'}
    for current, dirs, files in os.walk(INPUT):
        root = Path(current)
        depth = len(root.relative_to(INPUT).parts)
        if depth >= 6:
            dirs[:] = []
        else:
            dirs[:] = [d for d in dirs if d not in blocked and d != COMP]
        if marker in files:
            resolved = root.resolve()
            if resolved not in found:
                found.append(resolved)
    if len(found) == 1:
        log('Resolved renamed/nested mount by marker:', slug, found[0])
        return found[0]
    detail = {'requested_slug':slug, 'marker':marker, 'candidates':[str(p) for p in found],
              'attached_directories':mount_inventory()}
    OUT.mkdir(parents=True,exist_ok=True)
    save_json(OUT/'38_input_resolution.json',detail)
    raise K1Error('Missing or ambiguous input '+slug+'/'+marker+'. Candidates: '+str(found)+
                  '. For anatomy.pt attach heliosli/rsna-orthofoundation-fold0-epoch7-20261004 Version 1 '+
                  '(same anatomy/code dataset as successful 33). Merely attaching S6 weights is not sufficient. '+
                  'See 38_input_resolution.json; do not rerun until attachments are corrected.')
''')
module = '\n\n'.join(parts)+'\n'
compile(module, '<38 inspected component B>', 'exec')

def rows(archive, member):
    with zipfile.ZipFile(ROOT/archive) as z:
        return list(csv.DictReader(io.StringIO(z.read(member).decode('utf-8-sig'))))

truth = rows('results 30.zip', '30_gold_truth.csv')
labels = list(truth[0])[1:]
uids = [r['StudyInstanceUID'] for r in truth]
assert len(uids) == len(set(uids)) == 58
def values(archive, member):
    data = rows(archive, member)
    lookup = {r['StudyInstanceUID']: r for r in data}
    assert len(lookup) == len(data) and set(lookup) == set(uids)
    return [[float(lookup[u][c]) for c in labels] for u in uids]
ref = {'uids': uids, 'labels': labels,
       'truth': [[int(float(r[c])) for c in labels] for r in truth],
       'r6': values('results 30.zip', '30_r6_tta_gold.csv'),
       'raptor384': values('results 26.zip', '26_raptor_baseline.csv'),
       'proxy29B': values('results 30.zip', '30_gold_blend_r6_tta_current.csv')}

setup = r'''
import hashlib, importlib.util, json, os, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
OUT = Path('/kaggle/working/rsna_38_s6')
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
REPORT = {'experiment': '38_S6_complementarity_diagnostic', 'status': 'STARTED',
          'training_started': False, 'submission_created': False,
          'full_anchor33_gold_available': False, 'fallback_to_raptor_enabled': False,
          'public_reference_parity': 'UNVERIFIED: no pinned S6 visible reference available',
          'gold_caveat': 'Reused diagnostic set, not independent validation; proxy29B is NOT notebook33.'}
def save_report():
    REPORT['elapsed_seconds'] = round(time.time()-T0, 2)
    (OUT/'38_diagnostic_report.json').write_text(json.dumps(REPORT, indent=2, allow_nan=False))
save_report()
'''
setup += '\nREF = '+repr(ref)+'\nS6_SOURCE = '+repr(module)+'\n'
setup += r'''
source_path = OUT/'s6_component_b.py'
source_path.write_bytes(S6_SOURCE.encode('utf-8'))
spec = importlib.util.spec_from_file_location('rsna38_s6', source_path)
S6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S6)  # main and component-A runner removed at build time
S6.OUT = OUT/'worker_outputs'
S6.OUT.mkdir(exist_ok=True)
S6.TMP = Path('/tmp/rsna38_s6')
S6.TMP.mkdir(parents=True, exist_ok=True)
S6.MODE = 'parity'  # only ensures prepare_inputs reads train-series cost metadata
S6.B_GEOMETRY = dict(S6.B_GEOMETRY, chunk=96)
S6.B_WORKERS = 1  # bounded loader per GPU; model/preprocessing unchanged
S6._LOG = None
REPORT['public_source_sha256'] = '6e87350d5ae092b7a889546ca13517303523bc469f7e611ffe945913434d22fe'
REPORT['packaged_component_sha256'] = hashlib.sha256(S6_SOURCE.encode()).hexdigest()
REPORT['execution_changes'] = {'chunk': 96, 'loader_workers_per_gpu': 1,
                               'no_A_no_rank_blend_no_submission': True}
print('No training; S6 only. No Raptor or R6 checkpoint required.')
'''

preflight = r'''
try:
    REPORT['attached_directories'] = S6.mount_inventory()
    save_report()
    ds = S6.find_mount(S6.DS_SLUG, 'code_manifest.json')
    observed = S6.sha256_file(ds/'code_manifest.json')
    if observed != 'b66c574b5c0c13ccaa49d1219d41a8781431166186ef83e78cab6a9ba6b02d22':
        raise RuntimeError('S6 code manifest changed. Attach S6 version 5; do not bypass this guard: '+observed)
    I = S6.prepare_inputs()  # hashes checkpoint, anatomy and every staged code file
    train = pd.read_csv(I.comp/'train.csv', dtype={'StudyInstanceUID': str})
    gold = train.loc[train[S6.LAB].notna().all(axis=1)].set_index('StudyInstanceUID')
    if not gold.index.is_unique or set(gold.index) != set(REF['uids']) or S6.LAB != REF['labels']:
        raise RuntimeError('Gold UID/target identity mismatch')
    if not np.array_equal(gold.loc[REF['uids'], S6.LAB].to_numpy(float), np.asarray(REF['truth'])):
        raise RuntimeError('Competition gold labels differ from the embedded reference')
    if not (I.comp/'train_series').is_dir():
        raise RuntimeError('Attach competition training DICOM series, not just test files')
    REPORT['inputs'] = S6._RECEIPT['inputs']
    REPORT['environment'] = S6.probe_env()
    if not REPORT['environment'].get('cuda_available'):
        raise RuntimeError('Enable GPU T4x2 before running')
    S6._RECEIPT['env'] = REPORT['environment']
    REPORT['stack_probe'] = S6.probe_b_stack(I)  # strict state-dict + anatomy forward in disposable CPU process
    REPORT['status'] = 'PREFLIGHT_COMPLETE'
    save_report()
    print(json.dumps(REPORT, indent=2))
except BaseException as exc:
    REPORT.update(status='FAILED_PREFLIGHT', error=repr(exc), traceback=traceback.format_exc())
    save_report()
    raise
'''

run = r'''
try:
    # Diagnostic only. Each run enforces exact receipt/code identity, anchors and
    # complete UID coverage. A missing study is never replaced by Raptor or 0.5.
    visible_ids = I.test_ids[:3]
    p_visible, _, visible_info, failed = S6.run_b(I, 'visible', visible_ids, 'test', 1200)
    if failed:
        raise RuntimeError('S6 visible failures: '+repr(failed))
    REPORT['visible'] = visible_info
    pd.DataFrame(p_visible, index=pd.Index(visible_ids, name='StudyInstanceUID'),
                 columns=S6.LAB).to_csv(OUT/'38_s6_visible_probabilities.csv')
    save_report()
    # No hidden-test runtime extrapolation from only three studies.
    remaining = 7200 - (time.time()-T0)
    if remaining < 1800:
        raise RuntimeError('Insufficient diagnostic budget left before gold; stop safely')
    p_gold, _, gold_info, failed = S6.run_b(I, 'gold', REF['uids'], 'train', remaining)
    if failed or p_gold.shape != (58,12) or not np.isfinite(p_gold).all():
        raise RuntimeError('Incomplete/nonfinite S6 gold predictions')
    REPORT['gold'] = gold_info
    S6_GOLD = pd.DataFrame(p_gold, index=pd.Index(REF['uids'], name='StudyInstanceUID'), columns=S6.LAB)
    S6_GOLD.to_csv(OUT/'38_s6_gold_probabilities.csv')
    REPORT['status'] = 'INFERENCE_COMPLETE'
    save_report()
except BaseException as exc:
    S6.kill_all_children()
    REPORT.update(status='FAILED_INFERENCE', error=repr(exc), traceback=traceback.format_exc())
    save_report()
    raise
'''

analysis = r'''
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

def auc_vector(y, p):
    return np.array([roc_auc_score(y[:,j], p[:,j]) if len(np.unique(y[:,j])) == 2 else np.nan
                     for j in range(y.shape[1])])

def paired_bootstrap(y, a, b, n_boot=1000):
    rng = np.random.default_rng(38019)
    deltas = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(y), len(y))
        if any(len(np.unique(y[ix,j])) != 2 for j in range(y.shape[1])):
            continue  # use identical complete target sets, never variable-target macro means
        deltas.append(float(np.mean(auc_vector(y[ix],a[ix])-auc_vector(y[ix],b[ix]))))
    if not deltas:
        return {'valid_resamples': 0, 'ci95': None}
    return {'valid_resamples': len(deltas), 'ci95': np.quantile(deltas,[.025,.975]).tolist()}

def correlate(a,b):
    return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None

try:
    y = np.asarray(REF['truth'], dtype=int)
    s6p = S6_GOLD.loc[REF['uids'], REF['labels']].to_numpy(float)
    models = {name: np.asarray(REF[name],dtype=float) for name in ('r6','raptor384','proxy29B')}
    models['s6'] = s6p
    # Single preregistered diagnostic blend, NOT optimized on gold and NOT exact33.
    models['proxy29B_90_s6_10'] = .9*models['proxy29B']+.1*s6p
    for name,p in models.items():
        if p.shape != y.shape or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
            raise RuntimeError('Bad prediction matrix: '+name)
    aucs = {name: auc_vector(y,p) for name,p in models.items()}
    records = []
    for j,target in enumerate(REF['labels']):
        r = {'target':target, 'positives':int(y[:,j].sum())}
        r.update({name+'_auc':float(a[j]) for name,a in aucs.items()})
        for name in ('r6','raptor384'):
            r['s6_'+name+'_spearman'] = float(spearmanr(s6p[:,j],models[name][:,j]).statistic)
            r['s6_'+name+'_signed_error_pearson'] = correlate(s6p[:,j]-y[:,j], models[name][:,j]-y[:,j])
        records.append(r)
    pd.DataFrame(records).to_csv(OUT/'38_per_target_metrics.csv', index=False)
    REPORT['macro_auc_descriptive_only'] = {n:float(np.mean(a)) for n,a in aucs.items()}
    REPORT['paired_macro_auc_delta_ci95'] = {
        's6_minus_r6': paired_bootstrap(y,s6p,models['r6']),
        's6_minus_raptor384': paired_bootstrap(y,s6p,models['raptor384']),
        'fixed_proxy_blend_minus_proxy29B': paired_bootstrap(y,models['proxy29B_90_s6_10'],models['proxy29B'])}
    cases = []
    for j,target in enumerate(REF['labels']):
        for k,u in enumerate(REF['uids']):
            cases.append({'StudyInstanceUID':u,'target':target,'gold':int(y[k,j]),
                          's6':float(s6p[k,j]),'r6':float(models['r6'][k,j]),
                          'raptor384':float(models['raptor384'][k,j]),
                          'abs_s6_r6_delta':float(abs(s6p[k,j]-models['r6'][k,j]))})
    pd.DataFrame(cases).sort_values('abs_s6_r6_delta',ascending=False).to_csv(OUT/'38_discordant_cases.csv',index=False)
    REPORT['limitations'] = [
        'Gold is small and reused upstream and in project diagnostics; confidence intervals do not remove selection bias.',
        'Signed probability-error correlation is descriptive and partly reflects calibration and shared truth.',
        'No exact notebook33 gold comparison: Raptor224 gold predictions are unavailable.',
        'Visible S6 execution is not parity against published values; no pinned reference provided.',
        'No hidden-test runtime or Private score guarantee. No blend weight selected from these metrics.']
    REPORT['status'] = 'COMPLETE'
    save_report()
    display(pd.DataFrame(records))
    print(json.dumps(REPORT['macro_auc_descriptive_only'],indent=2))
    print('COMPLETE. Download Output ZIP. Do NOT submit notebook38: it deliberately creates no submission.csv.')
except BaseException as exc:
    REPORT.update(status='FAILED_ANALYSIS',error=repr(exc),traceback=traceback.format_exc())
    save_report()
    raise
'''

intro = '''# 38 — S6 complementarity diagnostic (not a submission)

Protect notebook33, user-confirmed Public 0.958. Evaluate independently retrained Nartaa S6, using its exact public component-B worker and checkpoint; no new model training, changed pooling, Raptor fallback, rank fusion or submission file. Original publisher reports S6 Public 0.950; this notebook has no score.

Attach exactly **three inputs**:
1. RSNA Knee competition, including `train_series` DICOMs.
2. [nartaa/rsna-knee-s6fall-weights, Version5](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights/versions/5).
3. [heliosli/rsna-orthofoundation-fold0-epoch7-20261004, Version1](https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004/versions/1), supplying anatomy/code.

No R6 bundle, Raptor weights, study cache or prior Output needed. Reference predictions are embedded from results26/30. **GPU T4x2, Internet Off; Save & Run All.** Metadata pins versions but may not replace inputs in an existing draft: check actual attached inputs. Allow up to a two-hour diagnostic budget, not an ETA. Strict stack probing has an additional subprocess timeout; no guarantee of GPU memory or wall time is made without Kaggle execution.

Gold58 is reused, not independent CV. The stored ensemble reference is **29B proxy, NOT anchor33** (Raptor224 gold unavailable). All comparisons are descriptive. Visible predictions have no pinned S6 reference to assert published parity. Failure is recorded, not replaced by Raptor. Send Output ZIP, especially `38_diagnostic_report.json`, `38_per_target_metrics.csv`, `38_s6_gold_probabilities.csv`, and worker logs.

Credits: Nartaa S6 release and Heliosli code/anatomy; inherited publisher/third-party licence conditions apply. S6 learned weights are declared CC BY-NC4.0, code Apache2.0; these declarations are not a competition-eligibility determination.
'''
def cell(kind,text):
    d={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
    if kind=='code': d.update(execution_count=None,outputs=[])
    return d
nb={'nbformat':4,'nbformat_minor':5,'cells':[cell('markdown',intro)]+[cell('code',t) for t in (setup,preflight,run,analysis)],
    'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                'language_info':{'name':'python'},'kaggle':{'accelerator':'nvidiaTeslaT4','isGpuEnabled':True,
                'isInternetEnabled':False,'language':'python','sourceType':'notebook','dataSources':[
                    {'sourceType':'competition','sourceId':154281},
                    {'sourceType':'datasetVersion','sourceId':20844630},
                    {'sourceType':'datasetVersion','sourceId':20440357}]}}}
for i,c in enumerate(nb['cells']):
    c['id']=f'rsna38-{i}'
    if c['cell_type']=='code': compile(''.join(c['source']),f'<cell{i}>','exec')
target = ROOT/'38_S6_Complementarity_Gold_Diagnostic.ipynb'
target.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
(ROOT/'38_Build_Manifest.json').write_text(json.dumps({
    'notebook_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
    'public_source_sha256':PUBLIC_SHA,'component_sha256':hashlib.sha256(module.encode()).hexdigest(),
    'source_dataset_version':5,'source_notebook_version':2,
    'embedded_reference_zip_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('results 26.zip','results 30.zip')},
    'gpu_execution_verified':False,'training_started':False,'submission_notebook':False},indent=2),encoding='utf-8')
print(target.name, target.stat().st_size, 'bytes; all cells compile')

if __name__=='__main__':
    print('Packaged only; no downloaded code or MRI inference executed.')
