"""Audit 29 outputs and package one no-training, no-submission R6 diagnostic."""
import ast
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
nb28 = json.loads((ROOT / '28_R7_Anatomical_Mirror_Gold_Diagnostic.ipynb').read_text(encoding='utf-8'))
nb29 = json.loads((ROOT / '29A_R6_TTA_Raptor384_224_Reproduction_V1Pinned.ipynb').read_text(encoding='utf-8'))
source28 = lambda i: ''.join(nb28['cells'][i]['source'])
def table(z, name):
    return list(csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig'))))

archives = {key: zipfile.ZipFile(ROOT / f'results 29{key}.zip') for key in 'AB'}
audits = {key: json.loads(z.read(f'29{key}_run_audit.json')) for key, z in archives.items()}
for audit in audits.values():
    assert audit['status'] == 'COMPLETE' and audit['model_bundle'] == 'fe055406fad6a12f'
    assert all(audit['workers_flip_tta']) and not any(audit['failures'].values())
a, b = [table(archives[key], 'submission.csv') for key in 'AB']
assert [r['StudyInstanceUID'] for r in a] == [r['StudyInstanceUID'] for r in b]
labels = list(a[0])[1:]
differences = [{'target': c,
                'changed_cells': sum(abs(float(x[c])-float(y[c])) > 1e-7 for x,y in zip(a,b)),
                'max_abs_delta': max(abs(float(x[c])-float(y[c])) for x,y in zip(a,b))}
               for c in labels]
report = {'experiment': '29_output_audit', 'date': '2026-10-16',
          'public_scores_user_reported': {'29A': .957, '29B': .957},
          'zip_sha256': {key: hashlib.sha256((ROOT / f'results 29{key}.zip').read_bytes()).hexdigest() for key in 'AB'},
          'audits': audits, 'visible_rows_only': len(a), 'per_target_difference': differences,
          'r6_predictions_byte_identical': archives['A'].read('ours_sub.csv') == archives['B'].read('ours_sub.csv'),
          'interpretation': 'No test labels in these ZIPs. Three visible studies cannot explain hidden Public errors, establish Private gains, or select target weights.'}
(ROOT / '29AB_Output_Analysis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

pre = source28(1)
ref_node = next(n for n in ast.parse(pre).body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'REF' for t in n.targets))
ref = ast.literal_eval(ref_node.value)
visible = table(archives['B'], 'ours_sub.csv')
ref['visible_uids'] = [r['StudyInstanceUID'] for r in visible]
ref['visible_r7'] = [[float(r[c]) for c in labels] for r in visible]
lines = pre.splitlines(keepends=True)
lines[ref_node.lineno-1:ref_node.end_lineno] = ['REF = ' + repr(ref) + '\n']
pre = ''.join(lines).replace('rsna-knee-compact-r7cnxfsss', 'rsna-knee-r6-frozen-fe055406').replace('8c7b666e27dc0e8f', 'fe055406fad6a12f')
pre = pre.replace('28_', '30_').replace('28_INPUTS', '30_INPUTS').replace('R7', 'R6')
pre = pre.replace('results 26 and visible R6 from results 27', 'results 26 and visible R6 TTA from results 29B')
pre = pre.replace('verified 25A/26/27 runs', 'verified 29A/29B runs')
pre = pre.replace('Embedded exact Raptor/R6 outputs', 'Embedded exact Raptor384/R7-plain outputs')
pre = pre.replace("'r7_bundle_id'", "'r6_bundle_id'").replace("'r7_gold_in_training'", "'r6_gold_in_training'")
pre += '''
import hashlib
observed_manifest_sha = hashlib.sha256((r7 / 'manifest.json').read_bytes()).hexdigest()
if observed_manifest_sha != '6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648':
    raise RuntimeError('Attach R6 VERSION 1 (datasetVersion 20611930), not latest/R7')
preflight['r6_manifest_sha256'] = observed_manifest_sha
(WORK / '30_preflight.json').write_text(json.dumps(preflight, indent=2))
'''
setup = next(''.join(c['source']) for c in nb29['cells'] if 'SOURCES =' in ''.join(c['source']))
setup = setup.replace('DEADLINE_HOURS = 8.3', 'DEADLINE_HOURS = 3.0')
run = source28(3).replace('28_', '30_').replace('R7', 'R6')
run = run.replace("    receipt_path = output / 'submission_receipt.json'", """    # Save existing worker diagnostics; do not alter worker/model source.
    for worker_dir in sorted(output.glob('worker*')):
        for filename in ('fluid_stacker.csv', 'state_stacker.csv', 'predictions.csv'):
            path = worker_dir / filename
            if path.is_file():
                shutil.copyfile(path, WORK / ('30_' + tag + '_' + worker_dir.name + '_' + filename))
    receipt_path = output / 'submission_receipt.json'""")
run = run.replace('visible_plain', 'visible_tta').replace('VISIBLE_PLAIN', 'VISIBLE_TTA')
run = run.replace('R6 plain probabilities', 'R6 TTA probabilities')
gold = source28(4).replace('rsna28', 'rsna30').replace('28_', '30_').replace('r7_tta_gold', 'r6_tta_gold')
gold = gold.replace("WORKER_ARGS + ['--flip-tta']", 'list(WORKER_ARGS)')

analysis = r'''
from sklearn.metrics import roc_auc_score
uids, labels = REF['uids'], REF['labels']
y = np.asarray(REF['truth'], dtype=np.int64)
r6 = tta_gold.loc[uids, labels].copy()
r7_plain = pd.DataFrame(REF['r7'], index=uids, columns=labels)
raptor = pd.DataFrame(REF['raptor'], index=uids, columns=labels)

def dump_table(suffix):
    paths = sorted(WORK.glob('30_r6_tta_gold_worker*_' + suffix + '.csv'))
    if not paths:
        raise RuntimeError('Worker diagnostic is missing: ' + suffix)
    frame = pd.concat([pd.read_csv(p, dtype={'StudyInstanceUID': str}) for p in paths], ignore_index=True)
    if not frame.StudyInstanceUID.is_unique or set(frame.StudyInstanceUID) != set(uids):
        raise RuntimeError('Worker dump UID coverage mismatch: ' + suffix)
    return frame.set_index('StudyInstanceUID').loc[uids]

fluid, state = dump_table('fluid_stacker'), dump_table('state_stacker')
candidates = {'r6_tta_current': r6}
for name, target, frame in [('without_fluid_stacker', 'Effusion', fluid),
                            ('without_state_stacker', "Baker's", state)]:
    raw, deployed = frame[target+'|p_model'].to_numpy(float), frame[target+'|p_out'].to_numpy(float)
    if not np.isfinite(raw).all() or not ((raw >= 0) & (raw <= 1)).all():
        raise RuntimeError('Missing/invalid raw probabilities for ' + target)
    if np.max(np.abs(deployed-r6[target].values)) > 2e-6:
        raise RuntimeError('Worker stacker dump does not reproduce deployed output for ' + target)
    candidates[name] = r6.copy()
    candidates[name][target] = raw

ranks_raptor = raptor.rank(method='average', pct=True)
blends = {name: (.5*ranks_raptor + .5*frame.rank(method='average', pct=True)).astype(np.float32)
          for name, frame in candidates.items()}
# Fixed family-diversity hypothesis, not a fitted weight. R7 here is PLAIN,
# matching results 26, not the unmeasured R7 TTA configuration.
blends['r6_r7_plain_diversity'] = (.5*ranks_raptor + .4*r6.rank(method='average', pct=True)
                                + .1*r7_plain.rank(method='average', pct=True)).astype(np.float32)
base = blends['r6_tta_current']
rows = []
for j, label in enumerate(labels):
    row = {'target': label, 'positives': int(y[:,j].sum()), 'negatives': int(len(y)-y[:,j].sum()),
           'r6_tta_auc': float(roc_auc_score(y[:,j], r6[label])),
           'r7_plain_auc': float(roc_auc_score(y[:,j], r7_plain[label])),
           'raptor384_auc': float(roc_auc_score(y[:,j], raptor[label])),
           'r6_r7_plain_spearman': float(r6[label].corr(r7_plain[label], method='spearman'))}
    for name, frame in blends.items():
        row[name+'_auc'] = float(roc_auc_score(y[:,j], frame[label]))
        row[name+'_delta'] = row[name+'_auc'] - float(roc_auc_score(y[:,j], base[label]))
    rows.append(row)
metrics = pd.DataFrame(rows)
metrics.to_csv(WORK / '30_per_target_auc.csv', index=False)
for name, frame in blends.items():
    frame.to_csv(WORK / ('30_gold_blend_' + name + '.csv'), index_label='StudyInstanceUID')
gold.to_csv(WORK / '30_gold_truth.csv')
rng = np.random.default_rng(301016)
deltas = {name: [] for name in blends if name != 'r6_tta_current'}
for _ in range(1500):
    ii = rng.integers(0, len(y), len(y))
    yy = y[ii]
    if not np.all((yy.sum(0) > 0) & (yy.sum(0) < len(ii))):
        continue
    base_auc = roc_auc_score(yy, base.values[ii], average='macro')
    for name in deltas:
        deltas[name].append(float(roc_auc_score(yy, blends[name].values[ii], average='macro')-base_auc))
result = {'experiment': '30_R6_TTA_error_and_stacker_audit', 'status': 'COMPLETE',
          'gold_is_independent_validation': False, 'gold_studies': len(y),
          'reference_anchor': '29B: R6 TTA 50% + Raptor384 50%; NOT full 29A with Raptor224',
          'submission_created': False, 'training_started': False,
          'visible_reproduction_max_abs_error': visible_error,
          'fluid_missing_count': int(pd.to_numeric(fluid.eff_sag_p99_open, errors='coerce').isna().sum()),
          'fluid_status_counts': fluid.status.fillna('MISSING').value_counts().to_dict(),
          'macro_auc': {name: float(roc_auc_score(y, frame.values, average='macro')) for name,frame in blends.items()},
          'paired_resampling': {name: {'draws': len(values), 'interval_95pct': np.quantile(values,[.025,.975]).tolist() if values else None}
                                for name, values in deltas.items()},
          'elapsed_seconds': round(time.time()-NB_T0, 2),
          'warning': '58 gold studies were reused for upstream development/selection. Do not fit weights or stackers here, or treat these exploratory intervals as independent CV. Confirm any chosen fixed change by a prospective submit.'}
(WORK / '30_diagnostic_report.json').write_text(json.dumps(result, indent=2))
assert not (WORK / 'submission.csv').exists(), 'Diagnostic must not create root submission.csv'
display(metrics.round(5))
print(json.dumps(result, indent=2))
'''

intro = '''# 30 — R6 TTA error decomposition and stacker audit

DIAGNOSTIC ONLY: no competition submission and no training. Keep 29A/B (both Public 0.957) unchanged.

First reproduce the exact R6 TTA outputs saved in results 29B on the three visible studies (tolerance 0.001). Then score the 58 gold studies once, saving the public worker's existing fluid/state diagnostic tables. Compare the current 29B rank blend with (1) raw Effusion before its fluid stacker, (2) raw Baker's before its state stacker, and (3) one fixed R6/R7-plain diversity hypothesis. No coefficients or weights are fitted. This does NOT evaluate Raptor224 or reproduce full 29A on gold.

The 58 gold labels have been reused upstream; this is error exploration, not independent validation. Missing fluid features are supported by the published stacker, not automatically a bug. A gold improvement does not guarantee Public or Private improvement. A three-hour timeout is a stop limit, not a runtime estimate.

Attach exactly: competition including train_series; **heliosli/rsna-knee-r6-frozen-fe055406 VERSION 1**; heliosli/rsna-orthofoundation-fold0-epoch7-20261004. No Raptor weights, output ZIPs or training cache required: small reference predictions are embedded. T4x2, Internet Off, Save & Run All. Metadata pins version ID 20611930 but Kaggle may retain old draft attachments; input identity is checked before model loading.

Send the entire Output ZIP, especially 30_diagnostic_report.json, 30_per_target_auc.csv and 30_r6_tta_gold_worker*_fluid_stacker.csv/state_stacker.csv. Do not submit this notebook. Credits and worker sources are retained below.
'''
credits_nb = json.loads((ROOT / '25A_Raptor_R7_Public0956_Reproduction.ipynb').read_text(encoding='utf-8'))
intro += '\nR6 checkpoint bundle and anatomical mirror implementation: Heliosli. Public-source reproduction credit: supplied one-click notebook and its authors.\n\n'
intro += ''.join(credits_nb['cells'][1]['source'])
def cell(text, md=False):
    if not md:
        compile(text, '<30 notebook cell>', 'exec')
    d = {'cell_type': 'markdown' if md else 'code', 'metadata': {}, 'source': text.splitlines(keepends=True)}
    if not md:
        d.update(execution_count=None, outputs=[])
    return d
notebook = copy.deepcopy(nb29)
notebook['cells'] = [cell(intro, True)] + [cell(t) for t in (pre, setup, run, gold, analysis)]
notebook['metadata']['kaggle']['dataSources'] = [d for d in notebook['metadata']['kaggle']['dataSources'] if d['sourceId'] != 20297367]
# No inference-worker change: diagnostics are already emitted by the pinned worker.
def sources_dict(text):
    return next(ast.literal_eval(n.value) for n in ast.parse(text).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'SOURCES' for t in n.targets))
assert sources_dict(setup) == sources_dict(next(''.join(c['source']) for c in nb29['cells'] if 'SOURCES =' in ''.join(c['source'])))
path = ROOT / '30_R6_TTA_Error_Stacker_Gold_Diagnostic.ipynb'
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps({'notebook':path.name,'code_compiles':True,'29_changed_cells':sum(r['changed_cells'] for r in differences),'no_submission':True},indent=2))
