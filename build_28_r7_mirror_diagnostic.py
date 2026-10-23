"""Package the public R7 worker's existing anatomical mirror TTA as a paired gold diagnostic."""
from __future__ import annotations
import ast
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parent
base = json.loads((ROOT / '25A_Raptor_R7_Public0956_Reproduction.ipynb').read_text(encoding='utf-8'))
src = lambda i: ''.join(base['cells'][i]['source'])

def read_csv(z, name):
    return list(csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig'))))

with zipfile.ZipFile(ROOT / 'results 26.zip') as z:
    truth = read_csv(z, '26_gold_truth.csv')
    labels = list(truth[0])[1:]
    uids = [r['StudyInstanceUID'] for r in truth]
    ref = {'labels': labels, 'uids': uids,
           'truth': [[int(float(r[c])) for c in labels] for r in truth]}
    for key, name in [('r7', '26_r7_gold.csv'), ('raptor', '26_raptor_baseline.csv')]:
        rows = {r['StudyInstanceUID']: r for r in read_csv(z, name)}
        ref[key] = [[float(rows[u][c]) for c in labels] for u in uids]
with zipfile.ZipFile(ROOT / 'results 27.zip') as z:
    rows = read_csv(z, 'ours_sub.csv')
    receipt = json.loads(z.read('submission_receipt.json'))
    assert receipt['fallback_studies'] == 0 and all(not w.get('flip_tta') for w in receipt['workers'])
    ref['visible_uids'] = [r['StudyInstanceUID'] for r in rows]
    ref['visible_r7'] = [[float(r[c]) for c in labels] for r in rows]

PREFLIGHT = src(2)[:src(2).index("competition = require_mount(")]
PREFLIGHT += 'REF = ' + repr(ref) + '\n'
PREFLIGHT += r'''
import os, time
import numpy as np
import pandas as pd
NB_T0 = time.time()
competition = require_mount('rsna-knee-abnormality-detection', 'train.csv')
r7 = require_mount('rsna-knee-compact-r7cnxfsss', 'manifest.json')
code = require_mount('rsna-orthofoundation-fold0-epoch7-20261004', 'receipt.json')
manifest_input = json.loads((r7 / 'manifest.json').read_text())
if manifest_input.get('bundle_id') != '8c7b666e27dc0e8f':
    raise RuntimeError('R7 bundle differs from the verified 25A/26/27 runs')
if manifest_input.get('flip_tta'):
    raise RuntimeError('This bundle already enables mirror TTA; the proposed change is not new')
if torch.cuda.device_count() < 2:
    raise RuntimeError('Use GPU T4x2 for this diagnostic')
labels = REF['labels']
train = pd.read_csv(competition / 'train.csv', dtype={'StudyInstanceUID': str})
gold = train.loc[train[labels].notna().all(axis=1), ['StudyInstanceUID'] + labels].copy()
if not gold.StudyInstanceUID.is_unique or set(gold.StudyInstanceUID) != set(REF['uids']):
    raise RuntimeError('Gold UID set differs from results 26')
gold = gold.set_index('StudyInstanceUID').loc[REF['uids']]
if not np.array_equal(gold.to_numpy(), np.asarray(REF['truth'])):
    raise RuntimeError('Gold annotation identity differs from results 26')
visible_ids = pd.read_csv(competition / 'test.csv', dtype=str).StudyInstanceUID.tolist()
if set(visible_ids) != set(REF['visible_uids']):
    raise RuntimeError('This is a diagnostic on visible test and train gold; do not competition-submit it')
if not (competition / 'train_series').is_dir():
    raise RuntimeError('Full competition training MRI data is required')
preflight = {'experiment': '28_R7_anatomical_mirror_gold_diagnostic', 'status': 'INPUTS_READY',
             'r7_bundle_id': manifest_input['bundle_id'], 'gold_studies': len(gold),
             'base_flip_tta': bool(manifest_input.get('flip_tta')),
             'gold_is_independent_validation': False, 'training_started': False,
             'reference': 'Embedded exact Raptor/R7 outputs from results 26 and visible R7 from results 27',
             'r7_gold_in_training': manifest_input.get('gold_in_training')}
(WORK / '28_preflight.json').write_text(json.dumps(preflight, indent=2))
print('28_INPUTS_READY', json.dumps(preflight), flush=True)
'''

setup = src(5).replace('DEADLINE_HOURS = 8.3', 'DEADLINE_HOURS = 3.0').replace(
    "DATA = find_mount(COMPETITION, 'test.csv')", 'DATA = competition')
setup += r'''

# Test the pinned worker's permutation contract before decoding any images.
import ast
worker_tree = ast.parse(SOURCES['compact_infer.py'])
permutation_function = next(node for node in worker_tree.body
                            if isinstance(node, ast.FunctionDef) and node.name == 'mirror_perm')
permutation_scope = {'MIRROR_PAIRS': (('Medial Meniscus', 'Lateral Meniscus'), ('Medial OA', 'Lateral OA'))}
exec(compile(ast.Module(body=[permutation_function], type_ignores=[]), '<pinned R7 mirror_perm>', 'exec'), permutation_scope)
state_spec_for_check = (json.loads((BUNDLE / manifest['state_spec']).read_text())
                        if manifest.get('state_spec') else None)
label_permutation, _ = permutation_scope['mirror_perm'](REF['labels'], state_spec_for_check)
if '--flip-tta' in WORKER_ARGS:
    raise RuntimeError('Plain reproduction unexpectedly enables mirror TTA')
preflight['mirror_label_permutation'] = label_permutation
preflight['state_spec_mirror_compatible'] = True
(WORK / '28_preflight.json').write_text(json.dumps(preflight, indent=2))
'''

RUN = r'''
def run_r7_diagnostic(data_root, tag, worker_arguments):
    output = TMP / ('28_' + tag)
    deadline = NB_T0 + 3 * 3600
    remaining = deadline - time.time()
    if remaining < 120:
        raise RuntimeError('Diagnostic deadline exhausted before ' + tag)
    env = dict(os.environ, PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''),
               PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable, '-u', str(SRC / 'probe_dual.py'), '--data', str(data_root), '--split', 'test',
               '--code', str(CODE), '--bundle', str(BUNDLE), '--arm', ARM, '--output', str(output),
               '--amp', 'auto', '--loader-workers', '2', '--deadline', str(deadline),
               '--worker', WORKER, '--worker-args', json.dumps(worker_arguments),
               '--max-fallback-fraction', '0.0']
    rc = subprocess.run(command, env=env, timeout=remaining).returncode
    for log in sorted(output.glob('worker*/log.txt')):
        shutil.copyfile(log, WORK / ('28_' + tag + '_' + log.parent.name + '_log.txt'))
    receipt_path = output / 'submission_receipt.json'
    if receipt_path.is_file():
        shutil.copyfile(receipt_path, WORK / ('28_' + tag + '_receipt.json'))
    if rc != 0:
        raise RuntimeError('R7 stage failed: ' + tag + '; inspect its logs and receipt')
    stage_receipt = json.loads(receipt_path.read_text())
    ids = pd.read_csv(data_root / 'test.csv', dtype=str).StudyInstanceUID.tolist()
    if stage_receipt.get('fallback_studies') or stage_receipt.get('scored') != len(ids):
        raise RuntimeError('Incomplete R7 stage: ' + tag)
    frame = pd.read_csv(output / 'submission.csv', dtype={'StudyInstanceUID': str})
    if not frame.StudyInstanceUID.is_unique or set(frame.StudyInstanceUID) != set(ids):
        raise RuntimeError('Prediction UID mismatch: ' + tag)
    frame = frame.set_index('StudyInstanceUID').loc[ids, REF['labels']].astype(float)
    if not np.isfinite(frame.values).all() or not ((frame.values >= 0) & (frame.values <= 1)).all():
        raise RuntimeError('Invalid probabilities: ' + tag)
    frame.to_csv(WORK / ('28_' + tag + '.csv'))
    return frame, stage_receipt

plain_visible, plain_receipt = run_r7_diagnostic(competition, 'visible_plain', WORKER_ARGS)
expected_visible = pd.DataFrame(REF['visible_r7'], index=REF['visible_uids'], columns=REF['labels'])
visible_error = float(np.max(np.abs(plain_visible.loc[REF['visible_uids']].values - expected_visible.values)))
preflight['visible_plain_max_abs_error'] = visible_error
preflight['status'] = 'VISIBLE_REPRODUCTION_PASS' if visible_error <= 1e-3 else 'VISIBLE_REPRODUCTION_FAILED'
(WORK / '28_preflight.json').write_text(json.dumps(preflight, indent=2))
if visible_error > 1e-3:
    raise RuntimeError('R7 plain probabilities do not reproduce the base within 0.001; stop before gold')
print('VISIBLE_PLAIN_REPRODUCED max_abs_error', visible_error, flush=True)
'''

GOLD = r'''
GOLD_ROOT = Path('/tmp/rsna28_gold_proxy')
GOLD_ROOT.mkdir(exist_ok=True)
pd.DataFrame({'StudyInstanceUID': REF['uids']}).to_csv(GOLD_ROOT / 'test.csv', index=False)
series = pd.read_csv(competition / 'train_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
series = series.loc[series.StudyInstanceUID.isin(set(REF['uids']))]
if set(series.StudyInstanceUID) != set(REF['uids']):
    raise RuntimeError('A gold study has no series metadata')
series.to_csv(GOLD_ROOT / 'test_series.csv', index=False)
sample = pd.DataFrame({'StudyInstanceUID': REF['uids']})
sample[REF['labels']] = 0.5
sample.to_csv(GOLD_ROOT / 'sample_submission.csv', index=False)
link = GOLD_ROOT / 'test_series'
if not link.exists():
    link.symlink_to(competition / 'train_series', target_is_directory=True)
if link.resolve() != (competition / 'train_series').resolve():
    raise RuntimeError('Diagnostic DICOM proxy points to unexpected data')
tta_arguments = WORKER_ARGS + ['--flip-tta']
tta_gold, tta_receipt = run_r7_diagnostic(GOLD_ROOT, 'r7_tta_gold', tta_arguments)
if not tta_receipt.get('workers') or not all(worker.get('flip_tta') for worker in tta_receipt['workers']):
    raise RuntimeError('A worker did not confirm active anatomical mirror TTA')
print('GOLD_TTA_COMPLETE', len(tta_gold), 'studies', flush=True)
'''

ANALYSIS = r'''
from sklearn.metrics import roc_auc_score
uids, labels = REF['uids'], REF['labels']
y = np.asarray(REF['truth'], dtype=np.int64)
r7_plain = pd.DataFrame(REF['r7'], index=uids, columns=labels)
raptor = pd.DataFrame(REF['raptor'], index=uids, columns=labels)
r7_tta = tta_gold.loc[uids, labels]
rank_raptor = raptor.rank(method='average', pct=True)
base_blend = (0.5 * rank_raptor + 0.5 * r7_plain.rank(method='average', pct=True)).astype(np.float32)
tta_blend = (0.5 * rank_raptor + 0.5 * r7_tta.rank(method='average', pct=True)).astype(np.float32)
base_blend.to_csv(WORK / '28_gold_blend_baseline.csv', index_label='StudyInstanceUID')
tta_blend.to_csv(WORK / '28_gold_blend_tta.csv', index_label='StudyInstanceUID')
r7_plain.to_csv(WORK / '28_r7_plain_gold_reference.csv', index_label='StudyInstanceUID')
gold.to_csv(WORK / '28_gold_truth.csv')
rows = []
for j, label in enumerate(labels):
    baseline_auc = float(roc_auc_score(y[:, j], base_blend[label]))
    candidate_auc = float(roc_auc_score(y[:, j], tta_blend[label]))
    rows.append({'target': label, 'positives': int(y[:, j].sum()),
                 'r7_plain_auc': float(roc_auc_score(y[:, j], r7_plain[label])),
                 'r7_tta_auc': float(roc_auc_score(y[:, j], r7_tta[label])),
                 'blend_baseline_auc': baseline_auc, 'blend_tta_auc': candidate_auc,
                 'delta_blend_auc': candidate_auc - baseline_auc,
                 'r7_plain_tta_spearman': float(r7_plain[label].corr(r7_tta[label], method='spearman'))})
metrics = pd.DataFrame(rows)
metrics.to_csv(WORK / '28_per_target_auc.csv', index=False)
rng = np.random.default_rng(281015)
bootstrap = []
for _ in range(2500):
    ii = rng.integers(0, len(uids), len(uids))
    yy = y[ii]
    if not np.all((yy.sum(0) > 0) & (yy.sum(0) < len(ii))):
        continue
    bootstrap.append(float(roc_auc_score(yy, tta_blend.values[ii], average='macro') -
                           roc_auc_score(yy, base_blend.values[ii], average='macro')))
    if len(bootstrap) == 2000:
        break
report = {
    'experiment': '28_R7_anatomical_mirror_gold_diagnostic', 'status': 'COMPLETE',
    'gold_studies': len(uids), 'training_started': False, 'submission_created': False,
    'gold_is_independent_validation': False,
    'base': '25A 50/50 rank blend; original Raptor windows and mirror retained',
    'candidate': 'Same Raptor; public R7 worker --flip-tta, with trained medial/lateral permutation and plain-only MCL',
    'visible_plain_max_abs_error': visible_error,
    'r7_gold_fallback_studies': tta_receipt['fallback_studies'],
    'baseline_gold_macro_auc': float(metrics.blend_baseline_auc.mean()),
    'candidate_gold_macro_auc': float(metrics.blend_tta_auc.mean()),
    'delta_macro_auc': float(metrics.delta_blend_auc.mean()),
    'paired_draws': len(bootstrap), 'paired_resampling_95pct': np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    'mcl_max_abs_probability_change': float(np.max(np.abs(r7_plain['MCL'] - r7_tta['MCL']))),
    'elapsed_seconds': round(time.time() - NB_T0, 2),
    'interpretation': 'Gold was reused in upstream model development/selection. This diagnoses a serving change; it does not establish independent CV or a Public/Private gain.'
}
(WORK / '28_diagnostic_report.json').write_text(json.dumps(report, indent=2))
assert not (WORK / 'submission.csv').exists(), 'Diagnostic must not create a root submission.csv'
display(metrics.round(5))
print(json.dumps(report, indent=2))
'''

INTRO = '''# 28 — R7 anatomical mirror TTA diagnostic

**Use Save & Run All. This diagnostic does not produce a competition submission.**

27 correctly applied the OA blend and scored 0.956 Public, matching the displayed incumbent. Its workers confirm R7 mirror TTA is inactive. This experiment uses the **existing, pinned public R7 worker's `--flip-tta` implementation**: sagittal slice reversal and position transformation, coronal/axial width reflection, coronal view exchange, and medial/lateral output permutation. MCL uses the plain pass, matching the worker's training contract. This changes the image evidence rather than the OA blend weight.

First reproduce plain R7 probabilities on the three visible studies from results 27. Then score all 58 gold studies with R7 TTA and compare with the stored exact R7/Raptor gold predictions from results 26. No Raptor inference is needed. Both blend conditions use the original 25A 50/50 weights and float32 submission format. Model workers and hashes are unchanged; only the supported TTA argument is added. The state-spec permutation is checked before decoding.

**Attach three inputs:** RSNA competition including train_series; `heliosli/rsna-knee-compact-r7cnxfsss`; `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`. **GPU T4x2, Internet Off.** No saved output attachment, Raptor weights or 05A/24C cache is needed: the small reference tables are embedded. TTA adds an extra R7 image pass; this run's duration is not measured yet. The three-hour limit is a timeout, not an ETA.

Send the Output ZIP, especially `28_diagnostic_report.json`, `28_per_target_auc.csv`, `28_preflight.json` and `28_r7_tta_gold_receipt.json`. Gold has been reused for upstream model development and selection, so these diagnostics are not independent CV. The Raptor author's current [training account](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792) says the 58 gold studies were excluded from gradient training but reused for selection; older project comments interpreting “all-data” as proof of gold-gradient exposure were too strong. The actual selection limitation remains.

'''
INTRO += src(1)

def cell(text, markdown=False):
    if markdown:
        return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}
    compile(text, '<28 generated cell>', 'exec')
    return {'cell_type': 'code', 'execution_count': None, 'outputs': [], 'metadata': {},
            'source': text.splitlines(keepends=True)}

out = copy.deepcopy(base)
out['cells'] = [cell(INTRO, True), cell(PREFLIGHT), cell(setup), cell(RUN), cell(GOLD), cell(ANALYSIS)]
# Keep the competition metadata; attach the two named datasets explicitly.
sources = out['metadata'].get('kaggle', {}).get('dataSources', [])
out['metadata'].setdefault('kaggle', {})['dataSources'] = [item for item in sources if item.get('sourceType') == 'competition']
out['metadata']['kaggle']['isInternetEnabled'] = False
target = ROOT / '28_R7_Anatomical_Mirror_Gold_Diagnostic.ipynb'
target.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')

# The source's mirrored label map is an involution and preserves MCL.
tree = ast.parse(src(5))
sources = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'SOURCES' for t in node.targets))
worker_tree = ast.parse(sources['compact_infer.py'])
fn = next(node for node in worker_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'mirror_perm')
scope = {'MIRROR_PAIRS': (('Medial Meniscus', 'Lateral Meniscus'), ('Medial OA', 'Lateral OA'))}
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<pinned mirror permutation>', 'exec'), scope)
perm, _ = scope['mirror_perm'](labels)
assert all(perm[perm[i]] == i for i in range(len(labels)))
assert perm[labels.index('MCL')] == labels.index('MCL')
assert labels[perm[labels.index('Medial OA')]] == 'Lateral OA'
assert ref['uids'] and len(ref['uids']) == len(set(ref['uids'])) == 58
setup_sources = next(ast.literal_eval(node.value) for node in ast.parse(setup).body if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'SOURCES' for t in node.targets))
assert setup_sources == sources
if '--selftest' in sys.argv:
    import numpy as np
    import pandas as pd
    with tempfile.TemporaryDirectory(prefix='rsna28_analysis_test_', dir=ROOT) as directory:
        test_root = Path(directory)
        assert test_root.resolve().parent == ROOT.resolve(), 'Cleanup must stay within the workspace'
        table = pd.DataFrame(ref['r7'], index=ref['uids'], columns=labels)
        truth_table = pd.DataFrame(ref['truth'], index=ref['uids'], columns=labels)
        truth_table.index.name = 'StudyInstanceUID'
        scope = {'REF': ref, 'WORK': test_root, 'np': np, 'pd': pd, 'json': json,
                 'time': time, 'NB_T0': time.time(), 'gold': truth_table,
                 'tta_gold': table, 'tta_receipt': {'fallback_studies': 0},
                 'visible_error': 0.0, 'display': lambda *a, **kw: None}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(ANALYSIS, '<28 actual analysis cell>', 'exec'), scope)
        tested = json.loads((test_root / '28_diagnostic_report.json').read_text())
        assert tested['delta_macro_auc'] == 0.0
        assert abs(tested['baseline_gold_macro_auc'] - 0.9354010743909692) < 1e-12
        assert tested['candidate_gold_macro_auc'] == tested['baseline_gold_macro_auc']
        assert not (test_root / 'submission.csv').exists()
    print('Actual analysis cell: aligned 58-row identity comparison, float32 baseline and resampling PASS')
print(json.dumps({'notebook': str(target), 'code_cells_compile': True,
                  'public_worker_sources_preserved': True, 'label_permutation_test': 'PASS',
                  'gold_reference_rows': len(uids), 'base_mirror_disabled_confirmed_from_27': True}, indent=2))
