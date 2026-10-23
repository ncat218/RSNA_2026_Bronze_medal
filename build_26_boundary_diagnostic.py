"""Build a paired diagnostic from the verified 25A source; no training or submission."""
from __future__ import annotations

import ast
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
BASE = HERE / '25A_Raptor_R7_Public0956_Reproduction.ipynb'
nb = json.loads(BASE.read_text(encoding='utf-8'))


def source(i):
    return ''.join(nb['cells'][i]['source'])


def cell(text, kind='code'):
    result = {'cell_type': kind, 'metadata': {}, 'source': text.splitlines(keepends=True)}
    if kind == 'code':
        checked = text.split('\n', 1)[1] if text.startswith('%%writefile ') else text
        compile(checked, '<26 notebook cell>', 'exec')
        result.update(execution_count=None, outputs=[])
    return result


with zipfile.ZipFile(HERE / 'results 25A.zip') as z:
    reference = list(csv.DictReader(io.StringIO(z.read('raptor_raw.csv').decode())))
reference = {r['StudyInstanceUID']: {k: float(v) for k, v in r.items() if k != 'StudyInstanceUID'}
             for r in reference}

KEEP_FUNCTION = '''def window_keep_indices(mask, centers, sizes):
    """Keep original centre positions; never manufacture or resample extra windows."""
    slot = []
    for slot_id, size in enumerate(sizes):
        slot.extend([slot_id] * int(size))
    if len(slot) != len(mask):
        raise ValueError('slot layout and mask differ')
    same_series, same_series_valid = [], []
    for j, center in enumerate(centers):
        c = int(center)
        if c < 1 or c + 1 >= len(mask):
            raise ValueError('window outside volume')
        if slot[c - 1] == slot[c] == slot[c + 1]:
            same_series.append(j)
            if all(int(mask[t]) > 0 for t in (c - 1, c, c + 1)):
                same_series_valid.append(j)
    return {'baseline': list(range(len(centers))),
            'same_series': same_series, 'same_series_valid': same_series_valid}
'''

DIAG_MAIN = r'''
# 26: paired heads on the SAME baseline features. Backbone, crop, normalization,
# mirror, source window positions and checkpoint remain the 25A implementation.
DIAG_REFERENCE = __REFERENCE__
__KEEP_FUNCTION__

def diagnose_study(model, vol, mask, device):
    centers = _eval_centers(mask, len(mask), K_EVAL)
    keep = window_keep_indices(mask, centers, SLOT_SIZES)
    if not keep['same_series_valid']:
        raise RuntimeError('no complete within-series window')
    probs = {name: [] for name in keep}
    removed_attention = {name: [] for name in keep if name != 'baseline'}
    with torch.inference_mode():
        for mirror in (False, True):
            windows = gpu_windows(vol, mask, K_EVAL, INPUT_RES, device, mirror=mirror)
            with torch.autocast('cuda', dtype=torch.float16):
                features = model.encode(windows.unsqueeze(0))
                # Same head call as model.forward in 25A.
                for name, indices in keep.items():
                    selected = features if name == 'baseline' else features[:, indices, :]
                    logits = model.head(selected).float()
                    if not torch.isfinite(logits).all():
                        raise RuntimeError('nonfinite logits: ' + name)
                    probs[name].append(torch.sigmoid(logits)[0].cpu().numpy())
                h = model.norm(features)
                attention = torch.softmax(model.att(h), dim=1)
                for name in removed_attention:
                    excluded = sorted(set(keep['baseline']) - set(keep[name]))
                    mass = attention[:, excluded, :].sum(dim=1)[0].float().cpu().numpy()
                    removed_attention[name].append(mass)
            del windows, features, h, attention
    return (
        {name: 0.5 * (values[0] + values[1]) for name, values in probs.items()},
        {name: 0.5 * (values[0] + values[1]) for name, values in removed_attention.items()},
        {name: len(indices) for name, indices in keep.items()},
    )


def main():
    import pandas as pd
    t0 = time.monotonic()
    work = Path('/kaggle/working')
    root = Path(os.environ['RSNA26_GOLD_ROOT'])
    competition = Path(os.environ['RSNA26_COMPETITION_ROOT'])
    receipt = {'experiment': '26_Raptor_series_boundary_paired_diagnostic',
               'status': 'RUNNING', 'training_started': False,
               'checkpoint_sha256': '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef',
               'source_25a_sha256': '__SOURCE_SHA__',
               'gold_is_independent_validation': False,
               'policies': ['baseline', 'same_series', 'same_series_valid'],
               'backbone_passes_per_study': 2, 'errors': []}
    def save_receipt():
        receipt['elapsed_seconds'] = round(time.monotonic() - t0, 2)
        (work / '26_raptor_receipt.json').write_text(json.dumps(receipt, indent=2))
    model = None
    try:
        torch.set_num_threads(2)
        model, _ = load_model(str(_weights), ARMS[0]['arch'], ARMS[0]['res'], 'cuda:0')
        builder = _load_fastread().make_builder(FASTREAD_VARIANT, IMG, SLOTS,
                                               hdr_threads=FR_HDR, px_threads=FR_PX)
        # Reproduce saved visible probabilities before interpreting gold differences.
        visible = pd.read_csv(competition / 'test.csv', dtype={'StudyInstanceUID': str})
        if set(visible.StudyInstanceUID) != set(DIAG_REFERENCE):
            raise RuntimeError('Diagnostic requires the same three visible studies; do not submit it')
        vser = pd.read_csv(competition / 'test_series.csv',
                           dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
        vrecords = {u: g.to_dict('records') for u, g in vser.groupby('StudyInstanceUID')}
        smoke = []
        for uid in visible.StudyInstanceUID:
            vol, mask = builder(uid, vrecords, str(competition / 'test_series'))
            p, _, counts = diagnose_study(model, vol, mask, 'cuda:0')
            expected = np.array([DIAG_REFERENCE[uid][label] for label in LAB], np.float32)
            err = float(np.max(np.abs(p['baseline'] - expected)))
            smoke.append({'StudyInstanceUID': uid, 'max_abs_probability_error': err,
                          'window_counts': counts})
            del vol, mask
        receipt['visible_reproduction'] = smoke
        receipt['visible_max_abs_error'] = max(row['max_abs_probability_error'] for row in smoke)
        save_receipt()
        if receipt['visible_max_abs_error'] > 1e-4:
            raise RuntimeError('25A baseline reproduction differs by > 1e-4; stop before gold audit')
        ids = pd.read_csv(root / 'test.csv', dtype=str).StudyInstanceUID.tolist()
        series = pd.read_csv(root / 'test_series.csv',
                            dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
        records = {u: g.to_dict('records') for u, g in series.groupby('StudyInstanceUID')}
        predictions = {name: [] for name in receipt['policies']}
        attention_rows, count_rows = [], []
        for index, uid in enumerate(ids, 1):
            if time.monotonic() - t0 > 2 * 3600:
                raise TimeoutError('Raptor diagnostic exceeded two hours')
            vol, mask = builder(uid, records, str(root / 'test_series'))
            p, mass, counts = diagnose_study(model, vol, mask, 'cuda:0')
            for name, values in p.items():
                predictions[name].append({'StudyInstanceUID': uid, **dict(zip(LAB, values))})
            for name, values in mass.items():
                attention_rows.append({'StudyInstanceUID': uid, 'policy': name,
                                       **dict(zip(LAB, values))})
            count_rows.append({'StudyInstanceUID': uid, 'valid_slices': int(mask.sum()), **counts})
            for name, rows in predictions.items():
                pd.DataFrame(rows).to_csv(work / ('26_raptor_' + name + '.csv'), index=False)
            pd.DataFrame(attention_rows).to_csv(work / '26_removed_attention.csv', index=False)
            pd.DataFrame(count_rows).to_csv(work / '26_window_counts.csv', index=False)
            receipt['completed_studies'] = index
            save_receipt()
            print('PAIRED', index, '/', len(ids), counts, 'seconds', round(time.monotonic() - t0), flush=True)
            del vol, mask, p, mass
        receipt.update(status='COMPLETE', requested_studies=len(ids),
                       cuda_max_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3))
        save_receipt()
    except Exception as exc:
        receipt['status'] = 'FAILED'
        receipt['errors'].append(repr(exc))
        save_receipt()
        raise
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
'''

original_raptor = source(3).split('\n', 1)[1]
prefix = original_raptor[:original_raptor.index('\ndef main():')]
source_sha = hashlib.sha256(original_raptor.encode()).hexdigest()
diag_script = prefix + DIAG_MAIN.replace('__REFERENCE__', repr(reference)).replace(
    '__KEEP_FUNCTION__', KEEP_FUNCTION).replace('__SOURCE_SHA__', source_sha)

PREFLIGHT = source(2).replace("'25_preflight.json'", "'26_preflight.json'") + r'''

import os, time
import numpy as np
import pandas as pd
NB_T0 = time.time()
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
          'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
train = pd.read_csv(competition / 'train.csv', dtype={'StudyInstanceUID': str})
gold = train.loc[train[LABELS].notna().all(axis=1), ['StudyInstanceUID'] + LABELS].copy()
if len(gold) != 58 or not gold.StudyInstanceUID.is_unique:
    raise RuntimeError('Expected exactly 58 unique gold studies')
if not np.isin(gold[LABELS].values, [0, 1]).all():
    raise RuntimeError('Gold labels must be binary')
if not (competition / 'train_series').is_dir():
    raise RuntimeError('Attach the complete competition training DICOMs')
GOLD_ROOT = Path('/tmp/rsna26_gold_proxy')  # never export a DICOM symlink in notebook Output
GOLD_ROOT.mkdir(exist_ok=True)
gold[['StudyInstanceUID']].to_csv(GOLD_ROOT / 'test.csv', index=False)
series = pd.read_csv(competition / 'train_series.csv',
                     dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
series = series.loc[series.StudyInstanceUID.isin(set(gold.StudyInstanceUID))]
if set(series.StudyInstanceUID) != set(gold.StudyInstanceUID):
    raise RuntimeError('Some gold studies have no series metadata')
series.to_csv(GOLD_ROOT / 'test_series.csv', index=False)
sample = gold[['StudyInstanceUID']].copy()
sample[LABELS] = 0.5
sample.to_csv(GOLD_ROOT / 'sample_submission.csv', index=False)
link = GOLD_ROOT / 'test_series'
if not link.exists():
    link.symlink_to(competition / 'train_series', target_is_directory=True)
if link.resolve() != (competition / 'train_series').resolve():
    raise RuntimeError('Gold proxy points to an unexpected DICOM directory')
gold.to_csv(WORK / '26_gold_truth.csv', index=False)
os.environ['RSNA26_GOLD_ROOT'] = str(GOLD_ROOT)
os.environ['RSNA26_COMPETITION_ROOT'] = str(competition)
preflight.update(gold_studies=len(gold), gold_series=len(series),
                 proxy=str(GOLD_ROOT), training_started=False,
                 gold_is_independent_validation=False,
                 r7_manifest_gold_in_training=r7_manifest.get('gold_in_training'),
                 r7_manifest_official_labels_read_by_trainer=r7_manifest.get('official_labels_read_by_trainer'))
(WORK / '26_preflight.json').write_text(json.dumps(preflight, indent=2))
print('26_GOLD_INPUTS_READY', len(gold), 'studies;', len(series), 'series', flush=True)
'''

RUN_RAPTOR = r'''import subprocess, sys, time
rc = subprocess.run([sys.executable, '-u', '/tmp/26_raptor_diagnostic.py'], timeout=2.1 * 3600).returncode
if rc != 0:
    raise RuntimeError('Paired Raptor diagnostic failed; inspect 26_raptor_receipt.json')
paired_receipt = json.loads((WORK / '26_raptor_receipt.json').read_text())
assert paired_receipt['status'] == 'COMPLETE' and paired_receipt['completed_studies'] == 58
RAPTOR_SECONDS = paired_receipt['elapsed_seconds']
'''

r7_setup = source(5).replace('DEADLINE_HOURS = 8.3', 'DEADLINE_HOURS = 3.0').replace(
    "DATA = find_mount(COMPETITION, 'test.csv')", "DATA = GOLD_ROOT  # 26: UID-only proxy of gold training images")
assert r7_setup != source(5)

ANALYSIS = r'''# Fixed comparisons; no weight fitting and no training.
from sklearn.metrics import roc_auc_score
gold = pd.read_csv(WORK / '26_gold_truth.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
uids = gold.index.tolist()
labels = list(gold.columns)
y = gold.to_numpy(dtype=np.int64)
if receipt.get('fallback_studies', 0) or receipt.get('scored') != len(uids):
    raise RuntimeError('R7 did not score all 58 gold studies; do not compare mean-filled fallback predictions')

def read_aligned(name):
    df = pd.read_csv(WORK / name, dtype={'StudyInstanceUID': str})
    if not df.StudyInstanceUID.is_unique or set(df.StudyInstanceUID) != set(uids):
        raise RuntimeError('UID mismatch in ' + name)
    result = df.set_index('StudyInstanceUID').loc[uids, labels].astype(float)
    if not np.isfinite(result.values).all() or not ((result.values >= 0) & (result.values <= 1)).all():
        raise RuntimeError('Invalid probabilities in ' + name)
    return result

r7_preds = read_aligned('ours_sub.csv')
r7_preds.to_csv(WORK / '26_r7_gold.csv')
raptor_preds = {policy: read_aligned('26_raptor_' + policy + '.csv')
                for policy in ('baseline', 'same_series', 'same_series_valid')}
blends = {policy: 0.5 * frame.rank(method='average', pct=True) +
                  0.5 * r7_preds.rank(method='average', pct=True)
          for policy, frame in raptor_preds.items()}
all_preds = {'R7': r7_preds, **{'Raptor_' + k: v for k, v in raptor_preds.items()},
             **{'Blend_' + k: v for k, v in blends.items()}}
for policy, frame in blends.items():
    frame.to_csv(WORK / ('26_gold_blend_' + policy + '.csv'))

metric_rows = []
for label_index, label in enumerate(labels):
    row = {'target': label, 'positives': int(y[:, label_index].sum()),
           'negatives': int((1 - y[:, label_index]).sum())}
    row.update({name: float(roc_auc_score(y[:, label_index], frame[label]))
                for name, frame in all_preds.items()})
    row['delta_blend_same_series'] = row['Blend_same_series'] - row['Blend_baseline']
    row['delta_blend_same_series_valid'] = row['Blend_same_series_valid'] - row['Blend_baseline']
    metric_rows.append(row)
metrics = pd.DataFrame(metric_rows)
metrics.to_csv(WORK / '26_per_target_auc.csv', index=False)
macro = {name: float(metrics[name].mean()) for name in all_preds}

# Resample study rows together, preserving paired predictions across conditions.
# These intervals describe this reused development set, not unseen-test uncertainty.
rng = np.random.default_rng(261014)
boot = {policy: [] for policy in ('same_series', 'same_series_valid')}
attempts = 0
while min(map(len, boot.values())) < 2000 and attempts < 3000:
    attempts += 1
    idx = rng.integers(0, len(uids), size=len(uids))
    yy = y[idx]
    if not np.all((yy.sum(axis=0) > 0) & (yy.sum(axis=0) < len(idx))):
        continue
    b = roc_auc_score(yy, blends['baseline'].to_numpy()[idx], average='macro')
    for policy in boot:
        a = roc_auc_score(yy, blends[policy].to_numpy()[idx], average='macro')
        boot[policy].append(float(a - b))
intervals = {policy: {'paired_draws': len(values),
                       'delta_macro': macro['Blend_' + policy] - macro['Blend_baseline'],
                       'resampling_95pct': np.quantile(values, [0.025, 0.975]).tolist()}
             for policy, values in boot.items()}
attention = pd.read_csv(WORK / '26_removed_attention.csv')
attention_summary = attention.groupby('policy')[labels].agg(['mean', 'max'])
attention_summary.to_csv(WORK / '26_attention_summary.csv')
counts = pd.read_csv(WORK / '26_window_counts.csv')
report = {
    'experiment': '26_Raptor_series_boundary_paired_diagnostic', 'status': 'COMPLETE',
    'gold_studies': len(uids), 'gold_is_independent_validation': False,
    'training_started': False, 'submission_created': False,
    'fixed_rank_weights': {'Raptor': 0.5, 'R7': 0.5},
    'macro_auc': macro, 'paired_resampling': intervals,
    'window_counts_mean': counts[['baseline', 'same_series', 'same_series_valid']].mean().to_dict(),
    'visible_reproduction_max_abs_error': paired_receipt['visible_max_abs_error'],
    'r7_fallback_studies': receipt.get('fallback_studies'),
    'elapsed_seconds': round(time.time() - NB_T0, 2),
    'interpretation': [
        'Raptor all-data weights and prior checkpoint selection expose the 58 gold studies; this is a diagnostic, not independent CV.',
        'Pooling is compared on identical backbone features. Crop 320, anatomical mirror, model hashes and R7 are held fixed.',
        'A gold improvement may reject an obviously harmful idea but cannot establish a Public or Private gain.',
        'Only a later prospective Kaggle probe can test the new window policy on hidden images.'
    ]
}
(WORK / '26_diagnostic_report.json').write_text(json.dumps(report, indent=2))
assert not (WORK / 'submission.csv').exists(), 'Diagnostic must not leave a root submission.csv'
display(metrics.round(5))
print(json.dumps(report, indent=2))
'''

INTRO = '''# 26 — Raptor series-boundary diagnostic on 58 gold studies

**Diagnostic only. Use Save & Run All; do not submit this notebook to the competition.**

25A and 25B both scored 0.956 Public. The base already uses the 320-pixel native crop and anatomical mirror TTA. This experiment asks whether pooling artificial 3-slice windows across the five series boundaries harms that base.

The frozen Raptor backbone is evaluated once per original/mirrored view. The exact same features are pooled three ways: (1) original 94-window policy, (2) omit windows crossing a selected-series boundary, (3) also omit windows containing zero/missing slices. With all 96 slices present there are 8 crossing windows and 86 within-series windows. R7 is run once and all three results use the fixed 50/50 rank blend.

First, the original policy must reproduce the three visible probabilities saved in results 25A within 1e-4. Then the notebook scores all 58 gold studies, records per-target AUC, removed attention mass and paired resampling intervals. Raptor's all-data training and prior gold-based model selection mean these are development diagnostics, **not independent CV**. No model is trained, weights are not tuned, and no root submission.csv is produced.

**Attach exactly the same inputs as 25A:** competition data (including train_series); `nartaa/rsna-knee-publication-swa-weights-20261007`; `heliosli/rsna-knee-compact-r7cnxfsss`; `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`. **GPU T4×2, Internet Off.** No 05A/24C cache, 224 checkpoint or saved output is needed. Runtime is not measured yet; the notebook caps the combined run at three hours, with the Raptor subprocess limited to about two hours.

Send the output ZIP, especially `26_diagnostic_report.json`, `26_per_target_auc.csv`, `26_attention_summary.csv`, `26_window_counts.csv` and `26_raptor_receipt.json`. Sources: the credited public source embedded in 25A; [Raptor author's measured recipe](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792); [MRNet's series-wise knee MRI processing](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699). MRNet motivates respecting series identity; it does not prove this pooling change improves Raptor.
'''

out = copy.deepcopy(nb)
out['cells'] = [cell(INTRO + '\n\n' + source(1), 'markdown'), cell(PREFLIGHT),
                cell('%%writefile /tmp/26_raptor_diagnostic.py\n' + diag_script),
                cell(RUN_RAPTOR), cell(r7_setup), cell(source(6)), cell(ANALYSIS)]
out['metadata'].setdefault('kaggle', {})['isInternetEnabled'] = False
target = HERE / '26_Raptor_SeriesBoundary_Gold_Diagnostic.ipynb'
target.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')

# Verify the concrete change and its degenerate cases using the same selection function.
scope = {}
exec(KEEP_FUNCTION, scope)
keep = scope['window_keep_indices']
sizes = [26, 22, 18, 12, 18]
centers = list(range(1, 95))
full = keep([1] * 96, centers, sizes)
assert [len(full[k]) for k in ('baseline', 'same_series', 'same_series_valid')] == [94, 86, 86]
crossing = [centers[i] for i in set(full['baseline']) - set(full['same_series'])]
assert sorted(crossing) == [25, 26, 47, 48, 65, 66, 77, 78]
missing = [1] * 96
missing[26:48] = [0] * 22
filtered = keep(missing, centers, sizes)
assert len(filtered['same_series']) == 86 and len(filtered['same_series_valid']) == 66
hole = [1] * 96
hole[10] = 0
assert len(keep(hole, centers, sizes)['same_series_valid']) == 83
assert not keep([0] * 96, centers, sizes)['same_series_valid']
# Verify source code integrity of model and decoder prefix, and unchanged R7 execution.
assert diag_script[:len(prefix)] == prefix
assert ''.join(out['cells'][5]['source']) == source(6)
for i, item in enumerate(out['cells']):
    if item['cell_type'] == 'code':
        text = ''.join(item['source'])
        if text.startswith('%%writefile '):
            text = text.split('\n', 1)[1]
        ast.parse(text, filename=f'{target.name}:cell{i}')
print(json.dumps({'notebook': str(target), 'cells': len(out['cells']),
                  'all_code_cells_compile': True, 'boundary_selection_tests': 'PASS',
                  'full_96_crossing_windows': sorted(crossing), 'full_96_safe_windows': 86,
                  'original_model_decoder_prefix_unchanged': True,
                  'r7_controller_cell_unchanged': True}, indent=2))
