"""Audit 26 and package one prospective OA-group blend probe of the verified 25A."""
from __future__ import annotations

import ast
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parent
OA = ['Medial OA', 'Lateral OA', 'PF OA']
R7_OA_WEIGHT = 0.65


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / (2.0 * len(values))
        for index in order[i:j]:
            result[index] = rank
        i = j
    return result


def auc(y, p):
    positive = sum(y)
    negative = len(y) - positive
    if not positive or not negative:
        raise ValueError('single-class target')
    rr = ranks(p)
    rank_sum = sum(r * len(y) for r, target in zip(rr, y) if target)
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


archive = ROOT / 'results 26.zip'
with zipfile.ZipFile(archive) as z:
    report = json.loads(z.read('26_diagnostic_report.json'))
    raptor_receipt = json.loads(z.read('26_raptor_receipt.json'))
    assert report['status'] == raptor_receipt['status'] == 'COMPLETE'
    assert report['gold_studies'] == raptor_receipt['completed_studies'] == 58
    assert report['r7_fallback_studies'] == 0 and not raptor_receipt['errors']
    assert report['visible_reproduction_max_abs_error'] <= 1e-4
    def table(name):
        rows = list(csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig'))))
        result = {r['StudyInstanceUID']: r for r in rows}
        assert len(result) == len(rows) == 58, name
        return result
    truth = table('26_gold_truth.csv')
    raptor = table('26_raptor_baseline.csv')
    r7 = table('26_r7_gold.csv')
    saved_baseline = table('26_gold_blend_baseline.csv')
    assert set(truth) == set(raptor) == set(r7) == set(saved_baseline)
uids = list(truth)
labels = [c for c in truth[uids[0]] if c != 'StudyInstanceUID']
metrics = []
preds = {}
for label in labels:
    y = [int(float(truth[u][label])) for u in uids]
    rp = ranks([float(raptor[u][label]) for u in uids])
    r7p = ranks([float(r7[u][label]) for u in uids])
    baseline = [0.5 * a + 0.5 * b for a, b in zip(rp, r7p)]
    assert max(abs(p - float(saved_baseline[u][label])) for u, p in zip(uids, baseline)) < 1e-12
    weight = R7_OA_WEIGHT if label in OA else 0.5
    candidate = [(1 - weight) * a + weight * b for a, b in zip(rp, r7p)]
    # Arithmetic order can split exactly tied sums by machine epsilon; match NumPy's float64 order.
    float32 = lambda value: struct.unpack('f', struct.pack('f', value))[0]
    # 25A and 27 write float32 scores. Respect their ties when estimating the probe.
    baseline = [float32(value) for value in baseline]
    candidate = [float32(value) for value in candidate]
    base_auc, candidate_auc = auc(y, baseline), auc(y, candidate)
    preds[label] = (y, baseline, candidate)
    metrics.append({'target': label, 'r7_weight': weight, 'baseline_auc': base_auc,
                    'candidate_auc': candidate_auc, 'delta': candidate_auc - base_auc})
base_macro = sum(row['baseline_auc'] for row in metrics) / len(labels)
candidate_macro = sum(row['candidate_auc'] for row in metrics) / len(labels)
loo = []
for omitted in range(len(uids)):
    deltas = []
    for label in labels:
        y, baseline, candidate = preds[label]
        ix = [i for i in range(len(uids)) if i != omitted]
        deltas.append(auc([y[i] for i in ix], [candidate[i] for i in ix]) -
                      auc([y[i] for i in ix], [baseline[i] for i in ix]))
    loo.append(sum(deltas) / len(labels))
analysis = {
    'source_zip': archive.name, 'source_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
    'gold_is_independent_validation': False,
    'probe_origin': 'One 65% R7 / 35% Raptor OA-group hypothesis, motivated after inspecting diagnostic 26; no weight search.',
    'base_macro_auc': base_macro, 'candidate_macro_auc': candidate_macro,
    'score_format': 'float32, matching the actual 25A/27 submission path and its ties',
    'diagnostic_26_float64_baseline_auc': report['macro_auc']['Blend_baseline'],
    'delta_macro_auc': candidate_macro - base_macro,
    'per_target': metrics,
    'leave_one_study_out_fixed_scores': {'min_delta': min(loo), 'max_delta': max(loo),
                                        'positive_delta_count': sum(x > 0 for x in loo), 'total': len(loo)},
    'runtime_26_seconds': report['elapsed_seconds'],
    'window_policy_decision': 'Retain original 25A windows; both filtering diagnostics decreased blend AUC.',
    'interpretation': 'Gold is reused development data. These numbers motivate a prospective hidden-test probe and do not estimate a Private gain.'
}
print(json.dumps(analysis, indent=2))
if '--analyze-only' in sys.argv:
    raise SystemExit(0)

nb = json.loads((ROOT / '25A_Raptor_R7_Public0956_Reproduction.ipynb').read_text(encoding='utf-8'))
out = copy.deepcopy(nb)
old = ''.join(nb['cells'][7]['source'])
expected = "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.5, 'PF OA': 0.5}"
assert old.count(expected) == 1
blend = old.replace(expected, "W_RAPTOR_C = {'Medial OA': 0.35, 'Lateral OA': 0.35, 'PF OA': 0.35}")
blend = blend.replace('# Per-finding override (user decision 10-10 17:15, fixed before any run): Medial OA 0.50, Lateral OA 0.50, PF OA 0.50 follow Raptor.',
                      '# 27: OA group uses 35% Raptor / 65% R7. Other nine targets remain the 25A 50/50 blend.')
blend = blend.replace('# Raptor trained on every training study and was selected on the 58 gold, so no clean labelled set exists to fit them.',
                      '# Hypothesis chosen after diagnostic 26 on reused gold studies; prospective Public test, not clean-CV weight fitting.')
blend += r'''

# Export the base computed on the SAME live test rows and verify the nine protected columns.
baseline_values = np.zeros_like(out)
for j, c in enumerate(labels):
    rr = pct_on(raw[c].values, ~rap_fail)
    ro = pct_on(ours[c].values, ~our_fail)
    b = 0.5 * rr + 0.5 * ro
    b[rap_fail] = ro[rap_fail]
    b[our_fail] = rr[our_fail]
    b[rap_fail & our_fail] = 0.5
    baseline_values[:, j] = b
baseline = pd.DataFrame(baseline_values.astype(np.float32), columns=labels)
baseline.insert(0, 'StudyInstanceUID', test_ids)
baseline[cols].to_csv(WORK / '27_base25A_reference.csv', index=False)
non_oa = [c for c in labels if c not in ('Medial OA', 'Lateral OA', 'PF OA')]
if not np.array_equal(sub[non_oa].values, baseline[non_oa].values):
    raise RuntimeError('27 modified a non-OA prediction; do not submit')
blend['experiment'] = '27_OA_group_R7_65_Raptor_35'
blend['diagnostic_origin'] = 'Hypothesis motivated by results 26; reused gold, not independent CV'
blend['protected_non_oa_columns'] = non_oa
blend['non_oa_byte_equal_as_float32'] = True
(WORK / 'blend_receipt.json').write_text(json.dumps(blend, indent=2))
'''
out['cells'][7]['source'] = blend.splitlines(keepends=True)
intro = '''# 27 — 0.956 base, OA-group R7 65% probe

One prospective Public-LB experiment. Keep the exact 25A Raptor and R7 inference, including the original 94 windows. Only Medial OA, Lateral OA and PF OA change from 50/50 ranks to **65% R7 / 35% Raptor**. The other nine predictions must match the base exactly on the same test rows.

Diagnostic 26 reproduced 25A, scored all 58 gold studies without fallback, and did not support removing cross-series windows. R7 has dedicated OA extent heads and was stronger than Raptor on all three gold OA targets. This single shared OA weight is motivated by that reused development set; it is not fitted or validated on independent CV, and no Public/Private score gain is guaranteed. Do not adjust this weight after seeing this probe's Public result.

**Attach exactly the same four inputs as 25A:** the RSNA competition; `nartaa/rsna-knee-publication-swa-weights-20261007`; `heliosli/rsna-knee-compact-r7cnxfsss`; `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`. **GPU T4x2, Internet Off, Save & Run All.** No 26 Output or 224 checkpoint is needed. It performs the same model inference work as 25A; hidden-test runtime is not measured by the three-row saved run.

After a successful saved run, submit once. Send its Public Score and Output ZIP, especially `27_run_audit.json`, `blend_receipt.json`, `submission_receipt.json` and `submission.csv`. `27_base25A_reference.csv` contains the original blend on those same live rows for comparison. Existing failure limits are inherited from 25A; inspect the scored row count and fallbacks.

'''
intro += ''.join(nb['cells'][1]['source'])
out['cells'][0]['source'] = intro.splitlines(keepends=True)
# Credits are already present in the new intro.
out['cells'][1]['source'] = ['The current best confirmed Public score remains **0.956 from 25A** until this prospective probe is scored.\n']
audit = ''.join(nb['cells'][8]['source']).replace("'25_run_audit.json'", "'27_run_audit.json'")
audit += "\naudit['experiment'] = '27_OA_group_R7_65_Raptor_35'\naudit['gold_is_independent_validation'] = False\naudit['non_oa_predictions_match_25a'] = True\n(work / '27_run_audit.json').write_text(json.dumps(audit, indent=2))\n"
# The original audit variable name is inspected below rather than assumed.
tree = ast.parse(''.join(nb['cells'][8]['source']))
dict_names = [t.id for node in tree.body if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
              for t in node.targets if isinstance(t, ast.Name)]
assert len(dict_names) == 1, dict_names
audit_name = dict_names[0]
audit = audit.replace("audit['", audit_name + "['").replace('json.dumps(audit, indent=2)', 'json.dumps(' + audit_name + ', indent=2)')
out['cells'][8]['source'] = audit.splitlines(keepends=True)
for index, item in enumerate(out['cells']):
    if item['cell_type'] == 'code':
        text = ''.join(item['source'])
        if text.startswith('%%writefile '):
            text = text.split('\n', 1)[1]
        compile(text, f'27:cell{index}', 'exec')
        item.update(execution_count=None, outputs=[])
for i in (3, 4, 5, 6):
    assert out['cells'][i]['source'] == nb['cells'][i]['source'], 'Model inference source drift'
out['metadata'].setdefault('kaggle', {})['isInternetEnabled'] = False
target = ROOT / '27_RSNA_0956_OA_Group_R7_65_Probe.ipynb'
target.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
(ROOT / '26_result_analysis_and_27_hypothesis.json').write_text(json.dumps(analysis, indent=2), encoding='utf-8')

# Execute the actual new blend cell on the 58 real diagnostic predictions.
# Exercise normal scoring and the inherited both-models-failed fallback route.
import contextlib
import numpy as np
import pandas as pd
for simulate_failure in (False, True):
    with tempfile.TemporaryDirectory(prefix='rsna27_blend_test_', dir=ROOT) as temporary:
        work = Path(temporary)
        assert work.resolve().parent == ROOT.resolve(), 'Temporary cleanup target must stay in this workspace'
        data = work / 'data'
        pred_dir = work / 'pred' / 'worker0'
        data.mkdir()
        pred_dir.mkdir(parents=True)
        pd.DataFrame({'StudyInstanceUID': uids}).to_csv(data / 'test.csv', index=False)
        sample = pd.DataFrame({'StudyInstanceUID': uids})
        sample[labels] = 0.5
        sample.to_csv(data / 'sample_submission.csv', index=False)
        rp = pd.DataFrame([raptor[u] for u in uids])
        op = pd.DataFrame([r7[u] for u in uids])
        rp[labels] = rp[labels].astype(float)
        op[labels] = op[labels].astype(float)
        if simulate_failure:
            rp.loc[0, labels] = 0.5
        rp.to_csv(work / 'raptor_raw.csv', index=False)
        ranked = rp.copy()
        ranked[labels] = rp[labels].rank(method='average', pct=True)
        ranked.to_csv(work / 'raptor_sub.csv', index=False)
        op.to_csv(work / 'ours_sub.csv', index=False)
        op.iloc[1 if simulate_failure else 0:].to_csv(pred_dir / 'predictions.csv', index=False)
        scope = {'WORK': work, 'DATA': data, 'pred': pred_dir.parent,
                 'RAPTOR_SECONDS': 0.0, 'NB_T0': time.time(), 'time': time,
                 'json': json,
                 'display': lambda *args, **kwargs: None}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(blend, '27 actual blend cell', 'exec'), scope)
        actual = pd.read_csv(work / 'submission.csv').set_index('StudyInstanceUID')
        reference_frame = pd.read_csv(work / '27_base25A_reference.csv').set_index('StudyInstanceUID')
        non_oa = [c for c in labels if c not in OA]
        assert np.array_equal(actual[non_oa].to_numpy(), reference_frame[non_oa].to_numpy())
        if simulate_failure:
            assert (actual.loc[uids[0]].to_numpy() == 0.5).all()
        else:
            expected_values = (0.35 * rp.set_index('StudyInstanceUID')[OA].rank(method='average', pct=True) +
                               0.65 * op.set_index('StudyInstanceUID')[OA].rank(method='average', pct=True))
            assert np.max(np.abs(actual[OA].to_numpy() - expected_values.astype(np.float32).to_numpy())) < 1e-7
            served_base = sum(auc(preds[label][0], reference_frame[label].tolist()) for label in labels) / len(labels)
            served_candidate = sum(auc(preds[label][0], actual[label].tolist()) for label in labels) / len(labels)
            assert abs(served_base - analysis['base_macro_auc']) < 1e-12
            assert abs(served_candidate - analysis['candidate_macro_auc']) < 1e-12
print('CREATED', target.name, '| code compiles | model source unchanged | real-prediction blend and fallback tests PASS')
