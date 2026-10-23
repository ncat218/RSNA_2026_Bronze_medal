"""Execute the actual 30 analysis on identity fixtures, not GPU model predictions."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
nb = json.loads((ROOT / '30_R6_TTA_Error_Stacker_Gold_Diagnostic.ipynb').read_text(encoding='utf-8'))
pre = ''.join(nb['cells'][1]['source'])
node = next(n for n in ast.parse(pre).body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'REF' for t in n.targets))
ref = ast.literal_eval(node.value)
labels, uids = ref['labels'], ref['uids']
frame = pd.DataFrame(ref['r7'], index=uids, columns=labels)
truth = pd.DataFrame(ref['truth'], index=uids, columns=labels)
truth.index.name = 'StudyInstanceUID'
analysis = ''.join(nb['cells'][-1]['source'])
with tempfile.TemporaryDirectory(prefix='rsna30_test_', dir=ROOT) as directory:
    work = Path(directory)
    assert work.resolve().parent == ROOT.resolve()
    for part in range(2):
        ii = np.arange(len(uids))[part::2]
        for suffix, target in [('fluid_stacker', 'Effusion'), ('state_stacker', "Baker's")]:
            d = pd.DataFrame({'StudyInstanceUID': np.array(uids)[ii],
                              target+'|p_model': frame[target].values[ii],
                              target+'|p_out': frame[target].values[ii]})
            if suffix == 'fluid_stacker':
                d['status'] = 'fixture_only'
                d['eff_sag_p99_open'] = np.nan
            d.to_csv(work / f'30_r6_tta_gold_worker{part}_{suffix}.csv', index=False)
    scope = {'REF':ref,'WORK':work,'np':np,'pd':pd,'json':json,'time':time,
             'NB_T0':time.time(),'tta_gold':frame,'gold':truth,'visible_error':0.0,
             'display':lambda *args:None}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(analysis, '<actual 30 analysis>', 'exec'), scope)
    r = scope['result']
    assert r['status'] == 'COMPLETE' and r['gold_studies'] == 58
    assert r['submission_created'] is False
    assert not (work / 'submission.csv').exists()
    scores = r['macro_auc']
    assert scores['r6_tta_current'] == scores['without_fluid_stacker'] == scores['without_state_stacker']
    # Identity source family has equivalent mathematical weights; allow float32 tie rounding.
    assert np.max(np.abs(scope['blends']['r6_r7_plain_diversity'].values-scope['base'].values)) < 1e-7
    corrupted = work / '30_r6_tta_gold_worker0_fluid_stacker.csv'
    bad = pd.read_csv(corrupted)
    bad.loc[0, 'Effusion|p_out'] = .123456
    bad.to_csv(corrupted, index=False)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(analysis, '<30 corrupted dump rejection>', 'exec'), scope)
    except RuntimeError as exc:
        assert 'does not reproduce' in str(exc)
    else:
        raise AssertionError('Corrupted diagnostic dump was not rejected')
print('30 analysis: 58-row alignment, identity stacker ablations, family formula, bootstrap, no submission, corrupt dump rejection PASS')
