"""Execute the actual three blend cells and audits against 58 prediction fixtures."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
with zipfile.ZipFile(ROOT / 'results 26.zip') as z:
    rp = pd.read_csv(io.BytesIO(z.read('26_raptor_baseline.csv')), dtype={'StudyInstanceUID': str})
    op = pd.read_csv(io.BytesIO(z.read('26_r7_gold.csv')), dtype={'StudyInstanceUID': str})
labels = list(rp.columns[1:])
for file in sorted(ROOT.glob('29[ABC]_*.ipynb')):
    tag = file.name[:3]
    use224, flip = tag != '29B', tag != '29C'
    nb = json.loads(file.read_text(encoding='utf-8'))
    scoring = next(''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('# Stage 3:'))
    audit_source = ''.join(nb['cells'][-1]['source'])
    for failure in (False, True):
        with tempfile.TemporaryDirectory(prefix='rsna29_test_', dir=ROOT) as directory:
            work = Path(directory)
            assert work.resolve().parent == ROOT.resolve()
            data, prediction = work / 'data', work / 'pred'
            data.mkdir()
            (prediction / 'worker0').mkdir(parents=True)
            a, c = rp.copy(), op.copy()
            b = rp.copy()
            b[labels] = np.roll(b[labels].values, 7, axis=0)
            if failure:
                a.loc[0, labels] = 0.5
                b.loc[0, labels] = 0.5
            a[['StudyInstanceUID']].to_csv(data / 'test.csv', index=False)
            sample = a.copy()
            sample[labels] = 0.5
            sample.to_csv(data / 'sample_submission.csv', index=False)
            a.to_csv(work / 'raptor_raw.csv', index=False)
            c.to_csv(work / 'ours_sub.csv', index=False)
            c.iloc[1 if failure else 0:].to_csv(prediction / 'worker0' / 'predictions.csv', index=False)
            for frame, name in [(a, 'raptor_sub.csv')] + ([(b, 'raptor224_sub.csv')] if use224 else []):
                ranked = frame.copy()
                ranked[labels] = frame[labels].rank(method='average', pct=True)
                ranked.to_csv(work / name, index=False)
            if use224:
                b.to_csv(work / 'raptor224_raw.csv', index=False)
            (work / 'submission_receipt.json').write_text(json.dumps({
                'workers': [{'flip_tta': {'active': True}} if flip else {}],
                'bundle_id': 'fe055406fad6a12f', 'scored': len(a) - int(failure)}))
            scope = {'WORK': work, 'DATA': data, 'pred': prediction, 'json': json,
                     'time': time, 'NB_T0': time.time(), 'RAPTOR_SECONDS': 0, 'RAPTOR224_SECONDS': 0,
                     'EXPERIMENT': tag, 'EXPECT_FLIP': flip, 'EXPECT_224': use224,
                     'display': lambda *args: None}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(scoring, tag + ':blend', 'exec'), scope)
                # Relocate only the audit output root to the isolated test fixture.
                patched_audit = audit_source.replace("work = Path('/kaggle/working')", 'work = WORK')
                exec(compile(patched_audit, tag + ':audit', 'exec'), scope)
            result = pd.read_csv(work / 'submission.csv')
            if failure:
                assert np.all(result.loc[0, labels].to_numpy(dtype=float) == 0.5)
                assert scope['audit']['status'] == 'FALLBACKS_PRESENT'
            else:
                expected = .5 * c[labels].rank(method='average', pct=True)
                expected += (.325 * a[labels].rank(method='average', pct=True) +
                             .175 * b[labels].rank(method='average', pct=True)) if use224 else .5 * a[labels].rank(method='average', pct=True)
                assert np.max(np.abs(result[labels].values - expected.to_numpy().astype(np.float32))) < 1e-7
                assert scope['audit']['status'] == 'COMPLETE'
            assert result.StudyInstanceUID.tolist() == a.StudyInstanceUID.tolist()
    print(tag, 'actual blend + receipt audit + fallback: PASS')
