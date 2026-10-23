"""Test the actual CPU audit cells without competition reports."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
nb = json.loads((root / '32_Synovitis_Effusion_Discordant_Case_Audit_CPU.ipynb').read_text(encoding='utf-8'))
sources = [''.join(c['source']) for c in nb['cells']]
node = next(n for n in ast.parse(sources[1]).body if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REF' for t in n.targets))
ref = ast.literal_eval(node.value)
with tempfile.TemporaryDirectory(prefix='rsna32_test_', dir=root) as directory:
    task = Path(directory)
    assert task.resolve().parent == root.resolve()
    competition = task / 'input' / 'competitions' / 'rsna-knee-abnormality-detection'
    competition.mkdir(parents=True)
    truth = pd.DataFrame(ref)[['StudyInstanceUID','gold_synovitis','gold_effusion']].rename(columns={'gold_synovitis':'Synovitis','gold_effusion':'Effusion'})
    truth['Report'] = 'PRIVATE_REPORT_SENTINEL_DO_NOT_EXPORT'
    truth.to_csv(competition / 'train.csv', index=False)
    setup = sources[1].replace("Path('/kaggle/input')", repr(task / 'input')).replace("Path('/kaggle/working/rsna32_case_audit')", repr(task / 'output'))
    # pathlib repr is WindowsPath on Windows; use Path constructors explicitly.
    setup = setup.replace(repr(task / 'input'), 'Path(' + repr(str(task / 'input')) + ')')
    setup = setup.replace(repr(task / 'output'), 'Path(' + repr(str(task / 'output')) + ')')
    scope = {'display':lambda *args:None}
    with contextlib.redirect_stdout(io.StringIO()):
        for s in [setup,sources[2],sources[3]]:
            exec(compile(s, '<actual 32 cell>', 'exec'), scope)
    assert scope['summary']['discordant_cases'] == 18
    assert scope['summary']['case_counts']['synovitis_only'] == 5
    assert len(list((task / 'output').iterdir())) == 4
    assert all('PRIVATE_REPORT_SENTINEL' not in p.read_text() for p in (task / 'output').iterdir())
    assert not list(task.rglob('submission.csv'))
    scope['train'].loc[0, 'Synovitis'] = 1-int(scope['train'].loc[0, 'Synovitis'])
    try:
        exec(compile(sources[2], '<32 mismatch test>', 'exec'), scope)
    except RuntimeError as exc:
        assert 'label identity mismatch' in str(exc)
    else:
        raise AssertionError('Changed gold label was not rejected')
print('32 PASS: actual cells, 58/18 UID alignment, case counts, no raw-report export, no submission, changed-label rejection')
