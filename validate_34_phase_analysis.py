"""Execute experiment 34's actual analysis on aligned identity fixtures."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
nb=json.loads((ROOT/'34_R6_Physical_Sampling_Phase_Gold_Probe.ipynb').read_text(encoding='utf-8'))
pre=''.join(nb['cells'][1]['source'])
node=next(n for n in ast.parse(pre).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REF' for t in n.targets))
ref=ast.literal_eval(node.value)
analysis=''.join(nb['cells'][-1]['source'])
with tempfile.TemporaryDirectory(prefix='rsna34_test_',dir=ROOT) as directory:
    work=Path(directory)
    assert work.resolve().parent==ROOT.resolve()
    log=work/'34_r6_tta_gold_worker0_log.txt'
    def write_logs(changed=False):
        log.write_text('\n'.join('PHASE_AUDIT '+json.dumps({'uid':u,'baseline_windows':100,
                       'candidate_windows':100,'added_windows':int(changed and i==0),
                       'removed_windows':int(changed and i==0),'phase':0}) for i,u in enumerate(ref['uids'])))
    frame=pd.DataFrame(ref['r6'],index=ref['uids'],columns=ref['labels'])
    scope={'REF':ref,'tta_gold':frame,'WORK':work,'np':np,'pd':pd,'json':json,
           'time':time,'NB_T0':time.time(),'DO_GPU_PROBE':True,'display':lambda *args:None}
    for changed in (False,True):
        write_logs(changed)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(analysis,'<actual 34 analysis>','exec'),scope)
        r=scope['result']
        assert r['changed_window_studies']==int(changed)
        assert r['delta_proxy_macro_auc']==0
        assert r['status']==('COMPLETE' if changed else 'NO_NEW_IMAGE_EVIDENCE')
        assert not (work/'submission.csv').exists()
    # Numerical drift on an unchanged bag must not be interpreted as phase gain.
    corrupted=frame.copy(); corrupted.iloc[1,0]=.123456
    scope['tta_gold']=corrupted
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(analysis,'<34 drift rejection>','exec'),scope)
    except RuntimeError as exc:
        assert 'Unchanged-window studies' in str(exc)
    else:
        raise AssertionError('Unchanged-bag drift was not rejected')
    # Full serving cells must be safe to skip without model imports.
    for c in nb['cells'][3:]:
        exec(compile(''.join(c['source']),'<34 no-eligible skip>','exec'),{'DO_GPU_PROBE':False})
print('34 PASS: actual analysis, aligned 58-row no-op, changed-window inventory, serving-drift rejection, no-GPU skip, no submission')
