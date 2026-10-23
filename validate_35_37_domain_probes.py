"""Test actual pooling, probability fusion and original 33 blend on CPU fixtures."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import zipfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
load=lambda p:json.loads((ROOT/p).read_text(encoding='utf-8'))
anchor=load('33_RSNA_0958_R6_Raptor_Submit.ipynb')
probe=load('36_Raptor_Series_Density_MIL_Submit.ipynb')
def model_class(nb):
    s=''.join(nb['cells'][2]['source']).split('\n',1)[1]
    c=next(n for n in ast.parse(s).body if isinstance(n,ast.ClassDef) and n.name=='RaptorClassifier')
    scope={'torch':torch,'nn':nn}
    exec(compile(ast.Module(body=[c],type_ignores=[]),'<actual Raptor head>','exec'),scope)
    return scope['RaptorClassifier']

torch.manual_seed(3601)
old=model_class(anchor)(nn.Identity(),F_dim=8,n=3,drop=0).eval()
new=model_class(probe)(nn.Identity(),F_dim=8,n=3,drop=0).eval()
new.load_state_dict(old.state_dict())
x=torch.randn(1,6,8)
assert torch.equal(old.head(x),new.head(x))
new._apply_series_prior=True
new._series_log_count=torch.zeros(6)
assert torch.equal(old.head(x),new.head(x))
# Duplicate every instance of one series and adjust its count. Pure corrected
# attention is invariant; original/corrected 50/50 mixture is intentionally not.
groups=np.array([0,0,1,1,2,2])
new._series_log_count=torch.tensor(np.log(np.bincount(groups)[groups]),dtype=torch.float32)
baseline=new.head(x)
ix=np.array([0,1,0,1,0,1,2,3,4,5])
gg=groups[ix]
new._series_log_count=torch.tensor(np.log(np.bincount(gg)[gg]),dtype=torch.float32)
assert torch.allclose(baseline,new.head(x[:,ix]),atol=1e-6)
assert np.isfinite(new.head(x[:,ix]).detach().numpy()).all()
print('36: actual head identity, singleton-prior identity, series-duplication invariance PASS')

with zipfile.ZipFile(ROOT/'results 26.zip') as z:
    raptor=pd.read_csv(io.BytesIO(z.read('26_raptor_baseline.csv')),dtype={'StudyInstanceUID':str})
    r6=pd.read_csv(io.BytesIO(z.read('26_r7_gold.csv')),dtype={'StudyInstanceUID':str})
labels=list(raptor.columns[1:]); ids=raptor.StudyInstanceUID.tolist()
for filename in ('35_R6_Phase_Ensemble_Submit.ipynb','37_R6_SameSize_ShiftedCrop_Submit.ipynb'):
    nb=load(filename); tag=filename[:2]
    stage=''.join(nb['cells'][7]['source'])
    scoring=''.join(nb['cells'][8]['source'])
    audit=''.join(nb['cells'][-1]['source'])
    for failure in (False,True):
        with tempfile.TemporaryDirectory(prefix='rsna_domain_test_',dir=ROOT) as directory:
            work=Path(directory); assert work.resolve().parent==ROOT.resolve()
            data=work/'data'; data.mkdir()
            raptor[['StudyInstanceUID']].to_csv(data/'test.csv',index=False)
            raptor.to_csv(data/'sample_submission.csv',index=False)
            base=r6.set_index('StudyInstanceUID'); candidate=base.copy()
            candidate.iloc[:,:]=np.roll(base.values,3,axis=0)
            bv=set(ids); cv=set(ids)
            if failure:
                bv-={ids[1],ids[2]}; cv-={ids[0],ids[2]}
            source_dir=work/'source'; source_dir.mkdir()
            def mock_controller(command,**kwargs):
                output=Path(command[command.index('--output')+1])
                output.mkdir(parents=True,exist_ok=True)
                original=output.name.endswith('_original')
                frame=base.copy() if original else candidate.copy()
                valid=bv if original else cv
                missing=[u for u in frame.index if u not in valid]
                if missing: frame.loc[missing]=.5
                frame.to_csv(output/'submission.csv',index_label='StudyInstanceUID')
                w=output/'worker0'; w.mkdir(exist_ok=True)
                frame.loc[[u for u in frame.index if u in valid]].to_csv(w/'predictions.csv',index_label='StudyInstanceUID')
                (w/'log.txt').write_text('MOCK CONTROLLER: no GPU inference')
                (output/'submission_receipt.json').write_text(json.dumps({'scored':len(valid),'fallback_studies':len(frame)-len(valid)}))
                return SimpleNamespace(returncode=0)
            def clean_scope():
                # Match parent setup names, deliberately NOT np/pd: old tests masked the missing import.
                return {'WORK':work,'TMP':work,'DATA':data,'SRC':source_dir,'CODE':work,
                        'BUNDLE':work,'ARM':'E_lrprior','WORKER':'compact_infer.py','WORKER_ARGS':[],
                        'BUNDLE_ID':'fe055406fad6a12f','NB_T0':time.time(),
                        'RAPTOR_SECONDS':0,'RAPTOR224_SECONDS':0,'display':lambda *a:None}
            scope=clean_scope()
            assert 'np' not in scope and 'pd' not in scope
            with patch('subprocess.run',side_effect=mock_controller), contextlib.redirect_stdout(io.StringIO()):
                exec(compile(stage,'<COMPLETE actual fusion cell>','exec'),scope)
            if tag=='35' and not failure:
                # Regression proof: removing the new import reproduces exactly the Kaggle error.
                broken=stage.replace('import numpy as np\n','')
                try:
                    with patch('subprocess.run',side_effect=mock_controller), contextlib.redirect_stdout(io.StringIO()):
                        exec(compile(broken,'<missing numpy regression>','exec'),clean_scope())
                except NameError as exc:
                    assert "'np'" in str(exc)
                else:
                    raise AssertionError('Regression fixture did not reproduce missing-numpy error')
            out=pd.read_csv(work/'ours_sub.csv').set_index('StudyInstanceUID')
            if failure:
                assert np.allclose(out.loc[ids[0]],base.loc[ids[0]])
                assert np.allclose(out.loc[ids[1]],candidate.loc[ids[1]])
                assert np.all(out.loc[ids[2]].values==.5)
            else: assert np.allclose(out.values,.5*base.values+.5*candidate.values)
            for name in ('raptor','raptor224'):
                raptor.to_csv(work/(name+'_raw.csv'),index=False)
                ranked=raptor.copy(); ranked[labels]=ranked[labels].rank(pct=True)
                ranked.to_csv(work/(name+'_sub.csv'),index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(scoring,'<33 preserved blend>','exec'),scope)
                exec(compile(audit,'<new final audit>','exec'),scope)
            assert scope['result']['status']==('FALLBACKS_PRESENT' if failure else 'COMPLETE')
            assert scope['result']['weights']['raptor_per_class']['Synovitis']==.8
            if failure: assert scope['result']['failures']['ours_failed']==1
            else:
                result=pd.read_csv(work/'submission.csv')[labels]
                rr=raptor[labels].rank(pct=True).values
                ro=out[labels].rank(pct=True).values
                weights=np.array([scope['W_RAPTOR_C'].get(c,.5) for c in labels])
                assert np.allclose(result.values,(weights*rr+(1-weights)*ro).astype(np.float32),atol=1e-7)
    print(tag+': COMPLETE fusion cell in clean parent scope, mocked controller, valid-UID union, fallbacks, protected 33 weights and audit PASS')

manifest=json.loads((ROOT/'35_37_Domain_Probe_Manifest.json').read_text())
import hashlib
assert hashlib.sha256((ROOT/'33_RSNA_0958_R6_Raptor_Submit.ipynb').read_bytes()).hexdigest()==manifest['anchor_sha256']
print('Protected 33 unchanged; no GPU/DICOM tests performed locally')
