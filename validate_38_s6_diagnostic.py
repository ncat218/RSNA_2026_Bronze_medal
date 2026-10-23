"""CPU regression tests. No GPU, downloaded entrypoint, network or Kaggle writes."""
import ast
import hashlib
import json
import tempfile
import types
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
nb = json.loads((ROOT/'38_S6_Complementarity_Gold_Diagnostic.ipynb').read_text(encoding='utf-8'))
codes = [''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
for source in codes:
    compile(source,'<38 cell>','exec')
assert len(codes)==4
assert nb['metadata']['kaggle']['dataSources'][1]['sourceId']==20844630

def assignment(source,name):
    return ast.literal_eval(next(n.value for n in ast.parse(source).body if isinstance(n,ast.Assign)
                          and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)))
ref = assignment(codes[0],'REF')
packaged = assignment(codes[0],'S6_SOURCE')
scope = {'__name__':'rsna38_test'}
# Import reviewed source definitions, never main/worker inference.
exec(compile(packaged,'<38 definitions>','exec'),scope)
assert 'main' not in scope and 'run_a' not in scope and 'A_B64' not in scope
assert scope['B_MODEL']['sha256']=='0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12'
assert scope['b_fail_cap'](58)==0
cmd = scope['b_cmd'](types.SimpleNamespace(launcher='launch',comp='comp',code='code',models='models',anatomy='anatomy'),
                     'train','assignment','out','tmp',12345)
assert cmd[cmd.index('--split')+1]=='train'
assert cmd[cmd.index('--crop')+1]=='288' and cmd[cmd.index('--views')+1]=='v4'
assert '--flip-tta' not in cmd

with tempfile.TemporaryDirectory(prefix='rsna38_test_', dir=ROOT) as temp:
    root=Path(temp)
    assert root.resolve().is_relative_to(ROOT.resolve())
    # Exercise the real resolver, not a mocked one, for layouts the old resolver missed.
    mounted=root/'input'
    nested=mounted/'datasets'/'someowner'/'renamed-anatomy'/'archive'
    nested.mkdir(parents=True)
    (nested/'anatomy.pt').write_bytes(b'resolver fixture, never loaded')
    scope['INPUT']=mounted
    scope['OUT']=root/'resolver_outputs'
    actual=scope['find_mount']('rsna-orthofoundation-fold0-epoch7-20261004','anatomy.pt')
    assert actual==nested.resolve()
    (nested/'anatomy.pt').unlink()
    try:
        scope['find_mount']('rsna-orthofoundation-fold0-epoch7-20261004','anatomy.pt')
    except scope['K1Error'] as exc:
        assert 'attach heliosli/' in str(exc)
    else:
        raise AssertionError('Missing anatomy mount accepted')
    assert (scope['OUT']/'38_input_resolution.json').is_file()
    # Run complete setup cell with only fixed filesystem locations redirected.
    setup=codes[0].replace("Path('/kaggle/working/rsna_38_s6')",'Path('+repr(str(root/'out'))+')')
    setup=setup.replace("Path('/tmp/rsna38_s6')",'Path('+repr(str(root/'tmp'))+')')
    clean={'__name__':'__main__','display':lambda x:None}
    exec(compile(setup,'<38 clean setup>','exec'),clean)
    s6=clean['S6']
    assert not hasattr(s6,'main') and not hasattr(s6,'run_a')
    comp=root/'competition'
    comp.mkdir()
    (comp/'train_series').mkdir()
    train=pd.DataFrame(ref['truth'],columns=ref['labels'])
    train.insert(0,'StudyInstanceUID',ref['uids'])
    train.to_csv(comp/'train.csv',index=False)
    fake_inputs=types.SimpleNamespace(test_ids=['visible1','visible2','visible3'],comp=comp)
    s6.find_mount=lambda *a:root
    s6.sha256_file=lambda p:'b66c574b5c0c13ccaa49d1219d41a8781431166186ef83e78cab6a9ba6b02d22'
    s6.prepare_inputs=lambda:fake_inputs
    s6._RECEIPT['inputs']={'fixture':True}
    s6.probe_env=lambda:{'cuda_available':True,'n_gpu':2,'fixture':True}
    s6.probe_b_stack=lambda i:{'fixture':True}
    exec(compile(codes[1],'<38 full preflight>','exec'),clean)
    assert clean['REPORT']['status']=='PREFLIGHT_COMPLETE'
    calls=[]
    def fake_run(i,tag,ids,split,budget_s):
        calls.append((tag,len(ids),split,budget_s))
        p=np.full((len(ids),12),.5) if tag=='visible' else np.asarray(ref['r6'])
        return p,np.zeros_like(p),{'seconds':1.,'failed_after_retry':{}},{}
    s6.run_b=fake_run
    exec(compile(codes[2],'<38 full runner>','exec'),clean)
    assert calls[0][:3]==('visible',3,'test') and calls[1][:3]==('gold',58,'train')
    assert clean['REPORT']['status']=='INFERENCE_COMPLETE'
    # Use real metrics; reduce only resample count for test speed.
    analysis=codes[3].replace('n_boot=1000','n_boot=20')
    exec(compile(analysis,'<38 full analysis>','exec'),clean)
    report=json.loads((root/'out/38_diagnostic_report.json').read_text())
    assert report['status']=='COMPLETE' and not report['submission_created']
    assert report['macro_auc_descriptive_only']['s6']==report['macro_auc_descriptive_only']['r6']
    metrics=pd.read_csv(root/'out/38_per_target_metrics.csv')
    assert len(metrics)==12 and (metrics.s6_r6_spearman>.99999).all()
    assert len(pd.read_csv(root/'out/38_discordant_cases.csv'))==58*12
    assert not (root/'submission.csv').exists()
    # A worker failure must produce an error receipt and propagate, never create predictions.
    def fail(*a,**kw): raise RuntimeError('injected GPU/worker failure')
    s6.run_b=fail
    try:
        exec(compile(codes[2],'<38 failure path>','exec'),clean)
    except RuntimeError as exc:
        assert 'injected' in str(exc)
    else:
        raise AssertionError('worker error swallowed')
    assert clean['REPORT']['status']=='FAILED_INFERENCE'

manifest=json.loads((ROOT/'38_Build_Manifest.json').read_text())
assert manifest['notebook_sha256']==hashlib.sha256((ROOT/'38_S6_Complementarity_Gold_Diagnostic.ipynb').read_bytes()).hexdigest()
assert not manifest['gpu_execution_verified']
anchor=ROOT/'33_RSNA_0958_R6_Raptor_Submit.ipynb'
assert hashlib.sha256(anchor.read_bytes()).hexdigest()=='685e6e0e65c3c5ed29b11c0b584887d894cd07f274de0b9f83370465ab7b4fde'
result={'status':'PASS','notebook_sha256':manifest['notebook_sha256'],
        'tests':['all cells compile','reviewed controller imports without original main/A runner',
                 'real mount resolver handles renamed nested datasets and records missing anatomy',
                 'clean-parent full setup/preflight/runner/analysis with mocked environment and inference',
                 'paired bootstrap and CSV output schema','injected worker failure propagates and records FAILED_INFERENCE',
                 'exact anchor33 hash unchanged'],
        'gpu_or_mri_inference_tested':False,'test_predictions':'synthetic/previous reference, not new S6 results'}
(ROOT/'38_Local_Validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('PASS: clean-parent full setup/preflight/run/analysis, identity/args, strict error propagation; CPU mocks only, no GPU validation.')
