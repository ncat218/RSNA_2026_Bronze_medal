"""CPU-only tests of 39 fusion, source adaptation and whole parent control flow."""
import ast, hashlib, json, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
nb=json.loads((ROOT/'39_Anchor0958_Plus_S6_Fixed10_Rank_Submit.ipynb').read_text(encoding='utf-8'))
source=''.join(nb['cells'][1]['source'])
tree=ast.parse(source)
def literal(name):
    return ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)
                                and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)))
anchor_script=literal('ANCHOR_SCRIPT')
module=literal('S6_COMPONENT')
for text in (source,anchor_script,literal('S6_SCRIPT'),module):compile(text,'<39 static>','exec')
anchor=json.loads((ROOT/'33_RSNA_0958_R6_Raptor_Submit.ipynb').read_text(encoding='utf-8'))
assert hashlib.sha256((ROOT/'33_RSNA_0958_R6_Raptor_Submit.ipynb').read_bytes()).hexdigest()=='685e6e0e65c3c5ed29b11c0b584887d894cd07f274de0b9f83370465ab7b4fde'
for i,c in enumerate(anchor['cells']):
    if c['cell_type']!='code':continue
    text=''.join(c['source'])
    if text.startswith('%%writefile '):
        _,body=text.split('\n',1)
        assert repr(body.encode()) in anchor_script
    else:
        if i==8:text=text.replace("sub.to_csv(WORK / 'submission.csv', index=False)","sub.to_csv(WORK / '39_anchor33.csv', index=False)")
        assert text in anchor_script

fusion=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fuse_anchor_s6')
scope={}
exec(compile(ast.Module(body=[fusion],type_ignores=[]),'<39 fusion>','exec'),scope)
fuse=scope['fuse_anchor_s6']
labels=['ACL','MCL']
ids=['a','b','c','d']
a=pd.DataFrame({'StudyInstanceUID':ids,'ACL':[.2,.4,.6,.8],'MCL':[.8,.6,.4,.2]})
b=pd.DataFrame({'StudyInstanceUID':['d','b','a','c'],'ACL':[.1,.9,.9,.5],'MCL':[.8,.7,.2,.4]})
out=fuse(a,b,ids,labels,{})
expected=.9*a[labels].to_numpy()+.1*b.set_index('StudyInstanceUID').loc[ids,labels].rank(method='average',pct=True).to_numpy()
np.testing.assert_allclose(out[labels],expected,atol=1e-7)
partial=b.copy()
partial.loc[partial.StudyInstanceUID=='b',labels]=np.nan
fallback=fuse(a,partial,ids,labels,{'b':'injected study decode failure'})
np.testing.assert_allclose(fallback.loc[1,labels].to_numpy(float),a.loc[1,labels].to_numpy(float),atol=1e-7)
for bad in (b.iloc[:-1],pd.concat([b,b.iloc[:1]])):
    try:fuse(a,bad,ids,labels,{})
    except RuntimeError:pass
    else:raise AssertionError('Bad UID coverage accepted')

# Execute the COMPLETE notebook parent with fixture I/O and mocked model stages.
# Mock only model inference and mount identity; retain report/export/error cleanup.
with tempfile.TemporaryDirectory(prefix='rsna39_test_',dir=ROOT) as folder:
    root=Path(folder)
    assert root.resolve().is_relative_to(ROOT.resolve())
    work=root/'working'
    work.mkdir()
    comp=root/'competition'
    comp.mkdir()
    full_labels=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',
                 'Effusion','Synovitis',"Baker's",'Contusion','Fracture']
    # Different IDs ensure reference-visible parity is not mistaken for hidden inference.
    test_ids=['hidden_a','hidden_b','hidden_c','hidden_d']
    pd.DataFrame({'StudyInstanceUID':test_ids}).to_csv(comp/'test.csv',index=False)
    for name in ('raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt','raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt'):
        (comp/name).write_bytes(b'fixture only; no model')
    fixture_module='''
from pathlib import Path
LAB=__LABELS__
COMP='competition'
HELIOS_SLUG='anatomy'
DS_SLUG='S6'
def find_mount(slug,marker):return Path(__COMP__)
def sha256_file(p):
    if 'r384_swa.pt' in p.name:return '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef'
    if 'r224_swa.pt' in p.name:return '3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6'
    return '6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648' if p.name=='manifest.json' else 'b66c574b5c0c13ccaa49d1219d41a8781431166186ef83e78cab6a9ba6b02d22'
def b_fail_cap(n):return 0
def mount_inventory():return ['CPU fixture']
'''.replace('__LABELS__',repr(full_labels)).replace('__COMP__',repr(str(comp)))
    fake_stage=r'''
def run_stage(name,script,max_seconds):
    if INJECT_FAILURE and name=='anchor33':
        (WORK/'submission.csv').write_text('PARTIAL')
        raise RuntimeError('injected stage failure')
    values=np.tile(np.array([.2,.4,.6,.8])[:,None],(1,len(M.LAB)))
    frame=pd.DataFrame(values,columns=M.LAB)
    frame.insert(0,'StudyInstanceUID',ids)
    if name=='s6':
        frame.to_csv(OUT/'39_s6_probabilities.csv',index=False)
        (OUT/'39_s6_receipt.json').write_text(json.dumps({'status':'COMPLETE','failed':{}}))
    else:
        frame.to_csv(WORK/'39_anchor33.csv',index=False)
        (WORK/'blend_receipt.json').write_text(json.dumps({'raptor384_failed':0,'raptor224_failed':0,'ours_failed':0}))
    REPORT[name]={'fixture':True}
'''
    for inject in (False,True):
        local_tree=ast.parse(source)
        for n in local_tree.body:
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='S6_COMPONENT' for t in n.targets):
                n.value=ast.Constant(fixture_module)
        replacement=ast.parse(fake_stage).body[0]
        local_tree.body=[replacement if isinstance(n,ast.FunctionDef) and n.name=='run_stage' else n for n in local_tree.body]
        local_text=ast.unparse(ast.fix_missing_locations(local_tree)).replace("Path('/kaggle/working')",'Path('+repr(str(work))+')')
        env={'__name__':'__main__','display':lambda x:None,'INJECT_FAILURE':inject}
        try:exec(compile(local_text,'<39 full parent fixture>','exec'),env)
        except RuntimeError as exc:
            assert inject and 'injected stage failure' in str(exc)
        else:
            assert not inject
        audit=json.loads((work/'rsna_39/39_run_audit.json').read_text())
        assert audit['status']==('FAILED' if inject else 'COMPLETE')
        assert (work/'submission.csv').exists()==(not inject)
        if not inject:
            final=pd.read_csv(work/'submission.csv')
            assert final.StudyInstanceUID.tolist()==test_ids
            assert len(final)==4 and np.isfinite(final[full_labels].to_numpy()).all()

manifest=json.loads((ROOT/'39_Build_Manifest.json').read_text())
assert manifest['notebook_sha256']==hashlib.sha256((ROOT/'39_Anchor0958_Plus_S6_Fixed10_Rank_Submit.ipynb').read_bytes()).hexdigest()
result={'status':'PASS','notebook_sha256':manifest['notebook_sha256'],
        'tests':['all sources compile','all anchor33 code bodies preserved except final export destination',
                 'rank fusion with ties and reordered IDs','missing/duplicate UID rejection',
                 'failed S6 row preserves anchor only','full parent clean-namespace success with fixture stages',
                 'full parent injected stage failure removes partial submission and writes FAILED audit'],
        'gpu_execution_tested':False,'hidden_submission_tested':False}
(ROOT/'39_Local_Validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('PASS: source preservation, fusion and full parent success/failure; GPU/hidden inference NOT tested.')
