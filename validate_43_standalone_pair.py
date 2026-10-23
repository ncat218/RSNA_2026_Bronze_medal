"""CPU-only full-parent and scientific-output tests for43A/B."""
import ast, hashlib, json, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'43_Standalone_Build_Manifest.json').read_text())
labels=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',
        'Effusion','Synovitis',"Baker's",'Contusion','Fracture']
def literal(source,name):
    return ast.literal_eval(next(n.value for n in ast.parse(source).body if isinstance(n,ast.Assign)
                                and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)))
results={}
for tag,spec in manifest.items():
    path=ROOT/spec['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
    nb=json.loads(path.read_text(encoding='utf-8'))
    source=''.join(nb['cells'][1]['source'])
    compile(source,'<'+tag+'>','exec')
    stage=literal(source,'S6_SCRIPT')
    compile(stage,'<'+tag+' stage>','exec')
    assert "if failed:raise RuntimeError('Standalone comparison requires every S6 row scored; failed='+repr(failed))" in stage
    assert 'ANCHOR_SCRIPT' not in source and 'W_RAPTOR_C' not in source
    assert len(nb['metadata']['kaggle']['dataSources'])==3
    func=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='make_standalone_submission')
    scope={'np':np,'pd':pd}
    exec(compile(ast.Module(body=[func],type_ignores=[]),'<43 rank export>','exec'),scope)
    export=scope['make_standalone_submission']
    ids=['a','b','c','d']
    raw=pd.DataFrame(np.tile([.01,.02,.9,.9],(12,1)).T,columns=labels)
    raw.insert(0,'StudyInstanceUID',ids)
    ranked=export(raw.iloc[::-1],ids,labels)
    assert ranked.StudyInstanceUID.tolist()==ids
    np.testing.assert_allclose(ranked[labels[0]], [.25,.5,.875,.875])
    for kind in ('nan','missing','duplicate'):
        bad=raw.copy()
        if kind=='nan':bad.loc[0,labels[0]]=np.nan
        elif kind=='missing':bad=bad.iloc[:-1]
        else:bad=pd.concat([bad,bad.iloc[:1]])
        try:export(bad,ids,labels)
        except RuntimeError:pass
        else:raise AssertionError('Bad standalone input accepted:'+kind)
    # Large-N synthetic check: rank ordering and no3-row-only assumptions.
    rng=np.random.default_rng(43)
    large=pd.DataFrame(rng.random((1500,12)),columns=labels)
    large_ids=[f'study_{i}' for i in range(1500)]
    large.insert(0,'StudyInstanceUID',large_ids)
    output=export(large,large_ids,labels)
    assert len(output)==1500 and np.isfinite(output[labels].to_numpy()).all()
    assert np.array_equal(np.argsort(output[labels[0]]),np.argsort(large[labels[0]]))

    with tempfile.TemporaryDirectory(prefix='rsna43_test_',dir=ROOT) as folder:
        root=Path(folder)
        assert root.resolve().is_relative_to(ROOT.resolve())
        work=root/'working';work.mkdir()
        comp=root/'competition';comp.mkdir()
        ids=['hidden_a','hidden_b','hidden_c','hidden_d']
        pd.DataFrame({'StudyInstanceUID':ids}).to_csv(comp/'test.csv',index=False)
        pd.DataFrame(columns=['StudyInstanceUID']+labels).to_csv(comp/'sample_submission.csv',index=False)
        for f in ('anatomy.pt','E.pt'):(comp/f).write_bytes(b'fixture; never loaded')
        fixture='''
from pathlib import Path
LAB=__LABELS__
COMP='competition'
DS_SLUG='S6'
HELIOS_SLUG='anatomy'
B_MODEL={'file':'E.pt','sha256':'fixture-model'}
def find_mount(slug,marker):return Path(__COMP__)
def sha256_file(p):
    if p.name=='anatomy.pt':return '5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e'
    if p.name=='E.pt':return 'fixture-model'
    return 'b66c574b5c0c13ccaa49d1219d41a8781431166186ef83e78cab6a9ba6b02d22'
def mount_inventory():return ['CPU fixture']
'''.replace('__LABELS__',repr(labels)).replace('__COMP__',repr(str(comp)))
        fake_stage='''
def run_stage(name,script,max_seconds):
    if TEST_CASE=='worker_error':
        (WORK/'submission.csv').write_text('PARTIAL')
        raise RuntimeError('injected worker failure')
    frame=pd.DataFrame(np.tile([.1,.2,.9,.8],(12,1)).T,columns=labels)
    frame.insert(0,'StudyInstanceUID',ids)
    if TEST_CASE=='missing':frame=frame.iloc[:-1]
    if TEST_CASE=='nan':frame.loc[0,labels[0]]=np.nan
    frame.to_csv(OUT/'39_s6_probabilities.csv',index=False)
    failed={'hidden_a':'injected decode error'} if TEST_CASE=='fallback' else {}
    (OUT/'39_s6_receipt.json').write_text(json.dumps({'status':'COMPLETE','failed':failed}))
    REPORT[name]={'fixture':True}
'''
        for case in ('success','worker_error','missing','nan','fallback'):
            tree=ast.parse(source)
            for n in tree.body:
                if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='S6_COMPONENT' for t in n.targets):
                    n.value=ast.Constant(fixture)
            replacement=ast.parse(fake_stage).body[0]
            tree.body=[replacement if isinstance(n,ast.FunctionDef) and n.name=='run_stage' else n for n in tree.body]
            local=ast.unparse(ast.fix_missing_locations(tree)).replace("Path('/kaggle/working')",'Path('+repr(str(work))+')')
            namespace={'__name__':'__main__','TEST_CASE':case,'display':lambda x:None}
            try:exec(compile(local,'<43 full parent fixture>','exec'),namespace)
            except RuntimeError:
                assert case!='success'
            else:assert case=='success'
            report=json.loads((work/('rsna_'+tag)/(tag+'_run_audit.json')).read_text())
            assert report['status']==('COMPLETE' if case=='success' else 'FAILED')
            assert (work/'submission.csv').exists()==(case=='success')
    results[tag]={'status':'PASS','notebook_sha256':spec['sha256'],
                  'tests':['all sources compile','3 input metadata; no anchor code',
                           'rank export with ties/reordered IDs','1500-row synthetic order preservation',
                           'full parent success with mocked inference',
                           'worker/missing-UID/NaN/fallback failures remove final CSV and record FAILED'],
                  'gpu_or_hidden_inference_tested':False}
(ROOT/'43_Local_Validation.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print('PASS43A/B CPU controller/output tests. Real models, GPU and hidden reruns remain unverified.')
