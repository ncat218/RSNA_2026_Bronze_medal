"""CPU tests of actual crop score function and inherited41 parent control flow."""
import ast, hashlib, json, tempfile, types
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent
file='41_Anchor0958_S6_TwoCrop_Fixed10_Rank_Submit.ipynb'
nb=json.loads((ROOT/file).read_text(encoding='utf-8'))
parent=''.join(nb['cells'][1]['source'])
def literal(text,name):
    return ast.literal_eval(next(n.value for n in ast.parse(text).body if isinstance(n,ast.Assign)
                                and any(isinstance(x,ast.Name) and x.id==name for x in n.targets)))
module=literal(parent,'S6_COMPONENT')
patched=literal(module,'CROP_TTA_PATCH')
expected_sha=literal(module,'CROP_TTA_PATCH_SHA')
assert hashlib.sha256(patched.encode()).hexdigest()==expected_sha
original=(ROOT/'research_38_s6/compact_infer_s6_original.py').read_text(encoding='utf-8')
def score_function(text,env):
    node=next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name=='score')
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<41 actual score function>','exec'),env)
    return env['score']
calls=[]
def run_model(model,img,meta,mask,mode):
    calls.append(tuple(img.shape))
    v=img.float().mean(dim=(1,2,3,4))/255
    return v[:,None]+torch.arange(12,dtype=torch.float32)[None,:]/100
env={'torch':torch,'a':types.SimpleNamespace(size=320,crop=288,chunk=96),
     'shared_trunk':False,'models2':[],'models':[object()],'amp':'fp16',
     'dev':torch.device('cpu'),'run_model':run_model,'off':16}
img=(torch.arange(320)[None,:]+torch.arange(320)[:,None]).remainder(256).to(torch.uint8)
images=img.repeat(4,3,1,1)
meta=torch.zeros(4,7)
plain=score_function(original,dict(env))(images,meta)
tta=score_function(patched,dict(env))(images,meta)
manual=.5*(run_model(None,images[None,:,:,16:304,16:304],None,None,None)[0]
             +run_model(None,images[None,:,:,24:312,24:312],None,None,None)[0])
torch.testing.assert_close(tta[0],manual,rtol=0,atol=0)
assert tta[1].shape==(1,12)
assert all(s==(1,4,3,288,288) for s in calls)
baseline_recipe=patched.replace('crop_offsets = ((16, 16), (24, 24))','crop_offsets = ((16, 16), (16, 16))')
control=score_function(baseline_recipe,dict(env))(images,meta)
torch.testing.assert_close(control[0],plain[0],rtol=0,atol=0)

# Original guard must precede patch; wrong input source cannot be silently accepted.
scope={'__name__':'rsna41_test'}
exec(compile(module,'<41 reviewed component definitions>','exec'),scope)
with tempfile.TemporaryDirectory(prefix='rsna41_test_',dir=ROOT) as temp:
    root=Path(temp)
    assert root.resolve().is_relative_to(ROOT.resolve())
    src=root/'src'
    src.mkdir()
    worker=src/'compact_infer.py'
    worker.write_bytes(original.encode())
    scope['OUT']=root
    scope['_41_original_prepare_inputs']=lambda:types.SimpleNamespace(src=src)
    scope['_RECEIPT']['inputs']={}
    scope['prepare_inputs']()
    assert hashlib.sha256(worker.read_bytes()).hexdigest()==expected_sha
    assert json.loads((root/'41_worker_patch_audit.json').read_text())['passes']==2
    worker.write_text('changed original source',encoding='utf-8')
    try:scope['prepare_inputs']()
    except scope['K1Error']:pass
    else:raise AssertionError('Changed source accepted before patch')

# Same complete parent harness as39: model stages mocked, export/error handling real.
harness=(ROOT/'validate_39_submit.py').read_text(encoding='utf-8')
harness=harness.replace('39_Anchor0958_Plus_S6_Fixed10_Rank_Submit.ipynb',file)
manifest=json.loads((ROOT/'41_Build_Manifest.json').read_text())
harness=harness.replace("manifest=json.loads((ROOT/'39_Build_Manifest.json').read_text())",
                        'manifest='+repr({'notebook_sha256':manifest['notebook_sha256']}))
harness=harness.replace('39_Local_Validation.json','41_Local_Validation.json')
exec(compile(harness,'<41 inherited full parent tests>','exec'),{'__file__':str(ROOT/'validate_39_submit.py'),'__name__':'__main__'})
report=json.loads((ROOT/'41_Local_Validation.json').read_text())
report['tests']+=['real PyTorch CPU score: exact offset16/24 crop geometry and study-logit averaging',
                  'repeated centre crop exactly reproduces original CPU scoring with toy model',
                  'verify original source before patch; reject changed original source']
report['real_model_gpu_inference_tested']=False
(ROOT/'41_Local_Validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('PASS41 CPU crop geometry, logit fusion, original-source guard and full parent fixtures. No GPU/hidden inference tested.')
