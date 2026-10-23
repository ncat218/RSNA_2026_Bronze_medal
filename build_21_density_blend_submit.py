"""Build a controlled 80/32 B2 standalone test inference notebook."""
from pathlib import Path
from io import BytesIO
import ast
import copy
import json
import zipfile

import pandas as pd


ROOT=Path(__file__).parent
base=json.loads((ROOT/'10_B2_RadImageNet_FrozenMIL_Submit.ipynb').read_text(encoding='utf-8'))
assert len(base['cells'])==5
cells=[copy.deepcopy(c) for c in base['cells']]


def source(i): return ''.join(cells[i]['source'])


def set_code(i,code):
    ast.parse(code,filename=f'notebook_21_cell_{i}')
    cells[i]['source']=code.splitlines(keepends=True)
    cells[i]['execution_count']=None
    cells[i]['outputs']=[]


intro='''# 21 — B2 frozen RadImageNet, fixed 80/32 density blend

One standalone Public-LB probe. Notebook 20 reproduced notebook 09 exactly at 80 slices and found 80/20 probability pooling of the 80- and 32-slice heads improved **weak-label OOF** by 0.00161. That is exploratory and not evidence of a leaderboard improvement. This notebook keeps the protected 0.943 ensemble untouched. Do not tune the 80/20 weight after seeing Public score.

**Attach:** RSNA competition data; notebook 20 **saved Output** containing `density_ablation_report.json` and all `density_80_fold_0_head.pt` through `density_80_fold_4_head.pt` plus the five corresponding density-32 heads; public `resnet-50-radimagenet-marwan` Dataset. Notebook 09 Output and the 05A study cache are **not required** for inference. **Settings:** T4×2, Internet Off. Only GPU 0 is used.

Cell 1 verifies the notebook 20 run was complete, the 80-token control reproduced notebook 09, all ten checkpoint identities match, and the encoder hash matches. Cell 4 checks the unchanged 80-token predictions against the three visible test rows of notebook 10 before writing `submission.csv`. If that reproduction fails, do **not** submit. Run Save & Run All, inspect `b2_density_blend_receipt.json` and `submission.csv`, then submit once if status is COMPLETE. Tell me its Public score and send the Output ZIP. A score gain on standalone B2 does not prove a gain when blended with the 0.943 ensemble.
'''
cells[0]['source']=intro.splitlines(keepends=True)

asset=source(1)
head_start=asset.index('head_roots = sorted(')
head_end=asset.index("print('ASSET CHECK PASSED'",head_start)
head_block='''head_roots = sorted({p.parent for p in shallow_files('density_80_fold_0_head.pt')
                     if all((p.parent/f'density_{d}_fold_{f}_head.pt').is_file()
                            for d in (80,32) for f in range(5))}, key=str)
if len(head_roots) != 1:
    raise RuntimeError(f'Expected one complete notebook 20 Output with 10 heads; found {head_roots}')
HEAD_ROOT = head_roots[0]
REPORT_PATH = HEAD_ROOT/'density_ablation_report.json'
if not REPORT_PATH.is_file(): raise FileNotFoundError('Attach complete notebook 20 saved Output')
REPORT = json.loads(REPORT_PATH.read_text(encoding='utf-8'))
if (REPORT.get('status') != 'COMPLETE' or REPORT.get('densities_completed') != [80,32,16]
    or REPORT.get('n_weak') != 4349 or REPORT.get('n_gold') != 58
    or REPORT.get('encoder_weight_sha256') != CONFIG['encoder_sha256']):
    raise RuntimeError('Notebook 20 report is incomplete or has wrong provenance')
control=REPORT.get('control_fold_auc_deviations')
if not isinstance(control,dict) or set(control) != {str(i) for i in range(5)} or any(
        abs(float(v)) > 1e-8 for v in control.values()):
    raise RuntimeError('Notebook 20 80-token control did not exactly reproduce notebook 09')
CHECKPOINTS = {80:[],32:[]}
for density in (80,32):
    for fold in range(5):
        ck = torch.load(HEAD_ROOT/f'density_{density}_fold_{fold}_head.pt',
                        map_location='cpu',weights_only=True)
        if ck.get('density') != density or ck.get('fold') != fold or ck.get('labels') != LABELS:
            raise RuntimeError(f'Wrong density/fold/label order in checkpoint {density}/{fold}')
        if ck.get('source_weight_sha256') != CONFIG['encoder_sha256']:
            raise RuntimeError(f'Encoder provenance mismatch in checkpoint {density}/{fold}')
        cfg = ck.get('config',{})
        if (cfg.get('experiment') != 'B2_RadImageNet_slice_density_16_32_80_v1'
            or cfg.get('max_tokens') != 80 or cfg.get('feature_dim') != 2048
            or cfg.get('head_dim') != 256 or list(cfg.get('active_folds',[])) != [0,1,2,3,4]):
            raise RuntimeError(f'Wrong notebook 20 head configuration: {density}/{fold}')
        if not isinstance(ck.get('state_dict'),dict) or not ck['state_dict']:
            raise RuntimeError(f'Empty head state: {density}/{fold}')
        CHECKPOINTS[density].append(ck)
'''
asset=asset[:head_start]+head_block+asset[head_end:]
asset=asset.replace("'heads':str(HEAD_ROOT)","'heads':str(HEAD_ROOT),'densities':[80,32]")
set_code(1,asset)

engine=source(3)
engine_start=engine.index('heads = []')
engine_end=engine.index('@torch.inference_mode()',engine_start)
engine=engine[:engine_start]+'''heads = {80:[],32:[]}
for density in (80,32):
    for fold,ck in enumerate(CHECKPOINTS[density]):
        model=StudyMIL()
        model.load_state_dict(ck['state_dict'],strict=True)
        heads[density].append(model.to(DEVICE).eval().requires_grad_(False))
del CHECKPOINTS
gc.collect()
print('MODELS READY',{'gpu':torch.cuda.get_device_name(0),'heads':{d:len(v) for d,v in heads.items()}})

'''+engine[engine_end:]
engine=engine.replace("return uid,np.full(len(LABELS),0.5,np.float32)",
                      "return uid,np.full(len(LABELS),0.5,np.float32),np.full(len(LABELS),0.5,np.float32),np.full(len(LABELS),0.5,np.float32)")
old='''    pred = torch.stack([torch.sigmoid(head(feat,s,p,m))[0] for head in heads]).mean(0)
    out = pred.cpu().numpy().astype(np.float32)
    if out.shape!=(12,) or not np.isfinite(out).all() or out.min()<0 or out.max()>1:
        raise RuntimeError(f'Bad prediction for {uid}')
    STATS['decoded_images'] += n
    return uid,out'''
new='''    # The 32-token arm takes uniform positions from the same successfully decoded
    # globally ordered 80-token sequence, exactly as notebook 20's FeatureDataset.
    take=np.rint(np.linspace(0,n-1,min(n,32))).astype(int)
    if len(np.unique(take)) != len(take): raise RuntimeError(f'Duplicate selected token: {uid}')
    t=len(take)
    idx=torch.as_tensor(take,device=DEVICE,dtype=torch.long)
    feat32=torch.zeros((1,32,CONFIG['feature_dim']),device=DEVICE,dtype=torch.float32)
    s32=torch.zeros((1,32),device=DEVICE,dtype=torch.long)
    p32=torch.zeros((1,32),device=DEVICE,dtype=torch.float32)
    m32=torch.zeros((1,32),device=DEVICE,dtype=torch.bool)
    feat32[0,:t]=feat[0,idx]; s32[0,:t]=s[0,idx]; p32[0,:t]=p[0,idx]; m32[0,:t]=True
    pred80=torch.stack([torch.sigmoid(head(feat,s,p,m))[0] for head in heads[80]]).mean(0)
    pred32=torch.stack([torch.sigmoid(head(feat32,s32,p32,m32))[0] for head in heads[32]]).mean(0)
    pred=.8*pred80+.2*pred32
    out=pred.cpu().numpy().astype(np.float32)
    base=pred80.cpu().numpy().astype(np.float32)
    aux=pred32.cpu().numpy().astype(np.float32)
    for arr in (out,base,aux):
        if arr.shape!=(12,) or not np.isfinite(arr).all() or arr.min()<0 or arr.max()>1:
            raise RuntimeError(f'Bad prediction for {uid}')
    STATS['decoded_images'] += n
    return uid,out,base,aux'''
assert old in engine
engine=engine.replace(old,new)
set_code(3,engine)

with zipfile.ZipFile(ROOT/'results_10.zip') as archive:
    old_visible=pd.read_csv(BytesIO(archive.read('submission.csv')),
                            dtype={'StudyInstanceUID':str})
reference={str(row['StudyInstanceUID']):[float(row[label]) for label in
    ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
     'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']]
    for _,row in old_visible.iterrows()}

finish='''REFERENCE_10_VISIBLE = __REFERENCE__
predictions=np.empty((len(manifest),len(LABELS)),np.float32)
base80=np.empty_like(predictions); aux32=np.empty_like(predictions)
smoke_n=min(3,len(manifest))
for i in range(smoke_n):
    uid,pred,base,aux=predict_study(manifest.iloc[i])
    if uid!=str(manifest.iloc[i].StudyInstanceUID): raise RuntimeError('Smoke identity drift')
    predictions[i]=pred; base80[i]=base; aux32[i]=aux
torch.cuda.synchronize(0)
print('VISIBLE SMOKE PASSED',{'studies':smoke_n,'elapsed_minutes':round((monotonic()-START)/60,2),
                              'probability_range':(float(predictions[:smoke_n].min()),
                                                   float(predictions[:smoke_n].max()))},flush=True)

for i in range(smoke_n,len(manifest)):
    uid,pred,base,aux=predict_study(manifest.iloc[i])
    if uid!=str(manifest.iloc[i].StudyInstanceUID): raise RuntimeError(f'Test identity drift at {i}')
    predictions[i]=pred; base80[i]=base; aux32[i]=aux
    if (i+1)%25==0 or i+1==len(manifest):
        torch.cuda.synchronize(0)
        elapsed=monotonic()-START
        print(f'predicted {i+1}/{len(manifest)}; elapsed={elapsed/60:.1f} min; '
              f'mean_seconds_per_study={elapsed/(i+1):.2f}',flush=True)

if not np.isfinite(predictions).all() or predictions.min()<0 or predictions.max()>1:
    raise RuntimeError('Invalid full prediction matrix')
pred=pd.DataFrame(predictions,columns=LABELS)
pred.insert(0,'StudyInstanceUID',manifest.StudyInstanceUID.astype(str).tolist())
submission=sample[['StudyInstanceUID']].merge(pred,on='StudyInstanceUID',how='left',
                                               sort=False,validate='one_to_one')
if (list(submission.columns)!=list(sample.columns)
    or list(submission.StudyInstanceUID)!=list(sample.StudyInstanceUID)
    or submission[LABELS].isna().any().any()):
    raise RuntimeError('Submission schema/order/coverage mismatch')

# On the three visible test IDs of notebook 10, verify the unchanged 80-token
# branch before allowing a changed standalone submission. Hidden IDs are not
# expected to match this local reference and are not gated by it.
matched=[i for i,uid in enumerate(manifest.StudyInstanceUID.astype(str))
         if uid in REFERENCE_10_VISIBLE]
if matched:
    deviations=[float(np.max(np.abs(base80[i]-np.asarray(
                REFERENCE_10_VISIBLE[str(manifest.iloc[i].StudyInstanceUID)],np.float32))))
                for i in matched]
    max_control_diff=max(deviations)
    if max_control_diff > 1e-4:
        raise RuntimeError(f'80-token inference differs from notebook 10: {max_control_diff}')
else:
    max_control_diff=None
if len(manifest)==3 and len(matched)!=3:
    raise RuntimeError('Visible three-study test IDs differ from notebook 10 reference')

def write_matrix(values,name):
    table=pd.DataFrame(values,columns=LABELS)
    table.insert(0,'StudyInstanceUID',manifest.StudyInstanceUID.astype(str).tolist())
    table=sample[['StudyInstanceUID']].merge(table,on='StudyInstanceUID',how='left',
                                             sort=False,validate='one_to_one')
    if list(table.columns)!=list(sample.columns) or table[LABELS].isna().any().any():
        raise RuntimeError(f'Schema/coverage mismatch in {name}')
    table.to_csv(WORKING/name,index=False)
    return table

write_matrix(base80,'b2_density80_control.csv')
write_matrix(aux32,'b2_density32_control.csv')
path=WORKING/'submission.csv'
submission.to_csv(path,index=False)
roundtrip=pd.read_csv(path,dtype={'StudyInstanceUID':str})
if (roundtrip.shape!=sample.shape or list(roundtrip.columns)!=list(sample.columns)
    or list(roundtrip.StudyInstanceUID)!=list(sample.StudyInstanceUID)
    or not np.isfinite(roundtrip[LABELS].to_numpy(float)).all()):
    raise RuntimeError('Saved submission roundtrip failed')

receipt={'status':'COMPLETE','experiment':'B2_density_80_32_fixed_80_20_standalone',
         'test_studies':len(manifest),'folds':[0,1,2,3,4],
         'encoder_weight_sha256':sha.hexdigest(),'head_root':str(HEAD_ROOT),
         'density_weights':{'80':0.8,'32':0.2},
         'visible_control_studies_matched':len(matched),
         'visible_control_max_probability_diff':max_control_diff,
         'elapsed_minutes':(monotonic()-START)/60,
         'probability_min':float(predictions.min()),
         'probability_max':float(predictions.max()),
         'dicom_stats':dict(STATS),
         'decode_failures_by_transfer_syntax':dict(DECODE_FAILURES),
         'warning':'Standalone diagnostic probe. Protected 0.943 ensemble is unchanged.'}
(WORKING/'b2_density_blend_receipt.json').write_text(
    json.dumps(receipt,indent=2),encoding='utf-8')
print('SUBMISSION_READY = YES',{'shape':submission.shape,
                                'runtime_min':round(receipt['elapsed_minutes'],2),
                                'empty_studies':STATS['empty_studies'],
                                'visible_control_max_diff':max_control_diff})
display(submission.head())
'''.replace('__REFERENCE__',repr(reference))
set_code(4,finish)

out=copy.deepcopy(base)
out['cells']=cells
out['metadata']['kaggle']['title']='21 B2 density 80 32 fixed blend standalone'
out['metadata']['kaggle']['description']='Controlled standalone 80/20 probability blend from notebook 20 heads'
dst=ROOT/'21_B2_Density80_32_FixedBlend_Submit.ipynb'
dst.write_text(json.dumps(out,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print(dst)
