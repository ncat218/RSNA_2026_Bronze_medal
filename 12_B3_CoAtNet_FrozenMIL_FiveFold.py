# %% [markdown]
# # 12 — B3 CoAtNet frozen encoder + study MIL, five-fold group CV
#
# Controlled strong-backbone successor to B2 (Public LB 0.914). Same 05A
# 224×224 uint8 cache, silver L1 labels, exact report-group folds, MIL head,
# optimizer and stopping rule as B2; the changed factor is the ImageNet-12k/1k
# pretrained CoAtNet encoder and its model-specific normalization. Gold58 is
# audit-only and never enters gradients, epoch selection, folds, or tuning.
#
# Attach **four inputs**: RSNA competition; notebook 02 Output containing
# `L1_family_mean_RECOMMENDED.csv`; `Study_case 05_RSNA` Dataset containing
# `rsna_05a_dense_mil/study_cache` and `fold_assignments.csv`; successful
# notebook 11 Output containing `rsna_b3_coatnet_preflight/b3_preflight.json`
# and its `model.safetensors`. Set **T4×2, Internet Off**. This notebook uses
# GPU 0 only and makes **no submission**. Run Save & Run All only after cell 1
# prints `ASSET CHECK PASSED`. Send the Output ZIP containing `run_report.json`,
# `weak_oof.csv`, `gold_metrics.csv`, and five `fold_*_head.pt` files.
#
# A complete preflight is evidence of technical feasibility, not evidence B3
# beats 0.914 or the protected 0.943 ensemble. Do not submit or tune a blend
# until the full OOF comparison is audited.

# %%
from pathlib import Path
from time import monotonic
import copy
import gc
import hashlib
import json
import os
import random

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import timm
from safetensors.torch import load_file

LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
          'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
CONFIG = {
    'experiment': 'B3_CoAtNet_ImageNet12k_1k_frozen_MIL_fivefold_v1',
    'seed': 111, 'active_folds': [0,1,2,3,4],
    'model_name': 'coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k',
    'encoder_sha256': '12e9f7a9707028b2125f03c389f80d8d3202dd830b381f309a9f28c7a3eeb59f',
    'image_microbatch': 4, 'max_tokens': 80, 'feature_dim': 768,
    'head_dim': 256, 'head_batch': 32,
    'max_epochs': 20, 'patience': 4, 'head_lr': 3e-4, 'weight_decay': 1e-4,
    'min_available_ram_gib': 5.0, 'max_gpu_reserved_gib': 12.0,
    'min_early_encoder_images_per_sec': 60.0, 'max_total_hours': 2.5,
}
INPUT = Path('/kaggle/input')
OUT = Path('/kaggle/working/rsna_b3_coatnet_frozen5')
OUT.mkdir(parents=True, exist_ok=True)
START = monotonic()
random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
torch.backends.cudnn.benchmark = False

def shallow_files(name, max_depth=9):
    found = []
    for base, dirs, files in os.walk(INPUT):
        depth = len(Path(base).relative_to(INPUT).parts)
        dirs[:] = [d for d in dirs if d not in {'train_series','test_series','study_cache','dicom','dicoms'}]
        if name in files:
            found.append(Path(base) / name)
        if depth >= max_depth:
            dirs[:] = []
    return sorted(found, key=str)

def cache_directories(max_depth=9):
    found = []
    for base, dirs, _ in os.walk(INPUT):
        depth = len(Path(base).relative_to(INPUT).parts)
        dirs[:] = [d for d in dirs if d not in {'train_series','test_series','dicom','dicoms'}]
        if 'study_cache' in dirs:
            found.append(Path(base) / 'study_cache')
            dirs.remove('study_cache')
        if depth >= max_depth:
            dirs[:] = []
    return found

cache_counts = []
for directory in cache_directories():
    names = [entry.name for entry in os.scandir(directory) if entry.is_file()]
    cache_counts.append((directory,
                         sum(n.startswith('weak_') and n.endswith('.npz') for n in names),
                         sum(n.startswith('gold_') and n.endswith('.npz') for n in names)))
valid_cache = [x for x in cache_counts if x[1] >= 4349 and x[2] >= 58]
if not valid_cache:
    raise FileNotFoundError('Attach complete Study_case 05_RSNA cache: '
                            f'{[(str(p),w,g) for p,w,g in cache_counts]}')
CACHE = sorted(valid_cache, key=lambda x: (-x[1],str(x[0])))[0][0]

receipts = []
for path in shallow_files('b3_preflight.json'):
    try:
        item = json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):
        continue
    if item.get('status') != 'COMPLETE':
        continue
    if item.get('config',{}).get('model_name') != CONFIG['model_name']:
        continue
    if item.get('weight_sha256') != CONFIG['encoder_sha256']:
        continue
    if item.get('cache_weak_studies',0) < 4349 or item.get('cache_gold_studies',0) < 58:
        continue
    weight = path.parent / Path(item.get('saved_weight_path','model.safetensors')).name
    if weight.is_file():
        receipts.append((path,weight,item))
if not receipts:
    raise FileNotFoundError('Attach successful notebook 11 Output with '
                            'rsna_b3_coatnet_preflight/b3_preflight.json and model.safetensors')
PREFLIGHT_PATH, WEIGHT, PREFLIGHT = sorted(receipts,key=lambda x:str(x[0]))[0]
digest = hashlib.sha256()
with WEIGHT.open('rb') as handle:
    for block in iter(lambda:handle.read(8<<20),b''):
        digest.update(block)
if digest.hexdigest() != CONFIG['encoder_sha256']:
    raise RuntimeError('B3 pretrained weight SHA256 mismatch; no GPU work started')
if WEIGHT.stat().st_size != int(PREFLIGHT['weight_bytes']):
    raise RuntimeError('B3 pretrained weight size mismatch')
if (PREFLIGHT.get('sample_studies_processed') != 64 or
        PREFLIGHT.get('empty_studies') != 0 or
        PREFLIGHT.get('feature_dim') != CONFIG['feature_dim']):
    raise RuntimeError('Notebook 11 preflight receipt is not the successful 64-study gate')

COMP = next((p for p in [INPUT/'competitions/rsna-knee-abnormality-detection',
                         INPUT/'rsna-knee-abnormality-detection']
             if (p/'train.csv').is_file()),None)
if COMP is None:
    raise FileNotFoundError('Attach RSNA Knee Abnormality Detection competition input')
TRAIN = pd.read_csv(COMP/'train.csv',dtype={'StudyInstanceUID':str})
if len(TRAIN) != 4407 or not TRAIN.StudyInstanceUID.is_unique:
    raise RuntimeError('Unexpected train.csv identities')
gold = TRAIN.loc[TRAIN[LABELS].notna().all(axis=1),['StudyInstanceUID']+LABELS].copy()
if len(gold) != 58:
    raise RuntimeError('Expected exactly 58 official gold studies')
silver_files = [p for p in shallow_files('L1_family_mean_RECOMMENDED.csv')
                if set(['StudyInstanceUID']+LABELS).issubset(pd.read_csv(p,nrows=1).columns)]
if not silver_files:
    raise FileNotFoundError('Attach notebook 02 Output containing L1_family_mean_RECOMMENDED.csv')
SILVER_PATH = silver_files[0]
silver = pd.read_csv(SILVER_PATH,dtype={'StudyInstanceUID':str})
if len(silver) != 4349 or not silver.StudyInstanceUID.is_unique:
    raise RuntimeError('Silver labels have unexpected identities/count')
if set(silver.StudyInstanceUID) & set(gold.StudyInstanceUID):
    raise RuntimeError('Gold studies leaked into weak silver labels')
if set(silver.StudyInstanceUID) | set(gold.StudyInstanceUID) != set(TRAIN.StudyInstanceUID):
    raise RuntimeError('Silver+gold does not cover train.csv exactly')
silver_y = silver[LABELS].to_numpy(np.float32)
if not np.isfinite(silver_y).all() or not ((silver_y>=0)&(silver_y<=1)).all():
    raise RuntimeError('Silver labels must be finite probabilities in [0,1]')

fold_files = [p for p in shallow_files('fold_assignments.csv')
              if set(['StudyInstanceUID','report_group','fold']).issubset(pd.read_csv(p,nrows=1).columns)]
fold_files = sorted(fold_files,key=lambda p:(str(CACHE.parent.parent) not in str(p),str(p)))
if not fold_files:
    raise FileNotFoundError('Attach 05A fold_assignments.csv with report_group')
FOLD_PATH = fold_files[0]
folds = pd.read_csv(FOLD_PATH,dtype={'StudyInstanceUID':str})[['StudyInstanceUID','report_group','fold']]
if not folds.StudyInstanceUID.is_unique:
    raise RuntimeError('Duplicate fold StudyInstanceUID')
weak = silver[['StudyInstanceUID']+LABELS].merge(folds,on='StudyInstanceUID',how='left',validate='one_to_one')
if weak.fold.isna().any() or weak.report_group.isna().any():
    raise RuntimeError('Missing weak fold/report_group')
weak['fold'] = weak.fold.astype(int)
if set(weak.fold) != set(range(5)):
    raise RuntimeError('Expected unchanged 05A folds 0–4')
if weak.fold.value_counts().sort_index().tolist() != [870,870,870,870,869]:
    raise RuntimeError('05A fold counts changed; B2 comparison would not be paired')
for fold in range(5):
    train_groups = set(weak.loc[weak.fold.ne(fold),'report_group'].astype(str))
    val_groups = set(weak.loc[weak.fold.eq(fold),'report_group'].astype(str))
    if not train_groups.isdisjoint(val_groups):
        raise RuntimeError(f'Report-group leakage in fold {fold}')

IDS = weak.StudyInstanceUID.astype(str).tolist()+gold.StudyInstanceUID.astype(str).tolist()
Y = np.vstack([weak[LABELS].to_numpy(np.float32),gold[LABELS].to_numpy(np.float32)])
if Y.shape != (4407,12):
    raise RuntimeError('Target matrix shape drift')
missing_cache = [uid for i,uid in enumerate(IDS)
                 if not (CACHE/f'{"weak" if i<len(weak) else "gold"}_{uid}.npz').is_file()]
if missing_cache:
    raise FileNotFoundError(f'{len(missing_cache)} cache studies missing; first {missing_cache[:3]}')
print('ASSET CHECK PASSED',{'weak':len(weak),'gold':len(gold),
                            'folds':weak.fold.value_counts().sort_index().to_dict(),
                            'cache':str(CACHE),'silver':str(SILVER_PATH),
                            'folds_path':str(FOLD_PATH),'preflight':str(PREFLIGHT_PATH),
                            'weight_sha256':digest.hexdigest(),
                            'weight_bytes':WEIGHT.stat().st_size,
                            'preflight_images_per_second':PREFLIGHT['images_per_second']},flush=True)

# %%
if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
    raise RuntimeError('Asset check passed; select GPU T4×2 before feature extraction')
DEVICE = torch.device('cuda:0')
feature_bytes = len(IDS)*CONFIG['max_tokens']*CONFIG['feature_dim']*2
available = psutil.virtual_memory().available
if available < feature_bytes + 8*2**30:
    raise RuntimeError(f'Not enough host RAM before feature allocation: {available/2**30:.2f} GiB')
features = np.zeros((len(IDS),CONFIG['max_tokens'],CONFIG['feature_dim']),np.float16)
slot_ids = np.zeros((len(IDS),CONFIG['max_tokens']),np.int64)
positions = np.zeros((len(IDS),CONFIG['max_tokens']),np.float32)
masks = np.zeros((len(IDS),CONFIG['max_tokens']),bool)

encoder = timm.create_model(CONFIG['model_name'],pretrained=False)
state = load_file(str(WEIGHT),device='cpu')
expected = encoder.state_dict()
missing = sorted(set(expected)-set(state))
extra = sorted(set(state)-set(expected))
bad_shapes = sorted(k for k in set(expected)&set(state)
                    if tuple(expected[k].shape)!=tuple(state[k].shape))
if missing or extra or bad_shapes:
    raise RuntimeError('CoAtNet/timm weight mismatch; refusing random initialization: '
                       f'missing={missing[:8]}, extra={extra[:8]}, shapes={bad_shapes[:8]}')
encoder.load_state_dict(state,strict=True)
del state,expected
data_config = timm.data.resolve_model_data_config(encoder)
if (tuple(data_config['input_size']) != (3,224,224) or
        tuple(data_config['mean']) != (0.5,0.5,0.5) or
        tuple(data_config['std']) != (0.5,0.5,0.5)):
    raise RuntimeError(f'CoAtNet normalization/input drift vs preflight: {data_config}')
encoder.reset_classifier(0)
if int(encoder.num_features) != CONFIG['feature_dim']:
    raise RuntimeError(f'CoAtNet feature_dim drift: {encoder.num_features}')
encoder = encoder.to(DEVICE).eval().requires_grad_(False)
mean = torch.tensor(data_config['mean'],device=DEVICE).view(1,3,1,1)
std = torch.tensor(data_config['std'],device=DEVICE).view(1,3,1,1)
gc.collect()
torch.cuda.reset_peak_memory_stats(0)
encode_start = monotonic()
n_images = 0
empty = 0
with torch.inference_mode():
    for i,uid in enumerate(IDS):
        split = 'weak' if i<len(weak) else 'gold'
        path = CACHE/f'{split}_{uid}.npz'
        with np.load(path,allow_pickle=False) as z:
            if not {'images','slot_ids','positions','slice_indices'}.issubset(z.files):
                raise RuntimeError(f'Invalid 05A cache keys: {path}')
            images = z['images']; s = z['slot_ids']; pos = z['positions']
            if images.dtype!=np.uint8 or images.ndim!=3 or images.shape[1:]!=(224,224):
                raise RuntimeError(f'Invalid 05A image shape/dtype: {path}')
            t = len(images)
            if t>CONFIG['max_tokens'] or len(s)!=t or len(pos)!=t or len(z['slice_indices'])!=t:
                raise RuntimeError(f'Invalid 05A token metadata: {path}')
            if t and (s.min()<0 or s.max()>=5 or not np.isfinite(pos).all()):
                raise RuntimeError(f'Invalid 05A slot/position metadata: {path}')
        if t==0:
            empty += 1
        slot_ids[i,:t]=s; positions[i,:t]=pos; masks[i,:t]=True; n_images+=t
        for offset in range(0,t,CONFIG['image_microbatch']):
            part = images[offset:offset+CONFIG['image_microbatch']]
            gray = torch.from_numpy(part).to(DEVICE).float().div_(255.0)
            rgb = gray.unsqueeze(1).expand(-1,3,-1,-1).contiguous()
            rgb = (rgb-mean)/std
            with torch.autocast(device_type='cuda',dtype=torch.float16):
                value = encoder(rgb)
            if value.shape!=(len(part),CONFIG['feature_dim']) or not bool(torch.isfinite(value).all()):
                raise RuntimeError(f'Nonfinite/wrong CoAtNet feature: {path}')
            half = value.to(torch.float16).cpu().numpy()
            if not np.isfinite(half).all():
                raise RuntimeError(f'Feature float16 overflow: {path}')
            features[i,offset:offset+len(part)] = half
            del gray,rgb,value,half
        del images,s,pos
        if i==63:
            torch.cuda.synchronize(0)
            early_rate = n_images/max(1e-6,monotonic()-encode_start)
            print('EARLY ENCODER RATE',round(early_rate,2),'images/sec',flush=True)
            if early_rate < CONFIG['min_early_encoder_images_per_sec']:
                raise RuntimeError(f'Early encoder rate {early_rate:.1f} images/sec below safety '
                                   'threshold; stopping before a long GPU run')
        if (i+1)%250==0:
            torch.cuda.synchronize(0)
            elapsed = monotonic()-encode_start
            reserved = torch.cuda.max_memory_reserved(0)/2**30
            available_gib = psutil.virtual_memory().available/2**30
            print(f'encoded {i+1}/{len(IDS)} studies; {n_images} images; '
                  f'{elapsed:.1f}s; peak reserved {reserved:.3f} GiB; '
                  f'host available {available_gib:.1f} GiB',flush=True)
            if reserved > CONFIG['max_gpu_reserved_gib']:
                raise RuntimeError('GPU memory safety threshold reached during encoding')
            if available_gib < CONFIG['min_available_ram_gib']:
                raise RuntimeError('Host RAM safety threshold reached during encoding')
            if (monotonic()-START)/3600 > CONFIG['max_total_hours']:
                raise RuntimeError('Overall configured time guard reached during encoding')
torch.cuda.synchronize(0)
encode_seconds = monotonic()-encode_start
encoder_peak_gib = torch.cuda.max_memory_allocated(0)/2**30
encoder_reserved_gib = torch.cuda.max_memory_reserved(0)/2**30
del encoder,mean,std
gc.collect(); torch.cuda.empty_cache()
if empty:
    raise RuntimeError(f'{empty} empty studies in 05A cache; do not train silently')
if not np.isfinite(features[masks]).all():
    raise RuntimeError('Nonfinite feature array')
print('FEATURE EXTRACTION COMPLETE',{'seconds':round(encode_seconds,1),'images':n_images,
                                     'empty_studies':empty,
                                     'host_feature_gib':round(feature_bytes/2**30,3),
                                     'encoder_peak_gpu_gib':round(encoder_peak_gib,3),
                                     'encoder_reserved_gpu_gib':round(encoder_reserved_gib,3)},flush=True)

# %%
class FeatureDataset(Dataset):
    def __init__(self,indices): self.indices=np.asarray(indices,dtype=int)
    def __len__(self): return len(self.indices)
    def __getitem__(self,k):
        i=int(self.indices[k])
        return (i,torch.from_numpy(features[i]),torch.from_numpy(slot_ids[i]),
                torch.from_numpy(positions[i]),torch.from_numpy(masks[i]),torch.from_numpy(Y[i]))

class StudyMIL(nn.Module):
    def __init__(self):
        super().__init__(); d=CONFIG['head_dim']; fd=CONFIG['feature_dim']
        self.project=nn.Sequential(nn.LayerNorm(fd),nn.Linear(fd,d),nn.GELU(),nn.Dropout(.10))
        self.slot=nn.Embedding(5,d)
        self.position=nn.Sequential(nn.Linear(1,d),nn.GELU(),nn.Linear(d,d))
        self.query=nn.Parameter(torch.randn(12,d)*.02)
        self.attention=nn.MultiheadAttention(d,4,dropout=.10,batch_first=True)
        self.norm=nn.LayerNorm(d)
        self.weight=nn.Parameter(torch.randn(12,d)*.02); self.bias=nn.Parameter(torch.zeros(12))
    def forward(self,x,s,p,m):
        m=m.clone(); m[~m.any(1),0]=True
        token=self.project(x.float())+self.slot(s)+self.position(p.unsqueeze(-1))
        q=self.query.unsqueeze(0).expand(len(x),-1,-1)
        pooled,_=self.attention(q,token,token,key_padding_mask=~m,need_weights=False)
        return (self.norm(pooled)*self.weight.unsqueeze(0)).sum(-1)+self.bias

def loader(indices,shuffle=False,seed=0):
    generator=torch.Generator().manual_seed(CONFIG['seed']+seed)
    return DataLoader(FeatureDataset(indices),batch_size=CONFIG['head_batch'],shuffle=shuffle,
                      generator=generator,num_workers=0,pin_memory=False,drop_last=False)

def pass_batches(model,data,optimizer=None):
    training=optimizer is not None
    model.train(training)
    ids=[]; probs=[]; loss_sum=0.0; count=0
    context=torch.enable_grad() if training else torch.inference_mode()
    with context:
        for ix,x,s,p,m,y in data:
            x=x.to(DEVICE).float(); s=s.to(DEVICE); p=p.to(DEVICE); m=m.to(DEVICE); y=y.to(DEVICE)
            logits=model(x,s,p,m)
            loss=F.binary_cross_entropy_with_logits(logits,y)
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            loss_sum+=float(loss.detach())*len(ix); count+=len(ix)
            if not training:
                ids.extend(ix.tolist()); probs.append(torch.sigmoid(logits).cpu().numpy())
    return loss_sum/max(1,count),np.asarray(ids,dtype=int),np.concatenate(probs) if probs else None

def target_auc(y,p):
    result=[]
    for j in range(12):
        yy=y[:,j]
        result.append(float(roc_auc_score(yy,p[:,j])) if len(np.unique(yy))==2 else np.nan)
    return result

def json_safe(value):
    if isinstance(value,dict): return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value): return None
    if isinstance(value,np.integer): return int(value)
    return value

history=[]; cv_rows=[]; gold_rows=[]; oof_parts=[]; gold_parts=[]
gold_indices=np.arange(len(weak),len(IDS))
gold_loader=loader(gold_indices)
for fold in CONFIG['active_folds']:
    if (monotonic()-START)/3600 > CONFIG['max_total_hours']:
        print('TIME GUARD: stopping before next fold'); break
    torch.manual_seed(CONFIG['seed']+fold)
    tr=np.flatnonzero(weak.fold.to_numpy()!=fold)
    va=np.flatnonzero(weak.fold.to_numpy()==fold)
    if not set(weak.iloc[tr].report_group.astype(str)).isdisjoint(
            set(weak.iloc[va].report_group.astype(str))):
        raise RuntimeError(f'Report-group leakage in fold {fold}')
    tr_loader=loader(tr,shuffle=True,seed=fold); va_loader=loader(va)
    model=StudyMIL().to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['head_lr'],
                                weight_decay=CONFIG['weight_decay'])
    best_bce=float('inf'); best_epoch=0; best_state=None; stale=0; fold_stop_reason=None
    for epoch in range(1,CONFIG['max_epochs']+1):
        if (monotonic()-START)/3600 > CONFIG['max_total_hours']:
            fold_stop_reason='time_guard'; print('TIME GUARD: ending fold with best checkpoint'); break
        t0=monotonic()
        train_bce,_,_=pass_batches(model,tr_loader,optimizer)
        val_bce,vi,vp=pass_batches(model,va_loader)
        auc=target_auc((Y[vi]>=.5).astype(int),vp)
        row={'fold':fold,'epoch':epoch,'train_bce':train_bce,'val_bce':val_bce,
             'diagnostic_weak_macro_auc':float(np.nanmean(auc)),'seconds':monotonic()-t0}
        history.append(row)
        pd.DataFrame(history).to_csv(OUT/'training_history.csv',index=False)
        print(row,flush=True)
        if val_bce < best_bce-1e-5:
            best_bce=val_bce; best_epoch=epoch; stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
        if psutil.virtual_memory().available/2**30 < CONFIG['min_available_ram_gib']:
            fold_stop_reason='host_ram_guard'; print('HOST RAM GUARD: ending fold'); break
        if stale>=CONFIG['patience']:
            break
    if best_state is None:
        raise RuntimeError(f'Fold {fold} produced no checkpoint')
    model.load_state_dict(best_state)
    torch.save({'state_dict':best_state,'fold':fold,'best_epoch':best_epoch,
                'best_val_bce':best_bce,'config':CONFIG,'labels':LABELS,
                'source_weight_sha256':digest.hexdigest(),
                'pretrained_data_config':data_config},OUT/f'fold_{fold}_head.pt')
    val_bce,vi,vp=pass_batches(model,va_loader)
    if len(vi)!=len(va) or set(vi)!=set(va):
        raise RuntimeError(f'Fold {fold} OOF identity mismatch')
    wauc=target_auc((Y[vi]>=.5).astype(int),vp)
    cv_rows.append({'fold':fold,'best_epoch':best_epoch,'val_bce':val_bce,
                    'stop_reason':fold_stop_reason,
                    'weak_macro_auc_proxy':float(np.nanmean(wauc)),'n_val':len(vi)})
    part=pd.DataFrame(vp,columns=LABELS)
    part.insert(0,'StudyInstanceUID',[IDS[i] for i in vi]); part['fold']=fold
    oof_parts.append(part)
    pd.concat(oof_parts).to_csv(OUT/'weak_oof.csv',index=False)
    _,gi,gp=pass_batches(model,gold_loader)
    if list(gi)!=list(gold_indices):
        raise RuntimeError('Gold audit identity mismatch')
    gauc=target_auc(Y[gi].astype(int),gp)
    gold_rows.append({'fold':fold,'gold_macro_auc_audit_only':float(np.nanmean(gauc)),
                      **{f'auc_{LABELS[j]}':gauc[j] for j in range(12)}})
    gold_parts.append(gp)
    pd.DataFrame(cv_rows).to_csv(OUT/'cv_metrics.csv',index=False)
    pd.DataFrame(gold_rows).to_csv(OUT/'gold_metrics.csv',index=False)
    del model,optimizer,best_state,tr_loader,va_loader
    gc.collect(); torch.cuda.empty_cache()

if gold_parts:
    mean_gold=np.mean(np.stack(gold_parts),axis=0)
    result=pd.DataFrame(mean_gold,columns=LABELS)
    result.insert(0,'StudyInstanceUID',gold.StudyInstanceUID.tolist())
    result.to_csv(OUT/'gold_predictions.csv',index=False)
    ensemble_auc=target_auc(gold[LABELS].to_numpy(int),mean_gold)
else:
    ensemble_auc=[np.nan]*12
completed_cleanly=(len(cv_rows)==len(CONFIG['active_folds']) and
                   all(row['stop_reason'] is None for row in cv_rows))
if completed_cleanly:
    full_oof = pd.concat(oof_parts,ignore_index=True)
    if (len(full_oof)!=len(weak) or not full_oof.StudyInstanceUID.is_unique or
            set(full_oof.StudyInstanceUID)!=set(weak.StudyInstanceUID) or
            len(gold_parts)!=5 or
            not all((OUT/f'fold_{fold}_head.pt').is_file() for fold in range(5))):
        raise RuntimeError('Five-fold output integrity failed; refusing COMPLETE status')
report={
    'status':'COMPLETE' if completed_cleanly else 'PARTIAL_GUARD',
    'config':CONFIG,'n_weak':len(weak),'n_gold':len(gold),
    'n_features':len(IDS),'n_images_encoded':n_images,'empty_studies':empty,
    'encode_seconds':encode_seconds,'encoder_peak_gpu_gib':encoder_peak_gib,
    'encoder_reserved_gpu_gib':encoder_reserved_gib,
    'host_feature_gib':feature_bytes/2**30,
    'total_elapsed_minutes':(monotonic()-START)/60,
    'folds_completed':[r['fold'] for r in cv_rows],
    'cv_metrics':cv_rows,'gold_metrics':gold_rows,
    'gold_mean_probability_macro_auc_audit_only':float(np.nanmean(ensemble_auc)),
    'gold_mean_probability_per_target_auc_audit_only':dict(zip(LABELS,ensemble_auc)),
    'silver_path':str(SILVER_PATH),'fold_path':str(FOLD_PATH),
    'cache_path':str(CACHE),'preflight_path':str(PREFLIGHT_PATH),
    'encoder_weight_path':str(WEIGHT),'encoder_weight_sha256':digest.hexdigest(),
    'pretrained_data_config':data_config,'timm_version':timm.__version__,
    'warning':'Gold58 is audit-only; Public LB 0.914/0.943 does not justify blend selection',
}
(OUT/'run_report.json').write_text(json.dumps(json_safe(report),indent=2,allow_nan=False),
                                   encoding='utf-8')
print('RUN REPORT:',json.dumps({'status':report['status'],
      'encode_minutes':round(encode_seconds/60,1),
      'folds_completed':report['folds_completed'],
      'cv_metrics':cv_rows,
      'gold_macro_auc_audit_only':report['gold_mean_probability_macro_auc_audit_only'],
      'elapsed_minutes':round(report['total_elapsed_minutes'],1)},indent=2),flush=True)
