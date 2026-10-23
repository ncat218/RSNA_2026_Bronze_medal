# %% [markdown]
# # 10 — B2 RadImageNet frozen MIL, five-fold standalone submission
#
# Controlled Public-LB probe. This notebook does not modify the 0.943 ensemble.
# Attach the RSNA competition, the exact successful **five-fold notebook 09 Output**
# containing `rsna_b2_rad_frozen_fold01/fold_0_head.pt` through `fold_4_head.pt`,
# and the public `resnet-50-radimagenet-marwan` Dataset. Internet Off; T4×2.
# The training cache Dataset and silver labels are **not** required for inference.
#
# Cell 1 verifies attachments and model provenance before any DICOM decoding.
# The test preprocessing below reproduces 05A's deterministic five-slot selection,
# 16 slices/slot, 140 mm physical crop and 224×224 uint8 representation. The frozen
# encoder uses exactly the notebook 09 normalization and float16 feature quantization.
# Run `Save & Run All`; inspect `b2_submit_receipt.json` and `submission.csv` before
# using a competition submission slot. Runtime and Public score are not guaranteed.

# %%
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from time import monotonic
import gc, hashlib, json, os, warnings
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

warnings.filterwarnings('ignore', category=RuntimeWarning)
START = monotonic()
INPUT = Path('/kaggle/input')
WORKING = Path('/kaggle/working')
LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
          'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
SLOTS = ['SAG_FLUID','SAG_NONFLUID','COR_FLUID','COR_NONFLUID','AX_FLUID']
ROUTING = {'SAG_FLUID':('Sagittal',1),'SAG_NONFLUID':('Sagittal',0),
           'COR_FLUID':('Coronal',1),'COR_NONFLUID':('Coronal',0),'AX_FLUID':('Axial',1)}
CONFIG = {'image_size':224,'physical_crop_mm':140.0,'slices_per_slot':16,
          'max_tokens':80,'image_microbatch':24,'feature_dim':2048,'head_dim':256,
          'encoder_sha256':'08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734'}
STATS = Counter()
DECODE_FAILURES = Counter()

def shallow_files(name, max_depth=8):
    found = []
    for base, dirs, files in os.walk(INPUT):
        depth = len(Path(base).relative_to(INPUT).parts)
        dirs[:] = [d for d in dirs if d not in {'train_series','test_series','study_cache','dicom','dicoms'}]
        if name in files: found.append(Path(base)/name)
        if depth >= max_depth: dirs[:] = []
    return sorted(found, key=str)

comp_candidates = [INPUT/'competitions/rsna-knee-abnormality-detection',
                   INPUT/'rsna-knee-abnormality-detection']
comp_candidates += [p.parent for p in shallow_files('test.csv')]
COMP = next((p for p in comp_candidates if all((p/n).exists() for n in
             ['test.csv','test_series.csv','sample_submission.csv','test_series'])), None)
if COMP is None: raise FileNotFoundError('Attach RSNA competition input')

weight_candidates = [INPUT/'datasets/marwanmath/resnet-50-radimagenet-marwan/ResNet50.pt',
                     INPUT/'resnet-50-radimagenet-marwan/ResNet50.pt']
WEIGHT = next((p for p in weight_candidates if p.is_file()), None)
if WEIGHT is None: raise FileNotFoundError('Attach resnet-50-radimagenet-marwan Dataset')
sha = hashlib.sha256()
with WEIGHT.open('rb') as fh:
    for block in iter(lambda: fh.read(8<<20), b''): sha.update(block)
if sha.hexdigest() != CONFIG['encoder_sha256']:
    raise RuntimeError('RadImageNet weight SHA256 differs from notebook 09')

head_roots = sorted({p.parent for p in shallow_files('fold_0_head.pt')
                     if all((p.parent/f'fold_{f}_head.pt').is_file() for f in range(5))}, key=str)
if len(head_roots) != 1:
    raise RuntimeError(f'Expected exactly one five-fold notebook 09 Output; found {head_roots}')
HEAD_ROOT = head_roots[0]
CHECKPOINTS = []
for fold in range(5):
    ck = torch.load(HEAD_ROOT/f'fold_{fold}_head.pt', map_location='cpu', weights_only=True)
    if ck.get('fold') != fold or ck.get('labels') != LABELS:
        raise RuntimeError(f'Wrong fold or target order in checkpoint {fold}')
    if ck.get('source_weight_sha256') != CONFIG['encoder_sha256']:
        raise RuntimeError(f'Encoder provenance mismatch in checkpoint {fold}')
    cfg = ck.get('config', {})
    if (cfg.get('experiment') != 'B2_RadImageNet_frozen_encoder_MIL_fold01_v1'
        or cfg.get('max_tokens') != 80 or cfg.get('feature_dim') != 2048
        or cfg.get('head_dim') != 256 or list(cfg.get('active_folds',[])) != [0,1,2,3,4]):
        raise RuntimeError(f'Not a full five-fold notebook 09 checkpoint: {fold}')
    if not isinstance(ck.get('state_dict'), dict) or not ck['state_dict']:
        raise RuntimeError(f'Empty state_dict in checkpoint {fold}')
    CHECKPOINTS.append(ck)
print('ASSET CHECK PASSED', {'competition':str(COMP),'heads':str(HEAD_ROOT),
                             'weight':str(WEIGHT),'weight_sha256':sha.hexdigest()})

# %%
TEST_ROOT = COMP/'test_series'
test = pd.read_csv(COMP/'test.csv', dtype={'StudyInstanceUID':str})
sample = pd.read_csv(COMP/'sample_submission.csv', dtype={'StudyInstanceUID':str})
series = pd.read_csv(COMP/'test_series.csv', dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
if not test.StudyInstanceUID.is_unique or not series.SeriesInstanceUID.notna().all():
    raise RuntimeError('Invalid competition test identities')
if list(sample.columns) != ['StudyInstanceUID']+LABELS:
    raise RuntimeError('sample_submission.csv schema changed')
if not sample.StudyInstanceUID.is_unique or set(sample.StudyInstanceUID) != set(test.StudyInstanceUID):
    raise RuntimeError('sample/test StudyInstanceUID mismatch')
if not {'StudyInstanceUID','SeriesInstanceUID','Anatomical_Plane','Fluid_Sensitive'}.issubset(series.columns):
    raise RuntimeError('test_series.csv schema changed')

def dcm_files(path):
    try: return [Path(e.path) for e in os.scandir(path) if e.is_file() and e.name.lower().endswith('.dcm')]
    except Exception: return []

def safe_float(value, default=np.nan):
    try: return float(value)
    except Exception: return default

def one_series_metadata(row):
    files = dcm_files(TEST_ROOT/str(row.StudyInstanceUID)/str(row.SeriesInstanceUID))
    out = {'n_slices':len(files),'PixelSpacingMean':np.nan,
           'FOVRow_mm':np.nan,'FOVCol_mm':np.nan}
    if not files: return out
    try:
        ds = pydicom.dcmread(str(files[0]),stop_before_pixels=True,force=True)
        ps = getattr(ds,'PixelSpacing',None)
        rs = safe_float(ps[0]) if ps is not None and len(ps)>=2 else np.nan
        cs = safe_float(ps[1]) if ps is not None and len(ps)>=2 else np.nan
        rows,cols = safe_float(getattr(ds,'Rows',np.nan)),safe_float(getattr(ds,'Columns',np.nan))
        return {'n_slices':len(files),'PixelSpacingMean':np.nanmean([rs,cs]),
                'FOVRow_mm':rows*rs,'FOVCol_mm':cols*cs}
    except Exception:
        STATS['selection_header_failures'] += 1
        return out

with ThreadPoolExecutor(max_workers=8) as ex:
    meta = list(ex.map(one_series_metadata,series.itertuples(index=False)))
series = pd.concat([series.reset_index(drop=True),pd.DataFrame(meta)],axis=1)

def route(row):
    try: key = (str(row.Anatomical_Plane),int(row.Fluid_Sensitive))
    except Exception: return None
    for name,value in ROUTING.items():
        if key == value: return name
    return None

series['slot'] = series.apply(route,axis=1)
series['usable_slice_count'] = series.n_slices.between(12,100).astype(int)
series['fov_ok'] = (series[['FOVRow_mm','FOVCol_mm']].min(axis=1)>=140.0).astype(int)
series['spacing_rank_value'] = series.PixelSpacingMean.fillna(999.0)
series['slice_count_distance'] = (series.n_slices-32).abs()
ranked = series.sort_values(['StudyInstanceUID','slot','usable_slice_count','fov_ok',
                             'spacing_rank_value','slice_count_distance','SeriesInstanceUID'],
                            ascending=[True,True,False,False,True,True,True])
selected = ranked.dropna(subset=['slot']).groupby(['StudyInstanceUID','slot'],as_index=False).first()
manifest = test[['StudyInstanceUID']].copy()
for slot in SLOTS:
    part = selected[selected.slot.eq(slot)][['StudyInstanceUID','SeriesInstanceUID']].rename(
        columns={'SeriesInstanceUID':f'{slot}__SeriesInstanceUID'})
    manifest = manifest.merge(part,on='StudyInstanceUID',how='left',validate='one_to_one')
if list(manifest.StudyInstanceUID) != list(test.StudyInstanceUID):
    raise RuntimeError('Test manifest order mismatch')
print('TEST MANIFEST', {'studies':len(manifest),'selected_series':int(manifest.iloc[:,1:].notna().sum().sum())})

# %%
def sort_dicom_series(series_dir):
    records = []
    for path in dcm_files(series_dir):
        try:
            ds = pydicom.dcmread(str(path),stop_before_pixels=True,force=True)
            iop,ipp = getattr(ds,'ImageOrientationPatient',None),getattr(ds,'ImagePositionPatient',None)
            position = np.nan
            if iop is not None and ipp is not None and len(iop)>=6 and len(ipp)>=3:
                a,b = np.asarray(iop[:6],float),np.asarray(ipp[:3],float)
                normal = np.cross(a[:3],a[3:])
                if np.isfinite(a).all() and np.isfinite(b).all() and np.linalg.norm(normal)>1e-8:
                    position = float(np.dot(b,normal))
            records.append((path,position,safe_float(getattr(ds,'InstanceNumber',np.nan))))
        except Exception:
            STATS['failed_headers'] += 1
    if not records: return []
    if all(np.isfinite(x[1]) for x in records):
        records.sort(key=lambda x:x[1]); STATS['geometry_sorted_series'] += 1
    elif all(np.isfinite(x[2]) for x in records):
        records.sort(key=lambda x:x[2]); STATS['instance_fallback_series'] += 1
    else:
        STATS['failed_series_order'] += 1
        return []
    return [x[0] for x in records]

def sample_slice_indices(n,budget):
    if n<=0: return np.empty(0,dtype=np.int32)
    if n<=budget: return np.arange(n,dtype=np.int32)
    return np.unique(np.rint(np.linspace(0,n-1,budget)).astype(np.int32))

def physical_center_crop(arr,row_spacing,col_spacing,crop_mm):
    h,w = arr.shape
    ch = min(h,max(1,int(round(crop_mm/row_spacing))))
    cw = min(w,max(1,int(round(crop_mm/col_spacing))))
    y0,x0 = max(0,(h-ch)//2),max(0,(w-cw)//2)
    return arr[y0:y0+ch,x0:x0+cw]

def read_dicom_slice(path):
    transfer = 'unknown'
    try:
        ds = pydicom.dcmread(str(path),force=True)
        transfer = str(getattr(getattr(ds,'file_meta',None),'TransferSyntaxUID','unknown'))
        arr = np.asarray(ds.pixel_array,dtype=np.float32).squeeze()
        if arr.ndim!=2: raise ValueError(f'Expected 2D slice, got {arr.shape}')
        arr = arr*safe_float(getattr(ds,'RescaleSlope',1.0),1.0)+safe_float(getattr(ds,'RescaleIntercept',0.0),0.0)
        finite = np.isfinite(arr)
        if not finite.any(): raise ValueError('All pixels non-finite')
        arr[~finite] = np.median(arr[finite])
        if str(getattr(ds,'PhotometricInterpretation','')).upper()=='MONOCHROME1':
            arr = float(arr.max()+arr.min())-arr
        lo,hi = np.percentile(arr,[1,99])
        arr = np.zeros_like(arr) if hi<=lo+1e-6 else np.clip((np.clip(arr,lo,hi)-lo)/(hi-lo),0,1)
        ps = getattr(ds,'PixelSpacing',None)
        if ps is None or len(ps)<2: raise ValueError('Missing PixelSpacing')
        rs,cs = float(ps[0]),float(ps[1])
        if not (np.isfinite(rs) and np.isfinite(cs) and rs>0 and cs>0):
            raise ValueError('Invalid PixelSpacing')
        arr = physical_center_crop(arr,rs,cs,CONFIG['physical_crop_mm'])
        tensor = torch.from_numpy(arr)[None,None].float()
        tensor = F.interpolate(tensor,size=(CONFIG['image_size'],CONFIG['image_size']),
                               mode='bilinear',align_corners=False)[0,0]
        return np.rint(tensor.clamp(0,1).numpy()*255).astype(np.uint8)
    except Exception:
        STATS['failed_pixel_decodes'] += 1
        DECODE_FAILURES[transfer] += 1
        return None

def load_study(row):
    uid = str(row['StudyInstanceUID'])
    images,slots,positions = [],[],[]
    for slot_id,slot in enumerate(SLOTS):
        series_uid = row.get(f'{slot}__SeriesInstanceUID',np.nan)
        if pd.isna(series_uid): continue
        ordered = sort_dicom_series(TEST_ROOT/uid/str(series_uid))
        for idx in sample_slice_indices(len(ordered),CONFIG['slices_per_slot']):
            image = read_dicom_slice(ordered[int(idx)])
            if image is None: continue
            images.append(image); slots.append(slot_id)
            positions.append(float(idx/max(1,len(ordered)-1)))
    if not images: STATS['empty_studies'] += 1
    return uid,images,slots,positions

class RadEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet50(weights=None).children())[:-2])
    def forward(self,x): return self.backbone(x).mean(dim=(2,3))

class StudyMIL(nn.Module):
    def __init__(self):
        super().__init__(); d=CONFIG['head_dim']
        self.project=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,d),nn.GELU(),nn.Dropout(.10))
        self.slot=nn.Embedding(5,d)
        self.position=nn.Sequential(nn.Linear(1,d),nn.GELU(),nn.Linear(d,d))
        self.query=nn.Parameter(torch.randn(12,d)*.02)
        self.attention=nn.MultiheadAttention(d,4,dropout=.10,batch_first=True)
        self.norm=nn.LayerNorm(d)
        self.weight=nn.Parameter(torch.randn(12,d)*.02)
        self.bias=nn.Parameter(torch.zeros(12))
    def forward(self,x,s,p,m):
        m=m.clone(); m[~m.any(1),0]=True
        token=self.project(x.float())+self.slot(s)+self.position(p.unsqueeze(-1))
        q=self.query.unsqueeze(0).expand(len(x),-1,-1)
        pooled,_=self.attention(q,token,token,key_padding_mask=~m,need_weights=False)
        return (self.norm(pooled)*self.weight.unsqueeze(0)).sum(-1)+self.bias

if not torch.cuda.is_available(): raise RuntimeError('Enable GPU before loading models')
DEVICE = torch.device('cuda:0')
encoder = RadEncoder()
encoder.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True),strict=True)
encoder = encoder.to(DEVICE).eval().requires_grad_(False)
heads = []
for fold,ck in enumerate(CHECKPOINTS):
    model = StudyMIL()
    model.load_state_dict(ck['state_dict'],strict=True)
    heads.append(model.to(DEVICE).eval().requires_grad_(False))
del CHECKPOINTS
gc.collect()
print('MODELS READY',{'gpu':torch.cuda.get_device_name(0),'heads':len(heads)})

@torch.inference_mode()
def predict_study(row):
    uid,images,slots,positions = load_study(row)
    if not images:
        return uid,np.full(len(LABELS),0.5,np.float32)
    if len(images)>CONFIG['max_tokens']: raise RuntimeError(f'Too many slices for {uid}')
    feat = torch.zeros((1,CONFIG['max_tokens'],CONFIG['feature_dim']),device=DEVICE,dtype=torch.float32)
    for offset in range(0,len(images),CONFIG['image_microbatch']):
        part = np.stack(images[offset:offset+CONFIG['image_microbatch']])
        gray = torch.from_numpy(part).to(DEVICE).float().div_(127.5).sub_(1.0)
        rgb = gray.unsqueeze(1).expand(-1,3,-1,-1).contiguous()
        with torch.autocast('cuda',dtype=torch.float16): value = encoder(rgb)
        if value.shape!=(len(part),2048) or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f'Bad encoder features for {uid}')
        feat[0,offset:offset+len(part)] = value.to(torch.float16).float()
    s = torch.zeros((1,CONFIG['max_tokens']),device=DEVICE,dtype=torch.long)
    p = torch.zeros((1,CONFIG['max_tokens']),device=DEVICE,dtype=torch.float32)
    m = torch.zeros((1,CONFIG['max_tokens']),device=DEVICE,dtype=torch.bool)
    n = len(images)
    s[0,:n] = torch.as_tensor(slots,device=DEVICE,dtype=torch.long)
    p[0,:n] = torch.as_tensor(positions,device=DEVICE,dtype=torch.float32)
    m[0,:n] = True
    pred = torch.stack([torch.sigmoid(head(feat,s,p,m))[0] for head in heads]).mean(0)
    out = pred.cpu().numpy().astype(np.float32)
    if out.shape!=(12,) or not np.isfinite(out).all() or out.min()<0 or out.max()>1:
        raise RuntimeError(f'Bad prediction for {uid}')
    STATS['decoded_images'] += n
    return uid,out

# %%
predictions = np.empty((len(manifest),len(LABELS)),np.float32)
smoke_n = min(3,len(manifest))
for i in range(smoke_n):
    uid,pred = predict_study(manifest.iloc[i])
    if uid!=str(manifest.iloc[i].StudyInstanceUID): raise RuntimeError('Smoke identity drift')
    predictions[i] = pred
torch.cuda.synchronize(0)
print('VISIBLE SMOKE PASSED',{'studies':smoke_n,'elapsed_minutes':round((monotonic()-START)/60,2),
                              'probability_range':(float(predictions[:smoke_n].min()),
                                                   float(predictions[:smoke_n].max()))},flush=True)

for i in range(smoke_n,len(manifest)):
    uid,pred = predict_study(manifest.iloc[i])
    if uid!=str(manifest.iloc[i].StudyInstanceUID): raise RuntimeError(f'Test identity drift at {i}')
    predictions[i] = pred
    if (i+1)%25==0 or i+1==len(manifest):
        torch.cuda.synchronize(0)
        elapsed = monotonic()-START
        print(f'predicted {i+1}/{len(manifest)}; elapsed={elapsed/60:.1f} min; '
              f'mean_seconds_per_study={elapsed/(i+1):.2f}',flush=True)

if not np.isfinite(predictions).all() or predictions.min()<0 or predictions.max()>1:
    raise RuntimeError('Invalid full prediction matrix')
pred = pd.DataFrame(predictions,columns=LABELS)
pred.insert(0,'StudyInstanceUID',manifest.StudyInstanceUID.astype(str).tolist())
submission = sample[['StudyInstanceUID']].merge(pred,on='StudyInstanceUID',how='left',
                                               sort=False,validate='one_to_one')
if (list(submission.columns)!=list(sample.columns)
    or list(submission.StudyInstanceUID)!=list(sample.StudyInstanceUID)
    or submission[LABELS].isna().any().any()):
    raise RuntimeError('Submission schema/order/coverage mismatch')
path = WORKING/'submission.csv'
submission.to_csv(path,index=False)
roundtrip = pd.read_csv(path,dtype={'StudyInstanceUID':str})
if (roundtrip.shape!=sample.shape or list(roundtrip.columns)!=list(sample.columns)
    or list(roundtrip.StudyInstanceUID)!=list(sample.StudyInstanceUID)
    or not np.isfinite(roundtrip[LABELS].to_numpy(float)).all()):
    raise RuntimeError('Saved submission roundtrip failed')

receipt = {'status':'COMPLETE','experiment':'B2_RadImageNet_frozen_MIL_five_fold_standalone',
           'test_studies':len(manifest),'folds':[0,1,2,3,4],
           'encoder_weight_sha256':sha.hexdigest(),'head_root':str(HEAD_ROOT),
           'elapsed_minutes':(monotonic()-START)/60,
           'probability_min':float(predictions.min()),'probability_max':float(predictions.max()),
           'dicom_stats':dict(STATS),'decode_failures_by_transfer_syntax':dict(DECODE_FAILURES),
           'warning':'Standalone B2 only; not the 0.943 ensemble. Empty studies use neutral 0.5.'}
(WORKING/'b2_submit_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print('SUBMISSION_READY = YES',{'shape':submission.shape,'runtime_min':round(receipt['elapsed_minutes'],2),
                                'empty_studies':STATS['empty_studies']})
display(submission.head())
