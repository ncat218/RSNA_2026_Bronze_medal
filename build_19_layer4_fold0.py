"""Build a paired, fold-0 RadImageNet frozen-versus-layer4 Kaggle experiment."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "18_B2_RadImageNet_Layer4_Finetune_Preflight.ipynb"
TARGET = ROOT / "19_B2_RadImageNet_Layer4_Fold0_Paired.ipynb"

INTRO = """# 19 — B2 RadImageNet fold-0 paired trainability test

This is a **full fold-0 diagnostic**, not a Kaggle submission. It trains two
arms on the same 3,479 weak training studies and validates on the same 870
held-out weak studies, using the pre-existing report-group fold assignments:

1. Frozen RadImageNet ResNet50 + new 12-target MIL head.
2. The same model and 16-slice input, but ResNet50 `layer4` is trainable.

The arms use the same head initialization, batch order, soft labels, 16
uniformly selected cached slices per study, optimizer hyperparameters, and
weak-validation BCE checkpoint rule. Only the encoder's `layer4` trainability
changes. Gold58 is evaluated **after** checkpoint selection; it never enters
the optimizer, stopping rule, or model selection. No test submission is made.

Attach the same inputs that worked for notebook 18: competition data, notebook
02 silver-label Output, complete `Study_case 05_RSNA` 05A cache Dataset with
`rsna_05a_dense_mil/study_cache`, and `resnet-50-radimagenet-marwan` weights.
Notebook 09 Output is **not required**. Use **T4×2, Internet Off**, Save & Run
All. The code uses GPU 0 only. Send back the notebook Output ZIP containing
`rsna_b2_layer4_fold0_paired/run_report.json`, `paired_metrics.csv`, the two
OOF CSVs, and `training_history.csv`. Checkpoints are saved for later use, but
this one fold alone is not a submit-ready ensemble. A successful local/static
check cannot guarantee Kaggle execution or a leaderboard improvement.
"""

CONFIG = """CONFIG = {
    'experiment': 'B2_RadImageNet_layer4_fold0_paired_v1',
    'seed': 111, 'active_folds': [0, 1, 2, 3, 4], 'selected_fold': 0,
    'max_tokens': 16, 'image_microbatch': 8, 'study_batch': 2,
    'feature_dim': 2048, 'head_dim': 256,
    'max_epochs': 12, 'patience': 4, 'max_total_hours': 3.0,
    'layer4_lr': 1e-5, 'head_lr': 3e-4, 'weight_decay': 1e-4,
    'min_available_ram_gib': 8.0, 'max_process_rss_gib': 15.0,
    'max_cuda_allocated_gib': 10.0,
    'encoder_sha256': '08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734',
}
"""

TRAIN = r"""# Fold-0 paired experiment. All metrics/checkpoints are persisted after each epoch.
import traceback

if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
    raise RuntimeError('Select T4x2 before running the full fold-0 diagnostic')
DEVICE = torch.device('cuda:0')
torch.cuda.reset_peak_memory_stats(0)

class RadEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet50(weights=None).children())[:-2])
    def forward(self, x):
        return self.backbone(x).mean(dim=(2, 3))

class StudyMIL(nn.Module):
    def __init__(self):
        super().__init__(); d = CONFIG['head_dim']
        self.project = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048,d),
                                     nn.GELU(), nn.Dropout(.10))
        self.slot = nn.Embedding(5,d)
        self.position = nn.Sequential(nn.Linear(1,d), nn.GELU(), nn.Linear(d,d))
        self.query = nn.Parameter(torch.randn(12,d)*.02)
        self.attention = nn.MultiheadAttention(d,4,dropout=.10,batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.weight = nn.Parameter(torch.randn(12,d)*.02)
        self.bias = nn.Parameter(torch.zeros(12))
    def forward(self,x,s,p,m):
        m=m.clone(); m[~m.any(1),0]=True
        token=self.project(x.float())+self.slot(s)+self.position(p.unsqueeze(-1))
        q=self.query.unsqueeze(0).expand(len(x),-1,-1)
        pooled,_=self.attention(q,token,token,key_padding_mask=~m,need_weights=False)
        return (self.norm(pooled)*self.weight.unsqueeze(0)).sum(-1)+self.bias

class CacheStudies(Dataset):
    def __init__(self, indices, allow_gold=False):
        self.indices = list(map(int, indices)); self.allow_gold = allow_gold
    def __len__(self): return len(self.indices)
    def __getitem__(self, index):
        i = self.indices[index]
        if i >= len(weak) and not self.allow_gold:
            raise RuntimeError('Gold study entered weak train/validation loader')
        uid = IDS[i]
        split = 'weak' if i < len(weak) else 'gold'
        path = CACHE / f'{split}_{uid}.npz'
        with np.load(path, allow_pickle=False) as z:
            if not {'images','slot_ids','positions','slice_indices'}.issubset(z.files):
                raise RuntimeError(f'Invalid cache keys: {path}')
            images=z['images']; slots=z['slot_ids']; positions=z['positions']
            if (images.dtype != np.uint8 or images.ndim != 3 or
                    images.shape[1:] != (224,224) or
                    len(slots) != len(images) or len(positions) != len(images) or
                    len(z['slice_indices']) != len(images)):
                raise RuntimeError(f'Invalid cache shape: {path}')
        if len(images) == 0 or not np.isfinite(positions).all():
            raise RuntimeError(f'Empty/nonfinite cache: {path}')
        if slots.min() < 0 or slots.max() >= 5:
            raise RuntimeError(f'Invalid slot index: {path}')
        take = np.rint(np.linspace(0,len(images)-1,
                         min(len(images),CONFIG['max_tokens']))).astype(int)
        if len(np.unique(take)) != len(take):
            raise RuntimeError(f'Duplicate selected slice: {path}')
        n=len(take); limit=CONFIG['max_tokens']
        image_buf=np.zeros((limit,224,224),np.uint8)
        slot_buf=np.zeros(limit,np.int64)
        pos_buf=np.zeros(limit,np.float32)
        mask_buf=np.zeros(limit,bool)
        image_buf[:n]=images[take]; slot_buf[:n]=slots[take]
        pos_buf[:n]=positions[take]; mask_buf[:n]=True
        return (i,torch.from_numpy(image_buf),torch.from_numpy(slot_buf),
                torch.from_numpy(pos_buf),torch.from_numpy(mask_buf),
                torch.from_numpy(Y[i]))

def make_loader(indices, shuffle=False, allow_gold=False):
    return DataLoader(CacheStudies(indices,allow_gold=allow_gold),
                      batch_size=CONFIG['study_batch'],shuffle=shuffle,
                      generator=torch.Generator().manual_seed(CONFIG['seed']),
                      num_workers=0,pin_memory=False,drop_last=False)

def snapshot():
    proc=psutil.Process()
    return {'process_rss_gib':proc.memory_info().rss/2**30,
            'system_available_gib':psutil.virtual_memory().available/2**30,
            'cuda_allocated_gib':torch.cuda.memory_allocated(0)/2**30,
            'cuda_reserved_gib':torch.cuda.memory_reserved(0)/2**30,
            'cuda_peak_allocated_gib':torch.cuda.max_memory_allocated(0)/2**30}

def guard_reason():
    mem=snapshot()
    if mem['system_available_gib'] < CONFIG['min_available_ram_gib']:
        return 'available_host_ram'
    if mem['process_rss_gib'] > CONFIG['max_process_rss_gib']:
        return 'process_rss'
    if mem['cuda_peak_allocated_gib'] > CONFIG['max_cuda_allocated_gib']:
        return 'cuda_peak_allocated'
    if (monotonic()-START)/3600 > CONFIG['max_total_hours']:
        return 'time_cap'
    return None

def make_model(train_layer4):
    # Reset before both arms so their newly initialized heads match exactly.
    torch.manual_seed(CONFIG['seed'])
    torch.cuda.manual_seed_all(CONFIG['seed'])
    encoder=RadEncoder()
    state=torch.load(WEIGHT,map_location='cpu',weights_only=True)
    expected=encoder.state_dict()
    missing=set(expected)-set(state); extra=set(state)-set(expected)
    wrong=[k for k in set(expected)&set(state) if expected[k].shape!=state[k].shape]
    if missing or extra or wrong:
        raise RuntimeError(f'RadImageNet checkpoint mismatch: {len(missing)} missing, '
                           f'{len(extra)} extra, {len(wrong)} shapes')
    encoder.load_state_dict(state,strict=True)
    del state,expected; gc.collect()
    encoder.requires_grad_(False)
    if train_layer4: encoder.backbone[-1].requires_grad_(True)
    encoder=encoder.to(DEVICE).eval()  # Keep all BatchNorm statistics frozen.
    head=StudyMIL().to(DEVICE)
    return encoder,head

def forward_studies(encoder,head,images,slots,positions,masks,train_layer4):
    batch,tokens=images.shape[:2]
    flat=images.reshape(batch*tokens,224,224)
    pieces=[]
    for start in range(0,len(flat),CONFIG['image_microbatch']):
        gray=flat[start:start+CONFIG['image_microbatch']].to(DEVICE).float()
        gray=gray.div_(127.5).sub_(1.0)
        rgb=gray.unsqueeze(1).expand(-1,3,-1,-1).contiguous()
        with torch.autocast('cuda',dtype=torch.float16):
            if train_layer4:
                piece=encoder(rgb)
            else:
                with torch.no_grad(): piece=encoder(rgb)
        pieces.append(piece)
    features=torch.cat(pieces,dim=0).reshape(batch,tokens,2048)
    return head(features,slots.to(DEVICE),positions.to(DEVICE),masks.to(DEVICE))

def predict(encoder,head,loader,train_layer4):
    encoder.eval(); head.eval()
    indices=[]; probs=[]; loss_sum=0.0; count=0
    with torch.inference_mode():
        for ix,images,slots,positions,masks,target in loader:
            logits=forward_studies(encoder,head,images,slots,positions,masks,
                                   train_layer4)
            loss=F.binary_cross_entropy_with_logits(logits,target.to(DEVICE))
            indices.extend(ix.tolist())
            probs.append(torch.sigmoid(logits).cpu().numpy())
            loss_sum+=float(loss)*len(ix); count+=len(ix)
            del logits,loss
    torch.cuda.synchronize(0)
    return loss_sum/count,np.asarray(indices,int),np.concatenate(probs)

def target_auc(binary,pred):
    result=[]
    for j in range(12):
        yy=binary[:,j]
        result.append(float(roc_auc_score(yy,pred[:,j]))
                      if len(np.unique(yy))==2 else float('nan'))
    return result

def csv_predictions(indices,pred,path):
    if len(indices)!=len(pred) or len(set(indices))!=len(indices):
        raise RuntimeError('Prediction identity/count mismatch')
    if not np.isfinite(pred).all() or ((pred<0)|(pred>1)).any():
        raise RuntimeError('Invalid prediction probability')
    table=pd.DataFrame(pred,columns=LABELS)
    table.insert(0,'StudyInstanceUID',[IDS[i] for i in indices])
    table.insert(1,'fold',CONFIG['selected_fold'])
    table.to_csv(path,index=False)

fold=CONFIG['selected_fold']
tr=np.flatnonzero(weak.fold.to_numpy()!=fold)
va=np.flatnonzero(weak.fold.to_numpy()==fold)
gi=np.arange(len(weak),len(IDS))
if len(tr)+len(va)!=4349 or len(gi)!=58 or len(va)!=870:
    raise RuntimeError(f'Unexpected fold sizes: train={len(tr)} val={len(va)} gold={len(gi)}')
if not set(weak.iloc[tr].report_group.astype(str)).isdisjoint(
        set(weak.iloc[va].report_group.astype(str))):
    raise RuntimeError('Report-group leakage in fold 0')
if set(map(int,tr)) & set(map(int,gi)) or set(map(int,va)) & set(map(int,gi)):
    raise RuntimeError('Gold leaked into weak fold')
val_loader=make_loader(va)
gold_loader=make_loader(gi,allow_gold=True)
print('FOLD CONTRACT',{'train':len(tr),'val':len(va),'gold_audit':len(gi),
                       'cache':str(CACHE)},flush=True)

report={'status':'RUNNING','experiment':CONFIG['experiment'],'config':CONFIG,
        'source_weight_sha256':digest.hexdigest(),'n_train':len(tr),
        'n_val':len(va),'n_gold_audit':len(gi),
        'train_uid_sha256':hashlib.sha256('\n'.join(IDS[i] for i in tr).encode()).hexdigest(),
        'val_uid_sha256':hashlib.sha256('\n'.join(IDS[i] for i in va).encode()).hexdigest(),
        'arms':{},'warning':'Fold-0 weak AUC is diagnostic, not Public or Private score; '
                            'Gold58 is audit-only and cannot select a model.'}
history=[]; arm_probs={}; arm_metrics={}
def persist():
    (OUT/'run_report.json').write_text(
        json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n',
        encoding='utf-8')
    if history: pd.DataFrame(history).to_csv(OUT/'training_history.csv',index=False)
persist()

try:
    for arm,train_layer4 in [('frozen',False),('layer4',True)]:
        if guard_reason():
            report['status']='PARTIAL_GUARD'; report['guard_reason']=guard_reason(); break
        gc.collect(); torch.cuda.empty_cache()
        encoder,head=make_model(train_layer4)
        params=list(head.parameters())
        groups=[{'params':params,'lr':CONFIG['head_lr']}]
        if train_layer4:
            layer4_params=[p for p in encoder.parameters() if p.requires_grad]
            if sum(p.numel() for p in layer4_params)!=14964736:
                raise RuntimeError('Unexpected layer4 trainable parameter count')
            groups.insert(0,{'params':layer4_params,'lr':CONFIG['layer4_lr']})
        optimizer=torch.optim.AdamW(groups,weight_decay=CONFIG['weight_decay'])
        scaler=torch.amp.GradScaler('cuda',enabled=True)
        train_loader=make_loader(tr,shuffle=True)
        best_bce=float('inf'); best_epoch=0; stale=0; stop_reason=None
        checkpoint=OUT/f'{arm}_fold0_best.pt'
        report['arms'][arm]={'status':'RUNNING','epochs_completed':0}
        persist()
        for epoch in range(1,CONFIG['max_epochs']+1):
            stop_reason=guard_reason()
            if stop_reason: break
            encoder.eval(); head.train()
            loss_sum=0.0; trained=0; t0=monotonic()
            for ix,images,slots,positions,masks,target in train_loader:
                stop_reason=guard_reason()
                if stop_reason: break
                optimizer.zero_grad(set_to_none=True)
                logits=forward_studies(encoder,head,images,slots,positions,masks,
                                       train_layer4)
                loss=F.binary_cross_entropy_with_logits(logits,target.to(DEVICE))
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f'Nonfinite loss: {arm} epoch {epoch}')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params+(layer4_params if train_layer4 else []),1.0)
                scaler.step(optimizer); scaler.update()
                loss_sum+=float(loss.detach())*len(ix); trained+=len(ix)
                del logits,loss
            torch.cuda.synchronize(0)
            if stop_reason:
                print('GUARD DURING TRAIN',arm,epoch,stop_reason,trained,flush=True)
                break
            if trained!=len(tr): raise RuntimeError('Incomplete training epoch')
            val_bce,indices,pred=predict(encoder,head,val_loader,train_layer4)
            if not np.array_equal(indices,va):
                raise RuntimeError('Validation UID/order mismatch')
            auc=target_auc((Y[indices]>=.5).astype(int),pred)
            epoch_seconds=monotonic()-t0
            row={'arm':arm,'epoch':epoch,'train_bce':loss_sum/trained,
                 'val_bce':val_bce,'weak_macro_auc_proxy':float(np.nanmean(auc)),
                 'epoch_seconds':epoch_seconds,'process_rss_gib':snapshot()['process_rss_gib'],
                 'cuda_peak_allocated_gib':snapshot()['cuda_peak_allocated_gib']}
            history.append(row)
            if val_bce < best_bce-1e-5:
                best_bce=val_bce; best_epoch=epoch; stale=0
                payload={'arm':arm,'fold':fold,'best_epoch':best_epoch,
                         'best_val_bce':best_bce,'head_state_dict':head.state_dict(),
                         'layer4_state_dict':encoder.backbone[-1].state_dict()
                             if train_layer4 else None,
                         'config':CONFIG,'labels':LABELS,
                         'source_weight_sha256':digest.hexdigest()}
                torch.save(payload,checkpoint)
            else:
                stale+=1
            report['arms'][arm].update({'epochs_completed':epoch,'best_epoch':best_epoch,
                                        'best_val_bce':best_bce,'last_row':row,
                                        'checkpoint':str(checkpoint)})
            persist()
            print('EPOCH',row,'best_epoch',best_epoch,flush=True)
            stop_reason=guard_reason()
            if stop_reason: break
            if stale>=CONFIG['patience']:
                stop_reason='early_stop'; break
        if best_epoch==0:
            report['arms'][arm]['status']='NO_COMPLETE_EPOCH'
            report['arms'][arm]['stop_reason']=stop_reason
            report['status']='PARTIAL_GUARD'; persist(); break
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved['source_weight_sha256']!=digest.hexdigest() or saved['arm']!=arm:
            raise RuntimeError('Checkpoint provenance mismatch')
        head.load_state_dict(saved['head_state_dict'],strict=True)
        if train_layer4:
            encoder.backbone[-1].load_state_dict(saved['layer4_state_dict'],strict=True)
        del saved; gc.collect()
        val_bce,indices,pred=predict(encoder,head,val_loader,train_layer4)
        if not np.array_equal(indices,va): raise RuntimeError('OOF UID/order mismatch')
        csv_predictions(indices,pred,OUT/f'{arm}_fold0_weak_oof.csv')
        weak_auc=target_auc((Y[indices]>=.5).astype(int),pred)
        arm_probs[arm]=pred
        gold_bce,gold_indices,gold_pred=predict(encoder,head,gold_loader,train_layer4)
        if not np.array_equal(gold_indices,gi):
            raise RuntimeError('Gold audit identity mismatch')
        csv_predictions(gold_indices,gold_pred,OUT/f'{arm}_fold0_gold_predictions.csv')
        gold_auc=target_auc(Y[gi].astype(int),gold_pred)
        arm_metrics[arm]={'best_epoch':best_epoch,'weak_val_bce':val_bce,
                          'weak_macro_auc_proxy':float(np.nanmean(weak_auc)),
                          'weak_per_target_auc_proxy':dict(zip(LABELS,weak_auc)),
                          'gold_macro_auc_audit_only':float(np.nanmean(gold_auc)),
                          'gold_per_target_auc_audit_only':dict(zip(LABELS,gold_auc))}
        report['arms'][arm].update({'status':'COMPLETE' if stop_reason in (None,'early_stop')
                                   else 'PARTIAL_GUARD','stop_reason':stop_reason,
                                   'metrics':arm_metrics[arm]})
        persist()
        del encoder,head,optimizer,scaler,train_loader,params
        if train_layer4: del layer4_params
        gc.collect(); torch.cuda.empty_cache()
        if stop_reason not in (None,'early_stop'):
            report['status']='PARTIAL_GUARD'; break
    if len(arm_metrics)==2:
        if arm_probs['frozen'].shape!=arm_probs['layer4'].shape:
            raise RuntimeError('Paired OOF shape mismatch')
        paired=[]
        for j,label in enumerate(LABELS):
            a=arm_metrics['frozen']['weak_per_target_auc_proxy'][label]
            b=arm_metrics['layer4']['weak_per_target_auc_proxy'][label]
            paired.append({'target':label,'frozen_weak_auc':a,'layer4_weak_auc':b,
                           'layer4_minus_frozen_auc':b-a})
        pd.DataFrame(paired).to_csv(OUT/'paired_metrics.csv',index=False)
        report['paired_weak_macro_auc_delta']=(arm_metrics['layer4']['weak_macro_auc_proxy']-
                                               arm_metrics['frozen']['weak_macro_auc_proxy'])
        report['status']='COMPLETE' if all(
            report['arms'][a]['status']=='COMPLETE' for a in ('frozen','layer4')) else 'PARTIAL_GUARD'
except Exception as exc:
    report['status']='ERROR'
    report['error']=f'{type(exc).__name__}: {str(exc)[:1000]}'
    report['traceback_tail']=traceback.format_exc()[-2000:]
    raise
finally:
    report['elapsed_minutes']=(monotonic()-START)/60
    report['final_memory']=snapshot()
    persist()
    print('FINAL RECEIPT',{'status':report['status'],
                           'elapsed_minutes':round(report['elapsed_minutes'],1),
                           'paired_weak_macro_auc_delta':report.get('paired_weak_macro_auc_delta')},
          flush=True)
"""


def source(cell: dict) -> str:
    return "".join(cell["source"])


def make_cell(kind: str, code: str) -> dict:
    cell = {"cell_type": kind, "metadata": {}, "source": code.splitlines(True)}
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) == 3
    asset = source(notebook["cells"][1])
    config_start = asset.index("CONFIG = {")
    config_end = asset.index("INPUT = Path(", config_start)
    asset = asset[:config_start] + CONFIG + asset[config_end:]
    asset = asset.replace("/kaggle/working/rsna_b2_layer4_preflight",
                          "/kaggle/working/rsna_b2_layer4_fold0_paired")
    if "rsna_b2_layer4_fold0_paired" not in asset:
        raise RuntimeError("Output-path replacement failed")
    for code in (asset, TRAIN):
        ast.parse(code)
    notebook["cells"] = [make_cell("markdown", INTRO), make_cell("code", asset),
                         make_cell("code", TRAIN)]
    notebook["metadata"]["kaggle"]["isInternetEnabled"] = False
    notebook["metadata"]["kaggle"]["isGpuEnabled"] = True
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                      encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
