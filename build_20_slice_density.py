"""Build controlled 16/32/80-token B2 ablation from notebook 09."""
from pathlib import Path
import ast
import copy
import json


ROOT = Path(__file__).parent
src_path = ROOT / '09_B2_RadImageNet_FrozenMIL_Fold01.ipynb'
dst_path = ROOT / '20_B2_RadImageNet_Slice_Density_Ablation.ipynb'
nb = json.loads(src_path.read_text(encoding='utf-8'))
assert len(nb['cells']) == 4


def source(i):
    return ''.join(nb['cells'][i]['source'])


def set_source(cell, value):
    cell['source'] = value.splitlines(keepends=True)
    cell['execution_count'] = None
    cell['outputs'] = []


intro = '''# 20 — B2 frozen RadImageNet: controlled slice-density ablation

One 80-slice feature extraction, then **80 / 32 / 16 tokens per study** on the same five grouped folds. All three arms use notebook 09's frozen encoder, MIL head, seed, batch size, optimizer, stopping rule and silver labels. The 16/32 subsets select uniformly from each study's valid cached token order, as in notebook 19. The 80-token arm reproduces notebook 09's unchanged dataset path and is run first.

This is **diagnostic only**, not a competition submission. Gold58 is audited after weak-validation checkpoint selection; it never affects training or choice of density. Compare the 80-token control with notebook 09 before interpreting 16/32. A mismatch may mean inputs or runtime changed. Even a replicated 80-token OOF result does not validate Public/Private LB improvements.

**Kaggle inputs:** competition data; notebook 02 output with `L1_family_mean_RECOMMENDED.csv`; complete `Study_case 05_RSNA` dataset with `rsna_05a_dense_mil/study_cache` and `fold_assignments.csv`; public `resnet-50-radimagenet-marwan` dataset. **Settings:** T4×2, Internet Off. Only GPU 0 is used. No notebook 09 output is required.

Run **Save & Run All** and send back the Output ZIP containing `density_ablation_report.json`, `density_summary.csv`, `density_target_auc.csv`, and the three `density_*_weak_oof.csv` files. The notebook writes per-fold files as it runs. Do not spend a submission slot on its diagnostic outputs.
'''

asset = source(1)
asset = asset.replace("'B2_RadImageNet_frozen_encoder_MIL_fold01_v1'", "'B2_RadImageNet_slice_density_16_32_80_v1'")
asset = asset.replace("'max_total_hours': 6.0", "'max_total_hours': 2.0")
asset = asset.replace("Path('/kaggle/working/rsna_b2_rad_frozen_fold01')", "Path('/kaggle/working/rsna_b2_slice_density')")
asset = asset.replace("'encoder_sha256':", "'densities': [80,32,16], 'encoder_sha256':")
assert "'densities': [80,32,16]" in asset
encode = source(2)

original_train = source(3)
head_start = original_train.index('class StudyMIL(nn.Module):')
head_end = original_train.index('history=[]; cv_rows=[]; gold_rows=[]')
head_code = original_train[head_start:head_end]
head_code = head_code.replace('def loader(indices,shuffle=False,seed=0):',
                             'def loader(indices,density,shuffle=False,seed=0):')
head_code = head_code.replace('DataLoader(FeatureDataset(indices),',
                             'DataLoader(FeatureDataset(indices,density),')

train = '''# Identical notebook 09 head and optimization; only the dataset token cap varies.
class FeatureDataset(Dataset):
    def __init__(self,indices,density):
        self.indices=np.asarray(indices,dtype=int)
        self.density=int(density)
        if self.density not in CONFIG['densities']: raise ValueError(self.density)
    def __len__(self): return len(self.indices)
    def __getitem__(self,k):
        i=int(self.indices[k]); d=self.density
        if d==80:
            # Preserve notebook 09's exact tensors and padding for the control.
            return (i,torch.from_numpy(features[i]),torch.from_numpy(slot_ids[i]),
                    torch.from_numpy(positions[i]),torch.from_numpy(masks[i]),torch.from_numpy(Y[i]))
        t=int(masks[i].sum())
        take=np.rint(np.linspace(0,t-1,min(t,d))).astype(int) if t else np.empty(0,dtype=int)
        if len(np.unique(take)) != len(take): raise RuntimeError(f'Duplicate token index: {IDS[i]}')
        x=np.zeros((d,CONFIG['feature_dim']),dtype=np.float16)
        s=np.zeros(d,dtype=np.int64); p=np.zeros(d,dtype=np.float32); m=np.zeros(d,dtype=bool)
        n=len(take)
        x[:n]=features[i,take]; s[:n]=slot_ids[i,take]; p[:n]=positions[i,take]; m[:n]=True
        return (i,torch.from_numpy(x),torch.from_numpy(s),torch.from_numpy(p),
                torch.from_numpy(m),torch.from_numpy(Y[i]))

''' + head_code + '''
# Treat StudyInstanceUID as an exact string key; never rely on positional joins.
expected_09_fold_auc={0:0.8648121354054118,1:0.8636259689332366,
                      2:0.862549066604258,3:0.8670961453049354,
                      4:0.8606571310358122}
history=[]; cv_rows=[]; gold_rows=[]; density_summaries=[]; target_rows=[]
gold_indices=np.arange(len(weak),len(IDS))
completed_densities=[]; guard_reason=None
control_fold_auc_deviations=None

def json_safe(value):
    if isinstance(value,dict): return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value): return None
    if isinstance(value,np.integer): return int(value)
    return value

for density in CONFIG['densities']:
    if (monotonic()-START)/3600 > CONFIG['max_total_hours']:
        guard_reason='time_guard_before_density'; break
    print(f'START DENSITY {density}',flush=True)
    oof_parts=[]; gold_parts=[]; density_cv=[]
    gold_loader=loader(gold_indices,density)
    for fold in CONFIG['active_folds']:
        if (monotonic()-START)/3600 > CONFIG['max_total_hours']:
            guard_reason=f'time_guard_before_density_{density}_fold_{fold}'; break
        torch.manual_seed(CONFIG['seed']+fold)
        tr=np.flatnonzero(weak.fold.to_numpy()!=fold)
        va=np.flatnonzero(weak.fold.to_numpy()==fold)
        if not set(weak.iloc[tr].report_group.astype(str)).isdisjoint(
                set(weak.iloc[va].report_group.astype(str))):
            raise RuntimeError(f'Fold {fold} report-group leakage')
        tr_loader=loader(tr,density,shuffle=True,seed=fold)
        va_loader=loader(va,density)
        model=StudyMIL().to(DEVICE)
        optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['head_lr'],
                                    weight_decay=CONFIG['weight_decay'])
        best_bce=float('inf'); best_epoch=0; best_state=None; stale=0; fold_stop_reason=None
        for epoch in range(1,CONFIG['max_epochs']+1):
            if (monotonic()-START)/3600 > CONFIG['max_total_hours']:
                fold_stop_reason='time_guard'; break
            t0=monotonic()
            train_bce,_,_=pass_batches(model,tr_loader,optimizer)
            val_bce,vi,vp=pass_batches(model,va_loader)
            auc=target_auc((Y[vi]>=.5).astype(int),vp)
            row={'density':density,'fold':fold,'epoch':epoch,'train_bce':train_bce,
                 'val_bce':val_bce,'diagnostic_weak_macro_auc':float(np.nanmean(auc)),
                 'seconds':monotonic()-t0}
            history.append(row)
            pd.DataFrame(history).to_csv(OUT/'training_history.csv',index=False)
            print(row,flush=True)
            if val_bce < best_bce-1e-5:
                best_bce=val_bce; best_epoch=epoch; stale=0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else: stale+=1
            if psutil.virtual_memory().available/2**30 < CONFIG['min_available_ram_gib']:
                fold_stop_reason='host_ram_guard'; break
            if stale>=CONFIG['patience']: break
        if best_state is None: raise RuntimeError(f'Density {density}, fold {fold} has no checkpoint')
        model.load_state_dict(best_state)
        torch.save({'state_dict':best_state,'density':density,'fold':fold,
                    'best_epoch':best_epoch,'best_val_bce':best_bce,
                    'config':CONFIG,'labels':LABELS,
                    'source_weight_sha256':digest.hexdigest()},
                   OUT/f'density_{density}_fold_{fold}_head.pt')
        val_bce,vi,vp=pass_batches(model,va_loader)
        if len(vi)!=len(va) or set(vi)!=set(va):
            raise RuntimeError(f'Density {density}, fold {fold} validation identity mismatch')
        wauc=target_auc((Y[vi]>=.5).astype(int),vp)
        fold_row={'density':density,'fold':fold,'best_epoch':best_epoch,
                  'val_bce':val_bce,'stop_reason':fold_stop_reason,
                  'weak_macro_auc_proxy':float(np.nanmean(wauc)),'n_val':len(vi)}
        density_cv.append(fold_row); cv_rows.append(fold_row)
        part=pd.DataFrame(vp,columns=LABELS)
        part.insert(0,'StudyInstanceUID',[IDS[i] for i in vi]); part['fold']=fold
        oof_parts.append(part)
        pd.concat(oof_parts,ignore_index=True).to_csv(
            OUT/f'density_{density}_weak_oof_partial.csv',index=False)
        _,gi,gp=pass_batches(model,gold_loader)
        if list(gi)!=list(gold_indices): raise RuntimeError('Gold audit identity mismatch')
        gauc=target_auc(Y[gi].astype(int),gp)
        gold_rows.append({'density':density,'fold':fold,
                          'gold_macro_auc_audit_only':float(np.nanmean(gauc)),
                          **{f'auc_{LABELS[j]}':gauc[j] for j in range(12)}})
        gold_parts.append(gp)
        pd.DataFrame(cv_rows).to_csv(OUT/'cv_metrics.csv',index=False)
        pd.DataFrame(gold_rows).to_csv(OUT/'gold_metrics.csv',index=False)
        del model,optimizer,best_state,tr_loader,va_loader
        gc.collect(); torch.cuda.empty_cache()
        if fold_stop_reason is not None:
            guard_reason=f'{fold_stop_reason}_density_{density}_fold_{fold}'; break
    del gold_loader
    if guard_reason: break
    if len(density_cv)!=5: raise RuntimeError(f'Density {density} did not finish all five folds')
    oof=pd.concat(oof_parts,ignore_index=True)
    if len(oof)!=len(weak) or not oof.StudyInstanceUID.is_unique:
        raise RuntimeError(f'Density {density} OOF identity/count mismatch')
    expected=weak[['StudyInstanceUID','fold']]
    check=expected.merge(oof,on=['StudyInstanceUID','fold'],how='left',validate='one_to_one',indicator=True)
    if not check['_merge'].eq('both').all() or check[LABELS].isna().any().any():
        raise RuntimeError(f'Density {density} OOF has missing or misaligned IDs/folds')
    oof.to_csv(OUT/f'density_{density}_weak_oof.csv',index=False)
    yy=(weak[LABELS].to_numpy()>=.5).astype(int)
    aucs=target_auc(yy,check[LABELS].to_numpy(float))
    mean_gold=np.mean(np.stack(gold_parts),axis=0)
    gold_pred=pd.DataFrame(mean_gold,columns=LABELS)
    gold_pred.insert(0,'StudyInstanceUID',gold.StudyInstanceUID.tolist())
    gold_pred.to_csv(OUT/f'density_{density}_gold_predictions.csv',index=False)
    gold_aucs=target_auc(gold[LABELS].to_numpy(int),mean_gold)
    summary={'density':density,'n_weak_oof':len(oof),
             'weak_oof_macro_auc_proxy':float(np.nanmean(aucs)),
             'gold58_macro_auc_audit_only':float(np.nanmean(gold_aucs)),
             'elapsed_minutes':(monotonic()-START)/60}
    density_summaries.append(summary)
    for j,label in enumerate(LABELS):
        target_rows.append({'density':density,'target':label,'weak_oof_auc_proxy':aucs[j],
                            'gold58_auc_audit_only':gold_aucs[j]})
    pd.DataFrame(density_summaries).to_csv(OUT/'density_summary.csv',index=False)
    pd.DataFrame(target_rows).to_csv(OUT/'density_target_auc.csv',index=False)
    completed_densities.append(density)
    print('DENSITY COMPLETE',summary,flush=True)
    if density==80:
        deviations={f:next(r['weak_macro_auc_proxy'] for r in density_cv if r['fold']==f)-old
                    for f,old in expected_09_fold_auc.items()}
        control_fold_auc_deviations=deviations
        print('NOTEBOOK 09 CONTROL FOLD-AUC DEVIATIONS',deviations,flush=True)
        if max(abs(v) for v in deviations.values()) > .005:
            guard_reason='notebook_09_control_not_reproduced'
            print('STOP: 80-token control differs by >0.005 AUC on at least one fold.',flush=True)
            break

complete=(completed_densities==CONFIG['densities'] and guard_reason is None)
paired_deltas=[]
if complete:
    base=pd.read_csv(OUT/'density_80_weak_oof.csv',dtype={'StudyInstanceUID':str})
    for density in [32,16]:
        other=pd.read_csv(OUT/f'density_{density}_weak_oof.csv',dtype={'StudyInstanceUID':str})
        joined=base.merge(other,on=['StudyInstanceUID','fold'],suffixes=('_80',f'_{density}'),
                          validate='one_to_one')
        if len(joined)!=len(weak) or not joined.StudyInstanceUID.is_unique:
            raise RuntimeError(f'Paired comparison {density} ID mismatch')
        label_check=weak[['StudyInstanceUID']+LABELS].merge(
            joined[['StudyInstanceUID']],on='StudyInstanceUID',validate='one_to_one')
        if len(label_check)!=len(weak): raise RuntimeError('Paired label count mismatch')
        ordered=label_check[['StudyInstanceUID']].merge(joined,on='StudyInstanceUID',
                   how='left',validate='one_to_one')
        for j,label in enumerate(LABELS):
            y=(label_check[label].to_numpy()>=.5).astype(int)
            # Align labels and predictions by UID, never by incidental row order.
            old=roc_auc_score(y,ordered[f'{label}_80'].to_numpy(float))
            new=roc_auc_score(y,ordered[f'{label}_{density}'].to_numpy(float))
            paired_deltas.append({'density':density,'target':label,
                                  'auc_80':old,'auc_new':new,'delta_vs_80':new-old})
    pd.DataFrame(paired_deltas).to_csv(OUT/'paired_target_deltas.csv',index=False)

report={'status':'COMPLETE' if complete else 'PARTIAL_GUARD',
        'guard_reason':guard_reason,'config':CONFIG,'n_weak':len(weak),'n_gold':len(gold),
        'n_features':len(IDS),'n_images_encoded':n_images,'empty_studies':empty,
        'encode_seconds':encode_seconds,'encoder_peak_gpu_gib':encoder_peak_gib,
        'host_feature_gib':feature_bytes/2**30,
        'total_elapsed_minutes':(monotonic()-START)/60,
        'densities_completed':completed_densities,'density_summary':density_summaries,
        'notebook_09_expected_fold_auc':expected_09_fold_auc,
        'control_fold_auc_deviations':control_fold_auc_deviations,
        'silver_path':str(SILVER_PATH),'fold_path':str(FOLD_PATH),
        'cache_path':str(CACHE),'encoder_weight_path':str(WEIGHT),
        'encoder_weight_sha256':digest.hexdigest(),
        'warning':'Diagnostic weak-OOF ablation. Gold58 is audit-only. No public/private score claim.'}
(OUT/'density_ablation_report.json').write_text(
    json.dumps(json_safe(report),indent=2,allow_nan=False),encoding='utf-8')
print('ABLATION REPORT',json.dumps({'status':report['status'],'guard_reason':guard_reason,
      'densities_completed':completed_densities,'density_summary':density_summaries,
      'elapsed_minutes':round(report['total_elapsed_minutes'],1)},indent=2),flush=True)
'''

# Build a clean, portable notebook rather than carrying executed outputs.
out=copy.deepcopy(nb)
out['cells'][0]['source']=intro.splitlines(keepends=True)
set_source(out['cells'][1],asset)
set_source(out['cells'][2],encode)
set_source(out['cells'][3],train)
out['metadata']['kaggle']['title']='20 B2 RadImageNet slice-density ablation 16 32 80'
out['metadata']['kaggle']['description']='Controlled 5-fold frozen-B2 slice-density ablation; shared feature extraction; diagnostic only'
for i,cell in enumerate(out['cells']):
    if cell['cell_type']=='code': ast.parse(''.join(cell['source']),filename=f'cell_{i}')
dst_path.write_text(json.dumps(out,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print(dst_path)
