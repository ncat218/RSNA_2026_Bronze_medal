"""Two standalone submissions to test S6 crop TTA without anchor dilution."""
import ast, copy, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def text(path):
    nb=json.loads((ROOT/path).read_text(encoding='utf-8'))
    return nb,''.join(nb['cells'][1]['source'])
def value(source,name):
    return ast.literal_eval(next(n.value for n in ast.parse(source).body if isinstance(n,ast.Assign)
                                and any(isinstance(x,ast.Name) and x.id==name for x in n.targets)))
nb39,parent39=text('39_Anchor0958_Plus_S6_Fixed10_Rank_Submit.ipynb')
_,parent42=text('42_Anchor0958_S6_SymmetricCrop_Fixed10_Rank_Submit.ipynb')
stage=next(n for n in ast.parse(parent39).body if isinstance(n,ast.FunctionDef) and n.name=='run_stage')
stage_source=ast.get_source_segment(parent39,stage)
visible=value(parent39,'VISIBLE_REF')
worker_base=value(parent39,'S6_SCRIPT')
manifest={}
for tag,module,kind in [('43A',value(parent39,'S6_COMPONENT'),'centre'),
                        ('43B',value(parent42,'S6_COMPONENT'),'symmetric_two_crop')]:
    folder='rsna_'+tag
    stage_worker=worker_base.replace("OUT=Path('/kaggle/working/rsna_39')",f"OUT=Path('/kaggle/working/{folder}')")
    stage_worker=stage_worker.replace("Path('/tmp/rsna39_s6')",f"Path('/tmp/rsna_{tag}_s6')")
    stage_worker=stage_worker.replace("budget=min(8400,float(os.environ['RSNA39_DEADLINE'])-time.time()-3600)",
                                    "budget=min(25200,float(os.environ['RSNA39_DEADLINE'])-time.time()-600)")
    stage_worker=stage_worker.replace("if budget<1200:raise RuntimeError('Insufficient global budget for S6 and protected anchor')",
                                    "if budget<1200:raise RuntimeError('Insufficient standalone S6 budget')")
    # A scientific standalone comparison: never replace S6 by Raptor, anchor or a
    # constant/mean-filled row. The inherited worker retries once; then fail closed.
    needle="p,z,info,failed=S6.run_b(I,'B',I.test_ids,'test',budget)"
    assert stage_worker.count(needle)==1
    stage_worker=stage_worker.replace(needle,needle+"\n    if failed:raise RuntimeError('Standalone comparison requires every S6 row scored; failed='+repr(failed))")
    controller=r'''
import hashlib, importlib.util, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
WORK=Path('/kaggle/working')
OUT=WORK/__FOLDER__
OUT.mkdir(parents=True,exist_ok=True)
START=time.time()
DEADLINE=START+8*3600
EXPERIMENT=__TAG__
REPORT={'experiment':EXPERIMENT,'status':'STARTED','mode':__KIND__,
        'standalone_S6':True,'uses_anchor33':False,'uses_Raptor':False,'uses_R6':False,
        'training_started':False,'gold_used':False,'weights_fitted':False,
        'per_study_model_fallback_allowed':False,'new_public_score':None,
        'output_transform':'per-target percentile rank, same transform for A/B',
        'caveat':'Public comparison of two fixed variants; not evidence of Private top10.'}

def save_report():
    REPORT['total_seconds']=round(time.time()-START,2)
    (OUT/(EXPERIMENT+'_run_audit.json')).write_text(json.dumps(REPORT,indent=2,allow_nan=False))

def clear_partial_submission():
    p=WORK/'submission.csv'
    if p.is_file():p.unlink()
'''.replace('__FOLDER__',repr(folder)).replace('__TAG__',repr(tag)).replace('__KIND__',repr(kind))
    controller+='\n'+stage_source+'\n'
    controller+='S6_COMPONENT = '+repr(module)+'\nS6_SCRIPT = '+repr(stage_worker)+'\nVISIBLE_REF = '+repr(visible)+'\n'
    controller+=r'''

def make_standalone_submission(raw,ids,labels):
    if list(raw.columns)!=['StudyInstanceUID']+labels:
        raise RuntimeError('S6 target order mismatch')
    if not raw.StudyInstanceUID.is_unique or set(raw.StudyInstanceUID)!=set(ids):
        raise RuntimeError('S6 UID coverage mismatch')
    p=raw.set_index('StudyInstanceUID').loc[ids,labels]
    vals=p.to_numpy(float)
    if not np.isfinite(vals).all() or (vals<0).any() or (vals>1).any():
        raise RuntimeError('Invalid/missing standalone probabilities; no fallback is allowed')
    # AUC is rank-based. Use exactly the same monotone output transform in A/B.
    result=p.rank(method='average',pct=True).astype(np.float64)
    return result.reset_index()

clear_partial_submission()
try:
    (OUT/'s6_component.py').write_bytes(S6_COMPONENT.encode('utf-8'))
    spec=importlib.util.spec_from_file_location('s6_standalone_preflight',OUT/'s6_component.py')
    M=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    M.OUT=OUT
    comp=M.find_mount(M.COMP,'test.csv')
    anatomy=M.find_mount(M.HELIOS_SLUG,'anatomy.pt')
    ds=M.find_mount(M.DS_SLUG,'code_manifest.json')
    if M.sha256_file(ds/'code_manifest.json')!='b66c574b5c0c13ccaa49d1219d41a8781431166186ef83e78cab6a9ba6b02d22':
        raise RuntimeError('Attach S6 weights Version5; source manifest differs')
    if M.sha256_file(anatomy/'anatomy.pt')!='5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e':
        raise RuntimeError('Attach exact Heliosli anatomy/code Version1')
    model=ds/M.B_MODEL['file']
    if not model.is_file() or M.sha256_file(model)!=M.B_MODEL['sha256']:
        raise RuntimeError('Pinned S6 checkpoint missing/changed')
    ids=pd.read_csv(comp/'test.csv',dtype={'StudyInstanceUID':str}).StudyInstanceUID.tolist()
    if not ids or len(set(ids))!=len(ids):raise RuntimeError('Invalid test study IDs')
    labels=list(M.LAB)
    columns=list(pd.read_csv(comp/'sample_submission.csv',nrows=0).columns)
    if columns!=['StudyInstanceUID']+labels:raise RuntimeError('Unexpected submission column order')
    REPORT['test_rows']=len(ids)
    REPORT['input_directories']=M.mount_inventory()
    REPORT['component_sha256']=hashlib.sha256(S6_COMPONENT.encode()).hexdigest()
    save_report()
    run_stage('s6',S6_SCRIPT,7.75*3600)
    receipt=json.loads((OUT/'39_s6_receipt.json').read_text())
    if receipt.get('status')!='COMPLETE' or receipt.get('failed'):
        raise RuntimeError('S6 incomplete; no standalone fallback allowed')
    raw=pd.read_csv(OUT/'39_s6_probabilities.csv',dtype={'StudyInstanceUID':str})
    result=make_standalone_submission(raw,ids,labels)
    raw.to_csv(OUT/(EXPERIMENT+'_probabilities.csv'),index=False,float_format='%.17g')
    if set(ids)=={r['StudyInstanceUID'] for r in VISIBLE_REF}:
        ref=pd.DataFrame(VISIBLE_REF).set_index('StudyInstanceUID').loc[ids,labels].to_numpy(float)
        actual=raw.set_index('StudyInstanceUID').loc[ids,labels].to_numpy(float)
        delta=float(np.max(np.abs(ref-actual)))
        REPORT['visible_vs38_max_abs_probability_delta']=delta
        if REPORT['mode']=='centre' and (not np.isfinite(delta) or delta>2e-4):
            raise RuntimeError('Centre-only S6 does not reproduce successful38 visible probabilities')
        if REPORT['mode']!='centre':REPORT['visible_difference_expected']=True
    REPORT['s6_receipt']=receipt
    REPORT['warnings']=[]
    if time.time()>DEADLINE:raise RuntimeError('Global deadline exceeded')
    pending=OUT/'submission.pending.csv'
    result.to_csv(pending,index=False,float_format='%.17g')
    final=pd.read_csv(pending,dtype={'StudyInstanceUID':str})
    if list(final.columns)!=columns or final.StudyInstanceUID.tolist()!=ids:
        raise RuntimeError('Final CSV row/column mismatch')
    values=final[labels].to_numpy(float)
    if not np.isfinite(values).all() or (values<0).any() or (values>1).any():
        raise RuntimeError('Invalid final scores')
    REPORT.update(status='COMPLETE',rows=len(ids),submission_sha256=hashlib.sha256(pending.read_bytes()).hexdigest())
    save_report()
    os.replace(pending,WORK/'submission.csv')
    print(EXPERIMENT+' COMPLETE',json.dumps(REPORT),flush=True)
    display(result.head())
except BaseException as exc:
    clear_partial_submission()
    REPORT.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
    save_report()
    raise
'''
    compile(controller,'<'+tag+'>','exec')
    compile(stage_worker,'<'+tag+' S6 subprocess>','exec')
    heading='S6 centre-only standalone control' if tag=='43A' else 'S6 symmetric two-crop TTA standalone'
    intro=f'''# {tag} — {heading}

**Creates submission.csv for ONE competition submission.** This is standalone S6, NOT anchor33/39 and NOT a10% blend. Its Public score may be lower than0.958; this is an experiment to isolate crop TTA. Do not replace selected final candidates33/39 with these controls merely because they are newer.

43A uses the original centre offset16,16 and unchanged S6 worker.43B uses crop offsets8,8 and24,24, equal mean of complete study logits, then the same S6 output transform as A. Both use the same learned S6 checkpoint, anatomy, windows, crop288/cache320, fp16, chunk96 and one loader worker/GPU. Both export per-target percentile ranks (same AUC ordering as their non-saturated probabilities). No gold fitting, training, Raptor fallback or R6 models.

Attach exactly **THREE inputs**:
1. RSNA Knee competition test DICOMs.
2. [Nartaa S6 all-data weights Version5](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights/versions/5).
3. [Heliosli OrthoFoundation code/anatomy Version1](https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004/versions/1).

**No R6 bundle, publication SWA/Raptor weights, study cache or prior Output needed. T4x2, Internet Off, Save & Run All.** Use separate notebooks for43A/B. Import metadata is not proof the draft attachments are correct. Require `rsna_{tag}/{tag}_run_audit.json` status COMPLETE before submitting. Return each Public score and Output ZIP.

Budgets: eight-hour global cap, S6 worker budget up to seven hours; these are safety timeouts, not measured ETAs.43B performs two S6 forwards per study, but decodes once. A/B can run independently, but submitting both uses today's two slots. Check available GPU quota first. No GPU/hidden-run guarantee:40A's hidden exception is unresolved. Removing anchor stages removes that branch, not proof that all rerun risks are fixed.

Scientific failure policy: one retry by inherited worker; any study still unscored causes a hard error and no submission.csv. No constant fill or silent different-model fallback. This strictness keeps the comparison interpretable but can cause a submission exception for a genuinely unprocessable hidden study.

{'Expected visible control parity against38 is checked.' if tag=='43A' else 'Crop changes are expected to change visible values. Original worker identity is verified before the explicit new patch, whose hash and two offsets are checked in receipts.'} AUC equality/difference can be hidden by rounded Public scores. A positive standalone result is not proof of an anchor gain; a null result does not prove there is no effect. Gold58 has already been reused and is not a clean validation set. Further weight fitting is not part of this test.

Credits/licence/provenance: Nartaa S6, Heliosli code/anatomy; inherit third-party terms. Code/weights have different declared licences; this is not an eligibility determination.
'''
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},
        'kaggle':{'accelerator':'nvidiaTeslaT4','isGpuEnabled':True,'isInternetEnabled':False,'language':'python','sourceType':'notebook',
                  'dataSources':[{'sourceType':'competition','sourceId':154281},{'sourceType':'datasetVersion','sourceId':20844630},
                                 {'sourceType':'datasetVersion','sourceId':20440357}]}},
        'cells':[{'cell_type':'markdown','id':tag+'-intro','metadata':{},'source':intro.splitlines(keepends=True)},
                 {'cell_type':'code','id':tag+'-controller','metadata':{},'execution_count':None,'outputs':[],
                  'source':controller.splitlines(keepends=True)}]}
    name=tag+'_S6_'+('CentreOnly' if tag=='43A' else 'SymmetricCropTTA')+'_Standalone_Submit.ipynb'
    path=ROOT/name
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    manifest[tag]={'file':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                   'component_sha256':hashlib.sha256(module.encode()).hexdigest(),
                   'stage_sha256':hashlib.sha256(stage_worker.encode()).hexdigest(),
                   'gpu_hidden_verified':False,'kind':kind,'no_anchor':True,'strict_no_failed_rows':True}
(ROOT/'43_Standalone_Build_Manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
