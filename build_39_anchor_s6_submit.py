"""Build a fixed rank-scale probe; preserve protected anchor33 on disk."""
import ast, copy, hashlib, json, zipfile, io
from pathlib import Path
import csv

ROOT=Path(__file__).resolve().parent
ANCHOR_SHA='685e6e0e65c3c5ed29b11c0b584887d894cd07f274de0b9f83370465ab7b4fde'
anchor_path=ROOT/'33_RSNA_0958_R6_Raptor_Submit.ipynb'
assert hashlib.sha256(anchor_path.read_bytes()).hexdigest()==ANCHOR_SHA
anchor=json.loads(anchor_path.read_text(encoding='utf-8'))
diag=json.loads((ROOT/'38_S6_Complementarity_Gold_Diagnostic.ipynb').read_text(encoding='utf-8'))
setup=''.join(diag['cells'][1]['source'])
module=ast.literal_eval(next(n.value for n in ast.parse(setup).body if isinstance(n,ast.Assign)
                            and any(isinstance(t,ast.Name) and t.id=='S6_SOURCE' for t in n.targets)))
with zipfile.ZipFile(ROOT/'results 38.zip') as z:
    visible=list(csv.DictReader(io.StringIO(z.read('rsna_38_s6/38_s6_visible_probabilities.csv').decode())))
    executed_module=z.read('rsna_38_s6/s6_component_b.py').decode()
assert executed_module==module

# A fresh process runs the original seven code cells. Convert %%writefile only;
# change just the final export destination, leaving all prediction/ranking logic intact.
anchor_parts=['from pathlib import Path\n'
              'def display(x):\n    print(x.to_string() if hasattr(x,"to_string") else str(x),flush=True)\n']
for i,c in enumerate(anchor['cells']):
    if c['cell_type']!='code':continue
    text=''.join(c['source'])
    if text.startswith('%%writefile '):
        first,body=text.split('\n',1)
        filename=first[len('%%writefile '):].strip()
        text='Path('+repr(filename)+').write_bytes('+repr(body.encode())+')\n'
    elif i==8:
        assert text.count("sub.to_csv(WORK / 'submission.csv', index=False)")==1
        text=text.replace("sub.to_csv(WORK / 'submission.csv', index=False)",
                          "sub.to_csv(WORK / '39_anchor33.csv', index=False)")
    compile(text,f'<anchor33 cell{i}>','exec')
    anchor_parts.append(text)
anchor_script='\n\n'.join(anchor_parts)
compile(anchor_script,'<isolated anchor33>','exec')

s6_script=r'''
import importlib.util, json, os, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
OUT=Path('/kaggle/working/rsna_39')
spec=importlib.util.spec_from_file_location('s6_39',OUT/'s6_component.py')
S6=importlib.util.module_from_spec(spec)
spec.loader.exec_module(S6)
S6.OUT=OUT/'s6_outputs'
S6.OUT.mkdir(parents=True,exist_ok=True)
S6.TMP=Path('/tmp/rsna39_s6')
S6.TMP.mkdir(parents=True,exist_ok=True)
S6.B_GEOMETRY=dict(S6.B_GEOMETRY,chunk=96)
S6.B_WORKERS=1
try:
    ds=S6.find_mount(S6.DS_SLUG,'code_manifest.json')
    if S6.sha256_file(ds/'code_manifest.json')!='b66c574b5c0c13ccaa49d1219d41a8781431166186ef83e78cab6a9ba6b02d22':
        raise RuntimeError('S6 manifest changed; attach Version5')
    I=S6.prepare_inputs()
    env=S6.probe_env()
    if not env.get('cuda_available'):raise RuntimeError('CUDA required')
    S6._RECEIPT['env']=env
    probe=S6.probe_b_stack(I)
    budget=min(8400,float(os.environ['RSNA39_DEADLINE'])-time.time()-3600)
    if budget<1200:raise RuntimeError('Insufficient global budget for S6 and protected anchor')
    p,z,info,failed=S6.run_b(I,'B',I.test_ids,'test',budget)
    pd.DataFrame(p,index=pd.Index(I.test_ids,name='StudyInstanceUID'),columns=S6.LAB).to_csv(OUT/'39_s6_probabilities.csv')
    S6.save_json(OUT/'39_s6_receipt.json',{'status':'COMPLETE','inputs':S6._RECEIPT['inputs'],
        'environment':env,'probe':probe,'run':info,'failed':failed,'model_fallback_to_raptor':False})
except BaseException as exc:
    S6.kill_all_children()
    S6.save_json(OUT/'39_s6_error.json',{'error':repr(exc),'traceback':traceback.format_exc()})
    raise
'''

fusion=r'''
def fuse_anchor_s6(anchor_frame,s6_frame,ids,labels,failed):
    import numpy as np
    import pandas as pd
    for name,frame in [('anchor33',anchor_frame),('S6',s6_frame)]:
        if list(frame.columns)!=['StudyInstanceUID']+labels:
            raise RuntimeError(name+' column identity mismatch')
        if not frame.StudyInstanceUID.is_unique or set(frame.StudyInstanceUID)!=set(ids):
            raise RuntimeError(name+' UID coverage mismatch')
    a=anchor_frame.set_index('StudyInstanceUID').loc[ids,labels].to_numpy(float)
    b=s6_frame.set_index('StudyInstanceUID').loc[ids,labels].to_numpy(float)
    if set(failed)-set(ids):raise RuntimeError('Unknown S6 failed UID')
    ok=np.array([u not in failed for u in ids])
    if not ok.any():raise RuntimeError('No S6 study scored')
    if not np.isfinite(a).all() or (a<0).any() or (a>1).any():raise RuntimeError('Invalid anchor33 output')
    if not np.isfinite(b[ok]).all() or (b[ok]<0).any() or (b[ok]>1).any():raise RuntimeError('Invalid scored S6 output')
    if np.isfinite(b[~ok]).any():raise RuntimeError('S6 failed rows unexpectedly contain predictions')
    # Preserve anchor's existing rank blend; rank S6 over its own scored rows only.
    out=a.copy()
    ranks=pd.DataFrame(b[ok]).rank(method='average',pct=True).to_numpy(float)
    out[ok]=.90*a[ok]+.10*ranks
    frame=pd.DataFrame(out.astype(np.float32),columns=labels)
    frame.insert(0,'StudyInstanceUID',ids)
    return frame
'''

controller=r'''
import hashlib, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
WORK=Path('/kaggle/working')
OUT=WORK/'rsna_39'
OUT.mkdir(parents=True,exist_ok=True)
START=time.time()
DEADLINE=START+8*3600
REPORT={'experiment':'39_anchor33_90_S6rank_10','status':'STARTED',
        'anchor_public_user_confirmed':.958,'new_public_score':None,
        'training_started':False,'weights_fitted_on_gold':False,
        'fusion':'0.90 * unchanged33_rank_blend + 0.10 * rank(S6)',
        'note':'Different from the probability blend measured on reused gold; improvement unproven.'}

def save_report():
    REPORT['total_seconds']=round(time.time()-START,2)
    (OUT/'39_run_audit.json').write_text(json.dumps(REPORT,indent=2,allow_nan=False))

def clear_partial_submission():
    # Only this run's final scoring file; never unlink attached input data.
    p=WORK/'submission.csv'
    if p.is_file():p.unlink()

def run_stage(name,script,max_seconds):
    path=OUT/(name+'.py')
    path.write_bytes(script.encode('utf-8'))
    log_path=OUT/(name+'.log')
    budget=min(max_seconds,DEADLINE-time.time()-120)
    if budget<=0:raise RuntimeError('Global eight-hour budget exhausted before '+name)
    env=dict(os.environ,RSNA39_DEADLINE=str(DEADLINE),PYTHONDONTWRITEBYTECODE='1')
    started=time.time()
    with log_path.open('w') as log:
        p=subprocess.Popen([sys.executable,'-u',str(path)],stdout=log,stderr=subprocess.STDOUT,
                           env=env,cwd=str(WORK),start_new_session=True)
        last=0
        try:
            while p.poll() is None:
                if time.time()-started>budget:raise TimeoutError(name+' exceeded stage/global budget')
                if time.time()-last>=60:
                    last=time.time()
                    tail=log_path.read_text(errors='replace')[-1000:]
                    print(name,'elapsed',round(time.time()-started),'seconds\n'+tail,flush=True)
                time.sleep(2)
            if p.returncode!=0:
                raise RuntimeError(name+' failed rc='+str(p.returncode)+'\n'+log_path.read_text(errors='replace')[-4500:])
        finally:
            # Reap any surviving DataLoader/model subprocesses, including on failure.
            # Some inherited launchers may create their own process groups.
            try:
                import psutil
                descendants=psutil.Process(p.pid).children(recursive=True)
                for child in reversed(descendants):
                    try:child.kill()
                    except psutil.NoSuchProcess:pass
                psutil.wait_procs(descendants,timeout=10)
            except (ImportError,ProcessLookupError):pass
            except Exception as cleanup_error:
                print('Cleanup diagnostic:',repr(cleanup_error),flush=True)
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            p.wait(timeout=30)
    REPORT[name]={'seconds':round(time.time()-started,2),'script_sha256':hashlib.sha256(script.encode()).hexdigest()}
    save_report()
'''
controller+='\nS6_COMPONENT = '+repr(module)+'\nS6_SCRIPT = '+repr(s6_script)+'\nANCHOR_SCRIPT = '+repr(anchor_script)+'\nVISIBLE_REF = '+repr(visible)+'\n'
controller+=fusion
controller+=r'''
clear_partial_submission()
try:
    (OUT/'s6_component.py').write_bytes(S6_COMPONENT.encode('utf-8'))
    # Resolve required inputs before spending GPU time on either model stage.
    import importlib.util
    spec=importlib.util.spec_from_file_location('rsna39_preflight',OUT/'s6_component.py')
    M=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    M.OUT=OUT
    comp=M.find_mount(M.COMP,'test.csv')
    M.find_mount(M.HELIOS_SLUG,'anatomy.pt')
    publication=M.find_mount('rsna-knee-publication-swa-weights-20261007','raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt')
    for filename,digest in {
        'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt':'7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef',
        'raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt':'3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6',
    }.items():
        if not (publication/filename).is_file() or M.sha256_file(publication/filename)!=digest:
            raise RuntimeError('Missing/changed anchor33 publication checkpoint: '+filename)
    r6=M.find_mount('rsna-knee-r6-frozen-fe055406-w1','manifest.json')
    if M.sha256_file(r6/'manifest.json')!='6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648':
        raise RuntimeError('R6 is not exact33 v1/w1 bundle; do not substitute latest/R7')
    ds=M.find_mount(M.DS_SLUG,'code_manifest.json')
    if M.sha256_file(ds/'code_manifest.json')!='b66c574b5c0c13ccaa49d1219d41a8781431166186ef83e78cab6a9ba6b02d22':
        raise RuntimeError('Attach S6 Version5')
    ids=pd.read_csv(comp/'test.csv',dtype={'StudyInstanceUID':str}).StudyInstanceUID.tolist()
    labels=list(M.LAB)
    if not ids or len(set(ids))!=len(ids):raise RuntimeError('Invalid test study IDs')
    REPORT['input_directories']=M.mount_inventory()
    REPORT['test_rows']=len(ids)
    save_report()
    run_stage('s6',S6_SCRIPT,9000)
    s6=pd.read_csv(OUT/'39_s6_probabilities.csv',dtype={'StudyInstanceUID':str})
    rec=json.loads((OUT/'39_s6_receipt.json').read_text())
    if rec.get('status')!='COMPLETE':raise RuntimeError('S6 receipt incomplete')
    failed=rec['failed']
    if len(failed)>M.b_fail_cap(len(ids)):raise RuntimeError('S6 failure allowance exceeded')
    REPORT['s6_failed_uids']=failed
    # Matching visible IDs are a regression check against the successful 38 run,
    # not a hard-coded replacement for hidden inference.
    if set(ids)=={r['StudyInstanceUID'] for r in VISIBLE_REF}:
        reference=pd.DataFrame(VISIBLE_REF).set_index('StudyInstanceUID').loc[ids,labels].to_numpy(float)
        actual=s6.set_index('StudyInstanceUID').loc[ids,labels].to_numpy(float)
        delta=float(np.max(np.abs(reference-actual)))
        REPORT['visible_s6_vs38_max_abs_delta']=delta
        if not np.isfinite(delta) or delta>2e-4:raise RuntimeError('S6 visible predictions drift from successful38; inspect before submitting')
    run_stage('anchor33',ANCHOR_SCRIPT,DEADLINE-time.time()-120)
    anchor_frame=pd.read_csv(WORK/'39_anchor33.csv',dtype={'StudyInstanceUID':str})
    result=fuse_anchor_s6(anchor_frame,s6,ids,labels,failed)
    # Keep underlying anchor fallbacks visible rather than claiming complete model coverage.
    anchor_receipt=json.loads((WORK/'blend_receipt.json').read_text())
    REPORT['anchor33_blend_receipt']=anchor_receipt
    warnings=[]
    if failed:warnings.append('S6_FAILED_ROWS_USE_ANCHOR33_ONLY')
    if any(anchor_receipt.get(k,0) for k in ('raptor384_failed','raptor224_failed','ours_failed')):
        warnings.append('ANCHOR33_INTERNAL_FALLBACKS')
    if time.time()>DEADLINE:raise RuntimeError('Deadline exceeded before final export')
    temp=OUT/'submission.pending.csv'
    result.to_csv(temp,index=False)
    check=pd.read_csv(temp,dtype={'StudyInstanceUID':str})
    if check.StudyInstanceUID.tolist()!=ids or not np.isfinite(check[labels].to_numpy()).all():
        raise RuntimeError('Final CSV identity/nonfinite error')
    REPORT.update(status='COMPLETE',warnings=warnings,rows=len(ids),
                  submission_sha256=hashlib.sha256(temp.read_bytes()).hexdigest())
    save_report()
    os.replace(temp,WORK/'submission.csv')
    print('39 COMPLETE',json.dumps(REPORT),flush=True)
    display(result.head())
except BaseException as exc:
    clear_partial_submission()
    REPORT.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
    save_report()
    raise
'''
compile(controller,'<39 parent>','exec')

intro='''# 39 — protected 0.958 anchor33 + fixed 10% S6 rank probe

This notebook **creates submission.csv and can be submitted**. It has no verified Public Score yet. 0.939505 in experiment38 was a reused Gold58 diagnostic, NOT a Public score and NOT exact33. This probe does not promise a gain over 0.958.

The original33 prediction and rank-fusion logic is preserved in an isolated process; only its final CSV export is renamed. S6 uses the checkpoint/worker that scored all58 gold in38, in another process, with no model fallback. Final = **90% unchanged33 rank blend + 10% S6 percentile rank**, fixed before this run. This differs from the probability-scale diagnostic blend. No gold labels or report text are used here. No training, weight fitting, phase/crop/pooling changes. Keep33 unchanged.

Attach **five inputs**: competition test DICOMs; Nartaa publication SWA weights Version1; the exact R6 frozen **w1 bundle used by33** (not latest/R7); Heliosli OrthoFoundation code/anatomy Version1; **Nartaa S6 all-data weights Version5**. Metadata pins them, but verify actual mounts in your draft. No old Output or study cache needed.

**T4x2, Internet Off, Save & Run All.** Stages run serially in separate process groups to release memory. Global eight-hour deadline, not an ETA. Hidden-run speed/OOM and score are unverified. Do not submit a failed/partial run. Read `rsna_39/39_run_audit.json`: require COMPLETE and inspect warnings. Full-stage failure removes partial submission.csv. A small capped number of S6 per-study failures uses anchor33 for those rows and is recorded; no silent full-component fallback.

Send Public Score and full Output ZIP. Credits: anchor33's original authors, Nartaa S6, Heliosli code/anatomy. Code/weight licences and external-data provenance remain separate; inherited conditions apply. This notebook is not an eligibility determination.
'''
def cell(kind,text,i):
    d={'cell_type':kind,'id':f'rsna39-{i}','metadata':{},'source':text.splitlines(keepends=True)}
    if kind=='code':d.update(execution_count=None,outputs=[])
    return d
metadata=copy.deepcopy(anchor['metadata'])
for key in ('papermill','widgets'):metadata.pop(key,None)
metadata['kaggle']['dataSources'].append({'sourceType':'datasetVersion','sourceId':20844630})
nb={'nbformat':4,'nbformat_minor':5,'metadata':metadata,
    'cells':[cell('markdown',intro,0),cell('code',controller,1)]}
target=ROOT/'39_Anchor0958_Plus_S6_Fixed10_Rank_Submit.ipynb'
target.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
manifest={'anchor_source_sha256':ANCHOR_SHA,'anchor_adapter_sha256':hashlib.sha256(anchor_script.encode()).hexdigest(),
          's6_component_sha256':hashlib.sha256(module.encode()).hexdigest(),
          'notebook_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
          'fusion':'0.9*unchanged33 + 0.1*rank(S6)','gpu_submission_verified':False}
(ROOT/'39_Build_Manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(target.name, target.stat().st_size,'bytes; all sources compile; no inference executed.')
