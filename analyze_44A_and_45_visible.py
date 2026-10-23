"""Local output audit; visible predictions only, without labels or metric claims."""
import ast, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from rsna45_fusion import final_fusion_probe, WEIGHTS_33
ROOT=Path(__file__).resolve().parent
def frame(z,name):
    return pd.read_csv(io.BytesIO(z.read(name)),dtype={'StudyInstanceUID':str})
def values(df,ids,labels):
    return df.set_index('StudyInstanceUID').loc[ids,labels].to_numpy(float)
def order_changes(a,b):
    return int(sum(np.count_nonzero(np.sign(a[i+1:]-a[i])!=np.sign(b[i+1:]-b[i])) for i in range(len(a)-1)))
with zipfile.ZipFile(ROOT/'results 40B.zip') as z40,zipfile.ZipFile(ROOT/'results 44A.zip') as z44:
    f40,f44=frame(z40,'submission.csv'),frame(z44,'submission.csv')
    ids=f40.StudyInstanceUID.tolist()
    labels=list(f40.columns[1:])
    a,b=values(f40,ids,labels),values(f44,ids,labels)
    delta=np.abs(a-b)
    run=json.loads(z44.read('rsna_39/39_run_audit.json'))
    ablation=json.loads(z44.read('44_postprocessor_audit.json'))
    assert run['status']=='COMPLETE' and not run['warnings']
    assert ablation['disabled']=='--fluid-stacker'
    args=ablation['original_worker_args'].copy()
    i=args.index('--fluid-stacker')
    del args[i:i+2]
    assert args==ablation['actual_worker_args'] and '--state-stacker' in args and '--flip-tta' in args
    audit={'status':'AUDITED','visible_rows':len(ids),'run_status':run['status'],
           'archive_sha256':hashlib.sha256((ROOT/'results 44A.zip').read_bytes()).hexdigest(),
           'elapsed_seconds':run['total_seconds'],'disabled':ablation['disabled'],
           'worker_argument_removal_verified':True,'warnings':run['warnings'],
           'changed_final_columns':[c for c,d in zip(labels,delta.max(axis=0)) if d>1e-10],
           'max_abs_final_delta_vs40B':float(delta.max()),
           'ordering_changes_vs40B':{c:order_changes(a[:,j],b[:,j]) for j,c in enumerate(labels)},
           's6_max_abs_delta_vs40B':float(np.abs(values(frame(z40,'rsna_39/39_s6_probabilities.csv'),ids,labels)-values(frame(z44,'rsna_39/39_s6_probabilities.csv'),ids,labels)).max()),
           'public_score_user_reported':.958,'hidden_predictions_available':False,
           'independent_validation':False,'auc_computed':False}
    (ROOT/'44A_Output_Audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    nb=json.loads((ROOT/'40B_Anchor0958_Plus_S6_Fixed10_Probability_Submit.ipynb').read_text(encoding='utf-8'))
    tree=ast.parse(''.join(nb['cells'][1]['source']))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fuse_anchor_s6')
    scope={}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original40Bfusion>','exec'),scope)
    anchor,s6=frame(z40,'39_anchor33.csv'),frame(z40,'rsna_39/39_s6_probabilities.csv')
    receipt=json.loads(z40.read('blend_receipt.json'))
    assert not any(receipt[k] for k in ('raptor384_failed','raptor224_failed','ours_failed'))
    failed=json.loads(z40.read('rsna_39/39_s6_receipt.json'))['failed']
    baseline=scope['fuse_anchor_s6'](anchor,s6,ids,labels,failed)
    np.testing.assert_allclose(values(baseline,ids,labels),a,rtol=0,atol=1e-7)
    manifest=json.loads((ROOT/'45_Build_Manifest.json').read_text())
    coverage={'schema':'45_coverage_v1','uids':ids,'labels':labels,
              'original_anchor_script_sha256':manifest['original_anchor_script_sha256'],
              'failed384':[False]*len(ids),'failed224':[False]*len(ids),'failed6':[False]*len(ids),
              'a224':.35,'default_raptor_weight':.5,'raptor_per_class':WEIGHTS_33}
    raw=[frame(z40,name) for name in ('raptor_raw.csv','raptor224_raw.csv','ours_sub.csv')]
    comparisons={}
    for tag in manifest['experiments']:
        output,detail=final_fusion_probe(tag,anchor,s6,*raw,ids,labels,failed,coverage,receipt,baseline,manifest['original_anchor_script_sha256'])
        m=values(output,ids,labels)
        detail['visible_pairwise_order_changes_vs40B']={c:order_changes(a[:,j],m[:,j]) for j,c in enumerate(labels)}
        comparisons[tag]=detail
    result={'status':'PASS','baseline40B_reproduced':True,'visible_rows':len(ids),
            'formulas_frozen_before_comparison':True,'comparisons':comparisons,
            'not_a_metric_or_validation':True,'gpu_or_hidden_execution_tested':False}
    (ROOT/'45_Visible_Regression.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('44A audited;40B reproduced;all five45 formulas pass real visible reconstruction. No AUC/private evidence.')
