"""Reconcile notebook 20 OOF and inspect fixed, exploratory blends offline."""
from io import BytesIO
from pathlib import Path
import json
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT=Path(__file__).parent
LABELS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
        'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']


def csv_from_zip(path,name):
    with zipfile.ZipFile(path) as archive:
        return pd.read_csv(BytesIO(archive.read(name)),dtype={'StudyInstanceUID':str})


result=ROOT/'results 10.zip'
silver=csv_from_zip(ROOT/'RSNA_Knee_Silver_Label_v1_Artifacts.zip',
                    'L1_family_mean_RECOMMENDED.csv')
folds=csv_from_zip(ROOT/'RSNA_05A_Fold0_DenseMIL_Artifacts.zip','fold_assignments.csv')
truth=silver[['StudyInstanceUID']+LABELS].merge(
    folds[['StudyInstanceUID','report_group','fold']],on='StudyInstanceUID',
    validate='one_to_one')
assert len(truth)==4349 and truth.StudyInstanceUID.is_unique
pred={}
for d in [80,32,16]:
    table=csv_from_zip(result,f'rsna_b2_slice_density/density_{d}_weak_oof.csv')
    assert len(table)==4349 and table.StudyInstanceUID.is_unique
    aligned=truth[['StudyInstanceUID','fold']].merge(
        table,on=['StudyInstanceUID','fold'],how='left',validate='one_to_one',indicator=True)
    assert aligned['_merge'].eq('both').all() and aligned[LABELS].notna().all().all()
    pred[d]=aligned[LABELS].to_numpy(float)

y=(truth[LABELS].to_numpy(float)>=.5).astype(int)
groups=truth.report_group.astype(str).to_numpy()
unique_groups,group_index=np.unique(groups,return_inverse=True)
members=[np.flatnonzero(group_index==i) for i in range(len(unique_groups))]


def macro_auc(rows,p):
    scores=[]
    for j in range(12):
        yy=y[rows,j]
        if len(np.unique(yy))<2: return np.nan
        scores.append(roc_auc_score(yy,p[rows,j]))
    return float(np.mean(scores))


all_rows=np.arange(len(y))
candidate={'80':pred[80],'32':pred[32],'16':pred[16],
           '80_20_probability':.8*pred[80]+.2*pred[32],
           '50_50_probability':.5*pred[80]+.5*pred[32]}
scores={name:macro_auc(all_rows,p) for name,p in candidate.items()}
print('Groups:',len(unique_groups),'Weak rows:',len(y))
print('OOF macro AUC:',json.dumps(scores,indent=2))

rng=np.random.default_rng(20261009)
bootstrap={name:[] for name in ['32','16','80_20_probability','50_50_probability']}
for _ in range(250):
    sampled_groups=rng.integers(0,len(unique_groups),len(unique_groups))
    rows=np.concatenate([members[i] for i in sampled_groups])
    base=macro_auc(rows,pred[80])
    for name in bootstrap:
        bootstrap[name].append(macro_auc(rows,candidate[name])-base)
for name,values in bootstrap.items():
    values=np.asarray(values,float)
    print(name,'delta_vs_80',scores[name]-scores['80'],
          'group_bootstrap_95',np.nanquantile(values,[.025,.975]).tolist())

for label_index,label in enumerate(LABELS):
    yy=y[:,label_index]
    print(label,'AUC_80',roc_auc_score(yy,pred[80][:,label_index]),
          'AUC_80_20',roc_auc_score(yy,candidate['80_20_probability'][:,label_index]))
