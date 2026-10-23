"""Static and checkpoint-provenance checks for notebook 21."""
from io import BytesIO
from pathlib import Path
import ast
import json
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


root=Path(__file__).parent
nb=json.loads((root/'21_B2_Density80_32_FixedBlend_Submit.ipynb').read_text(encoding='utf-8'))
assert len(nb['cells'])==5
for i,cell in enumerate(nb['cells']):
    if cell['cell_type']=='code': ast.parse(''.join(cell['source']),filename=f'cell_{i}')
first=''.join(nb['cells'][1]['source'])
engine=''.join(nb['cells'][3]['source'])
last=''.join(nb['cells'][4]['source'])
assert "REPORT.get('status') != 'COMPLETE'" in first
assert "REPORT.get('control_fold_auc_deviations')" in first
assert "ck.get('density') != density" in first
assert "pred=.8*pred80+.2*pred32" in engine
assert "if max_control_diff > 1e-4:" in last
assert "submission.to_csv(path,index=False)" in last

with zipfile.ZipFile(root/'results_09.zip') as old_zip,zipfile.ZipFile(root/'results 10.zip') as new_zip:
    old_oof=pd.read_csv(BytesIO(old_zip.read('rsna_b2_rad_frozen_fold01/weak_oof.csv')),
                        dtype={'StudyInstanceUID':str})
    new_oof=pd.read_csv(BytesIO(new_zip.read('rsna_b2_slice_density/density_80_weak_oof.csv')),
                        dtype={'StudyInstanceUID':str})
    aligned=old_oof.merge(new_oof,on=['StudyInstanceUID','fold'],
                          suffixes=('_old','_new'),validate='one_to_one')
    assert len(aligned)==4349
    labels=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
            'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
    oof_diff=max(float(np.abs(aligned[f'{label}_old']-aligned[f'{label}_new']).max())
                 for label in labels)
    print('max_80_oof_probability_diff',oof_diff)
    assert oof_diff<1e-4
    for fold in range(5):
        old=torch.load(BytesIO(old_zip.read(
            f'rsna_b2_rad_frozen_fold01/fold_{fold}_head.pt')),
            map_location='cpu',weights_only=True)['state_dict']
        new=torch.load(BytesIO(new_zip.read(
            f'rsna_b2_slice_density/density_80_fold_{fold}_head.pt')),
            map_location='cpu',weights_only=True)['state_dict']
        assert old.keys()==new.keys()
        max_diff=max(float((old[key]-new[key]).abs().max()) for key in old)
        print('fold',fold,'max_checkpoint_tensor_diff',max_diff)
        assert max_diff<1e-4

for n in [1,10,32,80]:
    take=np.rint(np.linspace(0,n-1,min(n,32))).astype(int)
    assert len(take)==min(n,32) and len(np.unique(take))==len(take)
    assert take[0]==0 and take[-1]==n-1

class FakeEncoder(nn.Module):
    def forward(self,x):
        return torch.ones((len(x),2048),device=x.device)


class FakeHead(nn.Module):
    def __init__(self,logit):
        super().__init__(); self.logit=logit
    def forward(self,x,s,p,m):
        assert x.shape[1] in (32,80)
        assert int(m.sum())<=x.shape[1]
        return torch.full((len(x),12),self.logit,device=x.device)


env={'torch':torch,'np':np,'nn':nn,'DEVICE':torch.device('cpu'),
     'LABELS':labels,'STATS':Counter(),
     'CONFIG':{'max_tokens':80,'feature_dim':2048,'image_microbatch':24},
     'encoder':FakeEncoder(),
     'heads':{80:[FakeHead(.2) for _ in range(5)],
              32:[FakeHead(-.2) for _ in range(5)]}}
function_code=engine[engine.index('@torch.inference_mode()'):]
exec(function_code,env)
for n in [0,1,10,32,80]:
    env['load_study']=lambda row,n=n:(str(row),
        [np.zeros((224,224),dtype=np.uint8) for _ in range(n)],
        [i%5 for i in range(n)],
        [i/max(1,n-1) for i in range(n)])
    uid,blend,control,aux=env['predict_study']('dummy')
    assert uid=='dummy' and blend.shape==control.shape==aux.shape==(12,)
    if n:
        expected=.8*torch.sigmoid(torch.tensor(.2))+.2*torch.sigmoid(torch.tensor(-.2))
        assert np.allclose(blend,float(expected),atol=1e-6)
    else:
        assert np.all(blend==.5)

print('Notebook 21: syntax, control provenance, token selection and mock inference passed.')
