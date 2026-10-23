"""Local CPU smoke checks for generated notebook 20; no Kaggle inputs needed."""
import ast
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


root=Path(__file__).parent
notebook=json.loads((root/'20_B2_RadImageNet_Slice_Density_Ablation.ipynb').read_text(encoding='utf-8'))
assert len(notebook['cells'])==4
for i,cell in enumerate(notebook['cells']):
    if cell['cell_type']=='code': ast.parse(''.join(cell['source']),filename=f'cell_{i}')
asset=''.join(notebook['cells'][1]['source'])
train=''.join(notebook['cells'][3]['source'])
assert "'densities': [80,32,16]" in asset
assert "'max_tokens': 80" in asset
assert 'gold_indices=np.arange(len(weak),len(IDS))' in train
assert "joined=base.merge(other,on=['StudyInstanceUID','fold']" in train

class Dataset:
    pass

env={'Dataset':Dataset,'np':np,'torch':torch,'nn':nn,
     'CONFIG':{'densities':[80,32,16],'feature_dim':2048,'head_dim':256},
     'features':np.zeros((4,80,2048),dtype=np.float16),
     'slot_ids':np.zeros((4,80),dtype=np.int64),
     'positions':np.zeros((4,80),dtype=np.float32),
     'masks':np.zeros((4,80),dtype=bool),
     'Y':np.zeros((4,12),dtype=np.float32),
     'IDS':['empty','ten','thirtytwo','eighty']}
for i,t in enumerate([0,10,32,80]):
    env['masks'][i,:t]=True
    env['features'][i,:t,0]=np.arange(t)
    env['positions'][i,:t]=np.arange(t)
    env['slot_ids'][i,:t]=np.arange(t)%5
exec(train[train.index('class FeatureDataset'):train.index('class StudyMIL')],env)
FeatureDataset=env['FeatureDataset']
for density in [80,32,16]:
    ds=FeatureDataset(range(4),density)
    for i,t in enumerate([0,10,32,80]):
        ix,x,s,p,m,y=ds[i]
        assert ix==i and x.shape==(density,2048)
        assert int(m.sum())==min(t,density)
        assert len(torch.unique(x[m,0]))==int(m.sum())
        if density==80:
            assert torch.equal(x,torch.from_numpy(env['features'][i]))
            assert torch.equal(m,torch.from_numpy(env['masks'][i]))
        elif t:
            expected=np.rint(np.linspace(0,t-1,min(t,density))).astype(int)
            assert np.array_equal(x[m,0].numpy(),expected)

exec(train[train.index('class StudyMIL'):train.index('def loader(')],env)
model=env['StudyMIL']()
ds=FeatureDataset([0,3],16)
batch=[ds[0],ds[1]]
x=torch.stack([row[1] for row in batch])
s=torch.stack([row[2] for row in batch])
p=torch.stack([row[3] for row in batch])
m=torch.stack([row[4] for row in batch])
logits=model(x,s,p,m)
assert logits.shape==(2,12) and torch.isfinite(logits).all()
logits.sum().backward()
assert model.weight.grad is not None and torch.isfinite(model.weight.grad).all()

print('Notebook 20: syntax, control tensors, token selection, padding and MIL forward/backward passed.')
