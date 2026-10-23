"""Build a short, bounded RadImageNet layer4 fine-tuning preflight notebook.

The input/fold contract and MIL head are copied from notebook 09. This is a
resource and execution gate, not an AUC experiment or Kaggle submission.
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "09_B2_RadImageNet_FrozenMIL_Fold01.ipynb"
OUTPUT = ROOT / "18_B2_RadImageNet_Layer4_Finetune_Preflight.ipynb"


def source(cell: dict) -> str:
    raw = cell.get("source", [])
    return "".join(raw) if isinstance(raw, list) else str(raw)


def markdown(value: str, identifier: str) -> dict:
    return {"cell_type": "markdown", "id": identifier, "metadata": {},
            "source": value.splitlines(keepends=True)}


def code(value: str, identifier: str) -> dict:
    ast.parse(value)
    return {"cell_type": "code", "id": identifier, "execution_count": None,
            "metadata": {}, "outputs": [], "source": value.splitlines(keepends=True)}


INTRO = """# 18 — RadImageNet ResNet50 layer4 fine-tuning preflight

This is a **short resource gate**, not a five-fold training run or submission.
The 0.943 submission remains unchanged. Notebook 17 showed that adding the
frozen B2 model to the 0.943 ensemble did not improve Public LB: G1/G2 scored
0.942 and G3 tied at 0.943. The previous CoAtNet frozen experiment (12) also
lost to B2 on exactly aligned weak OOF. We therefore test whether updating the
last ResNet50 block is feasible before spending hours training a new model.

Attach the same **competition, notebook 02 silver-label Output, complete
`Study_case 05_RSNA` cache Dataset, and public `resnet-50-radimagenet-marwan`
Dataset** used for notebook 09. Notebook 09 head Output is *not* required.
Use **T4×2, Internet Off**; this preflight intentionally uses only GPU 0.

The test runs two short epochs on a deterministic 64-study training subset of
folds 1–4 and a 16-study fold-0 validation subset. It selects at most 16
uniform cache slices per study. Only encoder `layer4` and a fresh MIL head
receive gradients; all BatchNorm running statistics stay frozen. Gold58 is
never loaded by the training dataset and is not used for model selection.
The validation BCE here is only a smoke check, **not** a quality estimate.

Use **Save & Run All**, then send `rsna_b2_layer4_preflight/preflight_18.json`.
The receipt has a 15-minute work cap plus RAM/VRAM checks and a conservative
single-GPU runtime projection. Do not start a full five-fold training run
until this preflight is complete and the projection fits the GPU budget.
"""


PREFLIGHT_CONFIG = """CONFIG = {
    'experiment': 'B2_RadImageNet_layer4_finetune_resource_preflight_v1',
    'seed': 111, 'active_folds': [0, 1, 2, 3, 4],
    'max_tokens': 16, 'image_microbatch': 8, 'study_batch': 2,
    'feature_dim': 2048, 'head_dim': 256,
    'sample_train_studies': 64, 'sample_val_studies': 16,
    'epochs': 2, 'max_preflight_seconds': 900,
    'layer4_lr': 1e-5, 'head_lr': 3e-4, 'weight_decay': 1e-4,
    'min_available_ram_gib': 8.0, 'max_cuda_allocated_gib': 10.0,
    'encoder_sha256': '08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734',
}
"""


TRAIN_CELL = r"""# One-GPU, bounded layer4+MIL training benchmark; no full-dataset optimization.
from contextlib import nullcontext

if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
    raise RuntimeError('Select T4x2 for the bounded layer4 preflight')
DEVICE = torch.device('cuda:0')
torch.cuda.reset_peak_memory_stats(0)

class RadEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet50(weights=None).children())[:-2])
    def forward(self, x):
        return self.backbone(x).mean(dim=(2, 3))

encoder = RadEncoder()
state = torch.load(WEIGHT, map_location='cpu', weights_only=True)
expected = encoder.state_dict()
missing = set(expected) - set(state)
extra = set(state) - set(expected)
bad_shape = [key for key in set(expected) & set(state)
             if expected[key].shape != state[key].shape]
if missing or extra or bad_shape:
    raise RuntimeError(f'RadImageNet checkpoint mismatch: {len(missing)} missing, '
                       f'{len(extra)} extra, {len(bad_shape)} bad shapes')
encoder.load_state_dict(state, strict=True)
del state, expected
gc.collect()
encoder.requires_grad_(False)
encoder.backbone[-1].requires_grad_(True)
encoder = encoder.to(DEVICE)
encoder.eval()  # BatchNorm statistics stay frozen, including in layer4.
trainable_encoder = [p for p in encoder.parameters() if p.requires_grad]
if not trainable_encoder:
    raise RuntimeError('No layer4 parameters were unfrozen')

class StudyMIL(nn.Module):
    def __init__(self):
        super().__init__(); d=CONFIG['head_dim']
        self.project=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,d),nn.GELU(),nn.Dropout(.10))
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

head = StudyMIL().to(DEVICE)
optimizer = torch.optim.AdamW([
    {'params': trainable_encoder, 'lr': CONFIG['layer4_lr']},
    {'params': list(head.parameters()), 'lr': CONFIG['head_lr']},
], weight_decay=CONFIG['weight_decay'])
scaler = torch.amp.GradScaler('cuda', enabled=True)

class CacheStudies(Dataset):
    def __init__(self, indices):
        self.indices = list(map(int, indices))
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, index):
        i = self.indices[index]
        uid = IDS[i]
        if i >= len(weak):
            raise RuntimeError('Gold study entered fine-tuning preflight')
        path = CACHE / f'weak_{uid}.npz'
        with np.load(path, allow_pickle=False) as z:
            if not {'images','slot_ids','positions','slice_indices'}.issubset(z.files):
                raise RuntimeError(f'Invalid cache keys: {path}')
            images = z['images']; slots = z['slot_ids']; pos = z['positions']
            if (images.dtype != np.uint8 or images.ndim != 3 or
                    images.shape[1:] != (224, 224) or
                    len(slots) != len(images) or len(pos) != len(images) or
                    len(z['slice_indices']) != len(images)):
                raise RuntimeError(f'Invalid cache shape: {path}')
        if len(images) == 0 or not np.isfinite(pos).all():
            raise RuntimeError(f'Empty/nonfinite cache: {path}')
        if slots.min() < 0 or slots.max() >= 5:
            raise RuntimeError(f'Invalid slot index: {path}')
        take = np.rint(np.linspace(0, len(images)-1,
                                   min(len(images), CONFIG['max_tokens']))).astype(int)
        if len(np.unique(take)) != len(take):
            raise RuntimeError(f'Duplicate selected slice: {path}')
        count = len(take); limit = CONFIG['max_tokens']
        image_buf = np.zeros((limit, 224, 224), np.uint8)
        slot_buf = np.zeros(limit, np.int64)
        pos_buf = np.zeros(limit, np.float32)
        mask_buf = np.zeros(limit, bool)
        image_buf[:count] = images[take]
        slot_buf[:count] = slots[take]
        pos_buf[:count] = pos[take]
        mask_buf[:count] = True
        return (torch.from_numpy(image_buf), torch.from_numpy(slot_buf),
                torch.from_numpy(pos_buf), torch.from_numpy(mask_buf),
                torch.from_numpy(Y[i]))

def forward_studies(images, slots, positions, masks):
    batch, tokens = images.shape[:2]
    pieces = []
    flat = images.reshape(batch*tokens, 224, 224)
    for start in range(0, len(flat), CONFIG['image_microbatch']):
        gray = flat[start:start+CONFIG['image_microbatch']].to(DEVICE).float()
        gray = gray.div_(127.5).sub_(1.0)
        rgb = gray.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
        with torch.autocast('cuda', dtype=torch.float16):
            pieces.append(encoder(rgb))
    features = torch.cat(pieces, dim=0).reshape(batch, tokens, 2048)
    return head(features, slots.to(DEVICE), positions.to(DEVICE), masks.to(DEVICE))

train_pool = np.flatnonzero(weak.fold.to_numpy() != 0)
val_pool = np.flatnonzero(weak.fold.to_numpy() == 0)
if not set(weak.iloc[train_pool].report_group.astype(str)).isdisjoint(
        set(weak.iloc[val_pool].report_group.astype(str))):
    raise RuntimeError('Report-group leakage in preflight fold 0')
rng = np.random.default_rng(CONFIG['seed'])
train_indices = sorted(rng.choice(train_pool, CONFIG['sample_train_studies'], replace=False))
val_indices = sorted(rng.choice(val_pool, CONFIG['sample_val_studies'], replace=False))
train_loader = DataLoader(CacheStudies(train_indices), batch_size=CONFIG['study_batch'],
                          shuffle=True, generator=torch.Generator().manual_seed(CONFIG['seed']),
                          num_workers=0, pin_memory=False, drop_last=False)
val_loader = DataLoader(CacheStudies(val_indices), batch_size=CONFIG['study_batch'],
                        shuffle=False, num_workers=0, pin_memory=False, drop_last=False)

def memory_snapshot():
    process = psutil.Process()
    return {'process_rss_gib': process.memory_info().rss / 2**30,
            'system_available_gib': psutil.virtual_memory().available / 2**30,
            'cuda_allocated_gib': torch.cuda.memory_allocated(0) / 2**30,
            'cuda_reserved_gib': torch.cuda.memory_reserved(0) / 2**30,
            'cuda_peak_allocated_gib': torch.cuda.max_memory_allocated(0) / 2**30}

def write_report(report):
    (OUT/'preflight_18.json').write_text(json.dumps(report, indent=2, sort_keys=True,
                                               allow_nan=False) + '\n', encoding='utf-8')

report = {
    'status': 'RUNNING', 'experiment': CONFIG['experiment'], 'config': CONFIG,
    'selected_fold': 0, 'train_studies': len(train_indices),
    'validation_studies': len(val_indices),
    'train_uid_sha256': hashlib.sha256('\n'.join(IDS[i] for i in train_indices).encode()).hexdigest(),
    'val_uid_sha256': hashlib.sha256('\n'.join(IDS[i] for i in val_indices).encode()).hexdigest(),
    'encoder_weight_sha256': digest.hexdigest(),
    'trainable_encoder_parameters': sum(p.numel() for p in trainable_encoder),
    'trainable_head_parameters': sum(p.numel() for p in head.parameters()),
    'measurements': [],
    'warning': 'Resource smoke test only. Small validation BCE is not a model-quality estimate.',
}
write_report(report)
started = monotonic()
train_seconds = []
try:
    for epoch in range(1, CONFIG['epochs']+1):
        encoder.eval(); head.train()
        train_start = monotonic(); train_loss = 0.0; trained = 0
        for images, slots, positions, masks, target in train_loader:
            if monotonic()-started > CONFIG['max_preflight_seconds']:
                report['status'] = 'TIME_CAP'; break
            optimizer.zero_grad(set_to_none=True)
            logits = forward_studies(images, slots, positions, masks)
            loss = F.binary_cross_entropy_with_logits(logits, target.to(DEVICE))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable_encoder + list(head.parameters()), 1.0)
            scaler.step(optimizer); scaler.update()
            train_loss += float(loss.detach()) * len(images); trained += len(images)
            mem = memory_snapshot()
            if (mem['system_available_gib'] < CONFIG['min_available_ram_gib'] or
                    mem['cuda_peak_allocated_gib'] > CONFIG['max_cuda_allocated_gib']):
                report['status'] = 'RESOURCE_GUARD'; break
        torch.cuda.synchronize(0)
        train_duration = monotonic()-train_start
        if trained:
            train_seconds.append(train_duration/trained)
        if report['status'] != 'RUNNING':
            break
        encoder.eval(); head.eval()
        val_loss = 0.0; validated = 0
        with torch.inference_mode():
            for images, slots, positions, masks, target in val_loader:
                logits = forward_studies(images, slots, positions, masks)
                loss = F.binary_cross_entropy_with_logits(logits, target.to(DEVICE))
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Nonfinite validation loss')
                val_loss += float(loss) * len(images); validated += len(images)
        torch.cuda.synchronize(0)
        row = {'epoch': epoch, 'train_bce_smoke': train_loss/trained,
               'val_bce_smoke': val_loss/validated,
               'train_studies_completed': trained, 'val_studies_completed': validated,
               'train_seconds': train_duration, 'memory': memory_snapshot()}
        report['measurements'].append(row)
        write_report(report)
        print('EPOCH', row, flush=True)
    if report['status'] == 'RUNNING':
        report['status'] = 'COMPLETE' if len(report['measurements']) == CONFIG['epochs'] else 'PARTIAL'
except RuntimeError as exc:
    report['status'] = 'RUNTIME_ERROR'
    report['error'] = f'{type(exc).__name__}: {str(exc)[:1200]}'
finally:
    report['elapsed_seconds'] = monotonic()-started
    report['final_memory'] = memory_snapshot()
    report['train_seconds_per_study_observed'] = train_seconds
    if train_seconds:
        # Lower bound: excludes validation, startup, checkpointing and Kaggle rerun.
        report['projected_5fold_4epoch_train_only_hours_single_gpu'] = (
            float(np.median(train_seconds))*len(train_pool)*5*4/3600)
    write_report(report)
    print('PREFLIGHT RECEIPT', {k:report.get(k) for k in
          ['status','elapsed_seconds','projected_5fold_4epoch_train_only_hours_single_gpu']},
          flush=True)
"""


def main() -> None:
    source_notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    if len(source_notebook["cells"]) != 4:
        raise RuntimeError("Notebook 09 structure changed")
    asset_cell = source(source_notebook["cells"][1])
    start = asset_cell.index("CONFIG = {")
    end = asset_cell.index("INPUT = Path(", start)
    asset_cell = asset_cell[:start] + PREFLIGHT_CONFIG + asset_cell[end:]
    old_out = "OUT = Path('/kaggle/working/rsna_b2_rad_frozen_fold01')"
    if asset_cell.count(old_out) != 1:
        raise RuntimeError("Notebook 09 output directory anchor changed")
    asset_cell = asset_cell.replace(
        old_out, "OUT = Path('/kaggle/working/rsna_b2_layer4_preflight')", 1)
    if "Report-group leakage" not in asset_cell and "report-group leakage" not in asset_cell:
        raise RuntimeError("Notebook 09 fold group-leakage check missing")
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {key: copy.deepcopy(source_notebook["metadata"][key])
                     for key in ("kernelspec", "language_info", "kaggle")
                     if key in source_notebook["metadata"]},
        "cells": [markdown(INTRO, "rsna18-intro"),
                  code(asset_cell, "rsna18-assets"),
                  code(TRAIN_CELL, "rsna18-train-preflight")],
    }
    notebook["metadata"]["kaggle"]["isGpuEnabled"] = True
    notebook["metadata"]["kaggle"]["isInternetEnabled"] = False
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                      encoding="utf-8")
    print(f"CREATED {OUTPUT.name}: {len(notebook['cells'])} cells, "
          f"{len(notebook['metadata']['kaggle']['dataSources'])} inherited inputs")


if __name__ == "__main__":
    main()
