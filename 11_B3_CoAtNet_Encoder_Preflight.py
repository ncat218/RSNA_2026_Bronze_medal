# %% [markdown]
# # 11 — B3 CoAtNet encoder preflight (short GPU gate)
#
# This is the next strong-backbone experiment after B2's 0.914 standalone
# Public score. It does **not** train, use Gold58 labels, blend with 0.943, or
# create a submission. It benchmarks a pinned ImageNet-12k/1k pretrained
# CoAtNet on 64 studies from the existing 05A 224px cache. The output contains
# the exact pretrained safetensors file for later offline five-fold training.
#
# Inputs: `Study_case 05_RSNA` Dataset containing
# `rsna_05a_dense_mil/study_cache` (at least 4,349 weak and 58 gold NPZ).
# The RSNA competition input is **not needed** for this preflight. Use GPU
# T4×2 (the code uses GPU 0 only). Enable Internet to download the pinned
# Hugging Face weight; alternatively attach a Dataset containing the exact
# `model.safetensors` and set `OFFLINE_WEIGHT_PATH` below. If neither source
# is available, the first cell fails before touching the GPU.
#
# Save & Run All, then send `b3_preflight.json` from Output. Do not start
# full B3 training solely because the model imports successfully.

# %%
from pathlib import Path
from time import monotonic
import gc
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import torch

try:
    import timm
    import huggingface_hub
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
except ImportError as exc:
    raise RuntimeError(
        'Kaggle environment needs timm, huggingface_hub, and safetensors. '
        'No GPU work was started; attach a compatible offline package or enable Internet.'
    ) from exc

CONFIG = {
    'experiment': 'B3_CoAtNet_ImageNet12k_1k_frozen_encoder_preflight_v1',
    'model_name': 'coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k',
    'hf_repo': 'timm/coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k',
    'hf_revision': 'd65969f6b8a7f779c76343b251b4decefff1973b',
    'sample_studies': 64,
    'microbatch': 4,
    'max_preflight_seconds': 900,
    'max_cache_slices': 80,
    'max_gpu_reserved_gib': 12.0,
}
OFFLINE_WEIGHT_PATH = ''  # Optional exact /kaggle/input/.../model.safetensors
INPUT = Path('/kaggle/input')
OUT = Path('/kaggle/working/rsna_b3_coatnet_preflight')
OUT.mkdir(parents=True, exist_ok=True)

def cache_directories(max_depth=9):
    found = []
    for base, dirs, _ in os.walk(INPUT):
        depth = len(Path(base).relative_to(INPUT).parts)
        dirs[:] = [d for d in dirs if d not in {'train_series', 'test_series', 'dicom', 'dicoms'}]
        if 'study_cache' in dirs:
            found.append(Path(base) / 'study_cache')
            dirs.remove('study_cache')
        if depth >= max_depth:
            dirs[:] = []
    return found

cache_candidates = []
for candidate in cache_directories():
    names = [entry.name for entry in os.scandir(candidate) if entry.is_file()]
    n_weak = sum(n.startswith('weak_') and n.endswith('.npz') for n in names)
    n_gold = sum(n.startswith('gold_') and n.endswith('.npz') for n in names)
    cache_candidates.append((candidate, n_weak, n_gold))
valid = [x for x in cache_candidates if x[1] >= 4349 and x[2] >= 58]
if not valid:
    raise FileNotFoundError(
        'Attach complete Study_case 05_RSNA Dataset with 4,349 weak_*.npz '
        'and 58 gold_*.npz in rsna_05a_dense_mil/study_cache. '
        f'Found: {[(str(p),w,g) for p,w,g in cache_candidates]}'
    )
CACHE, WEAK_COUNT, GOLD_COUNT = sorted(valid, key=lambda x: (-x[1], str(x[0])))[0]

if OFFLINE_WEIGHT_PATH:
    WEIGHT = Path(OFFLINE_WEIGHT_PATH)
    if not WEIGHT.is_file():
        raise FileNotFoundError(f'OFFLINE_WEIGHT_PATH does not exist: {WEIGHT}')
    weight_source = 'attached_dataset'
else:
    try:
        WEIGHT = Path(hf_hub_download(
            repo_id=CONFIG['hf_repo'], filename='model.safetensors',
            revision=CONFIG['hf_revision'], local_dir=OUT,
        ))
    except Exception as exc:
        raise RuntimeError(
            'Pinned CoAtNet weight could not be downloaded. Enable Kaggle Internet '
            'or attach model.safetensors as a Dataset and set OFFLINE_WEIGHT_PATH. '
            'No GPU work was started.'
        ) from exc
    weight_source = 'huggingface_pinned_revision'

digest = hashlib.sha256()
with WEIGHT.open('rb') as handle:
    for chunk in iter(lambda: handle.read(8 << 20), b''):
        digest.update(chunk)
weight_sha256 = digest.hexdigest()
if WEIGHT.stat().st_size < 100_000_000:
    raise RuntimeError(f'Pretrained weight unexpectedly small: {WEIGHT.stat().st_size} bytes')

print('ASSET CHECK PASSED', {
    'cache': str(CACHE), 'weak': WEAK_COUNT, 'gold': GOLD_COUNT,
    'weight': str(WEIGHT), 'weight_bytes': WEIGHT.stat().st_size,
    'weight_sha256': weight_sha256,
    'torch': torch.__version__, 'timm': timm.__version__,
    'huggingface_hub': huggingface_hub.__version__,
}, flush=True)

# %%
if not torch.cuda.is_available():
    raise RuntimeError('Asset check passed; select GPU T4×2 before benchmark')
if 'T4' not in torch.cuda.get_device_name(0):
    raise RuntimeError(f'Expected T4 on GPU 0, got {torch.cuda.get_device_name(0)}')

load_start = monotonic()
model = timm.create_model(CONFIG['model_name'], pretrained=False)
state = load_file(str(WEIGHT), device='cpu')
expected = model.state_dict()
missing = sorted(set(expected) - set(state))
extra = sorted(set(state) - set(expected))
bad_shapes = sorted(k for k in set(expected) & set(state)
                    if tuple(expected[k].shape) != tuple(state[k].shape))
if missing or extra or bad_shapes:
    raise RuntimeError(
        'Pinned CoAtNet checkpoint/installed timm mismatch; refusing random weights: '
        f'missing={missing[:8]}, extra={extra[:8]}, bad_shapes={bad_shapes[:8]}'
    )
model.load_state_dict(state, strict=True)
del state, expected
data_config = timm.data.resolve_model_data_config(model)
if tuple(data_config['input_size']) != (3, 224, 224):
    raise RuntimeError(f'Unexpected pretrained input_size: {data_config["input_size"]}')
model.reset_classifier(0)
feature_dim = int(model.num_features)
if feature_dim <= 0:
    raise RuntimeError(f'Unexpected CoAtNet feature_dim={feature_dim}')
model = model.to('cuda:0').eval().requires_grad_(False)
gc.collect()
torch.cuda.synchronize(0)
weight_load_seconds = monotonic() - load_start

all_weak = sorted(entry.path for entry in os.scandir(CACHE)
                  if entry.is_file() and entry.name.startswith('weak_')
                  and entry.name.endswith('.npz'))
if len(all_weak) < 4349:
    raise RuntimeError('Complete weak cache disappeared after asset check')
indices = np.linspace(0, len(all_weak) - 1, CONFIG['sample_studies'], dtype=int)
chosen = [all_weak[int(i)] for i in indices]
mean = torch.tensor(data_config['mean'], device='cuda:0').view(1, 3, 1, 1)
std = torch.tensor(data_config['std'], device='cuda:0').view(1, 3, 1, 1)

torch.cuda.reset_peak_memory_stats(0)
start = monotonic()
processed = 0
n_images = 0
n_empty = 0
slice_counts = []
feature_checksum_nonclinical = 0.0
status = 'COMPLETE'

with torch.inference_mode():
    for study_i, path in enumerate(chosen, 1):
        with np.load(path, allow_pickle=False) as z:
            if not {'images', 'slot_ids', 'slice_indices', 'positions'}.issubset(z.files):
                raise RuntimeError(f'Invalid 05A cache schema: {path}')
            images = z['images']
            if images.dtype != np.uint8 or images.ndim != 3 or images.shape[1:] != (224, 224):
                raise RuntimeError(f'Unexpected image shape/dtype: {path}, {images.shape}, {images.dtype}')
            if len(images) > CONFIG['max_cache_slices']:
                raise RuntimeError(f'Unexpected slice count: {path}, {len(images)}')
            if not (len(z['slot_ids']) == len(z['slice_indices']) ==
                    len(z['positions']) == len(images)):
                raise RuntimeError(f'Cache metadata length mismatch: {path}')
        if len(images) == 0:
            n_empty += 1
        slice_counts.append(len(images))
        n_images += len(images)
        for offset in range(0, len(images), CONFIG['microbatch']):
            part = images[offset:offset + CONFIG['microbatch']]
            gray = torch.from_numpy(part).to('cuda:0', non_blocking=False).float().div_(255.0)
            rgb = gray.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
            rgb = (rgb - mean) / std
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                feat = model(rgb)
            if feat.shape != (len(part), feature_dim) or not bool(torch.isfinite(feat).all()):
                raise RuntimeError(f'Invalid CoAtNet feature tensor at {path}: {tuple(feat.shape)}')
            feature_checksum_nonclinical += float(feat.float().mean().item())
            del gray, rgb, feat
        del images
        processed = study_i
        if study_i % 16 == 0:
            torch.cuda.synchronize(0)
            print(f'encoded {study_i}/{len(chosen)} studies, {n_images} images, '
                  f'{monotonic()-start:.1f}s', flush=True)
        if torch.cuda.max_memory_reserved(0) / 2**30 > CONFIG['max_gpu_reserved_gib']:
            status = 'PARTIAL_GPU_GUARD'
            print('GPU MEMORY GUARD: benchmark stopped safely', flush=True)
            break
        if monotonic() - start >= CONFIG['max_preflight_seconds']:
            status = 'PARTIAL_TIME_GUARD'
            print('TIME GUARD: benchmark stopped safely', flush=True)
            break

torch.cuda.synchronize(0)
elapsed = monotonic() - start
if processed == 0 or n_images == 0:
    raise RuntimeError('No valid images were encoded')

# Preserve exact pinned pretrained weights for a later Internet-Off training run.
saved_weight = OUT / 'coatnet_rmlp_1_rw2_224_pretrained.safetensors'
if WEIGHT.parent.resolve() == OUT.resolve():
    saved_weight = WEIGHT
elif WEIGHT.resolve() != saved_weight.resolve():
    shutil.copyfile(WEIGHT, saved_weight)
else:
    saved_weight = WEIGHT

report = {
    'status': status,
    'config': CONFIG,
    'cache_path': str(CACHE),
    'cache_weak_studies': WEAK_COUNT,
    'cache_gold_studies': GOLD_COUNT,
    'weight_source': weight_source,
    'weight_source_path': str(WEIGHT),
    'saved_weight_path': str(saved_weight),
    'weight_sha256': weight_sha256,
    'weight_bytes': WEIGHT.stat().st_size,
    'timm_version': timm.__version__,
    'torch_version': torch.__version__,
    'feature_dim': feature_dim,
    'pretrained_data_config': data_config,
    'device': torch.cuda.get_device_name(0),
    'sample_studies_processed': processed,
    'images_encoded': n_images,
    'empty_studies': n_empty,
    'mean_slices_per_study': float(np.mean(slice_counts)),
    'max_slices_per_study': int(np.max(slice_counts)),
    'weight_load_seconds': weight_load_seconds,
    'encoder_seconds': elapsed,
    'images_per_second': n_images / elapsed,
    'rough_full_4407_hours_one_T4': elapsed * 4407 / processed / 3600,
    'cuda_peak_allocated_gib': torch.cuda.max_memory_allocated(0) / 2**30,
    'cuda_peak_reserved_gib': torch.cuda.max_memory_reserved(0) / 2**30,
    'feature_checksum_nonclinical': feature_checksum_nonclinical,
    'caveat': '64-study preflight only; estimate excludes full feature writes, head training and inference',
}
(OUT / 'b3_preflight.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps({k: report[k] for k in (
    'status', 'feature_dim', 'sample_studies_processed', 'images_encoded',
    'encoder_seconds', 'rough_full_4407_hours_one_T4',
    'cuda_peak_allocated_gib', 'cuda_peak_reserved_gib')}, indent=2))
print('Send b3_preflight.json before starting a full five-fold B3 run.')
