"""Build a self-contained 0.943 + public ConvNeXt v1 fixed-30% probe.

Only reviewed source is embedded. The original 30 anchor cells are byte-for-byte
unchanged. Kaggle input discovery never imports arbitrary attached scripts.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'research_public_convnext'
ANCHOR = ROOT / 'rsna-knee-speedy-raptors-coatnet-d4-0943.ipynb'
OUTPUT = ROOT / '22C_RSNA_0943_Plus_ConvNeXt25D_Fixed30_HiddenSafe.ipynb'
ANCHOR_SHA = 'bb3993ecb592d1790e7bb12d2a92e72ce4e3aaa2c4cd788ebf8f0e538d3b78c7'
READER_WEIGHT_SHA = {
    'cnxt_v0_fold0.pt': '0b64baa9d1c29c8d5c4dffe06d9ab2c941119b37738e5a2346e882ab694e7b80',
    'cnxt_v0_fold1.pt': '61696b6e67474695242fc8049e96e2ea13987db6e20d04410dc2961a689585cb',
    'cnxt_v0_fold2.pt': 'f5cf19cf591600c1c1a04f1aa5ff467751ef01513426a4ea3baa789443656cb6',
}
READER_WEIGHT_BYTES = 123035271


def source(cell):
    return ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']


def cell(kind, value):
    result = {'cell_type': kind, 'metadata': {}, 'source': value.splitlines(keepends=True)}
    if kind == 'code':
        ast.parse(value)
        result.update(execution_count=None, outputs=[])
    return result


INTRO = '''# 22C — Protected 0.943 + fine-tuned 2.5D ConvNeXt, fixed 30% probe

**Purpose: one submission experiment, not training.** Run the exact protected
0.943 inference graph, then an independently fine-tuned ConvNeXt-Tiny reader,
then `rank(0.70*rank(anchor)+0.30*rank(reader))` on all 12 targets. No B2 heads,
density search, new CoAt/OAI checkpoints, fitting, or use of test reports.
The 0.943 source notebook is not edited. Keep that successful submission as backup.

**Attach:** keep every input of the successful 0.943 notebook (18 original
competition/dataset/notebook/model sources, inherited in metadata). **Add only**
[GoodPJW's RSNA Knee 2.5D ConvNeXt reader, version 1](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader/versions/1),
with its three `cnxt_v0_fold*.pt` checkpoints and source files. The notebook uses
the pinned `timm` wheel and OpenCV environment from the original anchor; it
smoke-tests actual DICOM pixel decoding before the anchor starts. You do **not**
need outputs 09/20/21, 05A study_cache, or our silver-label table. If import does
not restore original inputs, start from a copy of the successful 0.943 notebook
and import this notebook into it.

**Settings:** GPU T4×2, Internet Off, fresh session, Save Version → Save & Run All.
GPU 0/1 are used by the unchanged anchor; the new reader uses whichever GPU has
more free memory afterward. New reader: one study at a time, uint8 on CPU,
backbone microbatch 8, no DataLoader workers, no DataParallel, no training graph.
The whole experiment has an 8.5-hour safety budget for the competition's 9-hour
scoring limit. A 12-hour ordinary Kaggle session is not the submission limit.
Visible three-study runtime is not an estimate of the full hidden runtime.

The original reader's architecture/preprocessing is retained: 0.4 mm, 384-pixel
canonical series volumes (153.6 mm), up to 64 slices, six plane/FS slots, 12 native
adjacent-slice triplets per chosen series, 0.92 centre-view factor → 256 pixels,
ImageNet normalization, 2-layer transformer, finding-specific attention, mean
probability of the three released EMA checkpoints. Microbatching is checked
against the original forward on one real series per checkpoint before blending.
This is a batching check, not proof of full hidden/scored reproduction.

**Provenance:** public dataset v1 is Apache 2.0. Its released training code
excludes Gold58 from the optimizer and saves final EMA after a fixed schedule;
report-text GroupKFold and four public label sources are declared. No OAI raw
MRI appears in the inspected training code. Checkpoint/cohort membership and
upstream label ancestry are not independently certified by those statements.
Gold58 and the author's 0.929/0.944 Public results are not a new independent
holdout for us. The reference notebook also discloses multiple blend probes and
one-tick run variation. We choose 30% before our own submission; it is not an
OOF-optimal or Private-optimal weight. There is no guarantee of improvement.

Before submitting, inspect `convnext_probe_receipt.json`: `status=COMPLETE`,
20 DINO / 5 A5 / 4 Raptor / 4 CoAt, three SHA-pinned ConvNeXt checkpoints,
UID/schema coverage, no all-empty study or fallback, and
the reference-forward checks. A failed new reader does **not** silently submit
the anchor. Outputs keep `_anchor_0943_submission.csv`, reader probabilities,
per-fold predictions, source/provenance hashes, selected-series diagnostics,
and the final `submission.csv`. Send the Output ZIP and Public score afterward.

**Asset correction:** The local reference downloads of the three `.pt` files
were truncated (not valid torch ZIP archives). The checkpoint SHA-256 values
used here instead come from the official mounted GoodPJW Dataset as measured by
the CPU-only 22A diagnostic on 11 October 2026; each was 123,035,271 bytes.
The released `preprocess.py`, `knee.py`, and `infer.py` in that same mount match
the reviewed source byte-for-byte. Before GPU work, this notebook loads every
mounted checkpoint on CPU and checks fold, architecture, key tensor shapes,
and finite weights. The local truncated files are never used for inference.

**Hidden-cohort robustness correction (22C):** 22B stopped on any unreadable
DICOM probe or skipped slice, although the released reader's `load_series`
skips unreadable files and its `infer.py` skips a failed/short slot. 22C
matches those source semantics, recording skipped files/slots. It still
refuses an entirely unreadable study, changed assets, incomplete predictions,
and exceeded time/memory budgets. Kaggle does not expose hidden traceback, so
this fixes a confirmed code mismatch, not a verified diagnosis of the prior
submission failure. Inspect the 22B visible Output receipt and the elapsed
time of its failed scoring attempt before spending another submission.

Credit: [GoodPJW public reader](https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944),
the authors credited in the unchanged 0.943 anchor, and the
[archived public source fork](https://github.com/sinhaaditya5/kaggle-rsna-knee-abnormality/tree/main/kernels/k01-public-stack-baseline).
'''

PREFLIGHT = r'''# CPU asset/identity preflight BEFORE the anchor spends GPU time.
import gc as _c22_gc, hashlib as _c22_hash, json as _c22_json
import os as _c22_os, sys as _c22_sys, subprocess as _c22_sp, time as _c22_time
from pathlib import Path as _c22_Path
import pandas as _c22_pd
import torch as _c22_torch
_C22_START = _c22_time.monotonic()
_C22_TOTAL_SECONDS = 8.5 * 3600
_C22_WEIGHT = 0.30
_C22_WORK = _c22_Path('/kaggle/working')
_C22_SRC = _C22_WORK / '_convnext22_src'
_C22_OUTPUT = _C22_WORK / 'convnext22'
_C22_ENV = _C22_WORK / '_convnext22_env'
if (_C22_WORK / 'submission.csv').exists():
    raise RuntimeError('Use a fresh Save & Run All session, not an existing submission.csv')
_C22_SRC.mkdir(exist_ok=True); _C22_OUTPUT.mkdir(exist_ok=True)
for _c22_name, _c22_code in _C22_SOURCES.items():
    (_C22_SRC / _c22_name).write_text(_c22_code, encoding='utf-8')
_c22_sys.path.insert(0, str(_C22_SRC))
from rsna22_validation import LABELS as _C22_LABELS
from rsna22_reader import sha256 as _c22_sha, check_checkpoint as _c22_check
_c22_roots = [_c22_Path('/kaggle/input/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader'),
              _c22_Path('/kaggle/input/rsna-knee-2-5d-convnext-reader')]
_c22_hits = [p for p in _c22_roots if p.is_dir()]
if len(_c22_hits) != 1:
    raise FileNotFoundError('Attach exactly one public GoodPJW ConvNeXt reader Dataset v1: ' + str(_c22_hits))
_C22_ASSET = _c22_hits[0]
_C22_FILES = {}
for _c22_name, _c22_expected in _C22_PINNED.items():
    _c22_candidates = sorted(p for p in _C22_ASSET.rglob(_c22_name)
                             if p.is_file() and len(p.relative_to(_C22_ASSET).parts) <= 5)
    if not _c22_candidates:
        _c22_nearby = sorted(str(p.relative_to(_C22_ASSET)) for p in _C22_ASSET.rglob('*')
                             if p.is_file() and len(p.relative_to(_C22_ASSET).parts) <= 5)[:40]
        raise FileNotFoundError(f'Missing ConvNeXt v1 asset {_c22_name} under {_C22_ASSET}. '
                                f'First files: {_c22_nearby}. Attach dataset version 1 with all 3 checkpoints.')
    _c22_observed = [(str(p), _c22_sha(p)) for p in _c22_candidates]
    _c22_matching = [p for p, (_, digest) in zip(_c22_candidates, _c22_observed)
                     if digest == _c22_expected]
    if len(_c22_matching) != 1:
        raise RuntimeError(f'ConvNeXt asset identity mismatch for {_c22_name}: '
                           f'expected SHA256 {_c22_expected}; found {_c22_observed}. '
                           'Do not substitute a different checkpoint/version.')
    _C22_FILES[_c22_name] = _c22_matching[0]
    if _c22_name.endswith('.pt') and _C22_FILES[_c22_name].stat().st_size != 123035271:
        raise RuntimeError(f'ConvNeXt checkpoint byte-size mismatch: {_c22_name}')
_C22_CKPTS = []
for _c22_fold in range(3):
    _c22_name = f'cnxt_v0_fold{_c22_fold}.pt'
    _c22_ck = _c22_torch.load(_C22_FILES[_c22_name], map_location='cpu', weights_only=True)
    _c22_args = _c22_check(_c22_ck, _c22_fold)
    _C22_CKPTS.append({'fold': _c22_fold, 'name': _c22_name,
                       'path': str(_C22_FILES[_c22_name]),
                       'sha256': _C22_PINNED[_c22_name], 'args': _c22_args})
    del _c22_ck
    _c22_gc.collect()
_c22_comp = [_c22_Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
              _c22_Path('/kaggle/input/rsna-knee-abnormality-detection')]
_c22_data_hits = [p for p in _c22_comp if (p / 'test_series.csv').is_file()]
if len(_c22_data_hits) != 1:
    raise FileNotFoundError('Attach RSNA competition data once')
_C22_DATA = _c22_data_hits[0]
_C22_IDS = _c22_pd.read_csv(_C22_DATA / 'test.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
_c22_sample = _c22_pd.read_csv(_C22_DATA / 'sample_submission.csv', dtype={'StudyInstanceUID': str})
if (not _C22_IDS or len(_C22_IDS) != len(set(_C22_IDS)) or
        list(_c22_sample.columns) != ['StudyInstanceUID', *_C22_LABELS] or
        set(_c22_sample.StudyInstanceUID) != set(_C22_IDS)):
    raise RuntimeError('Invalid competition cohort/schema')
# Verify at least one real compressed/uncompressed DICOM decode before the
# expensive protected anchor runs. This uses Kaggle's pinned runtime decoder
# plugins, matching the public source's optional offline install (which has no
# decoder wheels in its released v1 Dataset).
import pydicom as _c22_pydicom
_c22_series = _c22_pd.read_csv(_C22_DATA / 'test_series.csv', dtype=str)
_C22_CODEC_PROBES = {}
_C22_CODEC_PROBE_FAILURES = []
for _c22_row in _c22_series.head(256).itertuples(index=False):
    _c22_series_dir = _C22_DATA / 'test_series' / str(_c22_row.StudyInstanceUID) / str(_c22_row.SeriesInstanceUID)
    try:
        _c22_first = next((p for p in _c22_series_dir.iterdir() if p.is_file() and p.suffix.lower() == '.dcm'), None)
    except OSError:
        _c22_first = None
    if _c22_first is None:
        continue
    try:
        _c22_header = _c22_pydicom.dcmread(str(_c22_first), stop_before_pixels=True, force=True)
        _c22_syntax = str(getattr(getattr(_c22_header, 'file_meta', None), 'TransferSyntaxUID', 'unknown'))
        if _c22_syntax not in _C22_CODEC_PROBES:
            _c22_pixels = _c22_pydicom.dcmread(str(_c22_first), force=True).pixel_array
            if _c22_pixels.size == 0:
                raise ValueError('empty pixel array')
            _C22_CODEC_PROBES[_c22_syntax] = {'path': str(_c22_first), 'shape': list(_c22_pixels.shape), 'status': 'PASS'}
            del _c22_pixels
    except Exception as _c22_exc:
        # The public loader skips unreadable DICOM files. A single bad file
        # among a larger hidden cohort must not veto the whole submission.
        if len(_C22_CODEC_PROBE_FAILURES) < 20:
            _C22_CODEC_PROBE_FAILURES.append({'path': str(_c22_first),
                'error': f'{type(_c22_exc).__name__}: {_c22_exc}'[:200]})
if not _C22_CODEC_PROBES:
    raise RuntimeError('DICOM decoder preflight found no readable test image in the first 256 series')
print('DICOM decoder preflight:', _C22_CODEC_PROBES,
      'skipped_probe_failures:', len(_C22_CODEC_PROBE_FAILURES), flush=True)
# Find the SAME pinned timm wheel already required by the anchor's D4 source.
_c22_timm = []
for _c22_base, _c22_dirs, _c22_files in _c22_os.walk('/kaggle/input'):
    _c22_dirs[:] = [d for d in _c22_dirs if d not in {'train_series','test_series','study_cache','dicom','dicoms'}]
    if 'timm-1.0.22-py3-none-any.whl' in _c22_files:
        _c22_wheel = _c22_Path(_c22_base) / 'timm-1.0.22-py3-none-any.whl'
        if _c22_sha(_c22_wheel) == '888981753e65cbaacfc07494370138b1700a27b1f0af587f4f9b47bc024161d0':
            _c22_timm.append(_c22_wheel)
    if len(_c22_Path(_c22_base).relative_to('/kaggle/input').parts) >= 7:
        _c22_dirs[:] = []
if not _c22_timm:
    raise FileNotFoundError('Restore the original D4 input containing the pinned timm 1.0.22 wheel')
_c22_sp.run([_c22_sys.executable, '-m', 'pip', 'install', '--quiet', '--no-index', '--no-deps',
             '--target', str(_C22_ENV), '--find-links', str(_C22_ASSET),
             str(sorted(_c22_timm, key=str)[0])], check=True, timeout=180)
_C22_MANIFEST = {'dataset': 'goodpjw2008/rsna-knee-2-5d-convnext-reader', 'version': 1,
                  'license': 'Apache-2.0', 'checkpoints': _C22_CKPTS,
                  'resolved_asset_paths': {n: str(p) for n,p in _C22_FILES.items()},
                  'dicom_decoder_probes': _C22_CODEC_PROBES,
                  'dicom_decoder_probe_failures': _C22_CODEC_PROBE_FAILURES,
                  'source_hashes': {n: _c22_sha(_C22_SRC / n) for n in _C22_SOURCES},
                  'attached_source_hashes': {n: v for n,v in _C22_PINNED.items() if n.endswith('.py')},
                  'anchor_source_sha256': _C22_ANCHOR_SHA,
                  'remaining_provenance_limits': ['No independently reconstructed cohort/fold membership',
                                                  'Upstream label ancestry not independently certified']}
_C22_MANIFEST_PATH = _C22_OUTPUT / 'asset_manifest.json'
_C22_MANIFEST_PATH.write_text(_c22_json.dumps(_C22_MANIFEST, indent=2) + '\n')
print('22 PREFLIGHT PASSED: pinned 3 EMA checkpoints; all original anchor inputs still required.', flush=True)
'''

CAPTURE = r'''# Remove root submission first; validation failures must not leave a candidate.
from rsna22_validation import validate_anchor_bundle as _c22_validate_anchor
from rsna22_validation import validate_frame as _c22_validate_frame
_C22_ANCHOR_PATH = _C22_WORK / '_anchor_0943_submission.csv'
_c22_os.replace(_C22_WORK / 'submission.csv', _C22_ANCHOR_PATH)
def _c22_read_bundle(relative):
    return (_C22_ANCHOR_PATH if relative == 'submission.csv' else _C22_WORK / relative).read_bytes()
_C22_ANCHOR_FRAME, _C22_ANCHOR_RECEIPT = _c22_validate_anchor(_c22_read_bundle, _C22_IDS)
_c22_reference = _c22_pd.DataFrame(_C22_VISIBLE_REFERENCE)
_C22_VISIBLE_PARITY = {'applicable': set(_C22_IDS) == set(_c22_reference.StudyInstanceUID),
                      'scope': 'same three-study visible cohort only'}
if _C22_VISIBLE_PARITY['applicable']:
    _c22_reference = _c22_validate_frame(_c22_reference, _C22_IDS, 'protected visible reference')
    _c22_diff = float(abs(_C22_ANCHOR_FRAME[_C22_LABELS] - _c22_reference[_C22_LABELS]).to_numpy().max())
    _C22_VISIBLE_PARITY['max_abs_diff'] = _c22_diff
    if _c22_diff > 1e-12:
        raise RuntimeError('Protected anchor visible ranking reproduction failed')
_c22_gc.collect(); _c22_torch.cuda.empty_cache()
print('EXACT ANCHOR CAPTURED; strict 20/5/4/4 family checks passed.', flush=True)
'''

READER = r'''# Isolated child with one common remaining budget; never imports mounted scripts.
_c22_remaining = _C22_TOTAL_SECONDS - (_c22_time.monotonic() - _C22_START)
if _c22_remaining <= 60:
    raise TimeoutError('No time remains for the required ConvNeXt reader')
_c22_free = [_c22_torch.cuda.mem_get_info(i)[0] for i in range(_c22_torch.cuda.device_count())]
if not _c22_free:
    raise RuntimeError('CUDA unavailable')
_c22_gpu = max(range(len(_c22_free)), key=lambda i: _c22_free[i])
if _c22_free[_c22_gpu] < 4 * 1024**3:
    raise MemoryError('Less than 4 GiB GPU memory available for ConvNeXt')
_c22_env = dict(_c22_os.environ)
_c22_env['CUDA_VISIBLE_DEVICES'] = str(_c22_gpu)
_c22_env['PYTHONPATH'] = _c22_os.pathsep.join([str(_C22_SRC), str(_C22_ENV),
                                            '/kaggle/working/_coat_env', _c22_env.get('PYTHONPATH','')])
_c22_log = _C22_OUTPUT / 'reader.log'
with _c22_log.open('w') as _c22_handle:
    _c22_process = _c22_sp.Popen([_c22_sys.executable, str(_C22_SRC/'rsna22_reader.py'),
                                '--data', str(_C22_DATA), '--manifest', str(_C22_MANIFEST_PATH),
                                '--output', str(_C22_OUTPUT), '--max-seconds', str(_c22_remaining - 30)],
                               env=_c22_env, stdout=_c22_handle, stderr=_c22_sp.STDOUT)
    try:
        _c22_returncode = _c22_process.wait(timeout=_c22_remaining)
    except BaseException:
        _c22_process.terminate()
        try: _c22_process.wait(timeout=10)
        except _c22_sp.TimeoutExpired:
            _c22_process.kill(); _c22_process.wait()
        raise
if _c22_returncode != 0:
    raise RuntimeError('ConvNeXt reader failed; no final submission published.\n' + _c22_log.read_text()[-6000:])
print(_c22_log.read_text()[-5000:], flush=True)
'''

FINAL = r'''# The only treatment: predeclared 70/30 globally ranked new-family addition.
from rsna22_validation import blend_frames as _c22_blend
from rsna22_validation import validate_reader_receipt as _c22_validate_reader
_c22_receipt = _c22_json.loads((_C22_OUTPUT/'convnext_reader_receipt.json').read_text())
_c22_validate_reader(_c22_receipt, _C22_IDS,
    expected_contract='goodpjw_convnext_v1_3ema_6slot_k12_256_micro8_v1',
    expected_checkpoint_hashes=[item['sha256'] for item in _C22_CKPTS])
_c22_reader_path = _C22_OUTPUT/'convnext_probabilities.csv'
if _c22_sha(_c22_reader_path) != _c22_receipt.get('output_sha256'):
    raise RuntimeError('Reader output SHA256 mismatch')
_c22_stats = _c22_receipt.get('dicom_stats', {})
if _c22_stats.get('empty_studies', 0):
    raise RuntimeError('Reader all-empty study must not be blended')
if len(_c22_receipt.get('parity_checks', [])) != 3:
    raise RuntimeError('Reader reference-forward checks missing')
_c22_reader_frame = _c22_validate_frame(
    _c22_pd.read_csv(_c22_reader_path, dtype={'StudyInstanceUID':str}), _C22_IDS, 'ConvNeXt')
_c22_final = _c22_blend(_C22_ANCHOR_FRAME, _c22_reader_frame, _C22_IDS, _C22_WEIGHT)
_c22_path = _C22_WORK/'submission.csv'
_c22_tmp = _C22_WORK/'_convnext22_candidate.csv.tmp'
with _c22_tmp.open('w', encoding='utf-8', newline='') as _c22_handle:
    _c22_final.to_csv(_c22_handle, index=False)
    _c22_handle.flush(); _c22_os.fsync(_c22_handle.fileno())
_c22_roundtrip = _c22_validate_frame(_c22_pd.read_csv(_c22_tmp, dtype={'StudyInstanceUID':str}),
                                    _C22_IDS, 'saved candidate')
if abs(_c22_roundtrip[_C22_LABELS]-_c22_final[_C22_LABELS]).to_numpy().max() > 1e-15:
    raise RuntimeError('Candidate CSV roundtrip mismatch')
_c22_final_receipt = {'status':'COMPLETE', 'experiment':'protected_0943_plus_goodpjw_convnext_v1_fixed30',
    'studies':len(_C22_IDS), 'labels':_C22_LABELS, 'formula':'rank(0.70*rank(anchor_0943)+0.30*rank(convnext_3EMA))',
    'reader_weight':_C22_WEIGHT, 'anchor_source_sha256':_C22_ANCHOR_SHA,
    'anchor_submission_sha256':_c22_sha(_C22_ANCHOR_PATH), 'anchor_counts':{'dino':20,'a5':5,'raptor':4,'coat':4},
    'reader_probabilities_sha256':_c22_sha(_c22_reader_path), 'reader_models':3,
    'reader_checkpoints':_C22_CKPTS, 'reader_dicom_stats':_c22_stats,
    'reader_reference_checks':_c22_receipt['parity_checks'], 'anchor_visible_parity':_C22_VISIBLE_PARITY,
    'elapsed_seconds':_c22_time.monotonic()-_C22_START, 'total_safety_budget_seconds':_C22_TOTAL_SECONDS,
    'submission_sha256':_c22_sha(_c22_tmp), 'provenance':_C22_MANIFEST,
    'limitations':['No full-ensemble independent OOF','No guarantee of Public/Private gain',
                   'Visible sample parity is not hidden-run parity','Upstream training/label ancestry not independently certified']}
(_C22_WORK/'convnext_probe_receipt.json').write_text(_c22_json.dumps(_c22_final_receipt, indent=2)+'\n')
_c22_os.replace(_c22_tmp, _c22_path)
print('22 COMPLETE', {'studies':len(_C22_IDS), 'formula':_c22_final_receipt['formula'],
                      'reader_models':3,'submission_sha256':_c22_final_receipt['submission_sha256']}, flush=True)
display(_c22_final.head())
'''


def main():
    if hashlib.sha256(ANCHOR.read_bytes()).hexdigest() != ANCHOR_SHA:
        raise RuntimeError('Protected anchor changed; inspect before rebuilding')
    anchor = json.loads(ANCHOR.read_text(encoding='utf-8'))
    if len(anchor['cells']) != 30:
        raise RuntimeError('Anchor structure changed')
    # The Kaggle CPU audit verified these three inference-critical files byte-for-byte.
    # Training-only files were not hashed in that audit and are not imported here.
    files = {n: ASSETS / n for n in ('preprocess.py', 'knee.py', 'infer.py')}
    pinned = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n,p in files.items()}
    pinned.update(READER_WEIGHT_SHA)
    sources = {n: (ASSETS/n).read_text(encoding='utf-8') for n in ('preprocess.py','knee.py')}
    for n in ('rsna22_reader.py','rsna22_validation.py'):
        sources[n] = (ROOT/n).read_text(encoding='utf-8')
    with zipfile.ZipFile(ROOT/'results_0.943.zip') as archive:
        import io
        import pandas as pd
        visible = pd.read_csv(io.BytesIO(archive.read('submission.csv')), dtype={'StudyInstanceUID':str}).to_dict('list')
    definitions = ('_C22_SOURCES = ' + repr(sources) + '\n_C22_PINNED = ' + repr(pinned) +
                   '\n_C22_ANCHOR_SHA = ' + repr(ANCHOR_SHA) + '\n_C22_VISIBLE_REFERENCE = ' + repr(visible) + '\n')
    notebook = copy.deepcopy(anchor)
    notebook['cells'] = [cell('markdown', INTRO), cell('code', definitions + PREFLIGHT)]
    for c in anchor['cells']:
        clean = copy.deepcopy(c)
        if c['cell_type'] == 'code':
            clean.update(execution_count=None, outputs=[])
        notebook['cells'].append(clean)
    notebook['cells'] += [cell('markdown','## Strict unchanged-anchor capture'), cell('code',CAPTURE),
                          cell('markdown','## Isolated ConvNeXt reader'), cell('code',READER),
                          cell('markdown','## Fixed new-family rank blend'), cell('code',FINAL)]
    for index, c in enumerate(notebook['cells']):
        c['id'] = f'rsna22-{index:03d}'
        if c['cell_type'] == 'code':
            ast.parse(source(c))
    assert all(source(notebook['cells'][i+2]) == source(c) for i,c in enumerate(anchor['cells']))
    notebook['metadata']['kaggle'].update(isGpuEnabled=True, isInternetEnabled=False)
    notebook['metadata']['rsna22'] = {'anchor_sha256':ANCHOR_SHA,'reader_dataset':'goodpjw2008/rsna-knee-2-5d-convnext-reader',
                                     'reader_version':1,'reader_dataset_must_be_attached_manually':True,
                                     'reader_checkpoint_sha256':{n:h for n,h in pinned.items() if n.endswith('.pt')}}
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    print('CREATED', OUTPUT.name, len(notebook['cells']), 'cells; 30 anchor cells unchanged; 3 reader hashes pinned')


if __name__ == '__main__':
    main()
