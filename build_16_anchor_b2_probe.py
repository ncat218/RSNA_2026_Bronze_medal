"""Assemble one controlled 0.943 + B2 Kaggle probe from pinned notebooks.

This is a mechanical composition: it preserves both scientific inference
pipelines, then applies one predeclared 80/20 rank blend. It never edits the
source notebooks. Run this builder locally when either source changes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANCHOR = ROOT / "rsna-knee-speedy-raptors-coatnet-d4-0943.ipynb"
B2 = ROOT / "10_B2_RadImageNet_FrozenMIL_Submit.ipynb"
OUTPUT = ROOT / "16_RSNA_0943_Anchor_Plus_B2_Fixed20_Probe.ipynb"


def source(cell: dict) -> str:
    value = cell.get("source", [])
    return "".join(value) if isinstance(value, list) else str(value)


def markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": value.splitlines(keepends=True)}


def code(value: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": value.splitlines(keepends=True)}


def clean(cell: dict) -> dict:
    result = copy.deepcopy(cell)
    if result["cell_type"] == "code":
        result["execution_count"] = None
        result["outputs"] = []
    result.pop("id", None)
    return result


intro = """# 16 — 0.943 anchor + B2 frozen MIL, fixed 20% rank probe

This is **one predeclared Public-LB experiment**. It runs the exact protected
0.943 notebook to completion, captures its `submission.csv`, runs the exact
successful notebook 10 B2 inference, and then applies 80% anchor rank + 20%
B2 rank per target. The source notebooks are unmodified. It does not train,
use gold labels, or search blend weights. An improvement is not guaranteed;
do not infer Private-LB gain from a Public score alone.

Kaggle settings: **T4×2, Internet Off**. Restore every input attached to the
successful 0.943 notebook (competition plus its public datasets/models/notebook
outputs). **Additionally attach the exact complete five-fold Output of notebook
09** containing `fold_0_head.pt` through `fold_4_head.pt`. The public
`resnet-50-radimagenet-marwan` Dataset with `ResNet50.pt` must also be present;
the B2 cell checks its SHA256. This notebook uses the sources as saved in the
two successful notebooks. Avoid attaching multiple different notebook 09
outputs containing `fold_0_head.pt` because the resolver fails on ambiguity.

Use **Save & Run All**. Before using a submission slot, inspect
`anchor_b2_probe_receipt.json`: `status=COMPLETE`, anchor CoAt members = 4,
B2 folds = 0–4, exact UID coverage, and no unexpected decode/availability
issues. Output also keeps `_anchor_0943_submission.csv` and
`_b2_standalone_submission.csv` for the visible test cohort. The final
`submission.csv` is the 80/20 probe. Kaggle's visible Output may contain only
three sample studies; it does not prove hidden-rerun behavior.

Evidence for this single probe is weak-label diagnostic only: on 4,349
UID-matched studies, a **partial** DINO+E13 public OOF proxy rose from 0.863469
to 0.870346 after a fixed 20% B2 addition. That partial proxy excludes A5,
Raptor, and CoAt, and cannot estimate the actual 0.943 ensemble's OOF/Private
score. Preserve the original 0.943 notebook and selected submission as backup.
"""

preflight = """# Lightweight B2 attachment check before the protected anchor spends GPU time.
from pathlib import Path as _pre_Path
import hashlib as _pre_hashlib
import os as _pre_os

_pre_input = _pre_Path('/kaggle/input')
_pre_weights = [
    _pre_input / 'datasets/marwanmath/resnet-50-radimagenet-marwan/ResNet50.pt',
    _pre_input / 'resnet-50-radimagenet-marwan/ResNet50.pt',
]
_pre_weight = next((p for p in _pre_weights if p.is_file()), None)
if _pre_weight is None:
    raise FileNotFoundError('Attach public resnet-50-radimagenet-marwan Dataset')
_pre_sha = _pre_hashlib.sha256()
with _pre_weight.open('rb') as _pre_handle:
    for _pre_block in iter(lambda: _pre_handle.read(8 << 20), b''):
        _pre_sha.update(_pre_block)
if _pre_sha.hexdigest() != '08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734':
    raise RuntimeError('B2 RadImageNet weight SHA256 changed')
_pre_head_roots = []
for _pre_base, _pre_dirs, _pre_files in _pre_os.walk(_pre_input):
    _pre_depth = len(_pre_Path(_pre_base).relative_to(_pre_input).parts)
    _pre_dirs[:] = [d for d in _pre_dirs
                    if d not in {'train_series','test_series','study_cache','dicom','dicoms'}]
    if 'fold_0_head.pt' in _pre_files:
        _pre_parent = _pre_Path(_pre_base)
        if all((_pre_parent / f'fold_{f}_head.pt').is_file() for f in range(5)):
            _pre_head_roots.append(_pre_parent)
    if _pre_depth >= 8:
        _pre_dirs[:] = []
if len(_pre_head_roots) != 1:
    raise RuntimeError(f'Attach exactly one complete notebook 09 Output: {_pre_head_roots}')
print('B2 ATTACHMENT PRECHECK PASSED',
      {'head_root': str(_pre_head_roots[0]), 'weight': str(_pre_weight)}, flush=True)
"""

capture = """# Capture the protected anchor output before B2 writes submission.csv.
from pathlib import Path as _probe_Path
import hashlib as _probe_hashlib
import json as _probe_json
import shutil as _probe_shutil
import numpy as _probe_np
import pandas as _probe_pd

_probe_work = _probe_Path('/kaggle/working')
_probe_final = _probe_work / 'submission.csv'
_probe_anchor = _probe_work / '_anchor_0943_submission.csv'
_probe_audit_path = _probe_work / 'btkd_v559_complete.json'
_probe_coat_path = _probe_work / '_coat_raptor_blend_receipt.json'
if not all(p.is_file() for p in (_probe_final, _probe_audit_path, _probe_coat_path)):
    raise RuntimeError('Protected 0.943 anchor output/receipts missing')
_probe_audit = _probe_json.loads(_probe_audit_path.read_text())
_probe_coat = _probe_json.loads(_probe_coat_path.read_text())
if (_probe_audit.get('status') != 'COMPLETE' or
        _probe_audit.get('dino_members') != 20 or
        _probe_audit.get('a5_folds') != 5 or
        _probe_audit.get('raptor_views') != 4 or
        _probe_audit.get('coat_family_members') != 4 or
        len(_probe_coat.get('members', [])) != 4 or
        _probe_coat.get('family_reduction') != 'rank_of_member_probability_mean'):
    raise RuntimeError('0.943 anchor did not run with its expected 20/5/4/4 family contract')
_probe_sha = _probe_hashlib.sha256(_probe_final.read_bytes()).hexdigest()
if _probe_sha != _probe_audit.get('submission_sha256'):
    raise RuntimeError('Anchor submission SHA256 differs from its receipt')
_probe_shutil.copyfile(_probe_final, _probe_anchor)
_probe_frame = _probe_pd.read_csv(_probe_anchor, dtype={'StudyInstanceUID': str})
if (len(_probe_frame) != _probe_audit.get('cohort') or
        not _probe_frame.StudyInstanceUID.is_unique or
        not _probe_np.isfinite(_probe_frame.iloc[:, 1:].to_numpy(float)).all()):
    raise RuntimeError('Anchor submission shape, UID, or numeric values invalid')
print('ANCHOR CAPTURE PASSED', {'studies': len(_probe_frame),
                                'sha256': _probe_sha,
                                'coat_members': _probe_coat['members']}, flush=True)
del _probe_frame
import gc as _probe_gc
import torch as _probe_torch
_probe_gc.collect()
_probe_torch.cuda.empty_cache()
"""

blend = """# Fixed 80/20 rank blend. No fitting or weight search is performed.
from pathlib import Path as _mix_Path
import hashlib as _mix_hashlib
import json as _mix_json
import os as _mix_os
import shutil as _mix_shutil
import numpy as _mix_np
import pandas as _mix_pd

_mix_root = _mix_Path('/kaggle/working')
_mix_anchor_path = _mix_root / '_anchor_0943_submission.csv'
_mix_b2_path = _mix_root / '_b2_standalone_submission.csv'
_mix_final_path = _mix_root / 'submission.csv'
_mix_anchor_receipt = _mix_json.loads((_mix_root / 'btkd_v559_complete.json').read_text())
_mix_b2_receipt = _mix_json.loads((_mix_root / 'b2_submit_receipt.json').read_text())
if (_mix_b2_receipt.get('status') != 'COMPLETE' or
        _mix_b2_receipt.get('folds') != [0, 1, 2, 3, 4] or
        _mix_b2_receipt.get('encoder_weight_sha256') !=
            '08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734'):
    raise RuntimeError('B2 five-fold inference receipt incomplete or wrong checkpoint')
_mix_shutil.copyfile(_mix_final_path, _mix_b2_path)
_mix_anchor = _mix_pd.read_csv(_mix_anchor_path, dtype={'StudyInstanceUID': str})
_mix_b2 = _mix_pd.read_csv(_mix_b2_path, dtype={'StudyInstanceUID': str})
_mix_labels = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA',
               'Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
_mix_cols = ['StudyInstanceUID'] + _mix_labels
if (list(_mix_anchor.columns) != _mix_cols or list(_mix_b2.columns) != _mix_cols or
        len(_mix_anchor) != len(_mix_b2) or
        not _mix_anchor.StudyInstanceUID.is_unique or
        not _mix_b2.StudyInstanceUID.is_unique or
        list(_mix_anchor.StudyInstanceUID) != list(_mix_b2.StudyInstanceUID) or
        _mix_b2_receipt.get('test_studies') != len(_mix_anchor) or
        _mix_anchor_receipt.get('cohort') != len(_mix_anchor)):
    raise RuntimeError('Anchor/B2 schema, UID order, or cohort mismatch')
for _mix_name, _mix_frame in [('anchor', _mix_anchor), ('B2', _mix_b2)]:
    _mix_values = _mix_frame[_mix_labels].to_numpy(float)
    if (not _mix_np.isfinite(_mix_values).all() or
            (_mix_values < 0).any() or (_mix_values > 1).any()):
        raise RuntimeError(f'Nonfinite or out-of-range {_mix_name} predictions')

_mix_anchor_rank = _mix_anchor[_mix_labels].rank(method='average', pct=True)
_mix_b2_rank = _mix_b2[_mix_labels].rank(method='average', pct=True)
_mix_raw = 0.80 * _mix_anchor_rank + 0.20 * _mix_b2_rank
_mix_output = _mix_anchor[['StudyInstanceUID']].copy()
_mix_output[_mix_labels] = _mix_raw.rank(method='average', pct=True)
if (list(_mix_output.columns) != _mix_cols or
        not _mix_np.isfinite(_mix_output[_mix_labels].to_numpy(float)).all()):
    raise RuntimeError('Blended submission invalid')

_mix_tmp = _mix_final_path.with_suffix('.csv.tmp')
with _mix_tmp.open('w', encoding='utf-8', newline='') as _mix_handle:
    _mix_output.to_csv(_mix_handle, index=False)
    _mix_handle.flush()
    _mix_os.fsync(_mix_handle.fileno())
_mix_os.replace(_mix_tmp, _mix_final_path)
_mix_roundtrip = _mix_pd.read_csv(_mix_final_path, dtype={'StudyInstanceUID': str})
if (list(_mix_roundtrip.columns) != _mix_cols or
        list(_mix_roundtrip.StudyInstanceUID) != list(_mix_anchor.StudyInstanceUID) or
        not _mix_np.isfinite(_mix_roundtrip[_mix_labels].to_numpy(float)).all()):
    raise RuntimeError('Final saved submission roundtrip failed')
_mix_receipt = {
    'status': 'COMPLETE',
    'experiment': 'protected_0943_anchor_plus_B2_fixed20_rank_probe_v1',
    'studies': len(_mix_roundtrip),
    'formula': 'rank(0.80*rank(anchor_0943)+0.20*rank(B2_09_fivefold))',
    'anchor_sha256': _mix_hashlib.sha256(_mix_anchor_path.read_bytes()).hexdigest(),
    'b2_sha256': _mix_hashlib.sha256(_mix_b2_path.read_bytes()).hexdigest(),
    'submission_sha256': _mix_hashlib.sha256(_mix_final_path.read_bytes()).hexdigest(),
    'anchor_coat_members': _mix_anchor_receipt.get('coat_family_members'),
    'b2_folds': _mix_b2_receipt.get('folds'),
    'b2_empty_studies': _mix_b2_receipt.get('dicom_stats', {}).get('empty_studies', 0),
    'b2_decode_failures': _mix_b2_receipt.get('decode_failures_by_transfer_syntax', {}),
    'warning': 'Public-LB probe only; no full 0.943 OOF or Private score estimate',
}
(_mix_root / 'anchor_b2_probe_receipt.json').write_text(
    _mix_json.dumps(_mix_receipt, indent=2, sort_keys=True) + '\\n', encoding='utf-8')
print('FIXED 20% PROBE COMPLETE', {
    'studies': len(_mix_roundtrip),
    'anchor_coat_members': _mix_receipt['anchor_coat_members'],
    'b2_folds': _mix_receipt['b2_folds'],
    'b2_empty_studies': _mix_receipt['b2_empty_studies'],
    'submission_sha256': _mix_receipt['submission_sha256'],
}, flush=True)
display(_mix_roundtrip.head())
"""


def main() -> None:
    anchor = json.loads(ANCHOR.read_text(encoding="utf-8"))
    b2 = json.loads(B2.read_text(encoding="utf-8"))
    if len(anchor["cells"]) != 30 or len(b2["cells"]) != 5:
        raise RuntimeError("Source notebook structure changed; inspect before recomposing")
    if not all(c["cell_type"] == "code" for c in b2["cells"][1:]):
        raise RuntimeError("B2 cells 1–4 are no longer all code")
    if "_coatnet_weight" not in source(anchor["cells"][27]):
        raise RuntimeError("Anchor final blend cell changed")
    if "b2_submit_receipt.json" not in source(b2["cells"][4]):
        raise RuntimeError("B2 final output cell changed")

    notebook = copy.deepcopy(anchor)
    notebook["cells"] = [markdown(intro), code(preflight)]
    notebook["cells"] += [clean(cell) for cell in anchor["cells"]]
    notebook["cells"] += [markdown("## Protected-anchor capture"), code(capture)]
    notebook["cells"] += [markdown("## Exact notebook 10 B2 inference")]
    notebook["cells"] += [clean(cell) for cell in b2["cells"][1:]]
    notebook["cells"] += [markdown("## Predeclared 80/20 rank probe"), code(blend)]
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"rsna16-{index:03d}"
    notebook.setdefault("metadata", {}).setdefault("kaggle", {})[
        "isGpuEnabled"] = True
    notebook["metadata"]["kaggle"]["isInternetEnabled"] = False

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile(source(cell), "<assembled-notebook-cell>", "exec")
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                      encoding="utf-8")
    print(f"CREATED {OUTPUT.name}: {len(notebook['cells'])} cells; "
          f"{len(notebook['metadata']['kaggle']['dataSources'])} inherited inputs")


if __name__ == "__main__":
    main()
