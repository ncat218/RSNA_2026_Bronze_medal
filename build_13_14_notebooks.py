"""Build the CPU baseline audit and controlled B2 plane-aware experiment.

The 14 notebook is mechanically derived from the pinned B2 notebook 09 so its
asset resolution, RadImageNet encoder, folds, targets, and optimizer stay fixed.
Only the head architecture and experiment bookkeeping change.
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "09_B2_RadImageNet_FrozenMIL_Fold01.ipynb"
AUDIT = ROOT / "13_RSNA_0943_Visible_Output_Integrity_CPU.ipynb"
PLANE = ROOT / "14_B2_RadImageNet_PlaneAwareMIL_FiveFold.ipynb"


def source(value: str) -> list[str]:
    return dedent(value).strip("\n").splitlines(keepends=True)


def code(value: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source(value)}


def markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source(value)}


def write_notebook(path: Path, notebook: dict) -> None:
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{path.name}:cell")
            cell["outputs"] = []
            cell["execution_count"] = None
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")


audit_intro = """
# 13 — Verify the saved 0.943 output (CPU, no submission)

Attach the **exact saved Output version** of the notebook that scored Public
0.943, or a private Dataset containing its complete Output files/ZIP. Required
files include `btkd_v559_complete.json`, `_coat_raptor_blend_receipt.json`,
`submission.csv`, four CoAt child receipts, and `diagnostics/phase_events.jsonl`.

Set Accelerator **None**, Internet **Off**. Use Save & Run All. Inspect the
generated `rsna_0943_integrity/baseline_integrity.json`. Only
`VISIBLE_OUTPUT_PASS` lets us proceed; it validates the **visible saved run**,
not Kaggle's hidden scoring rerun. This notebook never changes or submits the
0.943 baseline and cannot calculate AUC from unlabeled test data.
"""

audit_cell = r'''
from pathlib import Path
import csv
import hashlib
import io
import json
import math
import os
import zipfile

INPUT_ROOT = Path(os.environ.get('RSNA_AUDIT_INPUT', '/kaggle/input'))
SOURCE_OVERRIDE = os.environ.get('RSNA_AUDIT_SOURCE')  # Optional exact local path.
OUT = Path(os.environ.get('RSNA_AUDIT_OUTPUT', '/kaggle/working/rsna_0943_integrity'))
OUT.mkdir(parents=True, exist_ok=True)
LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
          'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
EXPECTED_COATS = {'resgated_top3','global96_top3','d4_swa3','repairv1_top3'}

class OutputReader:
    def __init__(self, root):
        self.root = Path(root)
        self.archive = zipfile.ZipFile(self.root) if self.root.is_file() else None
        self.names = ([i.filename for i in self.archive.infolist() if not i.is_dir()]
                      if self.archive else
                      [str(p.relative_to(self.root)).replace('\\','/')
                       for p in self.root.rglob('*') if p.is_file()])
    def close(self):
        if self.archive: self.archive.close()
    def locate(self, suffix):
        hits = [n for n in self.names if n == suffix or n.endswith('/'+suffix)]
        if len(hits) > 1: raise RuntimeError(f'Ambiguous {suffix}: {hits}')
        return hits[0] if hits else None
    def read(self, suffix):
        name = self.locate(suffix)
        if name is None: return None
        return self.archive.read(name) if self.archive else (self.root/name).read_bytes()
    def json(self, suffix):
        raw = self.read(suffix)
        return json.loads(raw) if raw is not None else None

if SOURCE_OVERRIDE:
    sources = [Path(SOURCE_OVERRIDE)]
else:
    sources = []
    if not INPUT_ROOT.exists(): raise FileNotFoundError(INPUT_ROOT)
    for base, dirs, files in os.walk(INPUT_ROOT):
        depth = len(Path(base).relative_to(INPUT_ROOT).parts)
        dirs[:] = [d for d in dirs if d not in
                   {'train_series','test_series','study_cache','dicom','dicoms'}]
        if 'btkd_v559_complete.json' in files:
            sources.append(Path(base))
        for name in files:
            if name.lower().endswith('.zip'):
                candidate = Path(base)/name
                if candidate.stat().st_size <= 2*1024**3:
                    try:
                        with zipfile.ZipFile(candidate) as z:
                            if any(n == 'btkd_v559_complete.json' or
                                   n.endswith('/btkd_v559_complete.json')
                                   for n in z.namelist()):
                                sources.append(candidate)
                    except (OSError, zipfile.BadZipFile):
                        pass
        if depth >= 9: dirs[:] = []
sources = sorted(set(sources), key=str)
if len(sources) != 1:
    raise RuntimeError(f'Attach exactly one saved 0.943 Output, found: {sources}')

reader = OutputReader(sources[0])
checks = []
def check(name, ok, detail=None):
    checks.append({'check':name,'pass':bool(ok),'detail':detail})

try:
    receipt = reader.json('btkd_v559_complete.json')
    blend = reader.json('_coat_raptor_blend_receipt.json')
    raw_csv = reader.read('submission.csv')
    check('required_root_files', receipt is not None and blend is not None and raw_csv is not None)
    if receipt is None or blend is None or raw_csv is None:
        raise RuntimeError('Output is missing a required root file')
    table = list(csv.DictReader(io.StringIO(raw_csv.decode('utf-8-sig'))))
    columns = list(table[0]) if table else []
    ids = [row['StudyInstanceUID'] for row in table] if columns == ['StudyInstanceUID']+LABELS else []
    check('schema_and_unique_ids', bool(ids) and len(set(ids)) == len(ids),
          {'rows':len(table),'columns':columns})
    values = []
    try:
        values = [float(row[label]) for row in table for label in LABELS]
    except (KeyError, TypeError, ValueError):
        pass
    check('finite_probability_range', len(values) == len(table)*12 and
          all(math.isfinite(v) and 0 <= v <= 1 for v in values))
    check('submission_sha256', hashlib.sha256(raw_csv).hexdigest() ==
          receipt.get('submission_sha256'))
    check('complete_and_study_count', receipt.get('status') == 'COMPLETE' and
          receipt.get('cohort') == len(table) and blend.get('study_count') == len(table),
          {'cohort':receipt.get('cohort'),'rows':len(table)})
    check('dino_a5_raptor_counts', receipt.get('dino_members') == 20 and
          receipt.get('a5_folds') == 5 and receipt.get('raptor_views') == 4,
          {'dino':receipt.get('dino_members'),'a5':receipt.get('a5_folds'),
           'raptor':receipt.get('raptor_views')})
    members = blend.get('members', [])
    check('four_exact_coat_members', len(members) == 4 and set(members) == EXPECTED_COATS
          and receipt.get('coat_family_members') == 4, members)
    check('coat_probability_mean_then_rank',
          blend.get('family_reduction') == 'rank_of_member_probability_mean' and
          receipt.get('family_reduction') == 'rank_of_member_probability_mean')
    within = blend.get('within_coat', {})
    check('coat_four_equal_weights', set(within) == EXPECTED_COATS and
          all(abs(float(within[k])-0.25) < 1e-8 for k in EXPECTED_COATS))
    suspicious = [e for e in receipt.get('events', []) if any(
        marker in str(e.get('kind','')).lower() for marker in
        ('failed','failure','fallback','unavailable','rejected','neutral','partial'))]
    check('no_failure_or_fallback_events', not suspicious, suspicious[:10])
    child_names = ['_coat_arm_receipt.json','_coat_g96_receipt.json',
                   '_coat_repairv1_receipt.json','d4_input/d4.receipt.json']
    for child_name in child_names:
        child = reader.json(child_name)
        check('child_'+Path(child_name).name,
              child is not None and child.get('fallback_studies') == 0 and
              child.get('studies') == len(table) and
              str(child.get('status','')).startswith('VALID_'),
              None if child is None else {'status':child.get('status'),
                                           'studies':child.get('studies'),
                                           'fallback_studies':child.get('fallback_studies')})
    raw_phases = reader.read('diagnostics/phase_events.jsonl')
    phase_events = ([json.loads(line) for line in raw_phases.decode().splitlines()
                     if line.strip()] if raw_phases else [])
    phase_errors = [e for e in phase_events if e.get('status') == 'FAILED'
                    or e.get('ok') is False]
    check('phase_log_no_failure', bool(phase_events) and not phase_errors,
          phase_errors[:10])
    phases = receipt.get('phases', {})
    missing_or_wrong = []
    for name, info in phases.items():
        phase_path = 'diagnostics/'+Path(info.get('path','')).name
        raw = reader.read(phase_path)
        if raw is None or hashlib.sha256(raw).hexdigest() != info.get('sha256'):
            missing_or_wrong.append(name)
    check('all_phase_sha256', bool(phases) and not missing_or_wrong,
          {'phase_count':len(phases),'missing_or_wrong':missing_or_wrong})
    report = {'status':'VISIBLE_OUTPUT_PASS' if all(c['pass'] for c in checks) else 'FAIL',
              'source':str(sources[0]),'visible_studies':len(table),
              'hidden_scoring_run_verified':False,
              'source_elapsed_seconds':receipt.get('elapsed_seconds'),
              'checks':checks,
              'warning':'Visible Output is not the hidden Kaggle scoring rerun; no AUC or Public/Private prediction is inferred.'}
finally:
    reader.close()

path = OUT/'baseline_integrity.json'
path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('AUDIT STATUS:',report['status'],'visible studies:',report['visible_studies'])
print('Failed checks:',[x for x in checks if not x['pass']])
print('Saved:',path)
'''

audit_notebook = {
    "cells": [markdown(audit_intro), code(audit_cell)],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "none", "dataSources": [], "isGpuEnabled": False,
                   "isInternetEnabled": False, "language": "python", "sourceType": "notebook"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
write_notebook(AUDIT, audit_notebook)


base = json.loads(BASE.read_text(encoding="utf-8"))
if len(base["cells"]) != 4 or [c["cell_type"] for c in base["cells"]] != [
        "markdown", "code", "code", "code"]:
    raise RuntimeError("Notebook 09 layout changed; refusing mechanical derivation")
plane = copy.deepcopy(base)
plane["cells"][0] = markdown("""
# 14 — B2 RadImageNet plane-aware MIL, controlled five-fold experiment

The **only scientific change from notebook 09 B2** is the study-level head:
separate sagittal/coronal/axial slice attention, followed by cross-plane
attention and a separate classifier for each of the 12 targets. Same frozen
RadImageNet ResNet50, 05A 224×224 cache, L1 soft labels, report-group folds,
optimizer, and weak-validation-BCE stopping. Gold58 is **audit-only**.

Attach: (1) RSNA competition, (2) notebook 02 saved Output containing
`L1_family_mean_RECOMMENDED.csv`, (3) private `Study_case 05_RSNA` Dataset
with complete `rsna_05a_dense_mil/study_cache` and `fold_assignments.csv`,
(4) public `resnet-50-radimagenet-marwan` Dataset with `ResNet50.pt`.
Use **T4×2, Internet Off**; only GPU 0 is used. First run the asset-check cell
and the following CPU architecture smoke-test cell. They must print
`ASSET CHECK PASSED` and `HEAD SMOKE PASSED` before Save & Run All.

This notebook makes **no submission**. Send the saved Output ZIP containing
`rsna_b2_planeaware5/run_report.json`, `weak_oof.csv`, `cv_metrics.csv`,
`gold_metrics.csv`, and `fold_0_head.pt` through `fold_4_head.pt`.
Do not attach its predictions to the 0.943 ensemble until the paired B2 OOF
comparison is inspected. `COMPLETE` means engineering completion, not gain.
""")

asset = "".join(plane["cells"][1]["source"])
replacements = {
    "B2_RadImageNet_frozen_encoder_MIL_fold01_v1":
        "B2_RadImageNet_planeaware_MIL_fivefold_v1",
    "/kaggle/working/rsna_b2_rad_frozen_fold01":
        "/kaggle/working/rsna_b2_planeaware5",
    "'max_total_hours': 6.0": "'max_total_hours': 3.0",
}
for old, new in replacements.items():
    if asset.count(old) != 1:
        raise RuntimeError(f"Notebook 09 asset anchor drift: {old}")
    asset = asset.replace(old, new)
plane["cells"][1]["source"] = asset.splitlines(keepends=True)

head = r'''
SLOT_TO_PLANE = (0,0,1,1,2)  # SAG_FLUID,SAG_NONFLUID,COR_FLUID,COR_NONFLUID,AX_FLUID
class StudyMIL(nn.Module):
    def __init__(self):
        super().__init__()
        d = CONFIG['head_dim']; fd = CONFIG['feature_dim']
        self.project = nn.Sequential(nn.LayerNorm(fd),nn.Linear(fd,d),nn.GELU(),nn.Dropout(.10))
        self.slot = nn.Embedding(5,d)
        self.position = nn.Sequential(nn.Linear(1,d),nn.GELU(),nn.Linear(d,d))
        self.plane_query = nn.Parameter(torch.randn(3,12,d)*.02)
        self.slice_attention = nn.MultiheadAttention(d,4,dropout=.10,batch_first=True)
        self.cross_plane = nn.MultiheadAttention(d,4,dropout=.10,batch_first=True)
        self.fusion_query = nn.Parameter(torch.randn(12,d)*.02)
        self.fusion_attention = nn.MultiheadAttention(d,4,dropout=.10,batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.weight = nn.Parameter(torch.randn(12,d)*.02)
        self.bias = nn.Parameter(torch.zeros(12))
        self.register_buffer('slot_to_plane',torch.tensor(SLOT_TO_PLANE,dtype=torch.long))

    def forward(self,x,s,p,m):
        b,t,_ = x.shape
        if s.shape != (b,t) or p.shape != (b,t) or m.shape != (b,t):
            raise RuntimeError('Plane-aware head input shape mismatch')
        if bool(((s < 0) | (s >= 5)).any()):
            raise RuntimeError('Invalid 05A slot id')
        token = self.project(x.float())+self.slot(s)+self.position(p.unsqueeze(-1))
        plane_ids = self.slot_to_plane[s]
        plane_tokens, plane_present = [], []
        for plane_id in range(3):
            selected = m & (plane_ids == plane_id)
            present = selected.any(dim=1)
            safe = selected.clone()
            safe[~present,0] = True
            query = self.plane_query[plane_id].unsqueeze(0).expand(b,-1,-1)
            pooled,_ = self.slice_attention(query,token,token,
                                            key_padding_mask=~safe,need_weights=False)
            pooled = torch.where(present[:,None,None],pooled,torch.zeros_like(pooled))
            plane_tokens.append(pooled)
            plane_present.append(present)
        stacked = torch.stack(plane_tokens,dim=2)  # [B,12,3,D]
        present = torch.stack(plane_present,dim=1)  # [B,3]
        flat = stacked.reshape(b*12,3,-1)
        missing = (~present[:,None,:].expand(b,12,3)).reshape(b*12,3).clone()
        all_empty = ~present.any(dim=1)
        if bool(all_empty.any()):
            missing[all_empty.repeat_interleave(12),0] = False
        contextual,_ = self.cross_plane(flat,flat,flat,
                                         key_padding_mask=missing,need_weights=False)
        query = self.fusion_query[None].expand(b,-1,-1).reshape(b*12,1,-1)
        fused,_ = self.fusion_attention(query,contextual,contextual,
                                        key_padding_mask=missing,need_weights=False)
        fused = fused.reshape(b,12,-1)
        fused = torch.where(all_empty[:,None,None],torch.zeros_like(fused),fused)
        return (self.norm(fused)*self.weight[None]).sum(dim=-1)+self.bias

# Run on CPU before committing to multi-study GPU feature extraction.
torch.manual_seed(CONFIG['seed'])
smoke = StudyMIL().cpu()
fake_x = torch.randn(3,8,CONFIG['feature_dim'])
fake_s = torch.tensor([[0,0,2,2,4,4,0,0],
                       [0,1,0,0,0,0,0,0],
                       [0,0,0,0,0,0,0,0]],dtype=torch.long)
fake_p = torch.linspace(0,1,8).repeat(3,1)
fake_m = torch.tensor([[1,1,1,1,1,1,0,0],
                       [1,1,0,0,0,0,0,0],
                       [0,0,0,0,0,0,0,0]],dtype=torch.bool)
fake_logits = smoke(fake_x,fake_s,fake_p,fake_m)
assert fake_logits.shape == (3,12) and bool(torch.isfinite(fake_logits).all())
fake_logits.square().mean().backward()
assert all(p.grad is not None and bool(torch.isfinite(p.grad).all())
           for p in smoke.parameters() if p.requires_grad)
del smoke,fake_x,fake_s,fake_p,fake_m,fake_logits
gc.collect()
print('HEAD SMOKE PASSED: 3 planes, sagittal-only, empty study; finite gradients',flush=True)
'''
plane["cells"].insert(2, code(head))

encode = "".join(plane["cells"][3]["source"])
anchor = "if not np.isfinite(features[masks]).all(): raise RuntimeError('Nonfinite feature array')"
if encode.count(anchor) != 1:
    raise RuntimeError("Notebook 09 encoder completion anchor drift")
encode = encode.replace(anchor,
    "if empty: raise RuntimeError(f'{empty} empty studies; refusing training')\n" + anchor)
plane["cells"][3]["source"] = encode.splitlines(keepends=True)

train = "".join(plane["cells"][4]["source"])
start = train.index("class StudyMIL(nn.Module):")
end = train.index("def loader(", start)
train = train[:start] + train[end:]
anchor = "if gold_parts:\n"
if train.count(anchor) != 1:
    raise RuntimeError("Notebook 09 completion anchor drift")
train = train.replace(anchor, """if empty:
    raise RuntimeError(f'{empty} empty studies; refusing a misleading COMPLETE report')
if len(cv_rows) == 5 and all(r['stop_reason'] is None for r in cv_rows):
    joined = pd.concat(oof_parts,ignore_index=True)
    if (len(joined) != len(weak) or not joined.StudyInstanceUID.is_unique or
            set(joined.StudyInstanceUID) != set(weak.StudyInstanceUID) or
            len(gold_parts) != 5 or
            not all((OUT/f'fold_{fold}_head.pt').is_file() for fold in range(5))):
        raise RuntimeError('Five-fold output integrity failed; refusing COMPLETE')
if gold_parts:
""", 1)
anchor = "'config':CONFIG,'labels':LABELS,'source_weight_sha256':digest.hexdigest()"
if train.count(anchor) != 1:
    raise RuntimeError("Notebook 09 checkpoint anchor drift")
train = train.replace(anchor, "'config':CONFIG,'labels':LABELS,'head_arch':'three_plane_cross_attention_v1',"
                      "'slot_to_plane':SLOT_TO_PLANE,'source_weight_sha256':digest.hexdigest()")
plane["cells"][4]["source"] = train.splitlines(keepends=True)
plane["metadata"]["kaggle"]["isInternetEnabled"] = False
plane["metadata"]["kaggle"]["isGpuEnabled"] = True
plane["metadata"]["kaggle"]["accelerator"] = "nvidiaTeslaT4"
write_notebook(PLANE, plane)

print(AUDIT.name, "and", PLANE.name, "written and AST-validated")
