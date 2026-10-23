"""Package controlled Kaggle reproductions of the public Raptor/R7 notebooks."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).parent
SOURCES = HERE / "Kaggle public"


def md_cell(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": value.splitlines(keepends=True)}


def code_cell(value: str) -> dict:
    compile(value, "generated Kaggle cell", "exec")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": value.splitlines(keepends=True)}


PREFLIGHT = r'''# Cheap input and hardware gate, before either DICOM decoding or model loading.
from pathlib import Path
import csv, json, torch

INPUT = Path('/kaggle/input')
WORK = Path('/kaggle/working')

def mount_candidates(slug):
    candidates = [INPUT / slug, INPUT / 'competitions' / slug]
    candidates.extend(INPUT.glob('datasets/*/' + slug))
    candidates.extend(INPUT.glob('notebooks/*/' + slug))
    return [p for p in candidates if p.is_dir()]

def require_mount(slug, required):
    found = [p for p in mount_candidates(slug) if (p / required).is_file()]
    if len(found) != 1:
        raise RuntimeError(f'Expected exactly one mount for {slug}/{required}; found {found}')
    return found[0]

competition = require_mount('rsna-knee-abnormality-detection', 'test.csv')
raptor = require_mount('rsna-knee-publication-swa-weights-20261007',
                       'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt')
r7 = require_mount('rsna-knee-compact-r7cnxfsss', 'manifest.json')
code = require_mount('rsna-orthofoundation-fold0-epoch7-20261004', 'receipt.json')
with (r7 / 'manifest.json').open(encoding='utf-8') as f:
    r7_manifest = json.load(f)
if r7_manifest.get('bundle_id') != '8c7b666e27dc0e8f':
    raise RuntimeError('R7 bundle_id differs from the tested public notebook')
if not (competition / 'test_series.csv').is_file() or not (competition / 'test_series').is_dir():
    raise RuntimeError('Competition test series CSV or DICOM directory is missing')
if r7_manifest.get('views') in ('v3k', 'v4') and not (code / 'anatomy.pt').is_file():
    raise RuntimeError('Missing anatomical router checkpoint required by this R7 view mode')
with (competition / 'test.csv').open(newline='', encoding='utf-8') as f:
    test_rows = list(csv.DictReader(f))
if not test_rows or len({r['StudyInstanceUID'] for r in test_rows}) != len(test_rows):
    raise RuntimeError('Empty or duplicate test StudyInstanceUIDs')
gpu_count = torch.cuda.device_count()
if gpu_count < 2:
    raise RuntimeError(f'T4x2 required, but only {gpu_count} CUDA device(s) available')
preflight = {
    'status': 'INPUTS_READY', 'test_rows_visible_to_this_run': len(test_rows),
    'gpu_count': gpu_count, 'competition': str(competition), 'raptor_weights': str(raptor),
    'r7_bundle': str(r7), 'r7_code': str(code),
    'r7_bundle_id': r7_manifest['bundle_id'],
    'note': 'This verifies presence and bundle identity only. Original stage code checks the checkpoint/module SHA256.'
}
(WORK / '25_preflight.json').write_text(json.dumps(preflight, indent=2), encoding='utf-8')
print('PREFLIGHT_OK', json.dumps(preflight), flush=True)
'''

PREFLIGHT_224 = PREFLIGHT.replace(
    "gpu_count = torch.cuda.device_count()",
    """weight_224 = 'raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt'
if not (raptor / weight_224).is_file():
    alternative = [p for p in mount_candidates('rsna-knee-raptor-alldata-d96-blendjev-oai-c08r224')
                   if (p / weight_224).is_file()]
    if len(alternative) != 1:
        raise RuntimeError('25B requires the exact Raptor-224 checkpoint before the long R7 stage')
gpu_count = torch.cuda.device_count()""",
)

POSTFLIGHT = r'''# Post-run diagnostics. This does not change the reference submission.csv.
from pathlib import Path
import json, numpy as np, pandas as pd

work = Path('/kaggle/working')
receipt = json.loads((work / 'blend_receipt.json').read_text(encoding='utf-8'))
sub = pd.read_csv(work / 'submission.csv', dtype={'StudyInstanceUID': str})
test_ids = pd.read_csv(competition / 'test.csv', dtype={'StudyInstanceUID': str})['StudyInstanceUID'].tolist()
sample_file = competition / 'sample_submission.csv'
expected_columns = (list(pd.read_csv(sample_file, nrows=1).columns)
                    if sample_file.is_file() else list(sub.columns))
if list(sub.columns) != expected_columns or sub['StudyInstanceUID'].tolist() != test_ids:
    raise RuntimeError('Final submission schema or row order differs from competition input')
if not np.isfinite(sub.iloc[:, 1:].to_numpy(dtype=float)).all():
    raise RuntimeError('Final submission has non-finite values')
if (sub.iloc[:, 1:].to_numpy(dtype=float) < 0).any() or (sub.iloc[:, 1:].to_numpy(dtype=float) > 1).any():
    raise RuntimeError('Final submission has values outside [0, 1]')
audit = {
    'status': 'COMPLETE' if receipt.get('raptor_failed') == 0 and receipt.get('ours_failed') == 0 else 'FALLBACKS_PRESENT',
    'rows': len(sub), 'raptor_failed': receipt.get('raptor_failed'),
    'r7_failed': receipt.get('ours_failed'), 'blend_weights': receipt.get('weights'),
    'rank_basis': receipt.get('rank_basis'), 'total_seconds': receipt.get('total_seconds'),
    'note': 'Only the live Kaggle leaderboard can establish Public AUC; the 3 visible rows cannot.'
}
(work / '25_run_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
print('RUN_AUDIT', json.dumps(audit), flush=True)
'''


def source_notebook(name: str) -> dict:
    path = SOURCES / name
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return nb


def code_digest(cells: list[dict]) -> str:
    body = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    return hashlib.sha256(body.encode()).hexdigest()


def save_notebook(nb: dict, filename: str) -> None:
    meta = nb.setdefault("metadata", {}).setdefault("kaggle", {})
    meta["isInternetEnabled"] = False
    meta["isGpuEnabled"] = True
    meta["accelerator"] = "nvidiaTeslaT4"
    output = HERE / filename
    output.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    verify = json.loads(output.read_text(encoding="utf-8"))
    assert verify["nbformat"] == 4
    assert verify["metadata"]["kaggle"]["isInternetEnabled"] is False
    for i, cell in enumerate(verify["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None and not cell["outputs"]
        source = "".join(cell["source"])
        if source.startswith("%%writefile "):
            source = source.split("\n", 1)[1]
        compile(source, f"{filename}:cell{i}", "exec")
    print(output, output.stat().st_size)


source_a = source_notebook("rsna-knee-0-956-raptor-convnext-r7-ensemble.ipynb")
source_a_hash = code_digest(source_a["cells"])
source_a["cells"][0] = md_cell("""# 25A — Raptor-384 + R7 public-source reproduction

This is a **controlled Public-LB reproduction**, not a newly trained competition-only model. The five source code cells from the public 0.956 notebook are unchanged; this copy adds only an early attachment gate and a final audit. The reference uses one Raptor CoAtNet-384 SWA checkpoint, the R7 four-fold ConvNeXt pipeline with its auxiliary models/stackers, and a per-finding percentile-rank blend of **50% Raptor + 50% R7**. Source code SHA-256: `""" + source_a_hash + """`.

Attach the RSNA competition and the three public datasets: `nartaa/rsna-knee-publication-swa-weights-20261007`, `heliosli/rsna-knee-compact-r7cnxfsss`, and `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`. Set **GPU T4x2**, **Internet Off**, then Save & Run All. Cell 1 checks mounts/GPU; the original source code verifies checkpoint/module hashes. Inspect `25_run_audit.json` and `submission.csv` before spending a submission slot. A successful run or visible-three-study output does not guarantee the hidden rerun or 0.956 Public.

**Provenance gate for a prize/final entry:** the Raptor checkpoint name and original authors document OAI training; the R7 pipeline includes external MRI-derived foundation/anatomy weights. Public-LB reproduction is separate from a verified eligibility determination. The 24C competition-only cache is not used by this inference notebook.
""")
source_a["cells"].insert(2, code_cell(PREFLIGHT))
source_a["cells"].append(code_cell(POSTFLIGHT))
save_notebook(source_a, "25A_Raptor_R7_Public0956_Reproduction.ipynb")


source_b = source_notebook("rsna-knee-one-unused-checkpoint-lifts-the-0-956.ipynb")
source_b_hash = code_digest(source_b["cells"])
source_b["cells"][0] = md_cell("""# 25B — R7 + dual-resolution Raptor (controlled 224-pixel probe)

Run this **after** 25A has completed and you know its Public score and hidden-run time. This public-source variant adds a CoAtNet-224, 0.8-crop checkpoint to the same Raptor-384 + R7 pipeline. Its fixed blend is **50% R7 + 32.5% Raptor-384 + 17.5% Raptor-224** (224 gets 35% of the Raptor half). This copy refuses to fall back silently to the two-model baseline if the 224 checkpoint is skipped or unusable. Original source code SHA-256: `""" + source_b_hash + """`.

Attach the same three datasets and competition as 25A. Set **GPU T4x2**, **Internet Off**, then Save & Run All. Inspect `25_run_audit.json`, `blend_receipt.json`, and `submission.csv` before submission. The public notebook only demonstrated success on three visible studies; a score gain and sub-nine-hour hidden run are **not guaranteed**. This is an external-MRI checkpoint experiment, not the 24C competition-only branch.
""")
source_b["cells"].insert(2, code_cell(PREFLIGHT_224))
stage2b = next(i for i, c in enumerate(source_b["cells"])
               if c["cell_type"] == "code" and "R224_OK = False" in "".join(c["source"]))
source_b["cells"].insert(stage2b + 1, code_cell("""# Controlled ablation: do not relabel the baseline as a 224-pixel experiment.
if not R224_OK:
    raise RuntimeError('Raptor-224 did not finish; 25B cannot silently become the 25A baseline')
"""))
blend_i = next(i for i, c in enumerate(source_b["cells"])
               if c["cell_type"] == "code" and "A224 = 0.35" in "".join(c["source"]))
blend_source = "".join(source_b["cells"][blend_i]["source"])
needle = "n = len(test_ids)\n"
assert blend_source.count(needle) == 1
blend_source = blend_source.replace(needle,
    "if r224 is None:\n    raise RuntimeError('Raptor-224 output is missing or invalid; no 25B submission written')\n" + needle)
source_b["cells"][blend_i]["source"] = blend_source.splitlines(keepends=True)
source_b["cells"].append(code_cell(POSTFLIGHT + "\nassert r224 is not None and R224_OK\n"))
save_notebook(source_b, "25B_Raptor384_Raptor224_R7_Probe.ipynb")
