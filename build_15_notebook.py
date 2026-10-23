"""Build a controlled sampled-neighbor 2.5D successor to B2 notebook 09."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "09_B2_RadImageNet_FrozenMIL_Fold01.ipynb"
TARGET = ROOT / "15_B2_RadImageNet_SampledNeighbor25D_FiveFold.ipynb"
notebook = copy.deepcopy(json.loads(SOURCE.read_text(encoding="utf-8")))
if len(notebook["cells"]) != 4:
    raise RuntimeError("B2 source layout changed")


def lines(text: str) -> list[str]:
    return dedent(text).strip("\n").splitlines(keepends=True)


notebook["cells"][0]["source"] = lines("""
# 15 — B2 RadImageNet sampled-neighbor 2.5D, five-fold group CV

Controlled input-representation experiment against notebook 09 B2. Keep the
same frozen RadImageNet ResNet50 checkpoint, 05A 224×224 cache, five
report-group folds, L1 soft labels, exact MIL head, optimizer and weak-BCE
checkpoint selection. The **only scientific change** is replacing repeated
grayscale RGB with `[previous sampled slice, current slice, next sampled
slice]` within each of the five MRI slots. Edge slices repeat themselves.
These are neighbors in the existing 05A **sampled cache**, not necessarily
immediately adjacent original DICOM slices. Gold58 remains audit-only.

Attach: RSNA competition; notebook 02 Output containing
`L1_family_mean_RECOMMENDED.csv`; private `Study_case 05_RSNA` Dataset with
complete `rsna_05a_dense_mil/study_cache` and `fold_assignments.csv`; public
`resnet-50-radimagenet-marwan` Dataset with `ResNet50.pt`. Internet Off.
First run the asset cell and CPU neighbor-mapping smoke-test cell. They must
print `ASSET CHECK PASSED` and `NEIGHBOR SMOKE PASSED`. Then switch to T4×2
and Save & Run All. This notebook uses GPU 0 only, makes **no submission**,
and has an internal 3-hour safety guard, not a promised runtime.

Send the Output ZIP containing `rsna_b2_sampled25d5/run_report.json`,
`weak_oof.csv`, `cv_metrics.csv`, `gold_metrics.csv`, and five head checkpoints.
Do not blend it into the 0.943 submission until its paired OOF comparison
against notebook 09 has been audited.
""")

asset = "".join(notebook["cells"][1]["source"])
replacements = {
    "B2_RadImageNet_frozen_encoder_MIL_fold01_v1":
        "B2_RadImageNet_sampled_neighbor_25D_fivefold_v1",
    "/kaggle/working/rsna_b2_rad_frozen_fold01":
        "/kaggle/working/rsna_b2_sampled25d5",
    "'max_total_hours': 6.0": "'max_total_hours': 3.0",
    "'feature_dim': 2048, 'head_dim': 256, 'head_batch': 32,":
        "'feature_dim': 2048, 'head_dim': 256, 'head_batch': 32,\n"
        "    'channel_mode': 'previous_sampled_current_next_sampled_within_slot',",
}
for old, new in replacements.items():
    if asset.count(old) != 1:
        raise RuntimeError(f"Notebook 09 asset anchor drift: {old}")
    asset = asset.replace(old, new)
notebook["cells"][1]["source"] = asset.splitlines(keepends=True)

smoke = """
def neighbors_within_slot(slot_ids):
    slot_ids = np.asarray(slot_ids)
    previous = np.arange(len(slot_ids),dtype=np.int32)
    following = previous.copy()
    for slot in range(5):
        idx = np.flatnonzero(slot_ids == slot)
        if idx.size:
            previous[idx] = np.r_[idx[0],idx[:-1]]
            following[idx] = np.r_[idx[1:],idx[-1]]
    return previous,following

# CPU-only test: channel neighbors must never cross an MRI slot boundary.
test_slots = np.array([0,0,2,2,4,4,1],dtype=np.int64)
test_images = np.arange(len(test_slots),dtype=np.uint8)[:,None,None]
test_prev,test_next = neighbors_within_slot(test_slots)
test_rgb = np.stack([test_images[test_prev],test_images,test_images[test_next]],axis=1)
assert test_prev.tolist() == [0,0,2,2,4,4,6]
assert test_next.tolist() == [1,1,3,3,5,5,6]
assert test_rgb[:,1,0,0].tolist() == list(range(7))
assert np.all(test_slots[test_prev] == test_slots)
assert np.all(test_slots[test_next] == test_slots)
del test_slots,test_images,test_prev,test_next,test_rgb
print('NEIGHBOR SMOKE PASSED: center retained, edges repeated, no cross-slot mixing',flush=True)
"""
notebook["cells"].insert(2, {
    "cell_type": "code", "id": "sampled-neighbor-smoke", "execution_count": None, "metadata": {},
    "outputs": [], "source": lines(smoke),
})

encoder = "".join(notebook["cells"][3]["source"])
changes = [
    (
        "        if t==0: empty+=1\n",
        "        if t==0: empty+=1\n"
        "        prev_idx,next_idx=neighbors_within_slot(s)\n",
    ),
    (
        "            gray=torch.from_numpy(part).to(DEVICE).float().div_(127.5).sub_(1.0)\n"
        "            rgb=gray.unsqueeze(1).expand(-1,3,-1,-1).contiguous()",
        "            selected=np.arange(offset,offset+len(part))\n"
        "            triplet=np.stack([images[prev_idx[selected]],part,\n"
        "                              images[next_idx[selected]]],axis=1)\n"
        "            rgb=torch.from_numpy(triplet).to(DEVICE).float().div_(127.5).sub_(1.0)",
    ),
    (
        "            del gray,rgb,value,half",
        "            del selected,triplet,rgb,value,half",
    ),
    (
        "        del images,s,pos",
        "        del images,s,pos,prev_idx,next_idx",
    ),
    (
        "        if (i+1)%250==0:",
        "        if i==63:\n"
        "            torch.cuda.synchronize(0)\n"
        "            early_rate=n_images/max(1e-6,monotonic()-encode_start)\n"
        "            print('EARLY 2.5D RATE',round(early_rate,1),'images/sec',flush=True)\n"
        "            if early_rate < 75:\n"
        "                raise RuntimeError('2.5D encoder too slow; stopped after 64 studies')\n"
        "        if (i+1)%250==0:",
    ),
    (
        "if not np.isfinite(features[masks]).all(): raise RuntimeError('Nonfinite feature array')",
        "if empty: raise RuntimeError(f'{empty} empty studies; refusing training')\n"
        "if not np.isfinite(features[masks]).all(): raise RuntimeError('Nonfinite feature array')",
    ),
]
for old, new in changes:
    if encoder.count(old) != 1:
        raise RuntimeError(f"Notebook 09 encoder anchor drift: {old[:80]}")
    encoder = encoder.replace(old, new)
notebook["cells"][3]["source"] = encoder.splitlines(keepends=True)

train = "".join(notebook["cells"][4]["source"])
anchor = "if gold_parts:\n"
if train.count(anchor) != 1:
    raise RuntimeError("Notebook 09 completion anchor drift")
train = train.replace(anchor, """if len(cv_rows) == 5 and all(r['stop_reason'] is None for r in cv_rows):
    joined = pd.concat(oof_parts,ignore_index=True)
    if (len(joined) != len(weak) or not joined.StudyInstanceUID.is_unique or
            set(joined.StudyInstanceUID) != set(weak.StudyInstanceUID) or
            len(gold_parts) != 5 or
            not all((OUT/f'fold_{fold}_head.pt').is_file() for fold in range(5))):
        raise RuntimeError('Five-fold output integrity failed; refusing COMPLETE')
if gold_parts:
""", 1)
notebook["cells"][4]["source"] = train.splitlines(keepends=True)
notebook["cells"][4]["id"] = "sampled-neighbor-fivefold-train"

notebook["metadata"]["kaggle"]["isInternetEnabled"] = False
notebook["metadata"]["kaggle"]["isGpuEnabled"] = True
notebook["metadata"]["kaggle"]["accelerator"] = "nvidiaTeslaT4"
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        ast.parse("".join(cell["source"]), filename=TARGET.name)
        cell["outputs"] = []
        cell["execution_count"] = None
TARGET.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n",
                  encoding="utf-8")
print(TARGET.name, "written and AST-validated")
