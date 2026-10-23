"""Create three disjoint target-group ablations of the 0.943+B2 probe.

The source inference cells are copied verbatim from notebook 16. Only the
header and final blend cell change. Each notebook writes one submission.csv.
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "16_RSNA_0943_Anchor_Plus_B2_Fixed20_Probe.ipynb"
LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
GROUPS = {
    "G1": ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"),
    "G2": ("Medial OA", "Lateral OA", "PF OA", "Effusion"),
    "G3": ("Synovitis", "Baker's", "Contusion", "Fracture"),
}


def source(cell: dict) -> str:
    value = cell.get("source", [])
    return "".join(value) if isinstance(value, list) else str(value)


def set_source(cell: dict, value: str) -> None:
    cell["source"] = value.splitlines(keepends=True)


def replace_once(value: str, old: str, new: str) -> str:
    if value.count(old) != 1:
        raise RuntimeError(f"Expected one source anchor, got {value.count(old)}: {old[:80]}")
    return value.replace(old, new, 1)


def intro(group_id: str, labels: tuple[str, ...]) -> str:
    return f"""# 17 {group_id} — B2 target-group ablation of the protected 0.943 anchor

This is one of three **predeclared diagnostic submissions**. The original
0.943 inference and notebook 09 five-fold B2 inference are unchanged. Only
these four targets use `rank(0.80*rank(anchor)+0.20*rank(B2))`:

{', '.join(labels)}.

The other eight target columns are copied from the 0.943 anchor. G1, G2 and G3
partition all 12 targets, so their Public-score changes versus the same 0.943
control approximately decompose the full 0.942 probe's change, subject to the
three-decimal leaderboard display and identical scoring cohorts. This does
not reveal per-target scores or establish Private-LB improvement.

Kaggle: **T4×2, Internet Off**. Attach the same 18 inputs that succeeded in
notebook 16, including the competition, public model assets, and the exact
complete notebook 09 Output with `fold_0_head.pt` through `fold_4_head.pt`.
Do not attach multiple notebook 09 Outputs with the same head filenames.

Use **Save & Run All** for this notebook. Inspect
`anchor_b2_group_probe_receipt.json` before submitting: status COMPLETE,
`group_id={group_id}`, four selected labels, anchor CoAt members = 4,
B2 folds = 0–4, and no unexpected empty/decode issues. The final
`submission.csv` is this group's probe. Keep the 0.943 submission selected
while diagnosing; do not select a group just because its rounded Public
score is higher. Preserve notebook 16 and all source notebooks unchanged.
"""


def final_cell(original: str, group_id: str, labels: tuple[str, ...]) -> str:
    value = replace_once(
        original,
        "# Fixed 80/20 rank blend. No fitting or weight search is performed.",
        f"# Predeclared {group_id} target-group ablation; other targets stay at 0.943 anchor.",
    )
    value = replace_once(
        value,
        "_mix_cols = ['StudyInstanceUID'] + _mix_labels\n",
        "_mix_cols = ['StudyInstanceUID'] + _mix_labels\n"
        f"_mix_group_id = {group_id!r}\n"
        f"_mix_group = {list(labels)!r}\n"
        "_mix_unchanged = [label for label in _mix_labels if label not in _mix_group]\n"
        "if (len(_mix_group) != 4 or len(set(_mix_group)) != 4 or\n"
        "        not set(_mix_group).issubset(_mix_labels) or len(_mix_unchanged) != 8):\n"
        "    raise RuntimeError('Target-group partition invalid')\n",
    )
    value = replace_once(
        value,
        "_mix_output = _mix_anchor[['StudyInstanceUID']].copy()\n"
        "_mix_output[_mix_labels] = _mix_raw.rank(method='average', pct=True)\n",
        "_mix_output = _mix_anchor.copy()\n"
        "_mix_output[_mix_group] = _mix_raw[_mix_group].rank(method='average', pct=True)\n"
        "if not _mix_np.array_equal(_mix_output[_mix_unchanged].to_numpy(float),\n"
        "                           _mix_anchor[_mix_unchanged].to_numpy(float)):\n"
        "    raise RuntimeError('Unselected anchor targets changed before saving')\n",
    )
    value = replace_once(
        value,
        "    raise RuntimeError('Final saved submission roundtrip failed')\n",
        "    raise RuntimeError('Final saved submission roundtrip failed')\n"
        "if not _mix_np.allclose(_mix_roundtrip[_mix_unchanged].to_numpy(float),\n"
        "                         _mix_anchor[_mix_unchanged].to_numpy(float),\n"
        "                         rtol=0, atol=1e-15):\n"
        "    raise RuntimeError('Unselected anchor targets changed after saving')\n",
    )
    value = replace_once(
        value,
        "    'experiment': 'protected_0943_anchor_plus_B2_fixed20_rank_probe_v1',\n"
        "    'studies': len(_mix_roundtrip),\n"
        "    'formula': 'rank(0.80*rank(anchor_0943)+0.20*rank(B2_09_fivefold))',\n",
        "    'experiment': 'protected_0943_anchor_plus_B2_group_ablation_v1',\n"
        "    'group_id': _mix_group_id,\n"
        "    'selected_labels': _mix_group,\n"
        "    'unchanged_labels': _mix_unchanged,\n"
        "    'studies': len(_mix_roundtrip),\n"
        "    'formula_selected': 'rank(0.80*rank(anchor_0943)+0.20*rank(B2_09_fivefold))',\n"
        "    'formula_unchanged': 'anchor_0943_exact',\n",
    )
    value = replace_once(
        value,
        "(_mix_root / 'anchor_b2_probe_receipt.json').write_text(\n",
        "(_mix_root / 'anchor_b2_group_probe_receipt.json').write_text(\n",
    )
    value = replace_once(
        value,
        "print('FIXED 20% PROBE COMPLETE', {\n",
        "print('TARGET-GROUP PROBE COMPLETE', {\n"
        "    'group_id': _mix_group_id,\n"
        "    'selected_labels': _mix_group,\n",
    )
    ast.parse(value)
    return value


def main() -> None:
    source_notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = source_notebook["cells"]
    if len(cells) != 41 or "Fixed 80/20 rank blend" not in source(cells[-1]):
        raise RuntimeError("Notebook 16 structure changed; inspect before rebuilding")
    if set().union(*[set(v) for v in GROUPS.values()]) != set(LABELS):
        raise RuntimeError("Groups do not cover all targets")
    if sum(len(v) for v in GROUPS.values()) != len(LABELS):
        raise RuntimeError("Groups overlap")

    for group_id, labels in GROUPS.items():
        notebook = copy.deepcopy(source_notebook)
        notebook["metadata"] = {
            key: copy.deepcopy(source_notebook["metadata"][key])
            for key in ("kernelspec", "language_info", "kaggle")
            if key in source_notebook["metadata"]
        }
        set_source(notebook["cells"][0], intro(group_id, labels))
        set_source(notebook["cells"][-2], f"## Predeclared {group_id} target-group probe\n")
        set_source(notebook["cells"][-1], final_cell(source(cells[-1]), group_id, labels))
        for index, cell in enumerate(notebook["cells"]):
            cell["id"] = f"rsna17-{group_id.lower()}-{index:03d}"
            if cell["cell_type"] == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
                ast.parse(source(cell))
        for index in range(1, len(cells) - 2):
            if source(notebook["cells"][index]) != source(cells[index]):
                raise RuntimeError(f"Source inference cell {index} changed in {group_id}")
        name = f"17_{group_id}_RSNA_0943_B2_Group_Probe.ipynb"
        path = ROOT / name
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
        print(f"CREATED {name}: {len(cells)} cells, "
              f"{len(notebook['metadata']['kaggle']['dataSources'])} inherited inputs")


if __name__ == "__main__":
    main()
