"""Build a strict, source-preserving Kaggle successor to the supplied 0.958 notebook."""

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Kaggle public" / "0.958.ipynb"
TARGET = ROOT / "31_RSNA_0958_Rebuild_Strict.ipynb"
MANIFEST = ROOT / "31_RSNA_0958_Rebuild_Manifest.json"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source match, found {count}")
    return source.replace(old, new, 1)


def set_cell(notebook: dict, index: int, source: str) -> None:
    notebook["cells"][index]["source"] = source.splitlines(keepends=True)


def main() -> None:
    original_bytes = SOURCE.read_bytes()
    notebook = json.loads(original_bytes)
    rebuilt = copy.deepcopy(notebook)

    # Saved papermill outputs and widget state describe an old three-study run.
    for cell in rebuilt["cells"]:
        cell["metadata"].pop("papermill", None)
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    rebuilt["metadata"].pop("widgets", None)
    kaggle = rebuilt["metadata"]["kaggle"]
    kaggle["accelerator"] = "nvidiaTeslaT4"
    kaggle["isGpuEnabled"] = True
    kaggle["isInternetEnabled"] = False

    cell6 = "".join(rebuilt["cells"][6]["source"])
    cell6 = cell6.replace("require_coat_members=0, global_asset_fallback=True,", "require_coat_members=4, global_asset_fallback=True,")
    if cell6.count("require_coat_members=4, global_asset_fallback=True,") != 2:
        raise RuntimeError("expected to pin all default presets to four CoAt members")
    cell6 = replace_once(
        cell6,
        "if _rk_override:\n    RK_CFG.update(_rk_json.loads(_rk_override))\n",
        """if _rk_override:
    _rk_changes = _rk_json.loads(_rk_override)
    if not isinstance(_rk_changes, dict) or set(_rk_changes) - set(RK_CFG):
        raise RuntimeError('unknown or invalid RSNA_TUNE_OVERRIDE keys')
    RK_CFG.update(_rk_changes)
if int(RK_CFG['require_coat_members']) != 4:
    raise RuntimeError('the rebuilt 0.958 pipeline requires all four CoAt readers')
if not 0.0 <= float(RK_CFG['private_alpha']) <= 1.0:
    raise RuntimeError('private_alpha must be in [0,1]')
if not 0.0 <= float(RK_CFG['base_coat_w']) <= 1.0:
    raise RuntimeError('base_coat_w must be in [0,1]')
if any(not 0.0 <= float(v) <= 1.0 for v in RK_CFG['coat_route'].values()):
    raise RuntimeError('coat_route weights must be in [0,1]')
""",
        "validated override and strict family count",
    )
    set_cell(rebuilt, 6, cell6)

    cell48 = "".join(rebuilt["cells"][48]["source"])
    cell48 = replace_once(
        cell48,
        "    if not _members:\n        rsna_event('coat_family_unavailable'",
        """    _expected_coat_members = (
        'resgated_top3', 'global96_top3', 'd4_swa3', 'repairv1_top3')
    _actual_coat_members = tuple(name for name, _ in _members)
    if _actual_coat_members != _expected_coat_members:
        raise RuntimeError('incomplete CoAt family: expected %s, got %s' %
                           (_expected_coat_members, _actual_coat_members))
    if not _members:
        rsna_event('coat_family_unavailable'""",
        "CoAt reader identity gate",
    )
    cell48 = replace_once(
        cell48,
        "    if len(_members) >= 2 and len(_probs) == len(_members) and RK_CFG['family_mode'] == 'prob_mean':",
        """    if set(_probs) != set(_expected_coat_members):
        raise RuntimeError('CoAt probability artifacts incomplete: %s' %
                           sorted(set(_expected_coat_members) - set(_probs)))
    if len(_members) >= 2 and len(_probs) == len(_members) and RK_CFG['family_mode'] == 'prob_mean':""",
        "CoAt probability identity gate",
    )
    cell48 = replace_once(
        cell48,
        """    if _RK_SRC_R is None or _RK_SRC_C is None:
        print('[v6] flat3 inputs unavailable; using the nested 2-way mix in probit space',
              flush=True)
        _rk_w = _ke_np.asarray([float(_coatnet_weight[l]) for l in _blend_labels], _ke_np.float64)
        _rk_parts = [_RK_SRC_T, _blend_cr[_blend_labels].to_numpy(_ke_np.float64)]
        _rk_wts = [1.0 - _rk_w, _rk_w]
        _RK_OUTER_MODE = 'probit_nested2_fallback'
    else:
""",
        """    if _RK_SRC_R is None or _RK_SRC_C is None:
        raise RuntimeError('flat3 requires both public Raptor and all four CoAt sources')
    else:
""",
        "flat3 fallback gate",
    )
    set_cell(rebuilt, 48, cell48)

    cell52 = "".join(rebuilt["cells"][52]["source"])
    cell52 = replace_once(
        cell52,
        "    rsna_event('raptor_preparation_identities_incomplete', missing=len(_expected_preparations - _ke_input_ids), extra=len(_ke_input_ids - _expected_preparations))\n",
        """    rsna_event('raptor_preparation_identities_incomplete', missing=len(_expected_preparations - _ke_input_ids), extra=len(_ke_input_ids - _expected_preparations))
    raise RuntimeError('Raptor study/arm input preparation is incomplete')
""",
        "Raptor preparation gate",
    )
    cell52 = replace_once(
        cell52,
        "rsna_save_predictions('final',_RSNA_TEST_IDS,_release_df[_RSNA_LABELS].to_numpy(),_RSNA_LABELS)\n",
        """if tuple(globals().get('_RK_FAMILY_MEMBERS') or ()) != (
        'resgated_top3', 'global96_top3', 'd4_swa3', 'repairv1_top3'):
    raise RuntimeError('final CoAt member contract failed')
if RSNA_TUNE_PRESET != 'parent' and globals().get('_RK_OUTER_MODE') != 'probit_flat3':
    raise RuntimeError('final outer blend is not the intended flat3 formula')
rsna_save_predictions('final',_RSNA_TEST_IDS,_release_df[_RSNA_LABELS].to_numpy(),_RSNA_LABELS)
""",
        "final blend contract",
    )
    set_cell(rebuilt, 52, cell52)

    asset_preflight = """# Resolve all four SHA-pinned CoAt manifests before the expensive MRI stages.
_RSNA31_COAT_MANIFESTS = (
    ('coat_resgated_ep10_top3_manifest.json', '98511a8fdeb9da0e6e70c78d013ff636e1476f31c80b5dc134d294b18c3f284e'),
    ('coatnet_global96_top3_manifest.json', '015f09030e76f86274cadae40777b3c3cf5c19f8c835d5ad88e15db41903779e'),
    ('coatnet_pairfilm_manifest.json', '7ada0605bca6b0530569c6454e988ace479606a3328ed591d090e5764fea661d'),
    ('coatnet_repairv1_top3_manifest.json', '99288a9a42f36ba544d7b0ad2fe645e3b1fc9c507c23802c16ca0d2e5085f684'),
)
_RSNA31_COAT_PATHS = {}
for _name, _digest in _RSNA31_COAT_MANIFESTS:
    _path = _asset_find_asset(_name, _digest)
    if _path is None:
        raise RuntimeError('missing SHA-pinned CoAt manifest: ' + _name)
    _RSNA31_COAT_PATHS[_name] = str(_path)
print('[31] four CoAt manifests resolved before MRI inference:', _RSNA31_COAT_PATHS, flush=True)
"""
    rebuilt["cells"].insert(7, {
        "cell_type": "code",
        "execution_count": None,
        "id": "rsna31-asset-preflight",
        "metadata": {},
        "outputs": [],
        "source": asset_preflight.splitlines(keepends=True),
    })

    preface = f"""# 31 — Rebuild the supplied 0.958 pipeline with strict completion gates

Source: `Kaggle public/0.958.ipynb` (SHA256 `{hashlib.sha256(original_bytes).hexdigest()}`).
The source's saved output covers **three visible studies**. The file name and saved output do not establish a new leaderboard or Private score. This notebook keeps the source's `v6a` scoring formula and the same 19 Kaggle inputs.

Changes made here: Kaggle GPU metadata enables 2×T4; four SHA-pinned CoAt manifests are checked before expensive inference; all four CoAt readers and their probability artifacts must be present; `flat3` cannot quietly become a two-source blend; every Raptor arm/study input must be prepared before publishing; overrides receive basic checks. Saved outputs were cleared so the next Kaggle run has fresh receipts.

Prior research used for the decision: `29AB_Analysis_and_Breakthrough_Plan_2026-10-16.md` and experiment 30 found no convincing independent gain from R6/R7 mixing or dropping the fluid/state stackers. The 58 gold studies were reused during model selection, so they cannot justify new blend weights here. This build makes execution more reliable; **a higher AUC is unverified**.

Run with the existing 19 attachments, Internet Off, **T4×2**, Save & Run All. Check `btkd_v559_complete.json`, `rsna_v6_blend_report.json`, model receipts and final `submission.csv` before submitting. Do not infer hidden-set runtime or AUC from the three visible studies.
"""
    rebuilt["cells"].insert(0, {
        "cell_type": "markdown",
        "id": "rsna31-preface",
        "metadata": {},
        "source": preface.splitlines(keepends=True),
    })

    TARGET.write_text(json.dumps(rebuilt, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    manifest = {
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "output": TARGET.name,
        "output_sha256": hashlib.sha256(TARGET.read_bytes()).hexdigest(),
        "scoring_formula": "v6a unchanged",
        "new_gates": ["gpu_metadata", "early_manifest_preflight", "four_coat_readers", "four_coat_probability_artifacts", "flat3_sources", "raptor_preparation", "final_blend_contract"],
        "gpu_inference_executed_locally": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(TARGET)
    print(MANIFEST)


if __name__ == "__main__":
    main()
