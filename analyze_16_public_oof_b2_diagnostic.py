"""Read-only diagnostic of owned B2 against available public-family OOF.

This is not OOF of the 0.943 ensemble: A5, Raptor, and CoAt family predictions
are absent. L1 report-derived labels are only a weak proxy, and the provenance
of public OOF checkpoints has not been independently established. No weight
selection or submission should follow directly from this script.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
PUBLIC = {
    "E9": "datasets/antoinegg1/rsna-knee-e9-radimagenet-heads-v15/v52_oof.csv",
    "E11_dataset": "datasets/antoinegg1/rsna-knee-e11-diverse-heads-v20/v52_e11_oof.csv",
    "E11_notebook": "notebooks/sofiaanjenje/rsna-knee-e11-train/rsna_rad_e11/v52_e11_oof.csv",
    "E13_notebook": "notebooks/sofiaanjenje/rsna-knee-e13-train/rsna_rad_e11/v52_e11_oof.csv",
}


def load_csv(archive_name: str, suffix: str) -> pd.DataFrame:
    with zipfile.ZipFile(ROOT / archive_name) as archive:
        matches = [name for name in archive.namelist()
                   if name == suffix or name.endswith("/" + suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one {suffix} in {archive_name}: {matches}")
        return pd.read_csv(io.BytesIO(archive.read(matches[0])),
                           dtype={"StudyInstanceUID": str})


def checked(frame: pd.DataFrame, n: int, name: str) -> pd.DataFrame:
    if (len(frame) != n or "StudyInstanceUID" not in frame or
            not frame.StudyInstanceUID.is_unique or
            not set(LABELS).issubset(frame.columns)):
        raise RuntimeError(f"Invalid row count, UID uniqueness, or labels: {name}")
    values = frame[LABELS].to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise RuntimeError(f"Invalid values: {name}")
    return frame.set_index("StudyInstanceUID").sort_index()


def auc(binary: pd.Series, score: pd.Series) -> float:
    positives = np.asarray(binary, dtype=bool)
    ranking = pd.Series(np.asarray(score, dtype=float)).rank(method="average")
    n_pos = int(positives.sum())
    n_neg = len(positives) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranking[positives].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def macro(truth: pd.DataFrame, predictions: pd.DataFrame) -> float:
    return float(np.mean([auc(truth[label] >= 0.5, predictions[label])
                          for label in LABELS]))


def ranked(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[LABELS].rank(method="average", pct=True)


def main() -> None:
    silver = checked(load_csv("RSNA_Knee_Silver_Label_v1_Artifacts.zip",
                              "L1_family_mean_RECOMMENDED.csv"), 4349, "L1")
    b2 = checked(load_csv("results_09.zip", "weak_oof.csv"), 4349, "B2")
    if not silver.index.equals(b2.index):
        raise RuntimeError("Silver and B2 IDs differ")
    if b2.fold.value_counts().sort_index().tolist() != [870, 870, 870, 870, 869]:
        raise RuntimeError("B2 fold sizes differ")

    public: dict[str, pd.DataFrame] = {}
    for name, suffix in PUBLIC.items():
        frame = checked(load_csv("results_07.zip", suffix), 4407, name)
        if not silver.index.isin(frame.index).all():
            raise RuntimeError(f"Missing weak IDs in {name}")
        public[name] = frame.loc[silver.index]

    with zipfile.ZipFile(ROOT / "results_07.zip") as archive:
        names = [name for name in archive.namelist() if name.endswith("oof.npz")]
        if len(names) != 1:
            raise RuntimeError(f"Expected one public DINO NPZ, got {names}")
        with np.load(io.BytesIO(archive.read(names[0]))) as data:
            if list(data["targets"]) != LABELS or len(data["ids"]) != 4407:
                raise RuntimeError("DINO OOF target order or ID count changed")
            dino = pd.DataFrame(data["pred"], columns=LABELS)
            dino.insert(0, "StudyInstanceUID", data["ids"].astype(str))
    public["DINO"] = checked(dino, 4407, "DINO").loc[silver.index]

    # A deliberately fixed diagnostic, not an optimized ensemble recipe.
    proxy = 0.5 * ranked(public["DINO"]) + 0.5 * ranked(public["E13_notebook"])
    proxy_plus_b2 = 0.8 * proxy + 0.2 * ranked(b2)

    print("COHORT: 4349 exact-matched weak UIDs; B2 five report-group folds")
    print("PROVENANCE: public OOF claimed by source, not independently verified")
    print("LABELS: L1 report-derived soft targets thresholded at 0.5; weak proxy only")
    for name, frame in [("B2", b2), *public.items(),
                        ("DINO_E13_proxy", proxy),
                        ("DINO_E13_plus_B2_fixed20", proxy_plus_b2)]:
        print(f"MACRO {name:26s} {macro(silver, frame):.6f}")
    print("FOLD DELTAS proxy_plus_B2 minus proxy:")
    for fold in range(5):
        mask = b2.fold.eq(fold)
        delta = macro(silver.loc[mask], proxy_plus_b2.loc[mask]) - macro(
            silver.loc[mask], proxy.loc[mask])
        print(f"  {fold}: {delta:+.6f}, n={int(mask.sum())}")
    print("PER TARGET: B2 vs partial public-family proxy and fixed 20% mix")
    for label in LABELS:
        truth = silver[label] >= 0.5
        base_auc = auc(truth, proxy[label])
        mix_auc = auc(truth, proxy_plus_b2[label])
        b2_auc = auc(truth, b2[label])
        rho = b2[label].rank().corr(proxy[label].rank())
        print(f"  {label:18s} proxy={base_auc:.4f} B2={b2_auc:.4f} "
              f"fixed20_delta={mix_auc-base_auc:+.4f} rank_rho={rho:.3f}")
    print("LIMIT: this partial proxy omits A5, Raptor, and all CoAt branches;")
    print("       no 0.943 OOF or Private-LB estimate can be inferred")


if __name__ == "__main__":
    main()
