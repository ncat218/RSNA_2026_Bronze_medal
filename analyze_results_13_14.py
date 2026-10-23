"""Read-only, ID-paired audit of B2 notebook 09 versus plane-aware notebook 14.

Uses the original L1 weak labels only as a diagnostic proxy. Official Gold58
is audit-only and cannot be used to pick blend weights or report Private AUC.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion",
    "Fracture",
]


def zipped_csv(filename: str, suffix: str) -> pd.DataFrame:
    with zipfile.ZipFile(ROOT / filename) as archive:
        matches = [n for n in archive.namelist()
                   if n == suffix or n.endswith("/" + suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {suffix} in {filename}: {matches}")
        return pd.read_csv(io.BytesIO(archive.read(matches[0])),
                           dtype={"StudyInstanceUID": str})


def zipped_json(filename: str, suffix: str) -> dict:
    with zipfile.ZipFile(ROOT / filename) as archive:
        matches = [n for n in archive.namelist()
                   if n == suffix or n.endswith("/" + suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {suffix} in {filename}: {matches}")
        return json.loads(archive.read(matches[0]))


def auc(binary: pd.Series, score: pd.Series) -> float:
    positives = np.asarray(binary, dtype=bool)
    ranking = pd.Series(np.asarray(score, dtype=float)).rank(method="average")
    n_pos = int(positives.sum())
    n_neg = len(positives) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranking[positives].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def checked_index(frame: pd.DataFrame, expected_n: int) -> pd.DataFrame:
    if len(frame) != expected_n or not frame.StudyInstanceUID.is_unique:
        raise RuntimeError(f"Unexpected row count/duplicate IDs: {len(frame)}")
    if not set(LABELS).issubset(frame.columns):
        raise RuntimeError("Missing target columns")
    if not np.isfinite(frame[LABELS].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite target/prediction values")
    return frame.set_index("StudyInstanceUID").sort_index()


def main() -> None:
    integrity = zipped_json("results_13.zip", "baseline_integrity.json")
    b2_report = zipped_json("results_09.zip", "run_report.json")
    p14_report = zipped_json("results_14.zip", "run_report.json")
    if integrity.get("status") != "VISIBLE_OUTPUT_PASS":
        raise RuntimeError("Notebook 13 baseline visible-output audit did not pass")
    for name, report in (("B2", b2_report), ("P14", p14_report)):
        if (report.get("status") != "COMPLETE" or
                report.get("folds_completed") != [0, 1, 2, 3, 4] or
                report.get("n_weak") != 4349 or
                report.get("n_gold") != 58 or
                report.get("empty_studies") != 0):
            raise RuntimeError(f"{name} incomplete: {report.get('status')}")
    if b2_report["encoder_weight_sha256"] != p14_report["encoder_weight_sha256"]:
        raise RuntimeError("Encoder checkpoint changed between B2 and P14")
    if b2_report["n_images_encoded"] != p14_report["n_images_encoded"]:
        raise RuntimeError("Different image count encoded")

    silver = checked_index(zipped_csv("RSNA_Knee_Silver_Label_v1_Artifacts.zip",
                                      "L1_family_mean_RECOMMENDED.csv"), 4349)
    b2 = checked_index(zipped_csv("results_09.zip", "weak_oof.csv"), 4349)
    p14 = checked_index(zipped_csv("results_14.zip", "weak_oof.csv"), 4349)
    if (not silver.index.equals(b2.index) or not b2.index.equals(p14.index)
            or not b2.fold.equals(p14.fold)):
        raise RuntimeError("OOF study IDs or folds differ; comparison invalid")
    if b2.fold.value_counts().sort_index().tolist() != [870, 870, 870, 870, 869]:
        raise RuntimeError("Fold sizes changed")
    weak_rows = []
    for label in LABELS:
        truth = silver[label] >= 0.5
        left, right = auc(truth, b2[label]), auc(truth, p14[label])
        rank_left = b2[label].rank(method="average")
        rank_right = p14[label].rank(method="average")
        weak_rows.append((label, left, right, right-left,
                          rank_left.corr(rank_right)))

    gold_b2 = checked_index(zipped_csv("results_09.zip", "gold_predictions.csv"), 58)
    gold_p14 = checked_index(zipped_csv("results_14.zip", "gold_predictions.csv"), 58)
    gold_truth = checked_index(zipped_csv("results_07.zip", "train.csv")
                               .dropna(subset=LABELS), 58)
    if not (gold_b2.index.equals(gold_p14.index) and
            gold_b2.index.equals(gold_truth.index)):
        raise RuntimeError("Gold58 IDs differ")
    gold_rows = []
    for label in LABELS:
        left = auc(gold_truth[label].astype(bool), gold_b2[label])
        right = auc(gold_truth[label].astype(bool), gold_p14[label])
        gold_rows.append((label, left, right, right-left))

    print("13:", integrity["status"], "visible studies:", integrity["visible_studies"])
    print("Matched OOF:", len(b2), "same 5 folds / encoder SHA / image count")
    print("Runtime min: B2", round(b2_report["total_elapsed_minutes"], 2),
          "P14", round(p14_report["total_elapsed_minutes"], 2))
    print("Weak OOF AUC proxy (thresholded L1), paired by study/fold:")
    for label, left, right, delta, rho in weak_rows:
        print(f"  {label:18s} B2={left:.4f} P14={right:.4f} "
              f"delta={delta:+.4f} rho={rho:.3f}")
    print("Weak macro:", round(float(np.mean([x[1] for x in weak_rows])), 6),
          round(float(np.mean([x[2] for x in weak_rows])), 6),
          "delta", round(float(np.mean([x[3] for x in weak_rows])), 6))
    print("Gold58 audit-only AUC, never a blend-selection metric:")
    for label, left, right, delta in gold_rows:
        print(f"  {label:18s} B2={left:.4f} P14={right:.4f} delta={delta:+.4f}")
    print("Gold macro:", round(float(np.mean([x[1] for x in gold_rows])), 6),
          round(float(np.mean([x[2] for x in gold_rows])), 6),
          "delta", round(float(np.mean([x[3] for x in gold_rows])), 6))
    print("Fold val BCE B2/P14:", [(a["fold"], round(a["val_bce"], 5),
                                      round(b["val_bce"], 5))
                                     for a, b in zip(b2_report["cv_metrics"],
                                                     p14_report["cv_metrics"])])


if __name__ == "__main__":
    main()
