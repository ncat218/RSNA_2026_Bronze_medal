"""CPU-only audit of a saved Speedy Raptors Kaggle notebook output.

Usage: python audit_raptor_0943.py OUTPUT_ZIP_OR_DIRECTORY

This reads predictions and receipts; it never changes a submission or tunes
weights against the public leaderboard. Test predictions have no ground truth,
so the resulting correlations measure diversity, not model accuracy.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
EXPECTED_COATS = {"resgated_top3", "global96_top3", "d4_swa3", "repairv1_top3"}


class OutputFiles:
    def __init__(self, root: Path):
        self.root = root
        self.archive = zipfile.ZipFile(root) if root.is_file() else None
        self.names = (
            [x.filename for x in self.archive.infolist() if not x.is_dir()]
            if self.archive else
            [str(x.relative_to(root)).replace("\\", "/") for x in root.rglob("*") if x.is_file()]
        )

    def close(self):
        if self.archive:
            self.archive.close()

    def find(self, suffix: str) -> str | None:
        hits = [n for n in self.names if n == suffix or n.endswith("/" + suffix)]
        if not hits:
            return None
        if len(hits) > 1:
            raise ValueError(f"Ambiguous {suffix}: {hits}")
        return hits[0]

    def read(self, name: str) -> bytes:
        if self.archive:
            return self.archive.read(name)
        return (self.root / name).read_bytes()

    def json(self, suffix: str) -> dict | None:
        name = self.find(suffix)
        return json.loads(self.read(name)) if name else None


def prediction_frame(files: OutputFiles, name: str) -> pd.DataFrame:
    with np.load(io.BytesIO(files.read(name)), allow_pickle=False) as data:
        ids = data["study_uids"].astype(str)
        labels = data["labels"].astype(str).tolist()
        values = np.asarray(data["values"], dtype=np.float64)
    if values.ndim == 3:
        values = values.mean(axis=0)
    if values.shape != (len(ids), len(labels)):
        raise ValueError(f"Unexpected prediction shape: {name}: {values.shape}")
    if labels != LABELS or len(set(ids)) != len(ids) or not np.isfinite(values).all():
        raise ValueError(f"Invalid labels, duplicate UIDs, or nonfinite values: {name}")
    return pd.DataFrame(values, index=ids, columns=labels)


def csv_frame(files: OutputFiles, suffix: str) -> pd.DataFrame | None:
    name = files.find(suffix)
    if not name:
        return None
    frame = pd.read_csv(io.BytesIO(files.read(name)), dtype={"StudyInstanceUID": str})
    if frame.columns.tolist() != ["StudyInstanceUID", *LABELS]:
        raise ValueError(f"Wrong CSV schema: {name}")
    if frame.StudyInstanceUID.duplicated().any():
        raise ValueError(f"Duplicate UIDs: {name}")
    return frame.set_index("StudyInstanceUID")[LABELS].astype(float)


def ranked(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(method="average", pct=True)


def per_target_correlation(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    return pd.Series({label: ranked(a)[label].corr(ranked(b)[label]) for label in LABELS})


def run(root: Path) -> int:
    files = OutputFiles(root)
    try:
        complete = files.json("btkd_v559_complete.json")
        blend = files.json("_coat_raptor_blend_receipt.json")
        submission = csv_frame(files, "submission.csv")
        if not complete or submission is None:
            raise FileNotFoundError("Need the saved notebook output with btkd_v559_complete.json and submission.csv")

        print(f"Submission rows: {len(submission)}; receipt status: {complete.get('status')}")
        print(f"Receipt cohort: {complete.get('cohort')}; DINO members: {complete.get('dino_members')}; "
              f"A5 folds: {complete.get('a5_folds')}; elapsed seconds: {complete.get('elapsed_seconds')}")
        if int(complete.get("cohort", -1)) != len(submission):
            print("ALERT: receipt cohort differs from submission rows")
        members = set((blend or {}).get("members", []))
        print(f"CoAt members: {sorted(members)}; reduction: {(blend or {}).get('family_reduction')}")
        missing = EXPECTED_COATS - members
        if missing:
            print(f"ALERT: missing CoAt members: {sorted(missing)}")
        if (blend or {}).get("family_reduction") != "rank_of_member_probability_mean":
            print("ALERT: CoAt reduction used a different path from the advertised probability mean")

        for suffix in (
            "_coat_arm_receipt.json", "_coat_g96_receipt.json",
            "_coat_repairv1_receipt.json", "d4_input/d4.receipt.json",
        ):
            receipt = files.json(suffix)
            if receipt is None:
                print(f"ALERT: missing branch receipt {suffix}")
                continue
            print(f"{suffix}: studies={receipt.get('studies')}, models={receipt.get('models')}, "
                  f"fallback_studies={receipt.get('fallback_studies')}, "
                  f"selection={receipt.get('checkpoint_selection')}")
            if receipt.get("fallback_studies") not in (0, None):
                print(f"ALERT: branch fallback in {suffix}")
            if "Gold58" in str(receipt.get("checkpoint_selection", "")):
                print("LIMIT: Gold58 selected checkpoints here; its reported AUC is not independent validation")

        events = complete.get("events", [])
        suspicious = [e for e in events if any(x in str(e.get("kind", "")) for x in
                      ("failed", "fallback", "unavailable", "rejected", "neutral", "partial"))]
        print(f"Fallback/failure/neutral events: {len(suspicious)}")
        for event in suspicious[:30]:
            print("  " + json.dumps(event, ensure_ascii=False)[:500])

        phase_name = files.find("diagnostics/phase_events.jsonl")
        if phase_name:
            phases = [json.loads(line) for line in files.read(phase_name).decode().splitlines() if line.strip()]
            for phase in phases:
                if phase.get("status") == "FAILED" or phase.get("ok") is False:
                    print("ALERT phase: " + json.dumps(phase, ensure_ascii=False)[:500])

        frames: dict[str, pd.DataFrame] = {"submission": submission}
        for suffix, key in [
            ("_raptor_public.csv", "raptor_public"),
            ("_coat_family_rank.csv", "coat_family"),
        ]:
            frame = csv_frame(files, suffix)
            if frame is not None:
                frames[key] = frame
        for name in files.names:
            if "/diagnostics/" not in "/" + name or not name.endswith(".npz"):
                continue
            key = Path(name).stem
            try:
                frames[key] = prediction_frame(files, name)
            except (KeyError, ValueError) as error:
                print(f"SKIP {key}: {error}")

        aligned: dict[str, pd.DataFrame] = {}
        for key, frame in frames.items():
            if len(frame) != len(submission) or set(frame.index) != set(submission.index):
                print(f"SKIP {key}: UID coverage differs from submission")
                continue
            aligned[key] = frame.reindex(submission.index)
        print(f"Aligned prediction sources: {len(aligned)}")
        if len(submission) < 50:
            print("LIMIT: visible sample is too small for branch similarity; hidden scoring output is not exposed")
            print("No correlation file written. Use an independent OOF/holdout cohort for model decisions.")
            return 0

        rows = []
        for key, frame in aligned.items():
            if key == "submission":
                continue
            corr = per_target_correlation(frame, submission)
            for label in LABELS:
                values = frame[label].to_numpy()
                rows.append({"source": key, "target": label, "spearman_to_final": corr[label],
                             "prediction_std": float(np.std(values)),
                             "unique_fraction": float(len(np.unique(values)) / len(values))})
        output = pd.DataFrame(rows)
        if not output.empty:
            output.to_csv("raptor_0943_branch_diagnostics.csv", index=False)
            summary = output.groupby("source", as_index=False).agg(
                mean_spearman_to_final=("spearman_to_final", "mean"),
                min_spearman_to_final=("spearman_to_final", "min"),
                mean_unique_fraction=("unique_fraction", "mean"),
            ).sort_values("mean_spearman_to_final")
            print("\nBranch similarity to final (lowest first; low similarity does NOT imply better AUC):")
            print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            print("\nWrote raptor_0943_branch_diagnostics.csv")
        print("Test predictions contain no labels: this audit cannot calculate AUC or choose blend weights.")
        return 0
    finally:
        files.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python audit_raptor_0943.py OUTPUT_ZIP_OR_DIRECTORY")
    raise SystemExit(run(Path(sys.argv[1])))
