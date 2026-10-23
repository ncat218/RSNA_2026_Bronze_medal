"""CPU-only contracts for the protected anchor and notebook 22 rank probe.

This module never opens checkpoints or performs model inference. Bundle readers
receive exact relative paths, so the same checks work on a ZIP or working folder.
"""
from __future__ import annotations

from io import BytesIO
import hashlib
import json
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd


LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
COLUMNS = ["StudyInstanceUID", *LABELS]
COAT_MEMBERS = ("resgated_top3", "global96_top3", "d4_swa3", "repairv1_top3")
CHILD_CONTRACTS = (
    ("_coat_arm_receipt.json", "VALID_COAT_RESGATED_EP10_TOP3_RANK_SUBMISSION", 3),
    ("_coat_g96_receipt.json", "VALID_COATNET_GLOBAL96_TOP3_T4X2_SUBMISSION", 3),
    ("d4_input/d4.receipt.json", "VALID_COATNET_D4_DEPTHZONE_SWA3_T4X2_SUBMISSION", 1),
    ("_coat_repairv1_receipt.json", "VALID_COATNET_REPAIRV1_TOP3_PROBABILITY_RANK_SUBMISSION", 3),
)


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _ids(values: Iterable[str]) -> list[str]:
    raw = list(values)
    _need(bool(raw), "Expected study IDs are empty")
    _need(all(v is not None and not pd.isna(v) and str(v).strip() for v in raw),
          "Expected study IDs contain null/empty IDs")
    result = [str(v) for v in raw]
    _need(len(result) == len(set(result)), "Expected study IDs are duplicated")
    return result


def validate_frame(frame: pd.DataFrame, expected_uid_list: Iterable[str],
                   name: str = "predictions") -> pd.DataFrame:
    """Validate exact columns/coverage and return a copy in expected UID order."""
    ids = _ids(expected_uid_list)
    _need(isinstance(frame, pd.DataFrame), f"{name}: expected a DataFrame")
    _need(list(frame.columns) == COLUMNS, f"{name}: column schema/order mismatch")
    _need(not frame.StudyInstanceUID.isna().any(), f"{name}: null study ID")
    actual = frame.StudyInstanceUID.astype(str)
    _need(actual.str.strip().ne("").all(), f"{name}: empty study ID")
    _need(actual.is_unique, f"{name}: duplicate study ID")
    _need(len(frame) == len(ids) and set(actual) == set(ids),
          f"{name}: study coverage mismatch")
    try:
        values = frame[LABELS].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: nonnumeric predictions") from exc
    _need(np.isfinite(values).all(), f"{name}: nonfinite prediction")
    _need(((values >= 0) & (values <= 1)).all(), f"{name}: prediction outside [0,1]")
    result = frame.copy(deep=True)
    result["StudyInstanceUID"] = actual.to_numpy()
    # Numeric strings are permissible CSV values, but must not subsequently
    # receive lexicographic ranks in blend_frames.
    result[LABELS] = values
    return result.set_index("StudyInstanceUID").loc[ids].reset_index()


def blend_frames(anchor: pd.DataFrame, reader: pd.DataFrame,
                 ids: Iterable[str], weight: float = .30) -> pd.DataFrame:
    """Global per-target rank(70% rank(anchor) + 30% rank(reader))."""
    expected = _ids(ids)
    try:
        valid_weight = not isinstance(weight, bool) and np.isfinite(weight) and 0 <= weight <= 1
    except (TypeError, ValueError):
        valid_weight = False
    _need(valid_weight, "Reader weight must be finite and in [0,1]")
    left = validate_frame(anchor, expected, "anchor")
    right = validate_frame(reader, expected, "reader")
    pooled = ((1.0 - float(weight)) * left[LABELS].rank(method="average", pct=True)
              + float(weight) * right[LABELS].rank(method="average", pct=True))
    output = left[["StudyInstanceUID"]].copy()
    output[LABELS] = pooled.rank(method="average", pct=True)
    return validate_frame(output, expected, "rank blend")


def _nonzero(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    try:
        return not np.isfinite(float(value)) or float(value) != 0
    except (TypeError, ValueError):
        return bool(value)


def _validate_failures(receipt: Mapping, name: str) -> None:
    # Missing acquisition metadata and successful numerical retries are logged
    # warnings, not proof of a model failure. These kinds explicitly alter output.
    bad_kinds = {
        "raptor_study_failed", "raptor_neutral_fill", "a5_study_failed",
        "a5_nonfinite_neutral", "coat_family_partial", "coat_family_unavailable",
        "coat_member_probability_unavailable", "rad_invalid_values_repaired",
    }
    for key in ("fallback_studies", "neutral_studies", "model_failures", "failures"):
        _need(not _nonzero(receipt.get(key)), f"{name}: {key} is nonzero")
    events = receipt.get("events", [])
    _need(isinstance(events, list), f"{name}: invalid events")
    for event in events:
        _need(isinstance(event, dict), f"{name}: invalid event entry")
        kind = str(event.get("kind", ""))
        bad = (kind in bad_kinds or kind.endswith("_child_failed")
               or kind.endswith("_output_rejected")
               or (kind.endswith("_fallback_studies") and _nonzero(event.get("count", 1))))
        _need(not bad, f"{name}: model failure/fallback event {kind}")
    for process in receipt.get("processes", []):
        _need(isinstance(process, dict) and process.get("returncode") == 0,
              f"{name}: worker process failed or omitted return code")
    for shard in receipt.get("shards", []):
        _need(isinstance(shard, dict), f"{name}: invalid shard")
        for key in ("fallback_studies", "neutral_studies", "model_failures", "failures"):
            _need(not _nonzero(shard.get(key)), f"{name}: shard {key} is nonzero")


def _read_json(read_bytes: Callable[[str], bytes], path: str) -> dict:
    try:
        value = json.loads(read_bytes(path))
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"Required receipt unavailable/invalid: {path}") from exc
    _need(isinstance(value, dict), f"{path}: receipt is not an object")
    return value


def _working_relative(path: str) -> str:
    prefix = "/kaggle/working/"
    _need(isinstance(path, str) and path.startswith(prefix), "Receipt output is outside working folder")
    relative = path[len(prefix):]
    _need(relative and not relative.startswith("/") and ".." not in relative.split("/"),
          "Receipt output path is invalid")
    return relative


def validate_anchor_bundle(read_bytes: Callable[[str], bytes],
                           expectedids: Iterable[str]) -> tuple[pd.DataFrame, dict]:
    """Require the protected 20/5/4/4 graph and unchanged output receipts.

    ``read_bytes('submission.csv')`` reads the still-uncaptured anchor output.
    Paths for the four child receipts are defined in CHILD_CONTRACTS above.
    """
    ids = _ids(expectedids)
    audit = _read_json(read_bytes, "btkd_v559_complete.json")
    family = _read_json(read_bytes, "_coat_raptor_blend_receipt.json")
    _need(audit.get("status") == "COMPLETE", "Anchor did not complete")
    for key, expected in (("dino_members", 20), ("a5_folds", 5),
                          ("raptor_views", 4), ("coat_family_members", 4), ("cohort", len(ids))):
        _need(audit.get(key) == expected, f"Anchor {key} mismatch")
    members = family.get("members", [])
    _need(isinstance(members, list) and len(members) == 4 and set(members) == set(COAT_MEMBERS),
          "Anchor CoAt member identities mismatch")
    weights = family.get("within_coat", {})
    _need(isinstance(weights, dict) and set(weights) == set(COAT_MEMBERS)
          and all(weights[name] == .25 for name in COAT_MEMBERS), "Anchor CoAt weights changed")
    _need(family.get("study_count") == len(ids), "CoAt family cohort mismatch")
    for receipt in (audit, family):
        _need(receipt.get("family_reduction") == "rank_of_member_probability_mean",
              "Anchor CoAt probability reduction changed")
        _need(receipt.get("inner_rerank") is False, "Anchor inner rerank changed")
    _need(audit.get("family_mix") == [.25] * 4 and audit.get("inner_public_coat_mix") == [.6, .4],
          "Anchor family mix changed")
    _need(family.get("private_alpha") == .4 and family.get("public_raptor_alpha") == .6
          and family.get("finding_specific_weights") is False, "Anchor inner family mix changed")
    expected_outer = {label: .6 for label in LABELS}
    expected_outer.update({"ACL": .75, "Medial Meniscus": .8, "Lateral Meniscus": 1.,
                           "Lateral OA": .75, "Fracture": .75})
    _need(audit.get("outer_weights") == expected_outer, "Anchor outer target weights changed")
    _validate_failures(audit, "anchor")
    _validate_failures(family, "CoAt family")
    for path, status, n_models in CHILD_CONTRACTS:
        child = _read_json(read_bytes, path)
        _need(child.get("status") == status and child.get("studies") == len(ids)
              and child.get("models") == n_models, f"{path}: child identity/count mismatch")
        _need("fallback_studies" in child, f"{path}: omitted fallback counter")
        _validate_failures(child, path)
        child_output = _working_relative(child.get("output"))
        try:
            payload = read_bytes(child_output)
        except (OSError, KeyError) as exc:
            raise ValueError(f"{path}: child output missing") from exc
        _need(hashlib.sha256(payload).hexdigest() == child.get("output_sha256"),
              f"{path}: child output SHA256 mismatch")
        validate_frame(pd.read_csv(BytesIO(payload), dtype={"StudyInstanceUID": str}), ids, path)
    try:
        payload = read_bytes("submission.csv")
    except (OSError, KeyError) as exc:
        raise ValueError("Anchor submission is missing") from exc
    _need(hashlib.sha256(payload).hexdigest() == audit.get("submission_sha256"),
          "Anchor submission SHA256 mismatch")
    frame = pd.read_csv(BytesIO(payload), dtype={"StudyInstanceUID": str})
    return validate_frame(frame, ids, "anchor submission"), audit


def validate_reader_receipt(receipt: Mapping, expected_uid_list: Iterable[str], *,
                            expected_contract: str | None = None,
                            expected_checkpoint_hashes: Iterable[str] | None = None) -> dict:
    """Validate a notebook-22 wrapper receipt, independent of reader architecture.

    Required: status COMPLETE and studies (or test_studies). Optional expected
    contract/hash arguments let callers pin their reader-specific assets. Frame
    validation and output SHA verification remain the caller's responsibility.
    """
    ids = _ids(expected_uid_list)
    _need(isinstance(receipt, Mapping), "Reader receipt is not an object")
    _need(receipt.get("status") == "COMPLETE", "Reader did not complete")
    count_keys = [key for key in ("studies", "test_studies") if key in receipt]
    _need(count_keys and all(receipt[key] == len(ids) for key in count_keys), "Reader cohort mismatch")
    for key in ("labels", "targets"):
        if key in receipt:
            _need(receipt[key] == LABELS, "Reader target order mismatch")
    if expected_contract is not None:
        _need(receipt.get("contract", receipt.get("runtime_contract")) == expected_contract,
              "Reader runtime contract mismatch")
    if expected_checkpoint_hashes is not None:
        expected = list(expected_checkpoint_hashes)
        _need(expected and len(expected) == len(set(expected)), "Expected reader checkpoint hashes invalid")
        checkpoints = receipt.get("checkpoints", [])
        _need(isinstance(checkpoints, list) and all(isinstance(ck, dict) for ck in checkpoints),
              "Reader checkpoint identities missing")
        actual = [ck.get("sha256") for ck in checkpoints]
        _need(len(actual) == len(expected) and set(actual) == set(expected),
              "Reader checkpoint SHA256 identities mismatch")
        if "models" in receipt:
            _need(receipt["models"] == len(expected), "Reader model count mismatch")
    _validate_failures(receipt, "reader")
    return dict(receipt)
