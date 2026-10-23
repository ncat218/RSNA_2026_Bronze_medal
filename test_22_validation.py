"""Read-only fixture and negative-contract tests for rsna22_validation."""
from io import BytesIO
from pathlib import Path
import hashlib
import json
import unittest
import zipfile

import numpy as np
import pandas as pd

from rsna22_validation import (
    CHILD_CONTRACTS, LABELS, blend_frames, validate_anchor_bundle,
    validate_frame, validate_reader_receipt,
)


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archive = zipfile.ZipFile(Path(__file__).with_name("results_0.943.zip"))
        cls.addClassCleanup(cls.archive.close)
        cls.frame = pd.read_csv(BytesIO(cls.archive.read("submission.csv")),
                                dtype={"StudyInstanceUID": str})
        cls.ids = cls.frame.StudyInstanceUID.tolist()

    def reader(self, overrides=None):
        overrides = overrides or {}
        return lambda name: overrides[name] if name in overrides else self.archive.read(name)

    def json_override(self, name, mutate):
        obj = json.loads(self.archive.read(name))
        mutate(obj)
        return {name: json.dumps(obj).encode()}

    def test_real_protected_anchor_fixture(self):
        frame, receipt = validate_anchor_bundle(self.reader(), self.ids)
        pd.testing.assert_frame_equal(frame, self.frame)
        self.assertEqual(receipt["coat_family_members"], 4)

    def test_frame_alignment_and_original_preserved(self):
        shuffled = self.frame.iloc[::-1].copy()
        before = shuffled.copy(deep=True)
        pd.testing.assert_frame_equal(validate_frame(shuffled, self.ids), self.frame)
        pd.testing.assert_frame_equal(shuffled, before)

    def test_frame_rejects_bad_identity_schema_and_values(self):
        cases = []
        duplicate = self.frame.copy(); duplicate.loc[1, "StudyInstanceUID"] = self.ids[0]
        cases.append(duplicate)
        extra = self.frame.copy(); extra.loc[0, "StudyInstanceUID"] = "extra"
        cases.append(extra)
        cases += [self.frame.iloc[:-1], self.frame.drop(columns=LABELS[-1]),
                  self.frame[["StudyInstanceUID", *LABELS[::-1]]]]
        for value in [np.nan, np.inf, -.01, 1.01]:
            bad = self.frame.copy(); bad.loc[0, LABELS[0]] = value; cases.append(bad)
        for bad in cases:
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                validate_frame(bad, self.ids)
        with self.assertRaises(ValueError):
            validate_frame(self.frame, [self.ids[0]] * len(self.ids))

    def test_global_70_30_blend_ties_and_roundtrip(self):
        anchor = self.frame.copy(); reader = self.frame.copy()
        for label in LABELS:
            anchor[label] = [.1, .1, .9]
            reader[label] = [.9, .1, .1]
        before_anchor = anchor.copy(); before_reader = reader.copy()
        got = blend_frames(anchor, reader.iloc[::-1], self.ids)
        expected = (.7 * anchor[LABELS].rank(method="average", pct=True)
                    + .3 * reader[LABELS].rank(method="average", pct=True)).rank(method="average", pct=True)
        np.testing.assert_allclose(got[LABELS], expected, rtol=0, atol=0)
        roundtrip = pd.read_csv(BytesIO(got.to_csv(index=False).encode()), dtype={"StudyInstanceUID": str})
        validate_frame(roundtrip, self.ids)
        pd.testing.assert_frame_equal(anchor, before_anchor)
        pd.testing.assert_frame_equal(reader, before_reader)
        one = blend_frames(anchor.iloc[:1], reader.iloc[:1], self.ids[:1])
        np.testing.assert_array_equal(one[LABELS].to_numpy(), np.ones((1, 12)))
        for weight in [np.nan, np.inf, -.1, 1.1, True, "invalid", None]:
            with self.assertRaises(ValueError):
                blend_frames(anchor, reader, self.ids, weight)

    def test_numeric_strings_are_ranked_numerically(self):
        numeric = self.frame.copy()
        for label in LABELS:
            numeric[label] = [.09, .1, .9]
        strings = numeric.copy()
        strings[LABELS] = numeric[LABELS].astype(str)
        pd.testing.assert_frame_equal(validate_frame(strings, self.ids), numeric)
        pd.testing.assert_frame_equal(blend_frames(strings, numeric, self.ids),
                                      blend_frames(numeric, numeric, self.ids))

    def test_anchor_rejects_missing_children_and_hash_corruption(self):
        for name, _, _ in CHILD_CONTRACTS:
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_anchor_bundle(self.reader({name: b"{}"}), self.ids)
        with self.assertRaises(ValueError):
            validate_anchor_bundle(self.reader({"submission.csv": self.archive.read("submission.csv") + b"\n"}), self.ids)
        overrides = self.json_override("_coat_arm_receipt.json", lambda obj: obj.update(output_sha256="0" * 64))
        with self.assertRaises(ValueError):
            validate_anchor_bundle(self.reader(overrides), self.ids)
        overrides = self.json_override("_coat_raptor_blend_receipt.json", lambda obj:
                                      obj.update(events=[{"kind": "coat_family_partial"}]))
        with self.assertRaises(ValueError):
            validate_anchor_bundle(self.reader(overrides), self.ids)

    def test_anchor_rejects_changed_formula_and_partial_family(self):
        mutations = [
            ("_coat_raptor_blend_receipt.json", lambda x: x.update(members=x["members"][:-1])),
            ("_coat_raptor_blend_receipt.json", lambda x: x.update(family_reduction="rank_mean")),
            ("_coat_raptor_blend_receipt.json", lambda x: x["within_coat"].update(d4_swa3=.30)),
            ("btkd_v559_complete.json", lambda x: x.update(a5_folds=4)),
            ("btkd_v559_complete.json", lambda x: x["outer_weights"].update(ACL=.6)),
        ]
        for name, mutate in mutations:
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_anchor_bundle(self.reader(self.json_override(name, mutate)), self.ids)

    def test_anchor_rejects_real_fallback_but_accepts_benign_warning(self):
        for kind in ["raptor_neutral_fill", "a5_study_failed", "coat_d4_child_failed",
                     "coat_member_probability_unavailable", "rad_invalid_values_repaired"]:
            overrides = self.json_override("btkd_v559_complete.json",
                lambda obj, kind=kind: obj["events"].append({"kind": kind}))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_anchor_bundle(self.reader(overrides), self.ids)
        overrides = self.json_override("btkd_v559_complete.json", lambda obj:
            obj["events"].extend([{"kind": "scratch_fallback"},
                                  {"kind": "raptor_order_index_fallback"},
                                  {"kind": "a5_fp16_nonfinite_retry_fp32"}]))
        validate_anchor_bundle(self.reader(overrides), self.ids)
        overrides = self.json_override("_coat_g96_receipt.json", lambda obj:
                                      obj["shards"][0].update(fallback_studies=1))
        with self.assertRaises(ValueError):
            validate_anchor_bundle(self.reader(overrides), self.ids)

    def test_reader_receipt_contract(self):
        digest = hashlib.sha256(b"fixed reader checkpoint").hexdigest()
        good = {"status": "COMPLETE", "studies": len(self.ids), "labels": LABELS,
                "contract": "convnext_reader_v1", "models": 1,
                "checkpoints": [{"sha256": digest}], "fallback_studies": 0,
                "processes": [{"returncode": 0}]}
        kwargs = {"expected_contract": "convnext_reader_v1", "expected_checkpoint_hashes": [digest]}
        validate_reader_receipt(good, self.ids, **kwargs)
        for changes in [{"status": "PARTIAL"}, {"studies": 999}, {"fallback_studies": 1},
                        {"labels": LABELS[::-1]}, {"models": 2},
                        {"checkpoints": [{"sha256": "0" * 64}]},
                        {"processes": [{"returncode": 1}]}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_reader_receipt({**good, **changes}, self.ids, **kwargs)


if __name__ == "__main__":
    unittest.main()
