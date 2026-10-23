"""CPU-only regression tests for the reader's released infer.py skip behavior."""

import ast
from collections import Counter
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


source = Path(__file__).with_name("rsna22_reader.py").read_text(encoding="utf-8")
function = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "load_windows")
scope = {"np": np}
exec(compile(ast.Module(body=[function], type_ignores=[]), "rsna22_reader.py", "exec"), scope)
load_windows = scope["load_windows"]


class SkipTests(unittest.TestCase):
    def setUp(self):
        self.volume = np.ones((4, 8, 8), np.uint8)
        self.knee = types.ModuleType("knee")
        self.knee.window_centres = lambda n, k, train: np.ones(k, dtype=int)
        self.preprocess = types.ModuleType("preprocess")
        self.preprocess.SIZE = 8
        self.table = {"study": [["bad", "alternate"], ["good"], [], [], [], []]}

    def test_bad_slice_is_skipped_but_usable_volume_is_kept(self):
        self.preprocess.load_series = lambda path: (self.volume, {"n_bad": 1})
        with patch.dict(sys.modules, {"knee": self.knee, "preprocess": self.preprocess}):
            _, mask, _ = load_windows("study", self.table, Path("fixture"), Counter(), [])
        np.testing.assert_array_equal(mask, [True, True, False, False, False, False])

    def test_unreadable_slot_does_not_try_second_candidate(self):
        calls = []
        def decode(path):
            calls.append(Path(path).name)
            if path.endswith("bad"):
                raise OSError("corrupt DICOM")
            return self.volume, {"n_bad": 0}
        self.preprocess.load_series = decode
        stats, rows = Counter(), []
        with patch.dict(sys.modules, {"knee": self.knee, "preprocess": self.preprocess}):
            _, mask, _ = load_windows("study", self.table, Path("fixture"), stats, rows)
        self.assertEqual(calls, ["bad", "good"])
        np.testing.assert_array_equal(mask, [False, True, False, False, False, False])
        self.assertEqual(stats["unreadable_slots_skipped"], 1)
        self.assertEqual(rows[0]["available_slots"], 1)

    def test_all_empty_still_fails_closed(self):
        def broken(path):
            raise OSError("corrupt DICOM")
        self.preprocess.load_series = broken
        with patch.dict(sys.modules, {"knee": self.knee, "preprocess": self.preprocess}):
            with self.assertRaisesRegex(RuntimeError, "No usable reader series"):
                load_windows("study", self.table, Path("fixture"), Counter(), [])


if __name__ == "__main__":
    unittest.main()
