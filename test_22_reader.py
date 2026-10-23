"""CPU batching/contract checks using the reviewed public KneeNet code.

The only substituted component is timm's image encoder: no public checkpoint,
network access, DICOM decoding, CUDA, or original-source edits are needed.
"""
from collections import Counter
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rsna22_reader import check_checkpoint, load_windows, microbatch_probability


class MeanEncoder(nn.Module):
    """Batch-independent 768-feature encoder with nonconstant channel weights."""

    num_features = 768

    def __init__(self):
        super().__init__()
        indices = torch.arange(self.num_features)
        self.register_buffer("channels", indices.remainder(3))
        self.register_buffer("scale", 0.5 + indices.remainder(17).float() / 17)
        self.register_buffer("bias", torch.sin(indices.float()) * 0.05)
        self.batch_sizes = []

    def forward(self, images):
        self.batch_sizes.append(len(images))
        means = images.mean(dim=(-1, -2))
        return means[:, self.channels] * self.scale + self.bias


class ReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.addClassCleanup(torch.set_num_threads, cls.previous_threads)
        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = lambda *args, **kwargs: MeanEncoder()
        spec = importlib.util.spec_from_file_location(
            "knee", Path(__file__).parent / "research_public_convnext" / "knee.py")
        cls.knee = importlib.util.module_from_spec(spec)
        cls.modules = patch.dict(sys.modules, {"timm": fake_timm, "knee": cls.knee})
        cls.modules.start()
        cls.addClassCleanup(cls.modules.stop)
        spec.loader.exec_module(cls.knee)
        torch.manual_seed(22)
        cls.model = cls.knee.KneeNet(pretrained=False).eval().requires_grad_(False)
        # Native public shape: six slots, twelve adjacent-slice triplets, 384 px.
        cls.windows = np.random.default_rng(22).integers(
            0, 256, size=(6, 12, 3, 384, 384), dtype=np.uint8)
        cls.pos = np.tile(np.linspace(.04, .96, 12, dtype=np.float32), (6, 1))

    def test_microbatch_matches_public_full_forward_partial_and_all_slots(self):
        for mask in [np.array([True, False, True, False, False, True]),
                     np.ones(6, bool)]:
            with torch.inference_mode():
                reference = self.model(
                    torch.from_numpy(self.windows).unsqueeze(0),
                    torch.from_numpy(mask).unsqueeze(0),
                    torch.from_numpy(self.pos).unsqueeze(0),
                    res=256).sigmoid()[0].numpy()
            for microbatch in (1, 8, 12):
                with self.subTest(slots=int(mask.sum()), microbatch=microbatch):
                    self.model.enc.batch_sizes.clear()
                    output = microbatch_probability(
                        self.model, self.windows, mask, self.pos,
                        res=256, microbatch=microbatch, device="cpu", amp=False)
                    self.assertEqual(output.shape, (12,))
                    self.assertTrue(np.isfinite(output).all())
                    self.assertTrue(((output >= 0) & (output <= 1)).all())
                    np.testing.assert_allclose(output, reference, rtol=1e-6, atol=1e-6)
                    self.assertLessEqual(max(self.model.enc.batch_sizes), microbatch)
                    self.assertEqual(sum(self.model.enc.batch_sizes), int(mask.sum()) * 12)

    def test_group_one_and_twelve_have_identical_deterministic_views(self):
        images = torch.from_numpy(self.windows[:1].reshape(12, 3, 384, 384))
        with torch.inference_mode():
            group_one = self.knee.make_views(images, 256, False, group=1)
            group_twelve = self.knee.make_views(images, 256, False, group=12)
        torch.testing.assert_close(group_one, group_twelve, rtol=0, atol=0)

    def test_all_masked_is_rejected_before_encoder(self):
        self.model.enc.batch_sizes.clear()
        with self.assertRaisesRegex(ValueError, "all-empty"):
            microbatch_probability(self.model, self.windows, np.zeros(6, bool),
                                   self.pos, device="cpu", amp=False)
        self.assertEqual(self.model.enc.batch_sizes, [])

    def test_reader_rejects_wrong_tensor_contract(self):
        mask = np.ones(6, bool)
        for windows, positions, micro in [
            (self.windows.astype(np.float32), self.pos, 8),
            (self.windows[:5], self.pos, 8),
            (self.windows, self.pos[:, :-1], 8),
            (self.windows, self.pos, 0),
        ]:
            with self.subTest(shape=windows.shape, microbatch=micro), self.assertRaises(ValueError):
                microbatch_probability(self.model, windows, mask, positions,
                                       microbatch=micro, device="cpu", amp=False)

    def test_window_centres_preserve_native_adjacent_triplets(self):
        for n in (3, 4, 64):
            got = self.knee.window_centres(n, 12, False)
            a, b = max(1., .04 * (n - 1)), min(n - 2., .96 * (n - 1))
            expected = np.clip(np.round(np.linspace(a, b, 12)).astype(int), 1, n - 2)
            np.testing.assert_array_equal(got, expected)
            self.assertEqual(len(got), 12)
            self.assertTrue((np.diff(got) >= 0).all())
            self.assertTrue(((got >= 1) & (got <= n - 2)).all())
            triplets = np.stack([got - 1, got, got + 1], axis=1)
            self.assertTrue(((triplets >= 0) & (triplets < n)).all())

    def test_slot_routing_largest_series_and_stable_ties(self):
        for plane_index, plane in enumerate(self.knee.PLANES):
            self.assertEqual(self.knee.slot_of(plane, True), plane_index)
            self.assertEqual(self.knee.slot_of(plane, False), plane_index + 3)
        rows = pd.DataFrame({
            "StudyInstanceUID": ["study"] * 5,
            "SeriesInstanceUID": ["small", "large", "tie", "short", "non_fs"],
            "Fat_Suppression": [True, True, True, True, False],
            "Anatomical_Plane": ["Sagittal"] * 5,
        })
        table = self.knee.build_study_table(rows, {
            "small": 3, "large": 64, "tie": 64, "short": 2, "non_fs": 4})
        self.assertEqual(table["study"][0], ["large", "tie", "small"])
        self.assertEqual(table["study"][3], ["non_fs"])
        self.assertEqual(table["study"][1], [])

    def test_checkpoint_metadata_and_head_dimensions(self):
        good = {"args": {"fold": 0, "res": 256, "k_eval": 12, "epochs": 14,
                         "backbone": "convnext_tiny.fb_in22k_ft_in1k"},
                "model": {key: value.detach().clone() for key, value in self.model.state_dict().items()}}
        self.assertEqual(check_checkpoint(good, 0)["res"], 256)
        for field, value in [("fold", 1), ("res", 224), ("k_eval", 8),
                             ("epochs", 13), ("backbone", "resnet50")]:
            bad = {**good, "args": {**good["args"], field: value}}
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_checkpoint(bad, 0)
        for key in ("cls_w", "cls_b", "slot_emb.weight", "proj.1.weight"):
            bad = {**good, "model": {**good["model"], key: torch.zeros(1)}}
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_checkpoint(bad, 0)
        bad = {**good, "model": {**good["model"], "cls_b": torch.full((12,), float("nan"))}}
        with self.assertRaises(ValueError):
            check_checkpoint(bad, 0)

    def test_load_windows_matches_public_skip_semantics(self):
        volume = np.arange(4, dtype=np.uint8)[:, None, None] * np.ones((4, 8, 8), np.uint8)
        calls = []
        module = types.ModuleType("preprocess")
        module.SIZE = 8
        def decode(path):
            calls.append(path)
            return volume, {"n_bad": 0, "n": 4}
        module.load_series = decode
        table = {"study": [["largest", "second"], [], [], [], [], []]}
        stats, rows = Counter(), []
        with patch.dict(sys.modules, {"preprocess": module}):
            x, mask, pos = load_windows("study", table, Path("fixture"), stats, rows)
            self.assertEqual(len(calls), 1)
            self.assertTrue(calls[0].endswith("largest"))
            self.assertEqual(x.dtype, np.uint8)
            self.assertEqual(x.shape, (6, 12, 3, 8, 8))
            np.testing.assert_array_equal(mask, [True, False, False, False, False, False])
            centres = self.knee.window_centres(4, 12, False)
            np.testing.assert_array_equal(x[0, :, :, 0, 0],
                                          np.stack([centres - 1, centres, centres + 1], axis=1))
            np.testing.assert_allclose(pos[0], centres / 3)
            self.assertEqual(rows[0]["available_slots"], 1)
            with self.assertRaisesRegex(RuntimeError, "No usable"):
                load_windows("missing", table, Path("fixture"), stats, rows)
            module.load_series = lambda path: (volume, {"n_bad": 1})
            _, mask, _ = load_windows("study", table, Path("fixture"), stats, rows)
            self.assertTrue(mask[0])
            self.assertEqual(stats["bad_slices_skipped"], 1)
            def broken(path):
                raise OSError("unreadable")
            module.load_series = broken
            with self.assertRaisesRegex(RuntimeError, "No usable"):
                load_windows("study", table, Path("fixture"), stats, rows)
            self.assertEqual(stats["unreadable_slots_skipped"], 1)
            # A failed first candidate must not silently route to a different
            # series; the public infer.py leaves only that slot empty.
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
