import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.fusion import fuse_anchor_s6


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.labels = ['ACL', 'MCL']
        self.ids = ['a', 'b']
        self.anchor = pd.DataFrame([['a', .2, .4], ['b', .6, .8]], columns=['StudyInstanceUID'] + self.labels)
        self.s6 = pd.DataFrame([['b', .3, .5], ['a', .9, .1]], columns=self.anchor.columns)

    def test_blend_aligns_by_uid(self):
        out = fuse_anchor_s6(self.anchor, self.s6, self.ids, self.labels, [])
        self.assertEqual(out.StudyInstanceUID.tolist(), self.ids)
        np.testing.assert_allclose(out[self.labels], [[.27, .37], [.57, .77]], rtol=1e-6)

    def test_failed_s6_row_retains_anchor(self):
        self.s6.loc[self.s6.StudyInstanceUID == 'b', self.labels] = np.nan
        out = fuse_anchor_s6(self.anchor, self.s6, self.ids, self.labels, ['b'])
        np.testing.assert_allclose(out.loc[1, self.labels].to_numpy(float), [.6, .8])

    def test_invalid_inputs_are_rejected(self):
        cases = []
        duplicate = self.s6.copy(); duplicate['StudyInstanceUID'] = ['a', 'a']; cases.append(duplicate)
        missing = self.s6.iloc[:1].copy(); cases.append(missing)
        invalid = self.s6.copy(); invalid.loc[0, 'ACL'] = 1.1; cases.append(invalid)
        nonfinite = self.s6.copy(); nonfinite.loc[0, 'ACL'] = np.nan; cases.append(nonfinite)
        for frame in cases:
            with self.subTest(frame=frame.to_dict()), self.assertRaises(RuntimeError):
                fuse_anchor_s6(self.anchor, frame, self.ids, self.labels, [])

    def test_notebook_checksum(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / 'configs/submission.json').read_text())
        self.assertEqual(hashlib.sha256((root / config['notebook']).read_bytes()).hexdigest(), config['sha256'])


if __name__ == '__main__':
    unittest.main()
