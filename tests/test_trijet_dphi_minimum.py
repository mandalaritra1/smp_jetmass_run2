"""Preserve the legacy minimum's missing-value and numerical semantics."""
import unittest

import awkward as ak
import numpy as np

from smp_jetmass_run2.trijet_processor import TrijetProcessor


class TrijetDphiMinimumTest(unittest.TestCase):
    def test_matches_legacy_reduction(self):
        cases = [
            ([1., 2., 3.], [2., 1., 4.], [3., 4., 1.]),
            ([None, 2., None], [1., None, None], [3., 1., None]),
            ([np.nan, np.inf, -np.inf], [1., 2., 3.], [4., 5., 6.]),
            ([0., -0.], [-0., 0.], [0., -0.]),
            ([], [], []),
        ]
        for dtype in (np.float32, np.float64):
            for rows in cases:
                with self.subTest(dtype=dtype, rows=rows):
                    arrays = [ak.values_astype(ak.Array(row), dtype) for row in rows]
                    expected = ak.min(arrays, axis=0)
                    actual = TrijetProcessor._minimum_pair_dphi(*arrays)
                    self.assertEqual(ak.to_list(ak.is_none(expected)),
                                     ak.to_list(ak.is_none(actual)))
                    expected_values = ak.to_numpy(ak.fill_none(expected, np.nan))
                    actual_values = ak.to_numpy(ak.fill_none(actual, np.nan))
                    np.testing.assert_array_equal(expected_values, actual_values)
                    np.testing.assert_array_equal(np.signbit(expected_values),
                                                  np.signbit(actual_values))
                    if len(actual_values):
                        self.assertEqual(actual_values.dtype, np.dtype(np.float64))


if __name__ == "__main__":
    unittest.main()
