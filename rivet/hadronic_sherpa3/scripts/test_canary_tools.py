#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


render_card = load_module("render_card")
validate_pair_split = load_module("validate_pair_split")
make_manifest = load_module("make_manifest")


class CanaryToolsTest(unittest.TestCase):
    def test_deterministic_seed_is_unique_for_test_grid(self):
        seeds = {
            render_card.deterministic_seed(310_000_000, slice_index, chunk_index)
            for slice_index in range(12)
            for chunk_index in range(100)
        }
        self.assertEqual(len(seeds), 1200)
        self.assertEqual(
            render_card.deterministic_seed(310_000_000, 2, 0), 310_020_000
        )

    def test_pair_split_flags_are_orthogonal(self):
        # Columns follow the event diagnostic schema.  Event one is dijet,
        # event two is trijet, and event three is exactly on the pair boundary.
        rows = np.zeros((3, 19), dtype=float)
        rows[:, 1] = 3
        rows[:, 2:5] = [500.0, 480.0, 200.0]
        rows[:, 5:8] = [0.2, -0.3, 0.5]
        rows[:, 12:14] = [1.5, 1.6]
        boundary = np.pi - 200.0 / 490.0
        rows[:, 11] = [boundary + 0.01, boundary - 0.01, boundary]
        dijet, trijet = validate_pair_split.expected_flags(rows)
        np.testing.assert_array_equal(dijet, [True, False, False])
        np.testing.assert_array_equal(trijet, [False, True, False])
        self.assertFalse(np.any(dijet & trijet))

    def test_event_ntuple_columns_are_read_from_the_header(self):
        # Regression: hardcoded indices drifted one column once the routine
        # started writing pair_boundary, so pass_dijet/pass_trijet were read
        # as pass_trijet/weight and every event looked like a mismatch.
        header = "# " + " ".join(validate_pair_split.EXPECTED_COLUMNS) + "\n"
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "events.txt"
            good.write_text(header + "1 2 " + " ".join(["0"] * 16) + "\n")
            col = validate_pair_split.read_columns(good)
            self.assertEqual(col["pass_dijet"], 15)
            self.assertEqual(col["pass_trijet"], 16)
            self.assertEqual(col["weight"], 17)
            self.assertEqual(col, validate_pair_split.CANONICAL_COLUMNS)

            # A layout change must fail loudly rather than shift silently.
            drifted = Path(tmp) / "drifted.txt"
            drifted.write_text(header.replace(" pair_boundary", "") + "1 2\n")
            with self.assertRaises(SystemExit):
                validate_pair_split.read_columns(drifted)

    def test_campaign_slices_are_contiguous_and_seed_unique(self):
        campaign_path = HERE.parent / "configs/campaign_2018_v1.json"
        campaign = json.loads(campaign_path.read_text())
        slices = campaign["slices_GeV"]
        self.assertTrue(all(low < high for low, high in slices))
        self.assertTrue(
            all(left[1] == right[0] for left, right in zip(slices, slices[1:]))
        )
        self.assertFalse(campaign["full_production_approved"])

    def test_manifest_hash_and_footer_helpers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ntuple.txt"
            path.write_text("1 2 3\n# sumw 42.5\n# sumw2 51.25\n")
            self.assertEqual(make_manifest.footer(path, "sumw"), 42.5)
            self.assertEqual(make_manifest.footer(path, "sumw2"), 51.25)
            self.assertEqual(
                make_manifest.sha256(path),
                "1bb4ef587dabfcb01b36ccb2b2c57f405203af8193921e3deb3516b4f055ad17",
            )


if __name__ == "__main__":
    unittest.main()
