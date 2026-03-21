"""Smoke tests for shared pipeline utilities."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipelines._split_utils import (
    prepare_feature_matrix,
    split_train_cal_select_test,
    split_train_cal_test,
)


class TestSplits(unittest.TestCase):
    def test_3way_split_no_overlap(self):
        y = np.array([0] * 80 + [1] * 20)
        fit, cal, test = split_train_cal_test(y, seed=42)
        all_idx = np.concatenate([fit, cal, test])
        self.assertEqual(len(all_idx), len(set(all_idx)))
        self.assertEqual(len(all_idx), len(y))

    def test_4way_split_no_overlap(self):
        y = np.array([0] * 80 + [1] * 20)
        fit, cal, sel, test = split_train_cal_select_test(y, seed=42)
        all_idx = np.concatenate([fit, cal, sel, test])
        self.assertEqual(len(all_idx), len(set(all_idx)))
        self.assertEqual(len(all_idx), len(y))

    def test_4way_split_proportions(self):
        y = np.array([0] * 800 + [1] * 200)
        fit, cal, sel, test = split_train_cal_select_test(y, seed=42)
        n = len(y)
        # Approximate: fit~48%, cal~16%, sel~16%, test~20%
        self.assertAlmostEqual(len(test) / n, 0.20, delta=0.03)
        self.assertAlmostEqual(len(fit) / n, 0.48, delta=0.05)


class TestPrepareFeatureMatrix(unittest.TestCase):
    def test_fit_transform_alignment(self):
        """Unseen categories in test data should become zero columns."""
        train_df = pd.DataFrame({"color": ["red", "blue", "red"], "size": [1, 2, 3]})
        test_df = pd.DataFrame({"color": ["red", "green"], "size": [4, 5]})
        full_df = pd.concat([train_df, test_df], ignore_index=True)

        _, fit_cols = prepare_feature_matrix(train_df, id_columns=())
        x_full, cols = prepare_feature_matrix(full_df, id_columns=(), fit_columns=fit_cols)

        # "green" should not appear in columns (it's test-only)
        self.assertNotIn("color_green", cols)
        # "red" and "blue" should appear
        self.assertIn("color_red", cols)
        self.assertIn("color_blue", cols)
        # Row for "green" should have 0 for color_red and color_blue
        green_row = x_full.iloc[4]
        self.assertEqual(green_row["color_red"], 0.0)
        self.assertEqual(green_row["color_blue"], 0.0)

    def test_missing_train_category_added(self):
        """If a fit-time column is missing from transform data, it should be added as 0."""
        train_df = pd.DataFrame({"color": ["red", "blue"], "val": [1, 2]})
        # Test data has only "red"
        test_df = pd.DataFrame({"color": ["red"], "val": [3]})

        _, fit_cols = prepare_feature_matrix(train_df, id_columns=())
        x_test, _ = prepare_feature_matrix(test_df, id_columns=(), fit_columns=fit_cols)

        self.assertIn("color_blue", x_test.columns.tolist())
        self.assertEqual(x_test["color_blue"].iloc[0], 0.0)


if __name__ == "__main__":
    unittest.main()
