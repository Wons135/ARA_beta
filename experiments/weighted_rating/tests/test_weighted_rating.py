from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from experiments.weighted_rating.build_inputs import validate_final_inputs
from experiments.weighted_rating.weighted_rating import (
    apply_alpha,
    change_summary,
    item_weighted_mean,
    kish_neff,
    permuted_weights_by_item,
)


class WeightedRatingMathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            {
                "review_id": [1, 2, 3, 4, 5],
                "parent_asin": ["A", "A", "A", "B", "B"],
                "rating": [1.0, 3.0, 5.0, 2.0, 4.0],
            }
        )

    def test_equal_weights_match_arithmetic_mean(self) -> None:
        weighted = item_weighted_mean(self.df, np.ones(len(self.df)))
        expected = self.df.groupby("parent_asin")["rating"].mean()
        pd.testing.assert_series_equal(weighted, expected, check_names=False)

    def test_alpha_zero_all_deltas_zero(self) -> None:
        base = item_weighted_mean(self.df, np.ones(len(self.df)))
        weighted = item_weighted_mean(self.df, apply_alpha(np.array([0.1, 0.3, 0.9, 0.2, 0.8]), 0.0))
        self.assertTrue(np.allclose((weighted - base).fillna(0.0), 0.0))

    def test_scale_invariance_per_title(self) -> None:
        weights = np.array([1.0, 2.0, 4.0, 3.0, 6.0])
        base = item_weighted_mean(self.df, weights)
        scaled = item_weighted_mean(self.df, weights * 10.0)
        pd.testing.assert_series_equal(base, scaled)

    def test_weighted_mean_within_title_rating_bounds(self) -> None:
        weights = np.array([0.1, 2.0, 10.0, 1.0, 4.0])
        weighted = item_weighted_mean(self.df, weights)
        mins = self.df.groupby("parent_asin")["rating"].min()
        maxs = self.df.groupby("parent_asin")["rating"].max()
        self.assertTrue(((weighted >= mins) & (weighted <= maxs)).all())

    def test_all_zero_weights_are_nan_and_not_covered(self) -> None:
        base = item_weighted_mean(self.df, np.ones(len(self.df)))
        weights = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
        weighted = item_weighted_mean(self.df, weights)
        self.assertTrue(np.isnan(weighted.loc["B"]))
        summary = change_summary(base, weighted, self.df.groupby("parent_asin").size())
        self.assertEqual(summary["items_compared"], 1)
        self.assertEqual(summary["coverage_%"], 50.0)

    def test_kish_neff_equal_and_unequal_weights(self) -> None:
        equal = kish_neff(self.df, np.ones(len(self.df)))
        counts = self.df.groupby("parent_asin").size().astype(float)
        pd.testing.assert_series_equal(equal, counts, check_names=False)
        unequal = kish_neff(self.df, np.array([1.0, 2.0, 5.0, 1.0, 3.0]))
        self.assertTrue((unequal < counts).all())

    def test_permutation_keeps_weights_inside_title(self) -> None:
        weights = np.array([1.0, 2.0, 5.0, 7.0, 11.0])
        rng = np.random.default_rng(42)
        permuted = permuted_weights_by_item(self.df, weights, rng)
        for parent in self.df["parent_asin"].unique():
            mask = self.df["parent_asin"] == parent
            self.assertEqual(sorted(weights[mask].tolist()), sorted(permuted[mask].tolist()))


class BuildInputAssertionTests(unittest.TestCase):
    def valid_inputs(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "review_id": [1, 2],
                "parent_asin": ["A", "A"],
                "rating": [1, 5],
                "helpful_vote": [0, 2],
                "review_length": [10, 20],
                "p_bert": [0.1, 0.9],
                "pred_source": ["saved", "inferred"],
                "yhat_xgb": [0.0, 3.0],
                "yhat_bert": [0.1, 2.5],
                "p_xgb": [np.nan, 0.7],
                "title": ["A title", "A title"],
            }
        )

    def test_duplicate_ids_fail(self) -> None:
        df = self.valid_inputs()
        df.loc[1, "review_id"] = 1
        with self.assertRaises(AssertionError):
            validate_final_inputs(df)

    def test_probability_above_one_fails(self) -> None:
        df = self.valid_inputs()
        df.loc[1, "p_bert"] = 1.1
        with self.assertRaises(AssertionError):
            validate_final_inputs(df)


if __name__ == "__main__":
    unittest.main()
