import math
import unittest

from src.eval.nij_scoring import fpr_by_race_at_threshold, nij_fair_and_accurate, nij_fp_term


class TestNijScoring(unittest.TestCase):
    def test_fp_term_formula(self):
        self.assertAlmostEqual(nij_fp_term(0.2, 0.5), 0.7)

    def test_fair_and_accurate_formula(self):
        fp = nij_fp_term(0.2, 0.5)  # 0.7
        fair_acc = nij_fair_and_accurate(brier=0.2, fp_term=fp)  # (1-0.2)*0.7 = 0.56
        self.assertAlmostEqual(fair_acc, 0.56)

    def test_fpr_nan_when_no_negatives(self):
        y_true = [1, 1, 1, 1]
        y_prob = [0.9, 0.8, 0.7, 0.6]
        race = ["BLACK", "BLACK", "WHITE", "WHITE"]
        out = fpr_by_race_at_threshold(y_true=y_true, y_prob=y_prob, race=race, threshold=0.5)
        self.assertTrue(math.isnan(out["BLACK"]))
        self.assertTrue(math.isnan(out["WHITE"]))


if __name__ == "__main__":
    unittest.main()
