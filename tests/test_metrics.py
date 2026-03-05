import unittest

from src.eval.fairness import (
    calibration_by_group,
    equalized_odds_gaps,
    fpr_gap,
    fnr_gap,
    predictive_parity_proxy,
    subgroup_auprc,
    subgroup_auroc,
    subgroup_brier,
    threshold_gap_summary,
    threshold_sweep,
)
from src.eval.metrics import (
    auprc,
    auroc,
    bootstrap_ci,
    brier_score,
    calibration_curve,
    expected_calibration_error,
    log_loss,
)
from src.eval.plots import plot_calibration, plot_fairness_bars, plot_pr_curve, plot_roc_curve


def _toy_data():
    y_true = [0, 0, 0, 1, 1, 1]
    y_prob = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
    group = ["A", "A", "B", "B", "A", "B"]
    return y_true, y_prob, group


class TestMetrics(unittest.TestCase):
    def test_core_metrics_basic_ranges(self):
        y_true, y_prob, _ = _toy_data()

        brier = brier_score(y_true, y_prob)
        auc = auroc(y_true, y_prob)
        aprc = auprc(y_true, y_prob)
        ll = log_loss(y_true, y_prob)
        ece = expected_calibration_error(y_true, y_prob, n_bins=5)

        self.assertTrue(0.0 <= brier <= 1.0)
        self.assertTrue(0.0 <= auc <= 1.0)
        self.assertTrue(0.0 <= aprc <= 1.0)
        self.assertTrue(ll >= 0.0)
        self.assertTrue(0.0 <= ece <= 1.0)

    def test_weighted_metrics_accept_sample_weight(self):
        y_true, y_prob, _ = _toy_data()
        weights = [1.0, 1.0, 1.0, 3.0, 3.0, 3.0]

        brier_w = brier_score(y_true, y_prob, sample_weight=weights)
        auc_w = auroc(y_true, y_prob, sample_weight=weights)
        aprc_w = auprc(y_true, y_prob, sample_weight=weights)
        ll_w = log_loss(y_true, y_prob, sample_weight=weights)

        self.assertTrue(0.0 <= brier_w <= 1.0)
        self.assertTrue(0.0 <= auc_w <= 1.0)
        self.assertTrue(0.0 <= aprc_w <= 1.0)
        self.assertTrue(ll_w >= 0.0)

    def test_calibration_curve_structure(self):
        y_true, y_prob, _ = _toy_data()
        cal = calibration_curve(y_true, y_prob, n_bins=4)

        self.assertEqual(
            set(cal.keys()),
            {"bin_edges", "pred_mean", "true_rate", "counts", "empty_bins"},
        )
        self.assertEqual(len(cal["bin_edges"]), 5)
        self.assertEqual(len(cal["pred_mean"]), 4)
        self.assertEqual(sum(cal["counts"]), len(y_true))
        self.assertEqual(len(cal["empty_bins"]), 4)

    def test_bootstrap_ci_shape_and_ordering(self):
        y_true, y_prob, _ = _toy_data()
        ci = bootstrap_ci(y_true, y_prob, metric_fn=brier_score, n_bootstrap=200, random_state=7)

        self.assertEqual(ci["n_bootstrap"], 200)
        self.assertGreater(ci["n_valid"], 0)
        self.assertLessEqual(ci["lower"], ci["estimate"])
        self.assertGreaterEqual(ci["upper"], ci["estimate"])
        self.assertTrue(ci["stratified"])

    def test_subgroup_metrics_present_for_each_group(self):
        y_true, y_prob, group = _toy_data()

        sb = subgroup_brier(y_true, y_prob, group=group)
        sa = subgroup_auroc(y_true, y_prob, group=group)
        sp = subgroup_auprc(y_true, y_prob, group=group)

        self.assertEqual(set(sb.keys()), {"A", "B"})
        self.assertEqual(set(sa.keys()), {"A", "B"})
        self.assertEqual(set(sp.keys()), {"A", "B"})

    def test_fairness_gap_metrics_non_negative(self):
        y_true, y_prob, group = _toy_data()

        gap_fpr = fpr_gap(y_true, y_prob, group=group, threshold=0.5)
        gap_fnr = fnr_gap(y_true, y_prob, group=group, threshold=0.5)
        eo = equalized_odds_gaps(y_true, y_prob, group=group, threshold=0.5)
        pp = predictive_parity_proxy(y_true, y_prob, group=group, threshold=0.5)

        self.assertGreaterEqual(gap_fpr, 0.0)
        self.assertGreaterEqual(gap_fnr, 0.0)
        self.assertGreaterEqual(eo["eo_gap_max"], eo["fpr_gap"])
        self.assertGreaterEqual(eo["eo_gap_max"], eo["fnr_gap"])
        self.assertGreaterEqual(pp["ppv_gap"], 0.0)

    def test_threshold_sweep_and_gap_summary(self):
        y_true, y_prob, group = _toy_data()

        out = threshold_sweep(y_true, y_prob, group=group, thresholds=[0.25, 0.5, 0.75])
        self.assertEqual(set(out.keys()), {"overall", "A", "B"})
        self.assertEqual(len(out["overall"]), 3)
        self.assertTrue(all("threshold" in row for row in out["overall"]))

        summary = threshold_gap_summary(y_true, y_prob, group=group, thresholds=[0.25, 0.5, 0.75])
        self.assertEqual(len(summary["thresholds"]), 3)
        self.assertEqual(len(summary["fpr_gap_per_threshold"]), 3)
        self.assertEqual(len(summary["fnr_gap_per_threshold"]), 3)
        self.assertGreaterEqual(summary["fpr_gap_max"], summary["fpr_gap_mean"])
        self.assertGreaterEqual(summary["fnr_gap_max"], summary["fnr_gap_mean"])

    def test_calibration_by_group_structure(self):
        y_true, y_prob, group = _toy_data()

        out = calibration_by_group(y_true, y_prob, group=group, n_bins=3)
        self.assertEqual(set(out.keys()), {"A", "B"})
        self.assertEqual(
            set(out["A"].keys()),
            {"bin_edges", "pred_mean", "true_rate", "counts", "empty_bins"},
        )

    def test_plot_builders_return_specs(self):
        y_true, y_prob, group = _toy_data()

        fig1 = plot_calibration(y_true, y_prob, group=group)
        fig2 = plot_roc_curve(y_true, y_prob)
        fig3 = plot_pr_curve(y_true, y_prob)
        fig4 = plot_fairness_bars(y_true, y_prob, group=group)

        for fig in (fig1, fig2, fig3, fig4):
            self.assertIsInstance(fig, dict)
            self.assertIn("title", fig)


if __name__ == "__main__":
    unittest.main()
