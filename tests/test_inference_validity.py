"""Statistical validity of the registered estimators, checked by simulation (synthetic data only).

These guard the findings of the independent review: plug-in JS depends on window transition count and on
soft-weight sharpness (so it is null-calibrated), and CR1 cluster inference is liberal with few clusters (so CR2 +
Bell-McCaffrey df is used).
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_ind

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyse_timestamped_transition_dynamics as m  # noqa: E402


def reference(rng, concentration):
    """Tensor reference q[a, s, d] with zero diagonal and Dirichlet(concentration) rows."""
    q = np.zeros((4, 6, 6))
    for a in range(4):
        for s in range(6):
            q[a, s, np.arange(6) != s] = rng.dirichlet(np.full(5, concentration))
    return q


def draw_window(rng, q, n, sharpness):
    """Window drawn EXACTLY from the reference: latent hard action ~ soft weights, destination ~ q[action, source]."""
    source = rng.integers(0, 6, n)
    weights = rng.dirichlet(np.full(4, sharpness), n)
    action = np.array([rng.choice(4, p=w) for w in weights])
    destination = np.array([rng.choice(6, p=q[a, s]) for a, s in zip(action, source)])
    return source, destination, weights


def excess(rng, q, n, sharpness, key):
    source, destination, weights = draw_window(rng, q, n, sharpness)
    counts = m.tensor_counts(source, destination, weights)
    js = m.weighted_js(counts, q)
    return js, js - m.null_expected_js(source, weights, q, key)


class NullCalibration(unittest.TestCase):
    def test_batch_js_matches_scalar_js(self):
        rng = np.random.default_rng(1)
        q = reference(rng, 1.0)
        stack = rng.integers(0, 6, (7, 4, 6, 6)).astype(float)
        stack[..., np.eye(6, dtype=bool)] = 0
        batch = m.weighted_js_batch(stack, q)
        np.testing.assert_allclose(batch, [m.weighted_js(x, q) for x in stack], rtol=1e-12)

    def test_excess_js_is_unbiased_where_raw_js_is_not(self):
        rng = np.random.default_rng(20261028)
        for concentration, sharpness, n in ((0.5, 1.0, 30), (0.5, 1.0, 150), (5.0, 0.3, 60), (2.0, 20.0, 60)):
            q = reference(rng, concentration)
            draws = [excess(rng, q, n, sharpness, f"cal|{concentration}|{n}|{i}") for i in range(120)]
            raw, calibrated = np.array(draws).T
            self.assertGreater(raw.mean(), 0.01, "raw JS should carry a clear positive finite-sample bias")
            self.assertLess(abs(calibrated.mean()), 0.0035, (concentration, sharpness, n, calibrated.mean()))

    def test_excess_js_does_not_depend_on_window_transition_count(self):
        rng = np.random.default_rng(7)
        q = reference(rng, 2.0)
        means = {}
        for n in (30, 150):
            values = np.array([excess(rng, q, n, 0.15, f"rate|{n}|{i}") for i in range(150)])
            means[n] = values.mean(axis=0)
        # raw log-JS moves with the window transition count (size/sign depend on reference sparsity and weight
        # sharpness, so no global n-correction exists); the calibrated excess does not
        self.assertGreater(abs(np.log(means[30][0]) - np.log(means[150][0])), 0.15)
        self.assertLess(abs(means[30][1] - means[150][1]), 0.004)

    def test_deterministic_given_key(self):
        rng = np.random.default_rng(3)
        q = reference(rng, 1.0)
        source, _, weights = draw_window(rng, q, 40, 1.0)
        a = m.null_expected_js(source, weights, q, "k")
        self.assertEqual(a, m.null_expected_js(source, weights, q, "k"))
        self.assertNotEqual(a, m.null_expected_js(source, weights, q, "other"))

    def test_real_grammar_change_is_detected(self):
        rng = np.random.default_rng(11)
        q = reference(rng, 1.0)
        shifted = 0.7 * q + 0.3 * reference(rng, 1.0)
        values = []
        for i in range(100):
            source, destination, weights = draw_window(rng, shifted, 60, 1.0)
            js = m.weighted_js(m.tensor_counts(source, destination, weights), q)
            values.append(js - m.null_expected_js(source, weights, q, f"sig|{i}"))
        # a genuine 30% grammar mixture leaves clearly positive excess (attenuated by soft action assignment)
        self.assertGreater(ttest_1samp(values, 0).statistic, 4)


class ClusterRobustInference(unittest.TestCase):
    def test_cr2_estimate_equals_ols_and_orders_df(self):
        rng = np.random.default_rng(5)
        groups = np.repeat(np.arange(8), 2)
        x = np.column_stack([np.ones(16), np.tile([0.0, 1.0], 8)])
        y = rng.normal(size=16) + 0.5 * x[:, 1]
        fitted = m.cluster_cr2(y, x, groups, np.array([0.0, 1.0]))
        np.testing.assert_allclose(fitted["estimate"], np.linalg.lstsq(x, y, rcond=None)[0][1])
        self.assertTrue(0 < fitted["df"] <= 16)

    def test_type_i_error_near_nominal_with_few_unbalanced_clusters(self):
        rng = np.random.default_rng(99)
        for n_hypo, n_hyper, n_both in ((9, 3, 0), (3, 3, 0), (5, 5, 5)):
            rows_p, p_values = [], []
            for _ in range(700):
                records = []
                person = 0
                for count, strata in ((n_hypo, ("hypo",)), (n_hyper, ("hyper",)), (n_both, ("hypo", "hyper"))):
                    for _ in range(count):
                        shared = rng.normal(0, 0.8)
                        for stratum in strata:
                            records.append({"participant": f"P{person}", "semiology": stratum,
                                            "value": shared + rng.normal(0, 0.5)})
                        person += 1
                p_values.append(m.clustered_hypo_hyper(pd.DataFrame(records))["p"])
            rate = np.mean(np.array(p_values) < 0.05)
            self.assertLess(rate, 0.075, (n_hypo, n_hyper, n_both, rate))
            self.assertGreater(rate, 0.01, (n_hypo, n_hyper, n_both, rate))

    def test_too_few_clusters_per_stratum_not_estimable(self):
        frame = pd.DataFrame({"participant": ["a", "b", "c", "d"], "semiology": ["hypo", "hypo", "hyper", "hyper"],
                              "value": [1., 2., 3., 4.]})
        self.assertTrue(np.isnan(m.clustered_hypo_hyper(frame)["p"]))


class PlanAndEligibility(unittest.TestCase):
    def test_family_structure(self):
        plan = m.confirmatory_plan()
        self.assertEqual(plan.family.value_counts().to_dict(), {"descriptive": 16, "secondary": 6, "primary": 2})
        primary = plan[plan.family.eq("primary")]
        self.assertEqual(set(primary.outcome), set(m.PRIMARY_OUTCOMES))
        self.assertTrue(primary.estimand.eq("final_minus_early").all() and primary.contrast.eq("FS versus ES").all())
        self.assertTrue(plan[plan.estimand.eq("final_minute")].family.eq("descriptive").all())

    def test_descriptive_cells_report_no_p_values(self):
        from test_timestamped import fixture
        result = m.run_confirmatory(fixture(), "software fixture")
        descriptive = result[result.family.eq("descriptive")]
        self.assertTrue(descriptive.p.isna().all() and descriptive.p_holm.isna().all())
        self.assertTrue(descriptive.estimate.notna().any())

    def test_secondary_holm_runs_over_six(self):
        from test_timestamped import fixture
        result = m.run_confirmatory(fixture(), "software fixture")
        secondary = result[result.family.eq("secondary")]
        self.assertEqual(len(secondary), 6)
        estimable = secondary[secondary.p.notna()]
        self.assertTrue((estimable.p_holm >= estimable.p).all())
        self.assertTrue((estimable.p_holm <= 1).all())

    def test_minimum_early_windows_enforced(self):
        from test_timestamped import fixture
        x = fixture()
        drop = x.participant.eq("FS0") & x.window_end_to_ictus_s.isin([620., 630., 640.])  # leaves 2 early windows
        cells = m.planned_interval_cells(x[~drop])
        events = m.event_changes(cells, "tensor_js_excess")
        self.assertNotIn("FS0", set(events.participant))
        self.assertIn("FS1", set(events.participant))

    def test_sensitivity_and_coverage_outputs(self):
        from test_timestamped import fixture
        x = fixture()
        shifted = x[x.window_end_to_ictus_s.eq(0.)].copy()
        shifted["window_end_to_ictus_s"] = 60.
        sensitivity = m.primary_sensitivity(pd.concat([x, shifted], ignore_index=True))
        self.assertEqual(set(sensitivity.analysis),
                         {"yuen_20pct_trimmed_welch", "studentised_permutation", "prefinal_window_-120_to_-60"})
        self.assertEqual(len(sensitivity), 12)
        permutation = sensitivity[sensitivity.analysis.eq("studentised_permutation")]
        self.assertTrue(((permutation.p > 0) & (permutation.p <= 1)).all())
        again = m.primary_sensitivity(pd.concat([x, shifted], ignore_index=True))
        pd.testing.assert_frame_equal(sensitivity, again)  # hash-seeded: bit-identical rerun
        index = x[["participant", "event_key", "group"]].drop_duplicates().assign(
            condition="preict", file_path=lambda d: d.event_key)
        coverage = m.coverage_table(index, x)
        self.assertTrue((coverage.events_complete <= coverage.events_in_index).all())
        self.assertEqual(coverage[coverage.group.eq("FS")].events_complete.max(), 4)

    def test_welch_type_i_error_with_rate_confounding_after_calibration(self):
        """Participant-level Welch on calibrated excess JS stays near nominal when FS final-minute rate is lower."""
        rng = np.random.default_rng(424242)
        rejections = {"raw_log": 0, "excess": 0}
        repetitions = 40
        q_by_participant = {}
        for _ in range(repetitions):
            changes = {"FS": {"raw_log": [], "excess": []}, "ES": {"raw_log": [], "excess": []}}
            for group, count, final_n in (("FS", 12, 42), ("ES", 25, 60)):  # FS final-minute transition rate x0.7
                for person in range(count):
                    q = reference(rng, 1.0)
                    early_raw, early_ex = [], []
                    for w in range(5):
                        js, ex = excess(rng, q, 60, 1.0, f"e|{group}|{person}|{w}|{len(changes[group]['raw_log'])}")
                        early_raw.append(np.log(js)); early_ex.append(ex)
                    js, ex = excess(rng, q, final_n, 1.0, f"f|{group}|{person}|{len(changes[group]['raw_log'])}")
                    changes[group]["raw_log"].append(np.log(js) - np.mean(early_raw))
                    changes[group]["excess"].append(ex - np.mean(early_ex))
            for kind in rejections:
                rejections[kind] += ttest_ind(changes["FS"][kind], changes["ES"][kind], equal_var=False).pvalue < .05
        # raw log JS is confounded by the 30% lower final-minute rate; the calibrated excess is not
        self.assertGreater(rejections["raw_log"] / repetitions, 0.15)
        self.assertLess(rejections["excess"] / repetitions, 0.2)


if __name__ == "__main__":
    unittest.main()
