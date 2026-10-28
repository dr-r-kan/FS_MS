"""Software fixtures only: no clinical files, cohort fits, or real registration."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.io import savemat
from scipy.stats import ttest_ind

import analyse_timestamped_transition_dynamics as m


def fixture():
    rows = []
    for group in ("FS", "ES"):
        for i in range(4):
            for endpoint in (600., 610., 620., 630., 640., 0.):
                row = dict(participant=f"{group}{i}", event_key=f"{group}{i}-1",
                           group=group, event_type=group, semiology="hypo" if i % 2 else "hyper",
                           condition="preict", window_end_to_ictus_s=endpoint)
                for j, outcome in enumerate(m.OUTCOMES):
                    row[outcome] = .1*i + (0 if endpoint else .05*i + (.2 if group == "FS" else -.1)) + .03*j
                rows.append(row)
    return pd.DataFrame(rows)


class MeasurementTests(unittest.TestCase):
    def test_existing_self_check(self):
        m.self_check()

    def test_missing_gradients_do_not_become_neutral(self):
        a = m.gradient_weights(np.array([0., np.nan]), np.array([0., np.nan]))
        np.testing.assert_allclose(a[0], [.25]*4)
        self.assertTrue(np.isnan(a[1]).all())

    def test_no_gap_interpolation(self):
        data = pd.DataFrame(dict(event_time_s=[0., 1., 2., 3., 4., 20., 21.],
                                 csi_z=[0., 1., np.nan, 3., 4., 20., 21.],
                                 cpi_z=[0., 2., np.nan, 6., 8., 40., 42.]))
        values = m.interpolate_native(data, np.array([.5, 1.5, 2.5, 3.5, 10., 20.5, 22.]))
        for value in values:
            self.assertTrue(np.isnan(value[[1, 2, 4, 6]]).all())
            self.assertTrue(np.isfinite(value[[0, 3, 5]]).all())

    def test_true_final_minute_and_whole_early_windows(self):
        x = fixture()
        wrong = x.iloc[[0]].copy(); wrong['window_end_to_ictus_s'] = 30.
        wrong[list(m.OUTCOMES)] = 999.
        outside = wrong.copy(); outside['window_end_to_ictus_s'] = 1190.
        expected = m.planned_interval_cells(x)
        actual = m.planned_interval_cells(pd.concat([x, wrong, outside], ignore_index=True))
        pd.testing.assert_frame_equal(expected, actual)

    def test_fit_labels_are_derived_and_validated(self):
        np.testing.assert_array_equal(m.fit_to_canonical(['A', 'F', 'C', 'G', 'B', 'E']),
                                      [m.STATES.index(x) for x in 'AFCGBE'])
        np.testing.assert_array_equal(m.fit_to_canonical(list('fegbac')), np.arange(6))
        for bad in ([], list('ABCDEF'), list('AFCGBB'), list('AFCGB')):
            with self.assertRaises(ValueError):
                m.fit_to_canonical(bad)

    def test_missing_fit_labels_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/'index.csv').write_text('manifest_index,participant,condition,file_path,group,cache_file\n')
            for name in ('native.csv', 'manifest.csv'):
                (root/name).write_text('x\n')
            (root/'recovery_audit.json').write_text(json.dumps({
                'compact_export_validated_against_full_backfit': True,
                'focused_window_fit_recovery_validation': {'transition_matrix_exact_match': True}}))
            run = subprocess.run([sys.executable, '-B', str(Path(m.__file__).resolve()), '--transition-index',
                                  str(root/'index.csv'), '--autonomic-native', str(root/'native.csv'),
                                  '--manifest', str(root/'manifest.csv'), '--out-dir', str(root/'out'), '--prepare'],
                                 capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('fit_labels', run.stderr)

    def test_kappa_identification(self):
        counts = np.ones((6, 6)); np.fill_diagonal(counts, 0)
        self.assertTrue(np.isnan(m.fit_kappa(counts, m.row_probabilities(counts))))
        reference = counts.copy()
        for i in range(6):
            reference[i, np.arange(6) != i] = [.05, .1, .15, .2, .5]
        observed = reference**1.7
        observed /= observed.sum(axis=1, keepdims=True)
        self.assertAlmostEqual(m.fit_kappa(observed*10000, reference), 1.7, places=6)


class InferenceTests(unittest.TestCase):
    def test_no_dd_is_unavailable_not_crash(self):
        result = m.run_confirmatory(fixture(), "software fixture")
        dd = result[result.contrast.eq("within-DD FS versus ES")]
        self.assertTrue(dd.p.isna().all())
        self.assertTrue(dd.n_first.eq(0).all())
        self.assertEqual(result.family.value_counts().to_dict(), {"descriptive": 16, "secondary": 6, "primary": 2})

    def test_outcome_missingness_requires_both_intervals(self):
        x = fixture()
        x.loc[x.participant.eq("FS0") & x.window_end_to_ictus_s.ge(600), 'tensor_js_excess'] = np.nan
        result = m.run_confirmatory(x, "software fixture")
        row = result[result.outcome.eq('tensor_js_excess') & result.estimand.eq('final_minute') & result.contrast.eq('FS versus ES')].iloc[0]
        self.assertEqual(row.n_first, 3)

    def test_welch_and_fixed_primary_family(self):
        x = fixture()
        x['log_tensor_kappa'] = np.nan
        result = m.run_confirmatory(x, "software fixture")
        row = result[result.family.eq('primary') & result.outcome.eq('tensor_js_excess')].iloc[0]
        expected = ttest_ind(.05*np.arange(4)+.2, .05*np.arange(4)-.1, equal_var=False)
        self.assertAlmostEqual(row.p, expected.pvalue)
        self.assertAlmostEqual(row.p_holm, min(1, 2*row.p))

    def test_independent_subject_overlap_rejected(self):
        x = fixture(); x.loc[x.participant.eq('ES0'), 'participant'] = 'FS0'
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            m.run_confirmatory(x, 'software fixture')


class LockTests(unittest.TestCase):
    def test_direct_cache_and_metric_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); cache = root/'cache.mat'

            def write_cache(shift=0):
                savemat(cache, {'transition': dict(sfreq=256., n_samples=2560, transition_sample=np.array([10., 40., 90.]) + shift,
                                                   from_state=np.array([1, 2, 3]), to_state=np.array([2, 3, 1]))})
            write_cache()
            index = root/'index.csv'; pd.DataFrame({'cache_file': [str(cache)]}).to_csv(index, index=False)
            metrics = root/'metrics.csv'; metrics.write_text('fixture')
            artifacts = {'transition_index': index, 'prepared_metrics': metrics}
            m.write_preregistration_lock(root, artifacts, 'plan', {})
            m.verify_preregistration_lock(root, artifacts, 'plan')
            write_cache(shift=1)
            with self.assertRaisesRegex(RuntimeError, 'transition cache'):
                m.verify_preregistration_lock(root, artifacts, 'plan')
            write_cache(); metrics.write_text('changed')
            with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
                m.verify_preregistration_lock(root, artifacts, 'plan')

    def test_cache_digest_ignores_file_bytes_but_not_content(self):
        with tempfile.TemporaryDirectory() as td:
            first, second, third = (Path(td)/f'{n}.mat' for n in 'abc')
            body = dict(sfreq=256., n_samples=2560, transition_sample=np.array([10., 40., 90.]),
                        from_state=np.array([1, 2, 3]), to_state=np.array([2, 3, 1]))
            savemat(first, {'transition': body})
            savemat(second, {'transition': body}, do_compression=True)  # different bytes, same content
            savemat(third, {'transition': {**body, 'to_state': np.array([2, 3, 2])}})
            self.assertNotEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(m.transition_cache_digest(first), m.transition_cache_digest(second))
            self.assertNotEqual(m.transition_cache_digest(first), m.transition_cache_digest(third))

    def test_json_artifacts_ignore_utc_timestamps_only(self):
        with tempfile.TemporaryDirectory() as td:
            a, b, c = (Path(td)/f'{n}.json' for n in 'abc')
            a.write_text(json.dumps({'k': 1, 'created_utc': '2026-01-01T00:00:00Z'}))
            b.write_text(json.dumps({'created_utc': '2027-02-02T00:00:00Z', 'k': 1}, indent=4))
            c.write_text(json.dumps({'k': 2, 'created_utc': '2026-01-01T00:00:00Z'}))
            self.assertEqual(m.artifact_digest(a), m.artifact_digest(b))
            self.assertNotEqual(m.artifact_digest(a), m.artifact_digest(c))

    def test_registration_placeholder_rejected(self):
        for value in ('', 'synthetic', 'https://osf.io/REPLACE_WITH_REGISTRATION_ID', 'http://osf.io/a1234'):
            with self.assertRaises(ValueError): m.validate_registration_id(value)

    def test_end_to_end_synthetic_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); rng = np.random.default_rng(1701)
            index, manifest, native = [], [], []
            for g in ('FS', 'ES'):
                for person in range(2):
                    for condition in ('ii', 'preict'):
                        number = len(index); participant = f'{g}{person}'
                        file = str(root/f'fixture_{number}.mat'); duration = 1260
                        times = np.arange(.5, duration, 1.)
                        source = rng.integers(1, 7, len(times)); destination = (source + rng.integers(1, 6, len(times))-1) % 6+1
                        cache = root/f'cache_{number}.mat'
                        savemat(cache, {'transition': dict(sfreq=256., n_samples=duration*256,
                            transition_sample=times*256+1, from_state=source, to_state=destination)})
                        index.append(dict(manifest_index=number, participant=participant, condition=condition,
                                          file_path=file, group=g, cache_file=str(cache)))
                        manifest.append(dict(file_path=file, seizure_type=g, hypo_hyper='hypo' if person else 'hyper'))
                        for t in np.arange(0, duration+1, 1.):
                            native.append(dict(file_path=file, event_time_s=t, csi_z=np.sin(t/40), cpi_z=np.cos(t/25)))
            pd.DataFrame(index).to_csv(root/'index.csv', index=False)
            pd.DataFrame(manifest).to_csv(root/'manifest.csv', index=False)
            pd.DataFrame(native).to_csv(root/'native.csv', index=False)
            (root/'recovery_audit.json').write_text(json.dumps({'compact_export_validated_against_full_backfit': True,
                'fit_labels': ['A', 'F', 'C', 'G', 'B', 'E'],
                'focused_window_fit_recovery_validation': {'transition_matrix_exact_match': True}}))
            out = root/'out'
            cmd = [sys.executable, '-B', str(Path(m.__file__).resolve()), '--transition-index', str(root/'index.csv'),
                   '--autonomic-native', str(root/'native.csv'), '--manifest', str(root/'manifest.csv'), '--out-dir', str(out)]
            prepared = subprocess.run(cmd+['--prepare'], capture_output=True, text=True)
            self.assertEqual(prepared.returncode, 0, prepared.stderr[-4000:])
            self.assertFalse((out/'tables/confirmatory_results.csv').exists())
            checked = subprocess.run(cmd+['--verify-lock'], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stderr[-4000:])
            lock_before = (out/'PREREGISTRATION_LOCK.json').read_bytes()
            executed = subprocess.run(cmd+['--run-confirmatory', '--preregistration-id', 'https://osf.io/synthetic-test'], capture_output=True, text=True)
            self.assertEqual(executed.returncode, 0, executed.stderr[-4000:])
            self.assertEqual(lock_before, (out/'PREREGISTRATION_LOCK.json').read_bytes())
            self.assertTrue((out/'tables/primary_sensitivity_results.csv').exists())
            self.assertTrue((out/'tables/coverage_by_group.csv').exists())
            json.loads((out/'CONFIRMATORY_EXECUTION.json').read_text())
            again = subprocess.run(cmd+['--prepare'], capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertTrue((out/'tables/confirmatory_results.csv').exists())


if __name__ == '__main__':
    unittest.main()
