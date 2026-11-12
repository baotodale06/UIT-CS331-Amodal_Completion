import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from finalize_surface_open_world import image_cluster_interval
from develop_surface_multimetric import ENDPOINTS, summarize
from analyze_surface_followup import policies, paired_interval


class SurfaceFollowupAnalysisTests(unittest.TestCase):
    def test_cached_bootstrap_keeps_paired_source_stratified_intervals(self):
        result = paired_interval([.1,.4,.5,.2,.3], [.12,.3,.35,.18,.33],
                                 ['VG','VG','COCOA','COCOA','free'], seed=18, repetitions=1000)
        np.testing.assert_allclose(result['absolute_ci'], [-.006, .094], atol=1e-14)
        np.testing.assert_allclose(result['relative_ci_percent'], [-10/3, 470/21], atol=1e-12)
        same = paired_interval([.1,.4,.5,.2,.3], [.1,.4,.5,.2,.3],
                               ['VG','VG','COCOA','COCOA','free'], seed=18, repetitions=1000)
        self.assertEqual(same['absolute_ci'], [0.,0.])

    def test_equal_budget_controls_receive_the_same_refinement(self):
        row = dict(raw=[{'error': .4}, {'error': .2}],
                   refined=[{'error': .3}, {'error': .1}],
                   calibrated=[{'error': .35}, {'error': .15}],
                   valid_indices=[0, 1], selection={'trained_surface': 1, 'semantic_only': 0})
        result = policies(row, 'error')
        self.assertAlmostEqual(result['uniform_raw'], .3)
        self.assertAlmostEqual(result['uniform_refined'], .2)
        self.assertAlmostEqual(result['semantic_only_refined'], .3)
        self.assertAlmostEqual(result['trained_surface_calibrated'], .15)
        self.assertAlmostEqual(result['trained_surface_refined'], .1)

    def test_development_weighting_uses_declared_cohort_counts_without_test_errors(self):
        metrics = (*ENDPOINTS, 'hidden_spatial_lpips')
        rows = [dict(case_id=str(i), source=source,
                     evaluation=dict(raw=[{key: 1. for key in metrics}],
                                     refined=[{key: value for key in metrics}]))
                for i, (source, value) in enumerate((('VG', .5), ('VG', .5), ('COCOA', 1.), ('laion', 4.)))]
        result = summarize(rows, {str(i): 0 for i in range(4)}, dict(VG=3, COCOA=1, laion=0))
        self.assertAlmostEqual(result['metrics']['hidden_core_mae']['method'], .625)
        self.assertAlmostEqual(result['metrics']['hidden_core_mae']['relative_gain_percent'], 37.5)
        with self.assertRaises(ValueError):
            summarize(rows, {str(i): 0 for i in range(4)}, dict(free_images=1))

    def test_repeated_targets_do_not_inflate_independent_sample_size(self):
        items = [dict(source='VG',image_identity=str(i),baseline={'LPIPS':.3+i*.05},
                      surface={'LPIPS':.31+i*.02}) for i in range(4)]
        first = image_cluster_interval(items,'LPIPS',repetitions=1000)
        duplicate = image_cluster_interval(items+items,'LPIPS',repetitions=1000)
        self.assertEqual(duplicate['images'],4)
        self.assertEqual(duplicate['n'],8)
        np.testing.assert_allclose(first['relative_ci_percent'],duplicate['relative_ci_percent'])
        self.assertGreater(first['relative_gain_percent'],0)

    def test_zero_baseline_resamples_do_not_become_fabricated_relative_intervals(self):
        items = [dict(source='VG', image_identity=str(i), baseline={'SSIM': baseline},
                      surface={'SSIM': method}) for i, (baseline, method) in
                 enumerate(((.4, .5), (0., .1)))]
        with np.errstate(divide='raise', invalid='raise'):
            result = image_cluster_interval(items, 'SSIM', repetitions=1000)
        self.assertAlmostEqual(result['relative_gain_percent'], 50.)
        self.assertEqual(result['relative_ci_percent'], [None, None])
        self.assertGreater(result['undefined_relative_bootstrap_samples'], 0)
        np.testing.assert_allclose(result['absolute_ci'], [.1, .1])
        for item in items:
            item['baseline']['SSIM'] = 0.
        result = image_cluster_interval(items, 'SSIM', repetitions=1000)
        self.assertIsNone(result['relative_gain_percent'])
        self.assertEqual(result['undefined_relative_bootstrap_samples'], 1000)


if __name__ == '__main__':
    unittest.main()
