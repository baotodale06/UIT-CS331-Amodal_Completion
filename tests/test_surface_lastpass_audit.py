"""Reject corrupted last-pass lineage and preserve paired failure denominators."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_surface_lastpass_result import check_target, paired_estimate, PENALTIES


class LastpassAuditTests(unittest.TestCase):
    def case(self):
        official = dict(id='VG/1_chair', source='VG', original_id='1')
        original = dict(status='success', input_sha256='a'*64, seed=11,
            evaluation_reference_artifacts={'crop': 'b'*64}, completion_sha256='c'*64,
            diagnostics={'surface_iterations': [dict(iteration=2, selected=3, decision_sha256='d'*64)]})
        records = {side: copy.deepcopy(original) for side in ('baseline', 'surface')}
        scores = {'CLIP': 28.0, 'LPIPS': .4, 'Feature Similarity': .5, 'SSIM': .6}
        cached = {side: scores.copy() for side in records}
        hashes = {side: 'e'*64 for side in records}
        row = dict(id=official['id'], source='VG', image_identity='VG/1',
            inputs=dict(freeze_sha256='f'*64, baseline_result_sha256='e'*64, surface_result_sha256='e'*64),
            baseline_status='success', surface_status='success', prior_cached_scores=copy.deepcopy(cached),
            scores={side: scores.copy() for side in ('baseline', 'surface', 'raw_final')},
            derivation=dict(iteration=2, selected_index=3, decision_sha256='d'*64,
                corrected_completion_sha256='c'*64, raw_candidate_sha256='1'*64,
                fixed_mask_sha256='2'*64, derived_sha256='3'*64, size=[20, 30],
                unchanged_alpha=True, unchanged_geometry=True))
        return row, official, records, cached, hashes, 'f'*64

    def test_original_candidate_and_caches_are_required(self):
        args = self.case()
        check_target(*args)
        for key, value in (('selected_index', 4), ('decision_sha256', '0'*64)):
            changed = copy.deepcopy(args)
            changed[0]['derivation'][key] = value
            with self.assertRaisesRegex(ValueError, 'lineage'):
                check_target(*changed)
        changed = copy.deepcopy(args)
        changed[0]['prior_cached_scores']['baseline']['LPIPS'] = .1
        with self.assertRaisesRegex(ValueError, 'cache'):
            check_target(*changed)

    def test_failure_must_keep_penalties_and_have_no_derivation(self):
        args = self.case()
        row, _, records, cached, _, _ = args
        records['surface']['status'] = row['surface_status'] = 'no_completion'
        row['derivation'] = None
        cached['surface'] = PENALTIES.copy()
        row['prior_cached_scores']['surface'] = PENALTIES.copy()
        row['scores']['surface'] = PENALTIES.copy()
        row['scores']['raw_final'] = PENALTIES.copy()
        check_target(*args)
        row['scores']['raw_final']['LPIPS'] = 0.0
        with self.assertRaisesRegex(ValueError, 'failed target'):
            check_target(*args)

    def test_missing_or_nonfinite_successful_metrics_are_rejected(self):
        for value in (None, float('nan'), float('inf')):
            args = self.case()
            args[0]['scores']['raw_final']['CLIP'] = value
            with self.assertRaisesRegex(ValueError, 'Invalid raw scores'):
                check_target(*args)

    def test_cluster_denominators_and_failure_penalties(self):
        rows = []
        for image, first, second in (('VG/1', .4, .3), ('VG/1', .6, .5), ('VG/2', 1., 1.)):
            rows.append(dict(source='VG', image_identity=image,
                scores={'baseline': {'LPIPS': first}, 'raw_final': {'LPIPS': second}}))
        result = paired_estimate(rows, 'baseline', 'raw_final', 'LPIPS')
        self.assertEqual(result['n'], 3)
        self.assertEqual(result['images'], 2)
        self.assertAlmostEqual(result['relative_gain_percent'], 10.)
        # Both targets of image 1 move together; draws contain 0, 1 or 2 copies.
        np.testing.assert_allclose(result['absolute_ci'], [0., .1], rtol=0, atol=1e-12)
        for row in rows:
            row['scores']['baseline']['CLIP'] = 20. if row['image_identity']=='VG/1' else None
            row['scores']['raw_final']['CLIP'] = 21. if row['image_identity']=='VG/1' else None
        clip = paired_estimate(rows, 'baseline', 'raw_final', 'CLIP')
        self.assertEqual((clip['n'], clip['images']), (2, 1))
        self.assertAlmostEqual(clip['relative_gain_percent'], 5.)


if __name__ == '__main__':
    unittest.main()
