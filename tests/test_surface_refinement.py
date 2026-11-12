import unittest
from importlib.util import find_spec

import numpy as np

from amodal_benchmark.surface_refinement import (RefinementConfig, refine_surface,
    rank_features, predict_rank, observation_in_completion_frame, pool_surface_features)


class SurfaceRefinementTests(unittest.TestCase):
    def setUp(self):
        yy, xx = np.indices((40, 48))
        self.truth = np.stack([80 + xx, 100 + yy, 90 + xx // 2], axis=-1).astype(np.uint8)
        self.visible = xx < 24
        self.predicted = np.ones((40, 48), bool)
        self.observed = self.truth.copy()
        self.observed[~self.visible] = 127
        self.candidate = self.truth + np.array([12, 8, 4], np.uint8)

    def test_observed_drift_recovers_hidden_surface_without_smoothing_texture(self):
        actual, details = refine_surface(self.observed, self.visible, self.candidate,
            self.predicted, RefinementConfig(mode='shift', strength=1))
        np.testing.assert_array_equal(actual, self.truth)
        self.assertTrue(details['applied'])

    def test_hidden_observation_values_cannot_affect_output(self):
        changed = self.observed.copy()
        changed[~self.visible] = 0
        for mode in ('shift', 'affine'):
            config = RefinementConfig(mode=mode)
            a, _ = refine_surface(self.observed, self.visible, self.candidate, self.predicted, config)
            b, _ = refine_surface(changed, self.visible, self.candidate, self.predicted, config)
            np.testing.assert_array_equal(a, b)

    @unittest.skipUnless(find_spec('scipy'), 'Optional research dependency scipy')
    def test_field_cannot_read_hidden_observation(self):
        changed = self.observed.copy()
        changed[~self.visible] = 0
        config = RefinementConfig(mode='field')
        a, _ = refine_surface(self.observed, self.visible, self.candidate, self.predicted, config)
        b, _ = refine_surface(changed, self.visible, self.candidate, self.predicted, config)
        np.testing.assert_array_equal(a, b)

    def test_correction_never_modifies_unpredicted_hidden_pixels(self):
        predicted = self.predicted.copy()
        predicted[:, 35:] = False
        actual, _ = refine_surface(self.observed, self.visible, self.candidate, predicted)
        np.testing.assert_array_equal(actual[:, 35:], self.candidate[:, 35:])
        np.testing.assert_array_equal(actual[self.visible], self.observed[self.visible])

    def test_no_support_is_explicit_noop(self):
        actual, details = refine_surface(self.observed, np.zeros_like(self.visible),
            self.candidate, self.predicted)
        np.testing.assert_array_equal(actual, self.candidate)
        self.assertFalse(details['applied'])

    def test_refuses_unaligned_or_ambiguous_images(self):
        with self.assertRaises(ValueError):
            refine_surface(self.observed / 255, self.visible, self.candidate, self.predicted)
        with self.assertRaises(ValueError):
            refine_surface(self.observed, self.visible[:, :20], self.candidate, self.predicted)

    def test_negative_offset_preserves_source_coordinates(self):
        observed, visible = observation_in_completion_frame(self.truth, self.visible,
                                                            (60, 70), (-7, -9))
        np.testing.assert_array_equal(observed[7:47, 9:57], self.truth)
        np.testing.assert_array_equal(visible[7:47, 9:57], self.visible)
        self.assertFalse(visible[:7].any())
        clipped, support = observation_in_completion_frame(self.truth, self.visible,
                                                            (10, 15), (4, 8))
        np.testing.assert_array_equal(clipped, self.truth[4:14, 8:23])
        np.testing.assert_array_equal(support, self.visible[4:14, 8:23])

    def test_ranker_ignores_evaluation_fields_and_has_fixed_fallback(self):
        score = dict(valid=False, boundary=.1, visible_mae=.2)
        other = dict(score, hidden_core_mae=0, hidden_ground_truth='never read')
        np.testing.assert_array_equal(rank_features(score), rank_features(other))
        n = len(rank_features(score))
        model = dict(medians=[0] * n, center=[0] * n, scale=[1] * n,
                     coefficients=[1] * n, valid_only=True)
        index, values = predict_rank([score, other], model)
        self.assertEqual(index, 0)
        self.assertEqual(values, [None, None])

    def test_sampling_consensus_uses_no_hidden_observation_rgb(self):
        changed = self.observed.copy()
        changed[~self.visible] = 255
        candidates = [self.candidate, self.truth]
        masks = [self.predicted, self.predicted]
        first = pool_surface_features(self.observed, self.visible, ~self.visible, candidates, masks)
        second = pool_surface_features(changed, self.visible, ~self.visible, candidates, masks)
        self.assertEqual(first, second)
        self.assertEqual(first[0]['mask_disagreement'], 0)

    def test_semantics_cannot_escape_the_frozen_surface_constraint(self):
        scores = [dict(valid=True,boundary=b,candidate_text_cosine=c)
                  for b,c in ((.1,.2),(.2,.3),(1.,.9))]
        model = dict(feature_names=['boundary'],medians=[0],center=[0],scale=[1],coefficients=[1],
                     valid_only=True,selection_rule=dict(kind='semantic_within_surface_topk',top_k=2))
        index,_ = predict_rank(scores,model)
        self.assertEqual(index,1)
        scores[1]['valid'] = False
        model['selection_rule']['top_k'] = 1
        index,_ = predict_rank(scores,model)
        self.assertEqual(index,0)


if __name__ == '__main__':
    unittest.main()
