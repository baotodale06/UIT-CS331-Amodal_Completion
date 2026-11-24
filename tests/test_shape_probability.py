import unittest
import importlib.util
import numpy as np
from scripts.diagnose_shape_probability import corrected_canvas, probability_scores


class ShapeProbabilityTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Pinned Torch environment required')
    def test_bias_preserves_support_and_zero_outside_crop(self):
        import torch
        support = np.zeros((4, 4), bool)
        support[1, 1] = True
        logits = torch.zeros((2, 2))
        p = corrected_canvas(logits, (1, 1, 3, 3), support, -1.)
        self.assertEqual(p[1, 1], 1.)
        self.assertEqual(p[0, 0], 0.)
        self.assertAlmostEqual(p[2, 2], 1/(1+np.exp(1)), places=7)
        zero = corrected_canvas(logits, (1, 1, 3, 3), support, 0.)
        self.assertEqual(zero[2, 2], .5)
        self.assertTrue(np.all(p<=zero))

    def test_native_probability_scores_exclude_only_observable_support(self):
        truth = np.array([[1, 1], [0, 0]], bool)
        support = np.array([[1, 0], [0, 0]], bool)
        p = np.array([[1., .8], [.2, 0.]])
        scores = probability_scores(p, truth, support)
        self.assertAlmostEqual(scores['brier'], .08/3)
        self.assertAlmostEqual(scores['nll'], (-2*np.log(.8)-np.log(1-1e-7))/3)
        self.assertEqual(probability_scores(p, truth, np.ones_like(support)), dict(brier=None, nll=None))
        with self.assertRaises(ValueError):
            probability_scores(p*2, truth, support)


if __name__=='__main__':
    unittest.main()
