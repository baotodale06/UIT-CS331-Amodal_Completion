import unittest

from scripts.explain_surface_results import decomposition, pair_concordance


class SurfaceExplanationTests(unittest.TestCase):
    def test_filtering_and_ranking_can_have_opposite_effects(self):
        result = decomposition([.1, .4, .3, .2], [None, 1., 2., 3.], 1)
        self.assertAlmostEqual(result['total_gain'], -.15)
        self.assertAlmostEqual(result['filter_gain'], -.05)
        self.assertAlmostEqual(result['ranking_gain'], -.1)
        self.assertAlmostEqual(result['exclusion_regret'], .1)
        self.assertAlmostEqual(result['ranking_regret'], .2)

    def test_empty_valid_set_uses_declared_fallback(self):
        result = decomposition([.4, .1, .2, .3], [None] * 4, 0)
        self.assertAlmostEqual(result['ranking_gain'], 0)
        self.assertAlmostEqual(result['ranking_regret'], 0)
        self.assertAlmostEqual(result['total_gain'], -.15)
        self.assertEqual(result['valid_count'], 0)

    def test_concordance_handles_direction_ties_and_unavailable_pairs(self):
        self.assertEqual(pair_concordance([.1, .2, .3], [1, 2, 3]), 1)
        self.assertEqual(pair_concordance([.1, .2, .3], [3, 2, 1]), 0)
        self.assertEqual(pair_concordance([.1, .2], [1, 1]), .5)
        self.assertIsNone(pair_concordance([.1, .1], [1, 2]))
        self.assertIsNone(pair_concordance([.1, .2], [None, 2]))


if __name__ == '__main__':
    unittest.main()
