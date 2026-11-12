"""Keep diagnostic uniform controls and per-metric oracle scope explicit."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from measure_annotated_surface_pools import policies


class AnnotatedPoolTests(unittest.TestCase):
    def fixture(self, valid):
        scores = [dict(CLIP=float(i + 1), LPIPS=(8-i)/10., **{'Feature Similarity': i/10., 'SSIM': i/20.}) for i in range(8)]
        observed = [dict(valid=i in valid, candidate_text_cosine=float(i)) for i in range(8)]
        return scores, observed

    def test_uniform_valid_does_not_include_rejected_candidates(self):
        scores, observed = self.fixture({1, 3})
        values, choices = policies(scores, observed, 1, scores[2])
        self.assertEqual(values['uniform_valid']['CLIP'], 3.)
        self.assertEqual(values['uniform_raw']['CLIP'], 4.5)
        self.assertEqual(choices['semantic_only'], 3)
        self.assertEqual(values['first_raw'], scores[0])
        self.assertEqual(values['frozen_raw'], scores[1])
        self.assertEqual(values['corrected'], scores[2])

    def test_all_invalid_uses_declared_first_candidate_fallback(self):
        scores, observed = self.fixture(set())
        values, choices = policies(scores, observed, 0, scores[0])
        self.assertEqual(values['uniform_valid'], scores[0])
        self.assertEqual(choices['semantic_only'], 0)
        self.assertNotEqual(values['uniform_raw'], values['uniform_valid'])


if __name__ == '__main__':
    unittest.main()
