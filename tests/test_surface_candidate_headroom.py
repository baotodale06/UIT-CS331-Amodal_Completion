"""Ensure oracle diagnostics cannot masquerade as observable candidate selection."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from measure_surface_candidate_headroom import derive_candidate, pool_policies
from measure_surface_lastpass_ablation import derive_raw_lastpass
import test_surface_lastpass_ablation as fixtures


class CandidateHeadroomTests(unittest.TestCase):
    def test_selected_candidate_exactly_reproduces_parent_derivation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, pool, record = fixtures.SurfaceLastpassAblationTests().make_case(root)
            original, _ = derive_raw_lastpass(source, root/'prior', record)
            decision = json.loads((pool/'decision.json').read_text())
            derived, details = derive_candidate(source, root/'new', record, decision, 3)
            self.assertEqual(fixtures.digest(original), fixtures.digest(derived))
            self.assertEqual(details['derived_sha256'], fixtures.digest(derived))
            bad = copy.deepcopy(decision)
            bad['crop_rc'][1] += 1
            with self.assertRaisesRegex(ValueError, 'Provided decision differs'):
                derive_candidate(source, root/'bad', record, bad, 3)

    def test_semantic_control_obeys_validity_and_each_oracle_has_one_output(self):
        scores = [dict(CLIP=20.+i, LPIPS=.2+.05*i, **{'Feature Similarity': .8-.05*i, 'SSIM': .5})
                  for i in range(8)]
        observed = [dict(valid=i<4, candidate_text_cosine=float(i)) for i in range(8)]
        outputs, choices = pool_policies(scores, observed, 2, scores[1])
        self.assertEqual(choices['semantic_only'], 3)
        self.assertEqual(choices['oracle_CLIP'], 7)
        self.assertEqual(choices['oracle_LPIPS'], 0)
        self.assertEqual(outputs['oracle_CLIP'], scores[7])
        self.assertEqual(outputs['oracle_LPIPS'], scores[0])
        self.assertGreater(outputs['oracle_CLIP']['LPIPS'], outputs['oracle_LPIPS']['LPIPS'])
        self.assertAlmostEqual(outputs['uniform_raw']['CLIP'], 23.5)
        # Evaluation values can change the oracle, never the semantic-only choice.
        scores[0]['CLIP'] = 100.
        _, changed = pool_policies(scores, observed, 2, scores[1])
        self.assertEqual(changed['semantic_only'], 3)
        self.assertEqual(changed['oracle_CLIP'], 0)

    def test_nonfinite_candidate_and_incomplete_pool_rejected(self):
        scores = [dict(CLIP=20., LPIPS=.3, **{'Feature Similarity': .8, 'SSIM': .5}) for _ in range(8)]
        observed = [dict(valid=False, candidate_text_cosine=1.) for _ in range(8)]
        _, choices = pool_policies(scores, observed, 0, scores[0])
        self.assertEqual(choices['semantic_only'], 0)
        with self.assertRaisesRegex(ValueError, 'Exactly eight'):
            pool_policies(scores[:7], observed, 0, scores[0])
        scores[2]['CLIP'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            pool_policies(scores, observed, 0, scores[0])


if __name__ == '__main__':
    unittest.main()
