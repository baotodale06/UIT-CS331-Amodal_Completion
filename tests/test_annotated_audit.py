"""Protect the independent audit from missing pools and invented failure scores."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_annotated_surface_pools import independent_policies
from measure_annotated_surface_pools import policies
from amodal_benchmark.protocol import failure_metrics


class AnnotatedAuditTests(unittest.TestCase):
    def record(self, valid):
        scores = [dict(CLIP=20. + i, LPIPS=(i % 3) / 10.,
                       **{'Feature Similarity': (7-i)/10., 'SSIM': i/20.}) for i in range(8)]
        return dict(surface_status='success', candidate_scores=scores,
            observable_scores=[dict(valid=i in valid, candidate_text_cosine=float(i % 2)) for i in range(8)],
            candidate_artifacts=[{} for _ in range(8)], selected_index=3,
            controls=dict(surface=scores[4]))

    def test_independent_replay_preserves_ties_invalids_and_oracle_directions(self):
        for valid in ({1, 3, 6}, set(), set(range(8))):
            record = self.record(valid)
            actual = independent_policies(record)
            expected = policies(record['candidate_scores'], record['observable_scores'],
                                record['selected_index'], record['controls']['surface'])
            self.assertEqual(actual, expected)
            self.assertEqual(actual[1]['oracle_LPIPS'], 0)
            self.assertEqual(actual[1]['oracle_Feature Similarity'], 0)
            self.assertEqual(actual[1]['oracle_CLIP'], 7)

    def test_incomplete_pool_is_rejected(self):
        record = self.record({1})
        for key in ('candidate_scores', 'observable_scores', 'candidate_artifacts'):
            changed = copy.deepcopy(record)
            changed[key].pop()
            with self.assertRaises(ValueError):
                independent_policies(changed)

    def test_failure_keeps_the_penalty_vector_for_every_policy(self):
        record = dict(surface_status='not_detected', candidate_scores=[], candidate_artifacts=[])
        values, choices = independent_policies(record)
        self.assertFalse(choices)
        self.assertEqual(len(values), 10)
        self.assertTrue(all(value == failure_metrics() for value in values.values()))
        record['candidate_scores'] = [{}]
        with self.assertRaises(ValueError):
            independent_policies(record)


if __name__ == '__main__':
    unittest.main()
