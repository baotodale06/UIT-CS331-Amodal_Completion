"""The exported nonlinear scorer must retain its learned candidate ordering."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from develop_surface_multimetric import fit_boosted
from amodal_benchmark.surface_refinement import predict_rank


@unittest.skipUnless(importlib.util.find_spec('sklearn'), 'Optional development-only sklearn')
class SurfaceBoostedTests(unittest.TestCase):
    def test_portable_scorer_learns_nonlinear_order_and_survives_json(self):
        rng = np.random.default_rng(73)
        rows = []
        for _ in range(40):
            x = rng.permutation(np.linspace(-1, 1, 8))
            rows.append(dict(scores=[dict(visible_mae=float(v), valid=True) for v in x],
                evaluation=dict(refined=[dict(hidden_core_mae=float((v - .4)**2 + .1)) for v in x])))
        model = fit_boosted(rows, ('visible_mae',), 2, 24, dict(hidden_core_mae=1.))
        restored = json.loads(json.dumps(model))
        for row in rows:
            index, values = predict_rank(row['scores'], restored)
            expected = int(np.argmin([item['hidden_core_mae'] for item in row['evaluation']['refined']]))
            self.assertEqual(index, expected)
            self.assertEqual((index, values), predict_rank(row['scores'], model))
            scores = [dict(score, valid=i != expected) for i, score in enumerate(row['scores'])]
            self.assertNotEqual(predict_rank(scores, restored)[0], expected)


if __name__ == '__main__':
    unittest.main()
