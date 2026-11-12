"""Fresh-test replay must catch altered errors and inconsistent deployed choices."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_surface_workflow import experiment_fixture
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.surface_policy import apply_frozen_surface
from amodal_benchmark.surface_refinement import FEATURE_NAMES
from scripts.validate_surface_followup import verify_case


class SurfaceFollowupValidationTests(unittest.TestCase):
    def test_replay_rejects_changed_mae_and_independent_choice(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = experiment_fixture(root, 'fresh', ['test'])
            case = json.loads((experiment / 'cases.json').read_text())[0]
            case_id = case['case_id']
            inputs = experiment / 'inputs' / case_id
            observed = np.array(Image.open(inputs / 'observed.png'))
            visible = np.array(Image.open(inputs / 'visible.png')) > 0
            occlusion = np.array(Image.open(inputs / 'occlusion.png')) > 0
            candidates, masks, scores = [], [], []
            folder = experiment / 'full/candidates' / case_id
            for seed in (11, 22, 33, 44):
                candidates.append(np.array(Image.open(folder / f'seed-{seed}.png')))
                masks.append(np.array(Image.open(folder / f'seed-{seed}-mask.png')) > 0)
                scores.append(json.loads((folder / f'seed-{seed}.json').read_text())['surface'])
            n = len(FEATURE_NAMES)
            freeze = dict(candidate_seeds=[11,22,33,44], refinement=dict(mode='none'),
                ranker=dict(feature_names=FEATURE_NAMES, medians=[0]*n, center=[0]*n, scale=[1]*n,
                            coefficients=[1]*n, valid_only=True))
            selected, mask, detail = apply_frozen_surface(observed, visible, occlusion, candidates, masks, freeze, scores)
            metrics, inference = root / 'metrics', root / 'inference'
            for parent in (metrics, inference):
                destination = parent / 'selected' / case_id
                destination.mkdir(parents=True)
                Image.fromarray(selected).save(destination / 'surface.png')
                Image.fromarray(mask.astype(np.uint8)*255).save(destination / 'mask.png')
            decision = inference / 'decisions' / (case_id + '.json')
            write_json(decision, dict(index=detail['index']))
            truth = np.array(Image.open(experiment / 'evaluation_only' / case_id / 'truth.png'))
            hidden = np.array(Image.open(experiment / 'evaluation_only' / case_id / 'hidden_core.png')) > 0
            values = [dict(hidden_core_mae=float(np.abs(rgb[hidden].astype(float)-truth[hidden]).mean()/255)) for rgb in candidates]
            row = dict(selection=dict(trained_surface=detail['index']), raw=values, refined=values, calibrated=values,
                selected_artifacts={path.name:sha256(path) for path in (metrics / 'selected' / case_id).iterdir()})
            measured = metrics / (case_id + '.json')
            write_json(measured, row)
            args = (case, list(map(str, (experiment, experiment/'full', root/'extra', metrics, inference, root/'features'))), freeze)
            self.assertEqual(verify_case(args)['mae_checks'], 12)
            altered = json.loads(json.dumps(row))
            altered['raw'][0]['hidden_core_mae'] += .01
            write_json(measured, altered)
            with self.assertRaisesRegex(ValueError, 'MAE replay mismatch'):
                verify_case(args)
            write_json(measured, row)
            write_json(decision, dict(index=(detail['index']+1) % 4))
            with self.assertRaisesRegex(ValueError, 'Independent selection disagrees'):
                verify_case(args)


if __name__ == '__main__':
    unittest.main()
