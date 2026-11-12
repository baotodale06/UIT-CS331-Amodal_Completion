"""Protect the development/test boundary and replication contract end to end."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.surface import score_surface
from scripts.analyze_surface_experiment import main
from scripts.validate_surface_experiment import validate_evaluation_rows, assert_equivalent


def experiment_fixture(root, name, splits, identity_prefix=None):
    experiment = root / name
    identity_prefix = identity_prefix or name
    seeds = [11, 22, 33, 44]
    write_json(experiment / 'protocol.json', dict(experiment=name, candidate_seeds=seeds,
        development_per_source=1, min_visible_recall=.5, mc_timestep=35))
    cases = []
    for index, split in enumerate(splits):
        case_id = f'{name}-{index}'
        observed = np.full((64, 64, 3), (150 + index, 60, 40), np.uint8)
        truth = observed.copy()
        visible = np.zeros((64, 64), bool)
        visible[8:56, 8:32] = True
        occlusion = np.zeros((64, 64), bool)
        occlusion[:, 32:] = True
        full = np.zeros((64, 64), bool)
        full[8:56, 8:56] = True
        hidden = full & occlusion
        observed[occlusion] = 127
        inputs, evaluation, folder = (experiment / 'inputs' / case_id,
            experiment / 'evaluation_only' / case_id, experiment / 'full/candidates' / case_id)
        for path in (inputs, evaluation, folder):
            path.mkdir(parents=True)
        Image.fromarray(observed).save(inputs / 'observed.png')
        Image.fromarray(visible.astype(np.uint8) * 255).save(inputs / 'visible.png')
        Image.fromarray(occlusion.astype(np.uint8) * 255).save(inputs / 'occlusion.png')
        Image.fromarray(truth).save(evaluation / 'truth.png')
        for key in ('hidden', 'hidden_core'):
            Image.fromarray(hidden.astype(np.uint8) * 255).save(evaluation / (key + '.png'))
        case = dict(case_id=case_id, source='VG', split=split, label='synthetic object', target_id=case_id,
            image_identity=f'{identity_prefix}/{index}',
            source_sha256=hashlib.sha256(f'{identity_prefix}/{index}'.encode()).hexdigest(),
            input_hashes={p.name: sha256(p) for p in inputs.iterdir()},
            evaluation_hashes={p.name: sha256(p) for p in evaluation.iterdir()})
        cases.append(case)
        for seed, change in zip(seeds, (50, 0, -35, 20)):
            candidate = truth.copy()
            candidate[hidden] = np.clip(candidate[hidden].astype(int) + change, 0, 255).astype(np.uint8)
            rgb_path, mask_path = folder / f'seed-{seed}.png', folder / f'seed-{seed}-mask.png'
            Image.fromarray(candidate).save(rgb_path)
            Image.fromarray(full.astype(np.uint8) * 255).save(mask_path)
            write_json(folder / f'seed-{seed}.json', dict(case_id=case_id, seed=seed,
                input_hashes=case['input_hashes'], status='generated',
                artifacts={p.name: sha256(p) for p in (rgb_path, mask_path)},
                surface=score_surface(observed, visible, occlusion, candidate, full)))
    write_json(experiment / 'cases.json', cases)
    return experiment


class SurfaceWorkflowTests(unittest.TestCase):
    def run_analysis(self, experiment, stage, freeze=None):
        argv = ['analyze_surface_experiment.py', '--experiment-dir', str(experiment),
                '--run-dir', str(experiment / 'full'), '--stage', stage]
        if freeze:
            argv.extend(['--freeze-from', str(freeze)])
        with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
            main()

    def test_freeze_does_not_read_test_truth_and_test_detects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = experiment_fixture(Path(temporary), 'pilot', ['development', 'test'])
            truth = experiment / 'evaluation_only/pilot-1/truth.png'
            truth.write_bytes(b'corrupted test truth must not be opened during development')
            self.run_analysis(experiment, 'freeze')
            self.assertTrue((experiment / 'analysis/development_freeze.json').exists())
            with self.assertRaisesRegex(ValueError, 'already frozen'):
                self.run_analysis(experiment, 'freeze')
            with self.assertRaisesRegex(ValueError, 'truth changed'):
                self.run_analysis(experiment, 'test')

    def test_frozen_selection_replays_on_disjoint_replication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilot = experiment_fixture(root, 'pilot', ['development', 'test'])
            replication = experiment_fixture(root, 'replication', ['test', 'test'])
            self.run_analysis(pilot, 'freeze')
            self.run_analysis(pilot, 'test')
            freeze = pilot / 'analysis/development_freeze.json'
            before = sha256(freeze)
            self.run_analysis(replication, 'test', freeze)
            self.assertEqual(sha256(freeze), before)
            rows = json.loads((replication / 'analysis/test_cases.json').read_text())
            self.assertTrue(all(r['selected']['surface_full']['index'] == 1 for r in rows))
            self.assertTrue(all(r['selected']['development_selected']['index'] == 1 for r in rows))

    def test_replication_rejects_shared_images_and_changed_scientific_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilot = experiment_fixture(root, 'pilot', ['development', 'test'])
            overlap = experiment_fixture(root, 'overlap', ['test', 'test'], identity_prefix='pilot')
            self.run_analysis(pilot, 'freeze')
            freeze = pilot / 'analysis/development_freeze.json'
            with self.assertRaisesRegex(ValueError, 'overlaps'):
                self.run_analysis(overlap, 'test', freeze)
            replication = experiment_fixture(root, 'replication', ['test', 'test'])
            protocol = json.loads((replication / 'protocol.json').read_text())
            protocol['mc_timestep'] = 20
            write_json(replication / 'protocol.json', protocol)
            with self.assertRaisesRegex(ValueError, 'scientific settings'):
                self.run_analysis(replication, 'test', freeze)

    def test_validator_recomputes_errors_instead_of_trusting_analysis(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = experiment_fixture(Path(temporary), 'pilot', ['development', 'test'])
            self.run_analysis(experiment, 'freeze')
            self.run_analysis(experiment, 'test')
            cases = json.loads((experiment / 'cases.json').read_text())
            protocol = json.loads((experiment / 'protocol.json').read_text())
            freeze = json.loads((experiment / 'analysis/development_freeze.json').read_text())
            result = validate_evaluation_rows(experiment, experiment / 'full', cases, protocol, freeze)
            self.assertEqual(result['candidate_metric_replays'], 8)
            path = experiment / 'analysis/test_cases.json'
            rows = json.loads(path.read_text())
            rows[0]['metrics'][0]['hidden_core_mae'] = 0.
            write_json(path, rows)
            with self.assertRaisesRegex(ValueError, 'Evaluation number changed'):
                validate_evaluation_rows(experiment, experiment / 'full', cases, protocol, freeze)

    def test_validator_rejects_nonfinite_numbers_and_missing_fields(self):
        with self.assertRaisesRegex(ValueError, 'number changed'):
            assert_equivalent(float('nan'), 0.)
        with self.assertRaisesRegex(ValueError, 'fields changed'):
            assert_equivalent({}, {'mae': 0.})


if __name__ == '__main__':
    unittest.main()
