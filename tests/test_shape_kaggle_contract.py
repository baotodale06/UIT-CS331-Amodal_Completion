"""Check the isolated Kaggle platform delta without training or GPU allocation."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class KaggleContractTests(unittest.TestCase):
    def test_only_declared_training_resource_settings_differ(self):
        original = json.loads((ROOT / 'configs/research/shape-sam-constrained-20261123.json').read_bytes())
        kaggle = json.loads((ROOT / 'configs/research/shape-sam-kaggle-20261123.json').read_bytes())
        self.assertEqual(kaggle['training_platform'], 'Kaggle')
        self.assertEqual(kaggle['training'], {**original['training'], 'world_size':2, 'workers_per_rank':2})
        for key in ('seed', 'initial_checkpoint_sha256', 'validation_diagnostic_result_sha256', 'inference_geometry'):
            self.assertEqual(kaggle[key], original[key])

    def test_two_rank_contract_rejects_wrong_platform_split_or_world(self):
        try:
            from scripts.train_shape_kaggle import validate_contract
        except ImportError as error:
            if error.name != 'torch':
                raise
            self.skipTest('Torch optional in local CPU tooling')
        config = json.loads((ROOT / 'configs/research/shape-sam-kaggle-20261123.json').read_bytes())
        manifest = dict(holdout_packaged=False, rows=[dict(split='train'),dict(split='validation')])
        validate_contract(config, manifest, 2)
        with self.assertRaises(ValueError):
            validate_contract(config, manifest, 4)
        with self.assertRaises(ValueError):
            validate_contract({**config,'training_platform':'Slurm'}, manifest, 2)
        with self.assertRaises(ValueError):
            validate_contract(config, {**manifest,'holdout_packaged':True}, 2)


if __name__ == '__main__':
    unittest.main()
