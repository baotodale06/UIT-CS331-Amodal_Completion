"""Regression coverage for the explicitly authorized Slurm training amendment."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


class ShapeSlurmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from scripts import train_shape_v2_slurm
        except ImportError:
            raise unittest.SkipTest("Optional torch environment required")
        cls.trainer = train_shape_v2_slurm
        cls.config = json.loads((train_shape_v2_slurm.ROOT / train_shape_v2_slurm.CONFIG).read_bytes())

    def test_contract_preserves_model_data_selection_and_global_batch(self):
        original = json.loads((self.trainer.ROOT / "configs/research/shape-v2-20261123.json").read_bytes())
        for key in ("seed", "split", "success", "planned_inference_policy"):
            self.assertEqual(self.config[key], original[key])
        for key, value in original["training"].items():
            self.assertEqual(self.config["training"][key], value)

    def test_allocation_and_holdout_checks(self):
        manifest = dict(holdout_packaged=False, rows=[dict(split="train"), dict(split="validation")])
        self.trainer.validate_contract(self.config, manifest, 4)
        for world in (1, 2, 8):
            with self.assertRaisesRegex(ValueError, "Allocation"):
                self.trainer.validate_contract(self.config, manifest, world)
        for split in ("holdout", "test"):
            contaminated = copy.deepcopy(manifest)
            contaminated["rows"].append(dict(split=split))
            with self.assertRaisesRegex(ValueError, "physically isolated"):
                self.trainer.validate_contract(self.config, contaminated, 4)

    def test_no_cpu_or_unallocated_training(self):
        with TemporaryDirectory() as folder:
            out = Path(folder) / "not-created"
            with patch.dict("os.environ", {}, clear=True), patch("sys.argv", ["train", "--data", "missing", "--output", str(out)]):
                with self.assertRaisesRegex(RuntimeError, "real Slurm CUDA"):
                    self.trainer.main()
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
