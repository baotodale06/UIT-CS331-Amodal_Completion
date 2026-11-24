"""External training rejection, native arithmetic and exact-pixel bridge checks."""
import copy
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from amodal_benchmark import shape_external_prior as prior
from amodal_benchmark.dataset import sha256, write_json
from scripts.validate_shape_external_adapter import binary_mask, index_rows, pixel_hash, safe_asset


def fixture(folder, schema):
    """Synthetic metadata in temporary tests only, never a trained checkpoint."""
    method, platform, config_name = prior.PROFILES[schema]
    config_path = prior.ROOT / "configs/research" / config_name
    config = prior.read(config_path)
    baseline, improved = dict(full_iou=.81, hidden_iou=.37), dict(full_iou=.82, hidden_iou=.38)
    contract = dict(config_sha256=sha256(config_path), data_sha256=prior.DATA_SHA256,
        initial_checkpoint_sha256=prior.INITIAL_SHA256, sam_training_manifest_sha256=prior.SAM_SHA256,
        validation_sam_result_sha256=config["validation_diagnostic_result_sha256"],
        sources={name: sha256(prior.ROOT / name) for name in prior.PROFILE_SOURCES[schema]},
        inherited_training_contract=dict(config_sha256=sha256(prior.ROOT / prior.SLURM_CONFIG),
            data_sha256=prior.DATA_SHA256,
            sources={name: sha256(prior.ROOT / name) for name in prior.SLURM_TRAINING_SOURCES}))
    receipt = dict(schema=schema, method=method, platform=platform, complete=True,
        holdout_used=False, benchmark_used=False, epochs_completed=20, contract=contract,
        preflight=False, seed=config["seed"], geometry_policy=config["inference_geometry"],
        selection=config["training"]["selection"], initial_validation=baseline,
        original_quadro_validation=baseline, selection_floor=baseline,
        selected_epoch=2, new_checkpoint_eligible=True, encoder_sha256="synthetic-encoder",
        history=[dict(epoch=epoch, validation=baseline if epoch == 1 else improved,
                      encoder_sha256="synthetic-encoder") for epoch in range(1, 21)],
        runs=[dict(world_size=2, job_id="synthetic", cuda_fft_round_trip=True,
            devices=[dict(rank=rank, gpu="Tesla T4", cuda_fft_round_trip=True) for rank in (0, 1)])],
        runtime=dict(torch="1.13.1", cuda="11.7", cudnn=8500, gpu="NVIDIA GeForce RTX 3060",
                     cuda_fft_round_trip=True))
    (folder / "best.pt").write_bytes(b"SYNTHETIC TEST ONLY")
    receipt["best_sha256"] = sha256(folder / "best.pt")
    write_json(folder / "best-validation-native.json", dict(epoch=2, aggregate=improved, contract=contract))
    receipt["best_validation_sha256"] = sha256(folder / "best-validation-native.json")
    write_json(folder / "training.json", receipt)
    replay = dict(verified=True, objects=1264, images=217, selected_epoch=2, new_checkpoint_eligible=True,
        training_sha256=sha256(folder / "training.json"), checkpoint_sha256=receipt["best_sha256"],
        benchmark_success=False, disposition="verified", state="COMPLETE",
        replay_source_sha256=sha256(prior.ROOT / prior.REPLAY_SOURCES[schema]),
        original_quadro_different_mask_objects=0, original_quadro_different_pixels=0)
    write_json(folder / "replay.json", replay)
    return receipt, replay


class ExternalPriorTests(unittest.TestCase):
    def test_all_profiles_require_completed_selected_independent_evidence(self):
        for schema in prior.PROFILES:
            with self.subTest(schema=schema), TemporaryDirectory() as temporary:
                folder = Path(temporary)
                receipt, replay = fixture(folder, schema)
                self.assertEqual(prior.training_evidence(folder, folder / "replay.json")[0], receipt)
                changes = [("complete", False), ("holdout_used", True), ("benchmark_used", True),
                           ("selected_epoch", 0), ("new_checkpoint_eligible", False), ("history", []),
                           ("method", "shape-plus-surface"), ("platform", "unverified")]
                for key, value in changes:
                    changed = copy.deepcopy(receipt)
                    changed[key] = value
                    write_json(folder / "training.json", changed)
                    # Rebind the receipt to ensure independent policy validation catches the problem.
                    updated = dict(replay, training_sha256=sha256(folder / "training.json"))
                    write_json(folder / "replay.json", updated)
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        prior.training_evidence(folder, folder / "replay.json")
                write_json(folder / "training.json", receipt)
                for key, value in (("verified", False), ("checkpoint_sha256", "changed"),
                                   ("replay_source_sha256", "changed")):
                    write_json(folder / "replay.json", dict(replay, **{key: value}))
                    with self.subTest(replay_key=key), self.assertRaises(ValueError):
                        prior.training_evidence(folder, folder / "replay.json")

    def test_missing_source_and_changed_encoder_are_rejected(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            receipt, replay = fixture(folder, 7)
            for kind in ("source", "parent", "encoder"):
                changed = copy.deepcopy(receipt)
                if kind == "source":
                    changed["contract"]["sources"].pop(next(iter(changed["contract"]["sources"])))
                elif kind == "parent":
                    changed["contract"]["inherited_training_contract"]["data_sha256"] = "changed"
                else:
                    changed["history"][2]["encoder_sha256"] = "changed"
                write_json(folder / "training.json", changed)
                write_json(folder / "replay.json", dict(replay, training_sha256=sha256(folder / "training.json")))
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    prior.training_evidence(folder, folder / "replay.json")

    def test_epoch_zero_regressions_nonfinite_values_and_stable_ties(self):
        with TemporaryDirectory() as temporary:
            for schema in prior.PROFILES:
                receipt, _ = fixture(Path(temporary), schema)
                self.assertEqual(prior.selected_epoch(receipt)["epoch"], 2)
                for row in receipt["history"]:
                    row["validation"] = dict(full_iou=.80, hidden_iou=.99)
                self.assertEqual(prior.selected_epoch(receipt)["epoch"], 0)
                receipt["history"][0]["validation"]["hidden_iou"] = float("nan")
                with self.assertRaises(ValueError):
                    prior.selected_epoch(receipt)

    def test_local_requires_both_original_and_local_floors(self):
        with TemporaryDirectory() as temporary:
            receipt, _ = fixture(Path(temporary), 6)
            receipt["original_quadro_validation"] = dict(full_iou=.83, hidden_iou=.39)
            receipt["selection_floor"] = dict(full_iou=.83, hidden_iou=.39)
            self.assertEqual(prior.selected_epoch(receipt)["epoch"], 0)

    def test_native_inventory_path_and_pixel_checks(self):
        selected = {"a": dict(image_id=1), "b": dict(image_id=2)}
        with self.assertRaises(ValueError):
            index_rows([dict(id="a", image_id=1), dict(id="a", image_id=1)], selected)
        with TemporaryDirectory() as temporary, self.assertRaises(ValueError):
            safe_asset(temporary, "../escaped.png")
        first = np.array([[255, 0], [0, 0]], np.uint8)
        second = np.array([[0, 255], [0, 0]], np.uint8)
        # Equal area (and potentially equal IoU) is insufficient for pixel replay.
        self.assertNotEqual(pixel_hash(first > 0), pixel_hash(second > 0))
        output = io.BytesIO()
        Image.fromarray(first).save(output, format="PNG")
        np.testing.assert_array_equal(binary_mask(output.getvalue(), (2, 2)), first > 0)
        with self.assertRaises(ValueError):
            binary_mask(output.getvalue(), (4, 1))

    def test_empty_support_never_calls_network(self):
        with patch.object(prior, "v2_prior", side_effect=AssertionError("network called")):
            probability, box, _, _ = prior.original_prior(None, np.zeros((3, 4, 3), np.uint8),
                                                        np.zeros((3, 4), bool), 6)
        self.assertFalse(probability.any())
        self.assertIsNone(box)

    def test_native_flags_restore_after_failure_and_preserve_observed_input(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch is optional in CPU environment")
        old = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
               torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        for schema in prior.PROFILES:
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                with prior.native_arithmetic(schema):
                    self.assertFalse(torch.backends.cudnn.deterministic)
                    self.assertFalse(torch.backends.cudnn.benchmark)
                    self.assertEqual(torch.backends.cudnn.allow_tf32, schema != 6)
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                    raise RuntimeError("synthetic")
            self.assertEqual(old, (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
                                  torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32))
        support = np.zeros((9, 13), bool)
        support[3:6, 5:8] = True
        first = np.full((9, 13, 3), 55, np.uint8)
        second = first.copy()
        second[~support] = 244
        captured = []
        def predict(model, rgb, mask):
            captured.append(rgb.copy())
            return np.full(mask.shape, .25, np.float32)
        with patch("amodal_benchmark.shape_v2_runtime.predict_probability", side_effect=predict):
            a = prior.original_prior(None, first, support, 6)
            b = prior.original_prior(None, second, support, 6)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(captured[0], captured[1])
        self.assertTrue(np.all(a[0][support] == 1))


if __name__ == "__main__":
    unittest.main()
