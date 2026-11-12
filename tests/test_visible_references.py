"""Test the visible-reference preparation recovered from the released code."""
import importlib.util
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class VisibleReferenceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("amodal_benchmark.references"),
                             "LISA visible-reference preparation is missing")
        from amodal_benchmark import references
        self.refs = references

    def test_erodes_mask_once_with_five_pixel_kernel_without_changing_rgb(self):
        rgb = np.full((100, 120, 3), [193, 41, 17], dtype=np.uint8)
        mask = np.zeros((100, 120), dtype=bool)
        mask[30:60, 40:80] = True
        full, cropped, geometry = self.refs.visible_rgba(rgb, mask)
        np.testing.assert_array_equal(full[..., :3], rgb)
        self.assertEqual(int((full[..., 3] > 0).sum()), 936)  # 26 x 36 after erosion
        self.assertEqual(geometry, {"original_size": [120, 100], "offset_rc": [-35, -20]})
        self.assertEqual(cropped.shape, (159, 159, 4))
        self.assertEqual(cropped[65, 60].tolist(), [193, 41, 17, 0])  # original mask corner eroded
        self.assertEqual(cropped[67, 62].tolist(), [193, 41, 17, 255])

    def test_visible_erosion_removes_pixels_at_source_boundary_on_zero_mask_canvas(self):
        rgb = np.full((100, 100, 3), 80, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=bool)
        mask[:20, :20] = True
        full, _, _ = self.refs.visible_rgba(rgb, mask)
        self.assertEqual(int((full[..., 3] > 0).sum()), 256)  # 16 x 16, includes top/left erosion
        self.assertEqual(full[0, 0, 3], 0)
        self.assertEqual(full[2, 2, 3], 255)

    def test_empty_lisa_prediction_is_recorded_as_a_reference_failure(self):
        with tempfile.TemporaryDirectory() as td:
            row = {"id": "VG/4_bus", "source": "VG", "label": "bus", "reference": "VG/4_bus.png"}
            record = self.refs.save_reference(Path(td), row, np.zeros((12, 16, 3), dtype=np.uint8),
                                              None, "input-hash", {"seed": 0})
            self.assertEqual(record["status"], "empty_mask")
            self.assertEqual(record["mask_pixels"], 0)
            self.assertTrue((Path(td) / "VG/4_bus.png").is_file())
            self.assertTrue((Path(td) / "masks/VG/4_bus.png").is_file())

    def test_reference_cache_checks_input_identity_and_generated_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            row = {"id": "VG/4_bus", "source": "VG", "label": "bus", "reference": "VG/4_bus.png"}
            mask = np.zeros((100, 120), dtype=bool)
            mask[30:60, 40:80] = True
            self.refs.save_reference(folder, row, np.ones((100, 120, 3), dtype=np.uint8),
                                     mask, "input-hash", {"seed": 0})
            self.assertIsNotNone(self.refs.cached_reference(folder, row, "input-hash"))
            with self.assertRaisesRegex(ValueError, "input"):
                self.refs.cached_reference(folder, row, "different-input")
            (folder / "masks/VG/4_bus.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                self.refs.cached_reference(folder, row, "input-hash")

    def test_raw_mask_is_saved_separately_from_eroded_reference_alpha(self):
        with tempfile.TemporaryDirectory() as td:
            row = {"id": "VG/4_bus", "source": "VG", "label": "bus", "reference": "VG/4_bus.png"}
            mask = np.zeros((100, 120), dtype=bool)
            mask[30:60, 40:80] = True
            self.refs.save_reference(Path(td), row, np.zeros((100, 120, 3), dtype=np.uint8),
                                     mask, "input-hash", {"seed": 0})
            with Image.open(Path(td) / "masks/VG/4_bus.png") as image:
                self.assertEqual(int((np.asarray(image) > 0).sum()), 1200)
            with Image.open(Path(td) / "VG/4_bus.png") as image:
                self.assertEqual(int((np.asarray(image)[..., 3] > 0).sum()), 936)

    def test_audit_accepts_a_recorded_empty_prediction_without_calling_it_corruption(self):
        from amodal_benchmark.dataset import audit_files
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "images/VG").mkdir(parents=True)
            path = root / "images/VG/4.jpg"
            Image.new("RGB", (16, 12)).save(path)
            from amodal_benchmark.dataset import sha256
            row = {"id": "VG/4_bus", "source": "VG", "label": "bus", "reference": "VG/4_bus.png",
                   "image": "images/VG/4.jpg", "original_id": "4"}
            self.refs.save_reference(root / "visible_references", row,
                                     np.zeros((12, 16, 3), dtype=np.uint8), None, sha256(path), {"seed": 0})
            report = audit_files(root, [row], root / "visible_references", decode=True)
            self.assertEqual(report["corrupt_references"], [])

    def test_reference_failure_is_not_counted_as_a_pdmc_failure(self):
        from amodal_benchmark.protocol import aggregate
        row = {"id": "VG/4_bus", "source": "VG", "status": "reference_failed"}
        stats = aggregate([row])["Overall"]
        self.assertEqual(stats["failures"], 0)
        self.assertEqual(stats["reference_failures"], 1)
        self.assertEqual(stats["LPIPS_n"], 0)


class ReferenceCLITests(unittest.TestCase):
    def test_preparation_preflight_needs_source_images_but_no_gpu_or_prior_references(self):
        script = ROOT / "prepare_visible_references.py"
        self.assertTrue(script.exists(), "Reference-preparation CLI is missing")
        from amodal_benchmark.dataset import read_official
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            shutil.copytree(ROOT / "tests/fixtures/official", dataset / "official")
            for row in read_official(dataset)[0][:2]:
                path = dataset / row["image"]
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (12, 12), "white").save(path)
            result = subprocess.run([sys.executable, str(script), "--dataset-root",
                                     str(dataset), "--sources", "VG",
                                     "--limit", "2", "--check-only"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("2", result.stdout)
        self.assertNotIn("No module named 'torch'", result.stderr)


if __name__ == "__main__":
    unittest.main()
