"""Regression checks for data leakage, geometry-only selection, and the success gate."""
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import zipfile
import numpy as np

from amodal_benchmark.shape_v2 import observation_tensor, score_masks, visible_box
from scripts.assess_shape_v2 import assess
from scripts.prepare_shape_v2_data import excluded_coco_ids, image_split


class ShapeV2Tests(unittest.TestCase):
    def test_hidden_rgb_cannot_change_shape_input_even_at_resize_boundaries(self):
        visible = np.zeros((19, 25), bool)
        visible[3:15, 8:13] = True
        first = np.full((19, 25, 3), 70, np.uint8)
        second = first.copy()
        second[~visible] = 255
        np.testing.assert_array_equal(observation_tensor(first, visible, size=32),
                                      observation_tensor(second, visible, size=32))
        self.assertTrue(np.isfinite(observation_tensor(first, visible)).all())

    def test_crop_uses_visible_support_and_allows_outpaint(self):
        visible = np.zeros((10, 12), bool)
        visible[:4, :2] = True
        self.assertEqual(visible_box(visible), (-3, -2, 5, 6))
        with self.assertRaises(ValueError):
            visible_box(np.zeros_like(visible))

    def test_soft_score_is_binary_iou_at_probability_extremes(self):
        prior = np.array([[1., 1.], [0., 0.]])
        candidate = np.array([[1, 0], [1, 0]], bool)
        self.assertEqual(score_masks(prior, [candidate, np.zeros((2, 2), bool)]), [1/3, None])
        with self.assertRaises(ValueError):
            score_masks(prior * float("nan"), [candidate])
        with self.assertRaises(ValueError):
            score_masks(prior, [np.ones((3, 3), bool)])

    def test_image_split_is_stable_and_excludes_coco_identity(self):
        self.assertEqual(image_split(321), image_split(321))
        self.assertEqual(excluded_coco_ids([dict(source="COCOA", original_id="COCO_train2014_000000000321"),
                                          dict(source="VG", original_id="321")]), {321})
        self.assertEqual(set(image_split(i) for i in range(100)), {"train", "validation", "holdout"})

    def test_gate_requires_every_metric_and_preserves_failed_rows(self):
        base = {"ok": {"CLIP": "20", "LPIPS": ".4", "Feature Similarity": ".5", "SSIM": ".5"},
                "failed": {"CLIP": "", "LPIPS": "1", "Feature Similarity": "0", "SSIM": "0"}}
        candidate = {key:dict(row) for key,row in base.items()}
        candidate["ok"].update({"CLIP": "21", "LPIPS": ".3", "Feature Similarity": ".55", "SSIM": ".55"})
        result = assess(base, candidate, ["ok", "failed"])
        self.assertTrue(result["eligible_for_success_claim"])
        self.assertEqual(result["metrics"]["LPIPS"]["control_n"], 2)
        candidate["ok"]["SSIM"] = ".51"
        self.assertFalse(assess(base, candidate, ["ok", "failed"])["eligible_for_success_claim"])
        candidate["failed"]["LPIPS"] = ""
        with self.assertRaisesRegex(ValueError, "Missing failure-inclusive"):
            assess(base, candidate, ["ok", "failed"])
        with self.assertRaisesRegex(ValueError, "All expected"):
            assess(base, {"ok": candidate["ok"]}, ["ok", "failed"])

    def test_network_and_hidden_loss_work_on_cpu_without_pretrained_download(self):
        try:
            import torch
            import torchvision
        except ImportError:
            self.skipTest("CPU torch/torchvision are optional for the core test environment")
        from amodal_benchmark.shape_v2 import make_network, shape_loss
        torch.set_num_threads(1)
        model = make_network(pretrained=False).eval()
        x = torch.zeros(1, 5, 64, 64)
        x[:, 3, 20:40, 20:40] = 1
        with torch.inference_mode():
            logits = model(x)
            loss = shape_loss(logits, x[:, 3:4], x[:, 3:4])
        self.assertEqual(tuple(logits.shape), (1, 1, 64, 64))
        self.assertTrue(torch.isfinite(loss).item())

    def test_training_package_contains_no_holdout_pixels_and_notebook_compiles(self):
        from amodal_benchmark.dataset import sha256, write_json
        from scripts import package_shape_v2_colab
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, out = root/"data", root/"package"
            data.mkdir()
            rows, hashes = [], {}
            for index, split in enumerate(("train", "validation", "holdout")):
                names = {"image": f"image-{index}.png", "mask": f"mask-{index}.png"}
                for name in names.values():
                    (data/name).write_bytes(name.encode())
                    hashes[name] = sha256(data/name)
                rows.append(dict(id=str(index), image_id=index, split=split, **names))
            write_json(data/"manifest.json", dict(status="complete", rows=rows, artifacts=hashes,
                                                   split_images={"train":1, "validation":1, "holdout":1}))
            with patch("sys.argv", ["package", "--data", str(data), "--output", str(out)]), patch("builtins.print"):
                package_shape_v2_colab.main()
            with zipfile.ZipFile(out/"shape-v2-training-data.zip") as archive:
                self.assertNotIn("data/image-2.png", archive.namelist())
                self.assertNotIn("data/mask-2.png", archive.namelist())
                manifest = json.loads(archive.read("data/manifest.json"))
                self.assertFalse(manifest["holdout_packaged"])
                self.assertEqual(manifest["counts"], {"train":1, "validation":1})
                self.assertEqual(manifest["split_images"], {"train":1, "validation":1})
            notebook = json.loads((out/"shape-v2-colab.ipynb").read_bytes())
            for row in notebook["cells"]:
                if row["cell_type"] == "code":
                    compile("".join(row["source"]), row["id"], "exec")

    def test_validation_loader_does_not_use_hidden_labels_to_define_input_crop(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU torch is optional for the core test environment")
        from PIL import Image
        from scripts.train_shape_v2 import AmodalData
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            visible = np.zeros((40, 40), bool)
            visible[12:24, 12:24] = True
            full = visible.copy()
            full[5:35, 5:35] = True
            encoded = full.astype(np.uint8) + 2*visible.astype(np.uint8)
            rgb = np.full((40, 40, 3), 70, np.uint8)
            Image.fromarray(rgb).save(root/"rgb.png")
            Image.fromarray(encoded).save(root/"mask.png")
            row = dict(image="rgb.png", mask="mask.png", image_id=1)
            loader = AmodalData(root, [row], False, size=64)
            first, truth_first, _ = loader[0]
            rgb[~visible] = 255
            Image.fromarray(rgb).save(root/"rgb.png")
            Image.fromarray(visible.astype(np.uint8)*3).save(root/"mask.png")
            second, truth_second, _ = loader[0]
            self.assertTrue(torch.equal(first, second))
            self.assertFalse(torch.equal(truth_first, truth_second))

    def test_training_refuses_cpu_before_loading_data_or_creating_outputs(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU torch is optional for the core test environment")
        from scripts import train_shape_v2
        with TemporaryDirectory() as temporary:
            out = Path(temporary)/"must-not-exist"
            with patch("sys.argv", ["train", "--data", "missing", "--output", str(out), "--expected-gpu", "H100"]), \
                    patch.object(torch.cuda, "is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "authorized Google Colab GPU"):
                    train_shape_v2.main()
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
