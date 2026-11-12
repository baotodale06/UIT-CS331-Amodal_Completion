"""CPU checks for dataset identity, target selection, and honest denominators."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import io
import tarfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("amodal_benchmark"),
                             "The benchmark adapter has not been implemented")
        from amodal_benchmark import dataset
        self.dataset = dataset

    def test_annotation_names_map_to_original_images_without_losing_targets(self):
        cases = [
            ("VG", "VG/1159295_bus.jpg", "bus", "1159295.jpg"),
            ("COCOA", "./COCO_test2014_000000296835_toothbrush.jpg", "toothbrush",
             "COCO_test2014_000000296835.jpg"),
            ("laion", "laion/00000/000000115_violin.jpg", "violin", "000000115.jpg"),
            ("free_images", "website/coins-1015125_640_potato.jpg", "potato",
             "coins-1015125_640_potato.jpg"),
        ]
        for source, name, label, original in cases:
            with self.subTest(source=source):
                row = self.dataset.target_record(source, {"filename": name, "labels": [label]})
                self.assertEqual(row["image"], f"images/{source}/{original}")
                self.assertEqual(row["id"], f"{source}/{Path(name).stem}")
                self.assertEqual(row["label"], label)

    def test_downloaded_annotations_match_the_paper_not_generic_coco(self):
        rows, counts = self.dataset.read_official(ROOT / "tests/fixtures")
        self.assertEqual(len(rows), 2565)
        self.assertEqual(len({r["label"] for r in rows}), 553)
        self.assertEqual({s: (v["images"], v["instances"], v["classes"])
                          for s, v in counts.items()},
                         {"VG": (1234, 1294, 269), "COCOA": (751, 845, 256),
                          "free_images": (228, 249, 147), "laion": (166, 177, 97)})

    def test_missing_images_stay_in_manifest_and_corrupt_jpegs_are_reported(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "images/VG").mkdir(parents=True)
            (root / "images/VG/4.jpg").write_bytes(b"truncated jpeg")
            rows = [self.dataset.target_record("VG", {"filename": "4_bus.jpg", "labels": ["bus"]}),
                    self.dataset.target_record("VG", {"filename": "16_car.jpg", "labels": ["car"]})]
            report = self.dataset.audit_files(root, rows, decode=True)
            self.assertEqual(report["total_targets"], 2)
            self.assertEqual(report["missing_images"], ["images/VG/16.jpg"])
            self.assertEqual(report["corrupt_images"], ["images/VG/4.jpg"])


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("amodal_benchmark"),
                             "The benchmark adapter has not been implemented")
        from amodal_benchmark import protocol
        self.p = protocol

    def test_transparent_pixels_are_composited_on_white(self):
        a = np.array([[[10, 20, 30, 255], [255, 0, 0, 0]]], dtype=np.uint8)
        self.assertEqual(self.p.white_rgb(a).tolist(), [[[10, 20, 30], [255, 255, 255]]])

    def test_failed_completion_penalties_are_included_without_fabricating_clip(self):
        rows = [{"source": "VG", "status": "success", "CLIP": 30., "LPIPS": .2,
                 "Feature Similarity": .8, "SSIM": .9},
                {"source": "VG", "status": "not_detected", **self.p.failure_metrics()}]
        s = self.p.aggregate(rows)["Overall"]
        self.assertEqual(s["targets"], 2)
        self.assertEqual(s["successes"], 1)
        self.assertAlmostEqual(s["LPIPS"], .6)
        self.assertAlmostEqual(s["Feature Similarity"], .4)
        self.assertAlmostEqual(s["SSIM"], .45)
        self.assertEqual(s["CLIP"], 30.)
        self.assertEqual(s["CLIP_n"], 1)

    def test_macro_source_average_is_not_used_for_overall(self):
        rows = [{"source": s, "status": "success", "CLIP": c, "LPIPS": .2,
                 "Feature Similarity": .8, "SSIM": .9}
                for s, c in [("VG", 10.), ("VG", 20.), ("laion", 60.)]]
        self.assertEqual(self.p.aggregate(rows)["Overall"]["CLIP"], 30.)

    def test_detailed_outcomes_separate_iteration_caps_and_adapter_rejections(self):
        rows = [{"source": "VG", "status": "success", "iteration_limit_reached": flag}
                for flag in (False, "False", "True")]
        rows.append({"source": "VG", "status": "success"})
        for status, reason in (("not_detected", "label_not_matched"),
                               ("ambiguous_target", "multiple_distinct_instances"),
                               ("not_detected", "segmentation_unavailable"),
                               ("no_completion", "query_mask_too_small")):
            rows.append({"source": "VG", "status": status, "failure_reason": reason})
        stat = self.p.aggregate(rows, detailed=True)["Overall"]
        self.assertEqual((stat["successes"], stat["completed_outputs"],
                          stat["iteration_limited_outputs"], stat["completion_state_unknown"]), (4, 2, 1, 1))
        self.assertEqual((stat["target_selection_failures"], stat["initial_segmentation_failures"],
                          stat["pdmc_failures"]), (2, 1, 1))
        self.assertEqual(stat["failure_stage_counts"], {"target_selection": 2, "initial_segmentation": 1,
                                                      "query_filter": 1})

    def test_summary_breaks_down_actual_rejection_reasons_and_keeps_exclusions_separate(self):
        rows = [{"source": "VG", "status": status, **extra} for status, extra in (
            ("success", {"iterations": 1}),
            ("no_completion", {"iterations": 0, "failure_reason": "query_mask_too_small"}),
            ("not_detected", {"failure_reason": "reference_iou_below_threshold"}),
            ("reference_failed", {}))]
        stat = self.p.aggregate(rows)["Overall"]
        self.assertEqual(stat["failure_reasons"], {"query_mask_too_small": 1,
                                                "reference_iou_below_threshold": 1})
        self.assertEqual(stat["status_counts"]["reference_failed"], 1)
        self.assertAlmostEqual(stat["failure_rate"], 2 / 3)

    def test_same_class_instances_require_identity_evidence(self):
        masks = np.zeros((2, 5, 5), dtype=np.uint8)
        masks[0, :2, :2] = 1
        masks[1, 3:, 3:] = 1
        self.assertEqual(self.p.choose_target(masks, ["cat", "cat"], "cat"), (None, "ambiguous_target"))
        self.assertEqual(self.p.choose_target(masks, ["cat", "cat"], "cat", masks[1]),
                         (1, "reference_iou"))

    def test_label_fallback_uses_word_boundaries_not_substrings(self):
        masks = np.ones((1, 4, 4), dtype=np.uint8)
        self.assertEqual(self.p.choose_target(masks, ["car"], "white car"), (0, "label_match"))
        self.assertEqual(self.p.choose_target(masks, ["car"], "carpet"), (None, "not_detected"))

    def test_best_reference_match_does_not_reject_duplicate_detections(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[4:16, 4:16] = 1
        masks = np.array([mask, mask])
        self.assertEqual(self.p.choose_target(masks, ["cat", "cat"], "cat", mask),
                         (None, "ambiguous_target"))
        details = {}
        result = self.p.choose_target(masks, ["cat", "cat"], "cat", mask,
                                     reference_policy="best", diagnostics=details)
        self.assertEqual(result, (0, "reference_iou"))
        self.assertEqual(details["best_iou"], 1.)
        self.assertEqual(details["candidates"][1]["class"], "cat")

    def test_best_reference_match_keeps_the_identity_threshold_and_detected_class(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[4:16, 4:16] = 1
        details = {}
        self.assertEqual(self.p.choose_target(np.array([mask]), ["bed"], "shirt", mask,
            reference_policy="best", diagnostics=details), (0, "reference_iou"))
        self.assertEqual(details["candidates"][0]["class"], "bed")
        self.assertEqual(self.p.choose_target(np.array([np.ones_like(mask)]), ["bed"], "shirt", mask,
            reference_policy="best"), (None, "not_detected"))
        reference = np.zeros((4, 4), dtype=np.uint8)
        reference[:2, :2] = 1
        candidate = np.zeros_like(reference)
        candidate[:2] = 1  # Four visible pixels / eight union pixels = exactly 0.5.
        self.assertEqual(self.p.choose_target(np.array([candidate]), ["cat"], "cat", reference,
            reference_policy="best"), (0, "reference_iou"))
        candidate[2, 0] = 1
        self.assertEqual(self.p.choose_target(np.array([candidate]), ["cat"], "cat", reference,
            reference_policy="best"), (None, "not_detected"))

    def test_resume_refuses_different_model_configuration(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run.json"
            self.p.ensure_run_config(path, {"seed": 0, "mc_timestep": 35})
            self.p.ensure_run_config(path, {"seed": 0, "mc_timestep": 35})
            with self.assertRaisesRegex(ValueError, "configuration"):
                self.p.ensure_run_config(path, {"seed": 0, "mc_timestep": 20})

    def test_alignment_places_an_object_crop_at_its_original_coordinates(self):
        self.assertTrue(hasattr(self.p, "align_rgba"), "Prediction/reference geometry alignment is missing")
        reference = np.zeros((4, 4, 4), dtype=np.uint8)
        reference[1:3, 1:3] = [255, 0, 0, 255]
        prediction = np.full((2, 2, 4), [255, 0, 0, 255], dtype=np.uint8)
        ref, pred = self.p.align_rgba(reference, prediction, {"original_size": [4, 4], "offset_rc": [1, 1]})
        np.testing.assert_array_equal(ref, pred)

    def test_alignment_keeps_completion_beyond_the_original_boundary(self):
        self.assertTrue(hasattr(self.p, "align_rgba"), "Prediction/reference geometry alignment is missing")
        reference = np.zeros((4, 4, 4), dtype=np.uint8)
        reference[1, 1] = [255, 0, 0, 255]
        prediction = np.full((2, 3, 4), [0, 0, 255, 255], dtype=np.uint8)
        ref, pred = self.p.align_rgba(reference, prediction, {"original_size": [4, 4], "offset_rc": [-1, -2]})
        self.assertEqual(ref.shape, (5, 6, 4))
        self.assertEqual(pred[0, 0].tolist(), [0, 0, 255, 255])
        self.assertEqual(ref[2, 3].tolist(), [255, 0, 0, 255])

    def test_resume_retries_previously_missing_inputs_and_checks_selection_reference(self):
        self.assertTrue(hasattr(self.p, "read_cached_result"), "Resume validation helper is missing")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "result.json"
            row = {"id": "laion/000000115_violin", "label": "violin"}
            path.write_text(json.dumps({**row, "status": "missing_image", "input_sha256": None}))
            self.assertIsNone(self.p.read_cached_result(path, row, "new-input-hash"))
            path.write_text(json.dumps({**row, "status": "not_detected", "input_sha256": "input",
                                        "selection_reference_sha256": "old-reference"}))
            with self.assertRaisesRegex(ValueError, "reference"):
                self.p.read_cached_result(path, row, "input", "changed-reference")


class PreparationTests(unittest.TestCase):
    def test_supplied_release_zip_reconstructs_all_official_annotations(self):
        archive = ROOT / "Open-World Amodal Appearance Completion-20261015T075717Z-1-001.zip"
        if not archive.is_file():
            self.skipTest("Optional original release archive is not installed")
        with tempfile.TemporaryDirectory() as td:
            result = subprocess.run([sys.executable, str(ROOT / "prepare_open_world.py"),
                                     "--root", td, "--release-zip", str(archive),
                                     "--allow-incomplete"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads((Path(td) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(data["targets"]), 2565)
            self.assertEqual(len(list((Path(td) / "images/free_images").glob("*.jpg"))), 249)
            report = json.loads((Path(td) / "verification.json").read_text())
            self.assertFalse(report["images_complete"])
            self.assertEqual(report["missing_targets_by_source"]["laion"], 177)

    def test_laion_import_keeps_only_requested_images_and_records_archive_hash(self):
        self.assertTrue((ROOT / "prepare_open_world.py").exists(), "Dataset installer is missing")
        import prepare_open_world as prep
        from amodal_benchmark.dataset import target_record
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            buf = io.BytesIO()
            Image.new("RGB", (12, 12), "white").save(buf, format="JPEG")
            path = root / "00000.tar"
            with tarfile.open(path, "w") as tar:
                for name in ("000000115.jpg", "000000116.jpg", "../escape.jpg"):
                    data = buf.getvalue()
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            row = target_record("laion", {"filename": "000000115_violin.jpg", "labels": ["violin"]})
            record = prep.import_laion(root, path, [row])
            self.assertEqual(list((root / "images/laion").iterdir()), [root / "images/laion/000000115.jpg"])
            self.assertEqual(len(record["archive_sha256"]), 64)
            self.assertFalse((root / "escape.jpg").exists())


class RunnerTests(unittest.TestCase):
    def test_cpu_preflight_reports_missing_data_before_importing_cuda_dependencies(self):
        p = ROOT / "evaluate_open_world.py"
        self.assertTrue(p.is_file(), "Evaluation entry point is missing")
        result = subprocess.run([sys.executable, str(p), "--dataset-root",
                                 str(ROOT / "tests/fixtures"), "--output-dir",
                                 tempfile.gettempdir(), "--check-only"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("input images missing", result.stderr.lower())
        self.assertNotIn("No module named 'torch'", result.stderr)

    def test_upstream_loop_and_rgba_output_use_the_actual_selected_target(self):
        self.assertIsNotNone(importlib.util.find_spec("amodal_benchmark.inference"),
                             "Pipeline integration is missing")
        from amodal_benchmark.inference import complete_target
        from types import SimpleNamespace
        class FakeUpstream:
            # Only the expensive model call is substituted; the adapter performs
            # target selection, iteration limits, and actual PNG/JPEG writes.
            QueryObject = lambda self, *args: SimpleNamespace(run_iter=True, iter_id=0,
                output_img_dir=args[-1], query_mask_canvas=None, init_outpaint_mask_canvas=None)
            compute_offset = lambda self, *args: (0, 0)
            def run_iteration(self, query, *args, **kwargs):
                query.iter_id += 1
                query.amodal_completion = Image.new("RGB", (12, 12), "red")
                query.amodal_segmentation = np.zeros((12, 12), dtype=np.uint8)
                query.amodal_segmentation[2:10, 2:10] = 1
                return query
        with tempfile.TemporaryDirectory() as td:
            result = complete_target(FakeUpstream(), [None] * 4,
                                     Image.new("RGB", (12, 12)), np.ones((12, 12, 3), dtype=np.uint8),
                                     np.ones((1, 12, 12), dtype=np.uint8), ["cat"], [.9],
                                     0, ["cat"], "input.jpg", Path(td), "sam.pth", "gray.jpg", 35, 1)
            self.assertEqual(result["iterations"], 2)  # matches upstream iter_id > max_iter_id
            self.assertEqual(result["geometry"], {"original_size": [12, 12], "offset_rc": [0, 0]})
            with Image.open(Path(td) / "completion.png") as im:
                self.assertEqual(im.mode, "RGBA")
                self.assertEqual(im.getpixel((0, 0))[3], 0)
                self.assertEqual(im.getpixel((5, 5))[3], 255)


if __name__ == "__main__":
    unittest.main()
