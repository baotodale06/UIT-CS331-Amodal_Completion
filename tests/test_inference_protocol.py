"""Run the pinned PD-MC gates/crops with only expensive model calls replaced."""
import ast
import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
import warnings

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark import inference


def cpu_upstream():
    source = ROOT / "tests/fixtures/pdmc_main.py"
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == inference.UPSTREAM_MAIN_SHA256
    names = {"QueryObject", "find_mask_sides", "check_touch_boundary", "check_valid_query",
             "create_canvas", "find_crop_region", "crop", "aggregate_occluders",
             "run_iteration", "check_occlusion", "filter_out_amodal_segmentation",
             "compute_iou", "compute_offset"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(raw)
    module = types.ModuleType("cpu_pdmc")
    module.__dict__.update(np=np, Image=Image, math=__import__("math"),
        cv2=types.SimpleNamespace(dilate=lambda mask, *a, **k: mask),
        analyze_masks=lambda *a: [], transform_image=lambda image: image, run_sam=None)
    exec(compile(ast.Module(body=[n for n in tree.body if getattr(n, "name", None) in names],
                            type_ignores=[]), str(source), "exec"), module.__dict__)
    module.diffusion_calls = 0
    def diffuse(query, model, lama, image, *args):
        module.diffusion_calls += 1
        return Image.fromarray(image.astype(np.uint8))
    module.run_mixed_context_diffusion = diffuse
    def segment(model, sam, checkpoint, image, *args):
        mask = np.zeros((1, image.height, image.width), dtype=np.uint8)
        mask[0, 65:90, 65:90] = 1
        return np.array(image), mask, ["cat"], [.9]
    module.segment = segment
    return module


class InferenceProtocolTests(unittest.TestCase):
    def patched(self, mode="paper"):
        install = getattr(inference, "install_runtime_hooks", None)
        self.assertTrue(callable(install), "Paper boundary repair and rejection tracing are missing")
        module = cpu_upstream()
        install(module, boundary_check=mode)
        return module

    def complete(self, module, mask, label="cat", score=.9):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            Image.new("RGB", (10, 10)).save(root / "gray.png")
            result = inference.complete_target(module, [None] * 4,
                Image.new("RGB", (100, 100), "red"), np.full((100, 100, 3), 70, np.uint8),
                np.array([mask]), [label], [score], 0, ["cat"], "input.jpg", root,
                "sam.pth", root / "gray.png", 35, 5)
            png = (root / "completion.png").read_bytes() if (root / "completion.png").exists() else None
            return result, png

    def test_released_boundary_check_rejects_a_large_object_touching_the_edge(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:40, 30:70] = 1
        self.assertEqual(cpu_upstream().check_touch_boundary(mask.copy()), set())
        repaired = self.patched()
        self.assertEqual(repaired.check_touch_boundary(mask.copy()), {"top"})
        result, _ = self.complete(repaired, mask)
        self.assertEqual(result["status"], "success")
        self.assertEqual(repaired.diffusion_calls, 1)
        self.assertEqual(result["diagnostics"]["iterations"][0]["boundary_sides"], ["top"])

    def test_each_boundary_and_interior_follow_the_paper_ten_pixel_rule(self):
        module = self.patched()
        for region, sides in [((slice(5, 45), slice(30, 70)), {"top"}),
                              ((slice(60, 100), slice(30, 70)), {"bottom"}),
                              ((slice(30, 70), slice(0, 40)), {"left"}),
                              ((slice(30, 70), slice(60, 100)), {"right"}),
                              ((slice(10, 45), slice(30, 70)), {"top"}),
                              ((slice(11, 45), slice(30, 70)), set()),
                              ((slice(50, 91), slice(30, 70)), {"bottom"}),
                              ((slice(50, 90), slice(30, 70)), set()),
                              ((slice(25, 75), slice(25, 75)), set())]:
            mask = np.zeros((100, 100), np.uint8)
            mask[region] = 1
            self.assertEqual(module.check_touch_boundary(mask), sides)

    def test_released_mode_preserves_old_boundary_stop_with_an_explicit_reason(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[:40, 30:70] = 1
        module = self.patched("released")
        result, png = self.complete(module, mask)
        self.assertEqual(result["status"], "no_completion")
        self.assertEqual(result["failure_reason"], "no_occlusion_detected")
        self.assertEqual(module.diffusion_calls, 0)
        self.assertIsNone(png)

    def test_query_rejections_record_the_actual_upstream_gate(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[25:75, 25:75] = 1
        tiny = np.zeros_like(mask)
        tiny[40:45, 40:45] = 1
        for query, label, score, reason in [(tiny, "cat", .9, "query_mask_too_small"),
                (mask, "unknown", .9, "unknown_detected_class"),
                (mask, "cat", .2, "detector_score_below_threshold"),
                (mask, "cat", .9, "no_occlusion_detected")]:
            with self.subTest(reason=reason):
                module = self.patched()
                result, _ = self.complete(module, query, label, score)
                self.assertEqual(result["failure_reason"], reason)
                self.assertEqual(result["detected_class"], label)
                self.assertEqual(result["detector_score"], score)
                self.assertEqual(result["iterations"], 0)
                self.assertEqual(module.diffusion_calls, 0)

    def test_failed_output_segmentation_is_distinct_from_pre_diffusion_rejection(self):
        module = self.patched()
        module.segment = lambda *a: (None, None, [], [])
        mask = np.zeros((100, 100), np.uint8)
        mask[:40, 30:70] = 1
        result, _ = self.complete(module, mask)
        self.assertEqual(result["failure_reason"], "completion_not_detected")
        self.assertEqual(result["iterations"], 1)
        self.assertEqual(module.diffusion_calls, 1)

    def test_empty_completion_array_follows_the_pinned_no_segmentation_path(self):
        source = ROOT / "tests/fixtures/pdmc_main.py"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source.read_bytes())
        analyze = next(n for n in tree.body if getattr(n, "name", None) == "analyze_masks")
        namespace = {"np": np}
        exec(compile(ast.Module(body=[analyze], type_ignores=[]), str(source), "exec"), namespace)
        module = cpu_upstream()
        module.segment = lambda *a: (None, np.empty((0, 100, 100), dtype=bool), [], [])
        module.analyze_masks = namespace["analyze_masks"]
        # Reproduce the real pinned function's masks[0] failure before hooks.
        query = types.SimpleNamespace(run_iter=True)
        args = (Image.new("RGB", (100, 100)), np.ones((100, 100)), "cat", 0,
                None, "sam", None, ["cat"], query, 0, 100, 0, 100)
        with self.assertRaises(IndexError):
            module.check_occlusion(*args)
        inference.install_runtime_hooks(module, "released")
        original_segment = module.segment
        result = module.check_occlusion(*args)
        self.assertFalse(result.run_iter)
        self.assertIsNone(result.amodal_segmentation)
        self.assertIs(module.segment, original_segment)

    def test_empty_annotation_queries_report_a_failed_completion_in_both_boundary_modes(self):
        from amodal_benchmark.detection import expanded_segmenter
        for mode in ("released", "paper"):
            with self.subTest(mode=mode):
                module = self.patched(mode)
                def no_detections(*args, **kwargs):
                    return None, None, [], []
                gdino = types.SimpleNamespace(tokenizer=lambda *a, **k: {"input_ids": [0, 1]}, max_text_len=256)
                expanded = expanded_segmenter(no_detections, ["cat"], ["plate"])
                module.segment = expanded
                mask = np.zeros((100, 100), np.uint8)
                mask[25:75, :40] = 1  # Left edge is recognized in both modes.
                with tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    Image.new("RGB", (10, 10)).save(root / "gray.png")
                    result = inference.complete_target(module, [gdino, None, None, None],
                        Image.new("RGB", (100, 100)), np.zeros((100, 100, 3), np.uint8),
                        np.array([mask]), ["cat"], [.9], 0, ["cat", "plate"], "input.jpg", root,
                        "sam", root / "gray.png", 35, 5)
                    self.assertEqual(result["status"], "no_completion")
                    self.assertEqual(result["failure_reason"], "completion_not_detected")
                    self.assertEqual(result["iterations"], 1)
                    self.assertEqual(module.diffusion_calls, 1)
                    self.assertFalse((root / "completion.png").exists())
                    trace = result["diagnostics"]["iterations"][0]
                    self.assertEqual(trace["completion_segmentation_failure"], "empty_mask_array")
                    self.assertIs(module.segment, expanded)

    def test_completion_segmenter_is_restored_if_a_model_raises(self):
        module = self.patched()
        def fail(*args, **kwargs):
            raise RuntimeError("unrelated model failure")
        module.segment = fail
        args = (Image.new("RGB", (100, 100)), np.ones((100, 100)), "cat", 0,
                None, "sam", None, ["cat"], types.SimpleNamespace(), 0, 100, 0, 100)
        with self.assertRaisesRegex(RuntimeError, "unrelated model failure"):
            module.check_occlusion(*args)
        self.assertIs(module.segment, fail)

    def test_nonempty_completion_retains_unwrapped_upstream_pixels_and_geometry(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[25:75, 25:75] = 1
        occluder = np.zeros_like(mask)
        occluder[30:70, 75:90] = 1
        outputs = []
        for patched in (False, True):
            module = cpu_upstream()
            module.analyze_masks = lambda *a: [occluder] if a[1].shape[:2] == (100, 100) else []
            if patched:
                inference.install_runtime_hooks(module, "released")
            result, png = self.complete(module, mask)
            self.assertEqual(result["status"], "success")
            outputs.append((png, {k: result[k] for k in (
                "status", "iterations", "geometry", "detected_class", "detector_score", "iteration_limit_reached")}))
        self.assertEqual(outputs[0], outputs[1])

    def test_supported_interior_completion_retains_identical_pixels_and_crop(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[25:75, 25:75] = 1
        outputs = []
        for mode in ("released", "paper"):
            module = self.patched(mode)
            occluder = np.zeros_like(mask)
            occluder[30:70, 75:90] = 1
            module.analyze_masks = lambda *a: [occluder] if a[1].shape[:2] == (100, 100) else []
            result, png = self.complete(module, mask)
            self.assertEqual(result["status"], "success")
            outputs.append((png, result["geometry"], result["iterations"]))
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
