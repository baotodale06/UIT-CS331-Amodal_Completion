"""Regression checks for denominators, identity joins and occlusion geometry."""
import csv
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from amodal_benchmark.appearance import (FAILURES, appearance_diagnostics, mask_diagnostics,
    ratio_bin, read_review, rgb_to_lab, stable_sample, taxonomy_counts, weighted_taxonomy_counts)
from amodal_benchmark.protocol import align_rgba

spec = importlib.util.spec_from_file_location("appearance_script", Path(__file__).resolve().parents[1] /
                                            "scripts/analyze_appearance_failures.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class AppearanceTests(unittest.TestCase):
    def test_bin_edges_are_left_closed_and_80_is_outside_requested_range(self):
        for ratio, expected in ((0, "[0,20)"), (.199999, "[0,20)"), (.2, "[20,30)"),
                                (.3, "[30,40)"), (.7, "[70,80)"), (.8, "[80,100]"),
                                (1, "[80,100]"), (None, "unavailable")):
            self.assertEqual(ratio_bin(ratio), expected)
        for ratio in (-.01, 1.01, math.nan):
            with self.assertRaises(ValueError):
                ratio_bin(ratio)

    def test_lost_visible_pixels_do_not_become_negative_occlusion(self):
        visible = np.array([[1, 1, 1, 0], [0, 0, 0, 0]], bool)
        predicted = np.array([[0, 1, 1, 1], [0, 0, 0, 1]], bool)
        values = mask_diagnostics(visible, predicted)
        self.assertEqual(values["predicted_expansion_ratio"], .5)
        self.assertEqual(values["visible_retention"], 2 / 3)
        self.assertEqual(values["lost_visible_pixels"], 1)

    def test_no_output_or_visible_evidence_is_unavailable_not_zero(self):
        for a, b in ((np.zeros((2, 2)), np.ones((2, 2))),
                     (np.ones((2, 2)), np.zeros((2, 2)))):
            self.assertIsNone(mask_diagnostics(a, b)["predicted_expansion_ratio"])

    def test_negative_outpainting_offset_preserves_pixel_correspondence(self):
        ref = np.zeros((3, 3, 4), np.uint8)
        ref[0, 0] = [100, 50, 20, 255]
        pred = np.zeros((3, 3, 4), np.uint8)
        pred[1, 2] = ref[0, 0]
        pred[0, 2] = [100, 50, 20, 255]
        a, b = align_rgba(ref, pred, {"original_size": [3, 3], "offset_rc": [-1, -2]})
        result = appearance_diagnostics(a, b)
        self.assertEqual(result["overlap_pixels"], 1)
        self.assertEqual(result["predicted_expansion_ratio"], .5)

    def test_color_conversion_and_identity(self):
        lab = rgb_to_lab(np.array([[0, 0, 0], [255, 255, 255]], np.uint8))
        np.testing.assert_allclose(lab[:, 0], [0, 100], atol=.001)
        rgba = np.full((20, 20, 4), [40, 120, 200, 255], np.uint8)
        result = appearance_diagnostics(rgba, rgba.copy())
        self.assertEqual(result["visible_delta_e76_median"], 0)

    def test_sample_is_stable_and_never_uses_failed_or_missing_outputs(self):
        rows = [{"id": str(i), "status": "success", "proxy_bin": "[20,30)"} for i in range(20)]
        rows += [{"id": "failed", "status": "no_completion", "proxy_bin": "unavailable"}]
        first = stable_sample(rows, 5)
        self.assertEqual(first, stable_sample(rows[::-1], 5))
        self.assertEqual(len(first), 5)
        self.assertEqual(first[0]["inclusion_probability"], .25)

    def test_unknown_and_inapplicable_are_not_negative_labels(self):
        review = [{key: value for key in FAILURES} for value in ("0", "1", "3", "U", "NA")]
        stat = taxonomy_counts(review)[0]
        self.assertEqual((stat["positive"], stat["assessable"], stat["unassessable"],
                          stat["not_applicable"], stat["moderate"]), (2, 3, 1, 1, 0))
        self.assertEqual(stat["frequency"], 2 / 3)

    def test_incomplete_visual_review_cannot_silently_report_zero_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["id", *FAILURES, "evidence", "reviewer"])
                writer.writeheader()
                writer.writerow({"id": "x", "evidence": "pending", "reviewer": "test"})
            with self.assertRaises(ValueError):
                read_review(path, ["x"])

    def test_stratum_weights_do_not_treat_oversampled_rare_bin_as_population(self):
        rows = [{**{key: label for key in FAILURES}, "stratum_population": population,
                 "stratum_sample": 1} for label, population in (("0", 90), ("3", 10))]
        stat = weighted_taxonomy_counts(rows)[0]
        self.assertEqual(stat["frequency"], .5)
        self.assertEqual(stat["weighted_frequency"], .1)

    def test_gt_join_requires_unique_exact_name_not_model_output(self):
        record = {"id": "COCOA/image_sofa", "source": "COCOA", "original_id": "image", "label": "sofa"}
        image = {"width": 10, "height": 10}
        region = {"name": "sofa", "order": 1, "occlude_rate": .35}
        index = {"image": (image, {"id": 42, "regions": [region]}, "external.json")}
        self.assertEqual(analysis.cocoa_match(record, index)["gt_bin"], "[30,40)")
        index["image"][1]["regions"].append(region.copy())
        self.assertIsNone(analysis.cocoa_match(record, index)["gt_occlusion_ratio"])
        index["image"][1]["regions"] = [{**region, "name": "couch"}]
        self.assertIsNone(analysis.cocoa_match(record, index)["gt_occlusion_ratio"])

    def test_bin_statistics_separate_penalties_from_conditional_metrics(self):
        rows = [{"gt_bin": "[20,30)", "status": "success", "CLIP": 30., "LPIPS": .2,
                 "SSIM": .8, "Feature Similarity": .7},
                {"gt_bin": "[20,30)", "status": "no_completion", "CLIP": None, "LPIPS": 1.,
                 "SSIM": 0., "Feature Similarity": 0.}]
        stat = analysis.summarize_bins(rows, "gt_bin")[1]
        self.assertEqual(stat["saved_rate"], .5)
        self.assertEqual(stat["LPIPS_all_mean"], .6)
        self.assertEqual(stat["LPIPS_saved_mean"], .2)
        self.assertEqual(stat["CLIP_all_n"], 1)


if __name__ == "__main__":
    unittest.main()
