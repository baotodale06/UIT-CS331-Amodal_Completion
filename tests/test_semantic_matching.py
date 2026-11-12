"""Regression cases for OpenWorld annotation matching without LISA."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class SemanticMatchingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("amodal_benchmark.matching"),
                             "The corrected semantic matcher is missing")
        from amodal_benchmark.matching import choose_semantic_target
        self.choose = choose_semantic_target
        self.mask = np.zeros((16, 16), dtype=bool)
        self.mask[3:11, 3:11] = True

    def test_aliases_and_descriptive_annotations_match_without_changing_detected_class(self):
        for label, name in [("television", "tv"), ("brown table", "dining table"),
                            ("white sofa", "couch"), ("small motorbike", "motorcycle"),
                            ("blue plane", "airplane"), ("toothbrush", "tooth brush"),
                            ("red cars", "car"), ("white car on the left", "car"),
                            ("white car", "white car")]:
            with self.subTest(label=label, name=name):
                details = {}
                selected, reason = self.choose([self.mask], [name], label, [.7], details)
                self.assertEqual(selected, 0)
                self.assertEqual(reason, "semantic_label_match")
                self.assertEqual(details["candidates"][0]["class"], name)
                self.assertEqual(details["selected_index"], 0)

    def test_identical_masks_collapse_with_highest_score_and_stable_ties(self):
        details = {}
        masks = [self.mask, self.mask.copy(), self.mask.copy()]
        result = self.choose(masks, ["tv", "television", "tv"], "television", [.4, .8, .8], details)
        self.assertEqual(result, (1, "semantic_label_match"))
        self.assertEqual(details["groups"], [{"members": [0, 1, 2], "representative": 1}])
        self.assertEqual(details["duplicate_count"], 2)

    def test_remaining_annotation_variants_match_the_detected_object(self):
        cases = [("teddy bear toy", "teddy bear"), ("pink teddy bear toys", "teddy bear"),
                 ("toy bear", "teddy bear"), ("toy bears on a bed", "teddy bear"),
                 ("motor bike", "motorcycle"), ("red motor bikes", "motorcycle"),
                 ("surfing board", "surfboard"), ("surfing boards beside a boat", "surfboard"),
                 ("snow board", "snowboard"), ("blue snow boards", "snowboard")]
        for label, detected in cases:
            with self.subTest(label=label):
                details = {}
                self.assertEqual(self.choose([self.mask], [detected], label, [.8], details),
                                 (0, "semantic_label_match"))
                self.assertEqual(details["annotation_class"], detected)
                self.assertEqual(details["candidates"][0]["class"], detected)
                self.assertEqual(details["policy"], "semantic-label-v2")

    def test_toy_bear_alias_does_not_select_a_live_bear_or_an_unrelated_toy(self):
        for label, detected in [("toy bear", "bear"), ("teddy bear toy", "bear"),
                                ("dog toy", "dog"), ("teddy dog toy", "teddy bear"),
                                ("teddy bear toy box", "teddy bear"),
                                ("motor bike helmet", "motorcycle"),
                                ("surfing board rack", "surfboard")]:
            with self.subTest(label=label, detected=detected):
                self.assertEqual(self.choose([self.mask], [detected], label, [.9]),
                                 (None, "not_detected"))

    def test_alias_uses_the_correct_class_without_resolving_distinct_instances_by_score(self):
        details = {}
        self.assertEqual(self.choose([self.mask, np.roll(self.mask, 4, axis=1)],
            ["bear", "teddy bear"], "toy bear", [.99, .4], details), (1, "semantic_label_match"))
        self.assertFalse(details["candidates"][0]["label_matched"])
        self.assertEqual(self.choose([self.mask, np.roll(self.mask, 4, axis=1)],
            ["teddy bear", "teddy bear"], "teddy bear toy", [.99, .4]),
            (None, "ambiguous_target"))

    def test_distinct_and_nearly_identical_instances_remain_ambiguous(self):
        for second in (np.roll(self.mask, 5, axis=1), self.mask.copy()):
            second[3, 3] = not second[3, 3]
            details = {}
            result = self.choose([self.mask, second], ["car", "car"], "white car", [.9, .4], details)
            self.assertEqual(result, (None, "ambiguous_target"))
            self.assertEqual(details["reason"], "multiple_distinct_instances")
            self.assertEqual(len(details["groups"]), 2)

    def test_compound_nouns_and_unrelated_classes_do_not_create_false_matches(self):
        for label, name in [("carpet", "car"), ("hot dog", "dog"),
                            ("table lamp", "dining table"), ("dog on a sofa", "couch"),
                            ("cat", "dog"), ("white sofa", "chair")]:
            with self.subTest(label=label, name=name):
                details = {}
                self.assertEqual(self.choose([self.mask], [name], label, [.9], details), (None, "not_detected"))
                self.assertEqual(details["reason"], "label_not_matched")
                self.assertEqual(details["candidates"][0]["detector_score"], .9)
                self.assertIn("bbox_xyxy", details["candidates"][0])

    def test_no_detection_and_missing_sam_masks_have_different_diagnostics(self):
        details = {}
        self.assertEqual(self.choose([], [], "cat", [], details), (None, "not_detected"))
        self.assertEqual(details["reason"], "no_detections")
        details = {}
        self.assertEqual(self.choose([], ["cat"], "cat", [.8], details), (None, "not_detected"))
        self.assertEqual(details["reason"], "segmentation_unavailable")
        self.assertEqual(details["candidates"][0]["class"], "cat")

    def test_empty_matching_masks_do_not_hide_a_usable_instance(self):
        details = {}
        result = self.choose([np.zeros_like(self.mask), self.mask], ["cat", "cat"], "cat", [.9, .7], details)
        self.assertEqual(result, (1, "semantic_label_match"))
        self.assertEqual(details["candidates"][0]["mask_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
