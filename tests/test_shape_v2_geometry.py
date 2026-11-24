"""Checks against inflation of geometry scores by crop selection or object count."""
import unittest
import numpy as np
from scripts.evaluate_shape_v2_geometry import geometry_scores, grouped_means


class ShapeV2GeometryTests(unittest.TestCase):
    def test_hidden_truth_outside_prediction_crop_is_retained(self):
        visible = np.zeros((10, 10), bool)
        visible[4:6, 4:6] = True
        truth = np.ones((10, 10), bool)
        probability = visible.astype(float)
        probability[3:7, 3:7] = 1
        scores = geometry_scores(probability, visible, truth)
        self.assertEqual(scores["full_iou"], .16)
        self.assertEqual(scores["hidden_iou"], 12/96)
        self.assertEqual(scores["visible_only_full_iou"], .04)

    def test_image_mean_does_not_overweight_images_with_many_objects(self):
        rows = [dict(image_id=1, full_iou=1., visible_only_full_iou=.5, hidden_iou=None)] * 10
        rows.append(dict(image_id=2, full_iou=0., visible_only_full_iou=0., hidden_iou=.25))
        result = grouped_means(rows)
        self.assertEqual(result["full_iou"], .5)
        self.assertEqual(result["hidden_iou"], .25)
        self.assertEqual(result["hidden_iou_images"], 1)

    def test_visible_support_is_preserved_and_invalid_probabilities_fail(self):
        visible = np.eye(3, dtype=bool)
        result = geometry_scores(np.zeros((3, 3)), visible, visible)
        self.assertEqual(result["full_iou"], 1.)
        self.assertIsNone(result["hidden_iou"])
        with self.assertRaises(ValueError):
            geometry_scores(np.full((3, 3), np.nan), visible, visible)


if __name__ == "__main__":
    unittest.main()
