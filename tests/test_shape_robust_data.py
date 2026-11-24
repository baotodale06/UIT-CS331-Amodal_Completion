"""Robust shape training cannot use noisy SAM support as scoring truth."""
import unittest
import numpy as np
from scripts.shape_robust_data import aggregate, cropped_sample, geometry, native_canvas


class RobustShapeDataTests(unittest.TestCase):
    def test_no_hard_support_restoration(self):
        probability = np.array([[.1,.9],[.2,.1]], np.float32)
        prediction = native_canvas(probability, (0,0,2,2), (2,2)) >= .5
        truth = np.array([[1,1],[1,0]], bool)
        annotated = np.array([[1,0],[1,0]], bool)
        self.assertEqual(geometry(prediction, truth, annotated), dict(full_iou=1/3, hidden_iou=1.))

    def test_native_truth_outside_crop_is_counted(self):
        prediction = native_canvas(np.ones((2,2), np.float32), (-1,-1,1,1), (3,3)) >= .5
        truth = np.ones((3,3), bool)
        annotated = np.zeros((3,3), bool)
        annotated[0,0] = True
        self.assertEqual(geometry(prediction, truth, annotated), dict(full_iou=1/9, hidden_iou=0.))

    def test_sam_false_positive_is_input_not_label(self):
        rgb = np.full((2,2,3), 200, np.uint8)
        truth = np.array([[1,1],[0,0]], bool)
        annotated = np.array([[1,0],[0,0]], bool)
        sam = np.array([[1,0],[0,1]], bool)
        x,y,v = cropped_sample(rgb, truth, annotated, sam, (0,0,2,2), 2)
        self.assertEqual(x[3,1,1], 1)
        self.assertEqual(y[0,1,1], 0)
        self.assertEqual(v[0,1,1], 0)
        self.assertEqual(y[0,0,1]-v[0,0,1], 1)

    def test_rgb_outside_support_is_not_observed(self):
        rgb = np.zeros((2,2,3), np.uint8)
        support = np.array([[1,0],[0,0]], bool)
        altered = rgb.copy()
        altered[~support] = 255
        first = cropped_sample(rgb, support, support, support, (0,0,2,2), 4)[0]
        second = cropped_sample(altered, support, support, support, (0,0,2,2), 4)[0]
        np.testing.assert_array_equal(first, second)

    def test_image_weighting_and_hidden_denominator(self):
        rows = [dict(image_id=1, full_iou=1., hidden_iou=.2), dict(image_id=1, full_iou=1., hidden_iou=None),
                dict(image_id=2, full_iou=0., hidden_iou=.6)]
        result = aggregate(rows)
        self.assertEqual(result["full_iou"], .5)
        self.assertEqual(result["hidden_iou"], .4)
        self.assertEqual(result["objects"], 3)


if __name__ == "__main__":
    unittest.main()
