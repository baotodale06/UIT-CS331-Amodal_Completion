"""SAM-mask diagnostics must keep scoring truth fixed and avoid oracle repair."""
import unittest
import numpy as np
from scripts.audit_shape_sam_validation import geometry, paired_images


class MaskValidationTests(unittest.TestCase):
    def test_prediction_is_not_repaired_with_annotated_visible_pixels(self):
        truth = np.array([[1, 1], [1, 0]], bool)
        visible = np.array([[1, 0], [1, 0]], bool)
        prediction = np.array([[1, 0], [0, 0]], bool)
        self.assertEqual(geometry(prediction, truth, visible), dict(full_iou=1/3, hidden_iou=0.))

    def test_hidden_region_stays_annotated_and_false_extensions_are_penalized(self):
        truth = np.array([[1, 1], [1, 0]], bool)
        visible = np.array([[1, 0], [1, 0]], bool)
        prediction = np.array([[1, 1], [1, 1]], bool)
        self.assertEqual(geometry(prediction, truth, visible), dict(full_iou=.75, hidden_iou=.5))
        self.assertEqual(geometry(np.zeros_like(truth), truth, visible), dict(full_iou=0., hidden_iou=0.))

    def test_each_image_gets_equal_weight_despite_object_counts(self):
        rows = [dict(image_id=image_id, annotated_input=dict(full_iou=a, hidden_iou=a),
                     sam_input=dict(full_iou=b, hidden_iou=b)) for image_id,a,b in [(1,1.,0.),(1,1.,0.),(2,0.,1.)]]
        result = paired_images(rows)
        self.assertEqual(result["full_iou"]["annotated_input"], .5)
        self.assertEqual(result["full_iou"]["sam_input"], .5)
        self.assertEqual(result["full_iou"]["images"], 2)
        self.assertEqual(result["full_iou"]["objects"], 3)

    def test_changing_hidden_denominator_between_conditions_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "denominators"):
            paired_images([dict(image_id=1, annotated_input=dict(full_iou=1.,hidden_iou=None),
                                sam_input=dict(full_iou=1.,hidden_iou=.5))])


if __name__ == "__main__":
    unittest.main()
