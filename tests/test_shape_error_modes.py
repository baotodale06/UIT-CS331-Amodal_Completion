import unittest
import numpy as np
from scripts.audit_shape_error_modes import error_counts,iou_components


class ShapeErrorModeTests(unittest.TestCase):
    def test_attribution_reconstructs_iou_and_separates_visible_hidden_misses(self):
        before = dict(truth_pixels=10,false_positive=2,false_negative_visible=2,false_negative_hidden=3)
        after = dict(truth_pixels=10,false_positive=6,false_negative_visible=1,false_negative_hidden=1)
        result = iou_components(before,after)
        self.assertAlmostEqual(result['total'],8/16-5/12)
        self.assertAlmostEqual(sum(result[key] for key in before if key!='truth_pixels'),result['total'])
        self.assertLess(result['false_positive'],0)
        self.assertGreater(result['false_negative_visible'],0)
        self.assertGreater(result['false_negative_hidden'],0)

    def test_fixed_support_false_positives_and_error_partition(self):
        truth = np.array([[1,1,1],[0,0,0]],bool)
        visible = np.array([[1,1,0],[0,0,0]],bool)
        support = np.array([[1,0,0],[1,0,0]],bool)
        prediction = np.array([[1,0,0],[1,1,0]],bool)
        counts = error_counts(prediction,truth,visible,support)
        self.assertEqual(counts,dict(truth_pixels=3,hidden_pixels=1,false_positive=2,false_negative=2,
            false_positive_in_support=1,false_positive_outside_support=1,false_negative_visible=1,false_negative_hidden=1))
        self.assertEqual(iou_components(counts,counts)['total'],0)


if __name__=='__main__':
    unittest.main()
