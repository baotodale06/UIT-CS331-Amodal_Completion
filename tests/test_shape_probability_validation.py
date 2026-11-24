import unittest
import numpy as np
from scripts.shape_probability_validation import native_pixel_hash, require_native_replay
from scripts.shape_robust_data import geometry


class ShapeProbabilityNativeTests(unittest.TestCase):
    def test_equal_iou_is_not_sufficient_for_exact_native_replay(self):
        truth = np.array([[1,0],[0,0]],bool)
        first = np.array([[1,1],[0,0]],bool)
        second = np.array([[1,0],[1,0]],bool)
        self.assertEqual(geometry(first,truth,np.zeros_like(truth)),geometry(second,truth,np.zeros_like(truth)))
        expected = native_pixel_hash(first)
        require_native_replay(first.copy(),expected)
        with self.assertRaises(ValueError):
            require_native_replay(second,expected)


if __name__=='__main__':
    unittest.main()
