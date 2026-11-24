"""Frozen correction selection and native probability semantics."""
import unittest
import numpy as np

from scripts.freeze_shape_probability_candidate import select_candidate
from amodal_benchmark.shape_calibrated_prior import original_prior


class CalibratedPriorTests(unittest.TestCase):
    def test_both_floors_and_stable_selection(self):
        values = dict(initial=dict(full_iou=.81, hidden_iou=.37),
                      adapted_bias_0=dict(full_iou=.80, hidden_iou=.45))
        with self.assertRaises(ValueError):
            select_candidate(values, [0])
        values["adapted_bias_-2"] = dict(full_iou=.82, hidden_iou=.39)
        self.assertEqual(select_candidate(values, [0, -2]), ("adapted_bias_-2", -2))
        values["adapted_bias_0"] = dict(full_iou=.82, hidden_iou=.39)
        self.assertEqual(select_candidate(values, [0, -2]), ("adapted_bias_0", 0))
        values["adapted_bias_-2"]["full_iou"] = float("nan")
        with self.assertRaises(ValueError):
            select_candidate(values, [0, -2])

    def test_exact_native_logit_order_visible_clamp_and_unobserved_rgb(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch is optional in CPU environment")
        class Fixed(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.))
                self.seen = []
            def forward(self, inputs):
                self.seen.append(inputs.detach().cpu().clone())
                return torch.tensor([[[[-1., 1.], [3., 5.]]]], device=inputs.device) + self.weight
        model = Fixed()
        support = np.zeros((9, 11), bool)
        support[3:5, 4:7] = True
        first = np.full((9, 11, 3), 60, np.uint8)
        second = first.copy()
        second[~support] = 220
        a, box, _, cropped = original_prior(model, first, support)
        b, _, _, _ = original_prior(model, second, support)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(torch.equal(model.seen[0], model.seen[1]))
        native = (torch.nn.functional.interpolate(torch.tensor([[[[-1., 1.], [3., 5.]]]]),
            cropped.shape, mode="bilinear", align_corners=False) - 2.).sigmoid()[0, 0].numpy()
        expected = np.zeros_like(a)
        x0, y0, x1, y1 = box
        expected[y0:y1, x0:x1] = native
        expected[support] = 1
        np.testing.assert_array_equal(a, expected)
        self.assertTrue(np.all(a[support] == 1))
        empty, empty_box, _, _ = original_prior(model, first, np.zeros_like(support))
        self.assertFalse(empty.any())
        self.assertIsNone(empty_box)
        with self.assertRaises(ValueError):
            original_prior(model, first, support, -1.5)


if __name__ == "__main__":
    unittest.main()
