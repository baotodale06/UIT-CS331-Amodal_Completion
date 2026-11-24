"""Verify mathematical microbatch equivalence and both-platform retention."""
import unittest

from scripts.shape_local_data import eligible, selection_floor


class LocalSelectionTests(unittest.TestCase):
    def test_new_epoch_must_retain_both_original_and_local_baselines(self):
        floor = selection_floor(dict(full_iou=.82,hidden_iou=.37),dict(full_iou=.81,hidden_iou=.38))
        self.assertEqual(floor,dict(full_iou=.82,hidden_iou=.38))
        self.assertFalse(eligible(dict(full_iou=.819,hidden_iou=.40),floor,(.38,.82)))
        self.assertFalse(eligible(dict(full_iou=.83,hidden_iou=.379),floor,(.38,.82)))
        self.assertFalse(eligible(dict(full_iou=.82,hidden_iou=.38),floor,(.38,.82)))
        self.assertTrue(eligible(dict(full_iou=.82,hidden_iou=.39),floor,(.38,.82)))

    def test_nonfinite_geometry_cannot_be_selected(self):
        with self.assertRaises(ValueError):
            selection_floor(dict(full_iou=float('nan'),hidden_iou=.2),dict(full_iou=.8,hidden_iou=.3))


class AccumulationTests(unittest.TestCase):
    def test_loss_and_gradient_match_unequal_global_hidden_pixel_counts(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch optional for local CPU tooling')
        from amodal_benchmark.shape_v2 import shape_loss
        from scripts.shape_local_data import accumulated_loss
        torch.manual_seed(7)
        for empty_hidden in (False,True):
            logits = torch.randn(7,1,9,11,dtype=torch.float64,requires_grad=True)
            truth = (torch.rand_like(logits)>.3).double()
            visible = truth.clone() if empty_hidden else truth*(torch.rand_like(logits)>.6)
            if not empty_hidden:
                visible[:2] = truth[:2]
            expected = shape_loss(logits,truth,visible)
            expected.backward()
            actual_logits = logits.detach().clone().requires_grad_()
            total_hidden = float((truth*(1-visible)).sum())
            actual = sum(accumulated_loss(actual_logits[start:end],truth[start:end],visible[start:end],len(truth),total_hidden)
                         for start,end in ((0,2),(2,5),(5,7)))
            actual.backward()
            torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(actual_logits.grad,logits.grad,rtol=1e-12,atol=1e-12)


if __name__ == '__main__':
    unittest.main()
