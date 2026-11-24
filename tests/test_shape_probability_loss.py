import importlib.util
import math
import unittest

HAS_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_TORCH, 'Pinned Torch environment required')
class ShapeProbabilityLossTests(unittest.TestCase):
    def test_empirical_probability_is_stationary_without_positive_bonus(self):
        import torch
        from scripts.shape_probability_loss import probability_loss
        parameter = torch.tensor(math.log(.25/.75), requires_grad=True)
        logits = parameter.expand(4,1,1,1)
        truth = torch.tensor([1.,0.,0.,0.]).reshape(4,1,1,1)
        loss = probability_loss(logits,truth,torch.zeros_like(truth))
        loss.backward()
        self.assertAlmostEqual(parameter.grad.item(),0.,places=7)
        self.assertAlmostEqual(loss.item(),-(.25*math.log(.25)+.75*math.log(.75)),places=6)

    def test_visible_support_cannot_change_loss_or_gradients(self):
        import torch
        from scripts.shape_probability_loss import probability_loss
        logits = torch.tensor([[[[30.,2.,-2.]]]],requires_grad=True)
        support = torch.tensor([[[[1.,0.,0.]]]])
        truth = torch.tensor([[[[0.,0.,1.]]]])
        loss = probability_loss(logits,truth,support)
        loss.backward()
        self.assertEqual(logits.grad[0,0,0,0].item(),0.)
        self.assertGreater(logits.grad[0,0,0,1].item(),0.)
        self.assertLess(logits.grad[0,0,0,2].item(),0.)
        self.assertAlmostEqual(loss.item(),torch.nn.functional.softplus(torch.tensor(2.)).item())

    def test_uneven_microbatches_preserve_global_loss_and_gradients(self):
        import torch
        from scripts.shape_probability_loss import probability_loss
        torch.manual_seed(17)
        x = torch.randn(5,1,4,4,requires_grad=True)
        truth = torch.randint(0,2,x.shape).float()
        support = torch.randint(0,2,x.shape).float()
        support[0] = 1
        full = probability_loss(x,truth,support)
        expected = torch.autograd.grad(full,x)[0]
        y = x.detach().clone().requires_grad_(True)
        accumulated = sum(probability_loss(y[a:b],truth[a:b],support[a:b],5) for a,b in ((0,2),(2,4),(4,5)))
        actual = torch.autograd.grad(accumulated,y)[0]
        self.assertTrue(torch.allclose(full,accumulated,atol=1e-7,rtol=0))
        self.assertTrue(torch.allclose(expected,actual,atol=1e-7,rtol=0))
        self.assertEqual(actual[0].abs().sum().item(),0.)

    def test_invalid_supervision_is_rejected(self):
        import torch
        from scripts.shape_probability_loss import probability_loss
        x = torch.zeros(2,1,2,2)
        with self.assertRaises(ValueError):
            probability_loss(x,x,torch.full_like(x,.5))
        with self.assertRaises(ValueError):
            probability_loss(x,x,x,1)


if __name__=='__main__':
    unittest.main()
