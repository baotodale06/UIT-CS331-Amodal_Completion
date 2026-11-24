import unittest
import numpy as np
from scripts.shape_constrained_data import eligible_improvement, freeze_encoder


class ConstrainedShapeTests(unittest.TestCase):
    def test_native_arithmetic_restores_training_policy_on_failure(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch optional in CPU tooling')
        from unittest.mock import patch
        from scripts.shape_constrained_data import evaluate_native
        previous = torch.backends.cudnn.deterministic
        try:
            torch.backends.cudnn.deterministic = True
            def fail(*args):
                self.assertFalse(torch.backends.cudnn.deterministic)
                raise RuntimeError('Test validation failure')
            with patch('scripts.shape_constrained_data._evaluate_native', side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, 'Test validation failure'):
                    evaluate_native(None, None, None, None, None)
            self.assertTrue(torch.backends.cudnn.deterministic)
        finally:
            torch.backends.cudnn.deterministic = previous

    def test_hidden_gain_cannot_override_full_shape_regression(self):
        baseline = dict(full_iou=.81, hidden_iou=.37)
        self.assertFalse(eligible_improvement(dict(full_iou=.80, hidden_iou=.50), baseline, (.37,.81)))
        self.assertTrue(eligible_improvement(dict(full_iou=.81, hidden_iou=.38), baseline, (.37,.81)))
        self.assertFalse(eligible_improvement(baseline, baseline, (.37,.81)))
        with self.assertRaises(ValueError):
            eligible_improvement(dict(full_iou=np.nan, hidden_iou=.5), baseline, (.37,.81))

    def test_encoder_weights_and_batchnorm_stay_fixed_during_decoder_update(self):
        try:
            import torch
            from torch import nn
        except ImportError:
            self.skipTest('Torch optional in CPU tooling')
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Sequential(nn.Conv2d(5,8,1), nn.BatchNorm2d(8))
                self.pool = nn.Identity()
                self.layers = nn.Sequential(nn.Conv2d(8,8,1), nn.BatchNorm2d(8))
                self.decoder = nn.Conv2d(8,1,1)
            def forward(self, x):
                return self.decoder(self.layers(self.pool(self.stem(x))))
        model = Model()
        model.train()
        freeze_encoder(model)
        before = {name:tensor.clone() for name,tensor in model.state_dict().items()}
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
        model(torch.ones(2,5,4,4)).sum().backward()
        optimizer.step()
        for name,tensor in model.state_dict().items():
            if not name.startswith('decoder.'):
                self.assertTrue(torch.equal(before[name],tensor),name)
        self.assertFalse(torch.equal(before['decoder.bias'],model.decoder.bias))


if __name__ == '__main__':
    unittest.main()
