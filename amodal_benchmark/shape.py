"""Observation-only silhouette prior and candidate selection for Shape-Guided PMC."""
import numpy as np
from PIL import Image


def overlap(first, second):
    first, second = np.asarray(first, bool), np.asarray(second, bool)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Shape masks must share a two-dimensional pixel frame")
    union = np.count_nonzero(first | second)
    return np.count_nonzero(first & second) / union if union else 0.0


def choose(scores, surface_losses=None):
    """Rank by prior IoU or by equal-weight within-pool midranks on valid masks."""
    valid = [i for i, value in enumerate(scores) if value is not None and
             (surface_losses is None or surface_losses[i] is not None)]
    if not valid:
        return 0, True
    if surface_losses is None:
        return max(valid, key=lambda i: scores[i]), False

    def midrank(values, i):
        return sum((values[j] < values[i]) + 0.5 * (values[j] == values[i]) for j in valid)

    shape_losses = [None if value is None else -value for value in scores]
    return min(valid, key=lambda i: midrank(shape_losses, i) + midrank(surface_losses, i)), False


def make_network():
    import torch
    from torch import nn

    def block(inputs, outputs):
        return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs),
                             nn.ReLU(), nn.Conv2d(outputs, outputs, 3, padding=1),
                             nn.GroupNorm(4, outputs), nn.ReLU())

    class ShapeUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoders = nn.ModuleList(block(i, o) for i, o in [(5, 12), (12, 24), (24, 48)])
            self.bridge = block(48, 96)
            self.decoders = nn.ModuleList(block(i, o) for i, o in [(144, 48), (72, 24), (36, 12)])
            self.output = nn.Conv2d(12, 1, 1)

        def forward(self, x):
            skips = []
            for encoder in self.encoders:
                x = encoder(x)
                skips.append(x)
                x = nn.functional.max_pool2d(x, 2)
            x = self.bridge(x)
            for decoder, skip in zip(self.decoders, reversed(skips)):
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                x = decoder(torch.cat((x, skip), dim=1))
            return self.output(x)

    return ShapeUNet()


def resized(array, size, rgb=False):
    image = Image.fromarray(array if rgb else np.asarray(array, np.uint8))
    return np.array(image.resize(size, Image.Resampling.BILINEAR if rgb else Image.Resampling.NEAREST))


def predict(model, observed, visible, occlusion):
    import torch
    visible, occlusion = np.asarray(visible, bool), np.asarray(occlusion, bool)
    if observed.shape != (*visible.shape, 3) or visible.shape != occlusion.shape:
        raise ValueError("Shape inputs must share one crop")
    rgb = observed.copy()
    rgb[occlusion & ~visible] = 127
    size = (256, 256)
    tensor = np.dstack((resized(rgb, size, True) / 255., resized(visible, size), resized(occlusion, size)))
    device = next(model.parameters()).device
    with torch.inference_mode():
        logits = model(torch.as_tensor(tensor, dtype=torch.float32, device=device).permute(2, 0, 1)[None])
        logits = torch.nn.functional.interpolate(logits, visible.shape, mode="bilinear", align_corners=False)
    return visible | (occlusion & (logits[0, 0].cpu().numpy() >= 0))


def score_masks(prior, masks):
    return [overlap(mask, prior) if np.any(mask) else None for mask in masks]

