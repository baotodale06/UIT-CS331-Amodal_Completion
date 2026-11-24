"""True-amodal shape prior. No appearance ranker or generated-RGB correction."""
import numpy as np
from PIL import Image


def visible_box(visible, padding=2.0):
    """Square input crop defined only by observable support, including outpaint."""
    yy, xx = np.where(visible)
    if not len(xx) or padding < 1:
        raise ValueError("Nonempty visible support and padding >=1 required")
    side = int(np.ceil(max(xx.max() - xx.min() + 1, yy.max() - yy.min() + 1) * padding))
    x0 = int(np.floor((xx.min() + xx.max() + 1 - side) / 2))
    y0 = int(np.floor((yy.min() + yy.max() + 1 - side) / 2))
    return x0, y0, x0 + side, y0 + side


def observation_tensor(rgb, visible, unknown=None, size=256):
    visible = np.asarray(visible, bool)
    unknown = ~visible if unknown is None else np.asarray(unknown, bool) & ~visible
    if rgb.shape != (*visible.shape, 3) or unknown.shape != visible.shape:
        raise ValueError("Shape observations must share one frame")
    neutral = np.full_like(rgb, 127)
    neutral[visible] = rgb[visible]
    rgb_small = np.asarray(Image.fromarray(neutral).resize((size, size), Image.Resampling.BILINEAR), np.float32)
    masks = [np.asarray(Image.fromarray(m.astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST))
             for m in (visible, unknown)]
    # Normalize only RGB using the pretrained encoder's fixed normalization.
    rgb_small = (rgb_small / 255 - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return np.concatenate([rgb_small.transpose(2, 0, 1), np.stack(masks)], axis=0).astype(np.float32)


def make_network(pretrained=False):
    import torch
    from torch import nn
    from torchvision.models import ResNet34_Weights, resnet34

    def block(inputs, outputs):
        return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1, bias=False), nn.GroupNorm(8, outputs),
                             nn.SiLU(), nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
                             nn.GroupNorm(8, outputs), nn.SiLU())

    class AmodalUNet(nn.Module):
        def __init__(self):
            super().__init__()
            encoder = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            conv = nn.Conv2d(5, 64, 7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                conv.weight[:, :3].copy_(encoder.conv1.weight)
                conv.weight[:, 3:].zero_()
            self.stem = nn.Sequential(conv, encoder.bn1, encoder.relu)
            self.pool = encoder.maxpool
            self.layers = nn.ModuleList([encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4])
            self.decoders = nn.ModuleList([block(512+256, 256), block(256+128, 128),
                                          block(128+64, 64), block(64+64, 64)])
            self.output = nn.Sequential(block(64+5, 32), nn.Conv2d(32, 1, 1))

        def forward(self, inputs):
            skips = [self.stem(inputs)]
            x = self.pool(skips[0])
            for layer in self.layers:
                x = layer(x)
                skips.append(x)
            for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
                x = nn.functional.interpolate(x, skip.shape[-2:], mode="bilinear", align_corners=False)
                x = decoder(torch.cat([x, skip], 1))
            x = nn.functional.interpolate(x, inputs.shape[-2:], mode="bilinear", align_corners=False)
            return self.output(torch.cat([x, inputs], 1))

    return AmodalUNet()


def shape_loss(logits, truth, visible):
    import torch.nn.functional as F
    probability = logits.sigmoid()
    hidden = truth * (1 - visible)
    predicted_hidden = probability * (1 - visible)
    axes = (1, 2, 3)
    def dice(pred, target):
        return (1 - (2 * (pred * target).sum(axes) + 1) / (pred.sum(axes) + target.sum(axes) + 1)).mean()
    positive = (F.softplus(-logits) * hidden).sum() / hidden.sum().clamp_min(1)
    return F.binary_cross_entropy_with_logits(logits, truth) + dice(probability, truth) + dice(predicted_hidden, hidden) + .5 * positive


def predict_probability(model, observed, visible, unknown=None):
    import torch
    device = next(model.parameters()).device
    inputs = torch.from_numpy(observation_tensor(observed, visible, unknown)).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(inputs)
        probability = torch.nn.functional.interpolate(logits, visible.shape, mode="bilinear", align_corners=False).sigmoid()
    result = probability[0, 0].cpu().numpy()
    if unknown is not None:
        result[~np.asarray(unknown, bool) & ~np.asarray(visible, bool)] = 0
    result[np.asarray(visible, bool)] = 1
    return result


def score_masks(probability, masks):
    """Expected intersection/union, retaining uncertainty rather than hard shrinkage."""
    probability = np.asarray(probability, np.float64)
    if probability.ndim != 2 or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Expected a finite 2D probability mask")
    scores = []
    for mask in masks:
        mask = np.asarray(mask, bool)
        if mask.shape != probability.shape:
            raise ValueError("Candidate and prior frames differ")
        scores.append(float(probability[mask].sum() / (mask.sum() + probability[~mask].sum())) if mask.any() else None)
    return scores
