"""Decoder-only SAM adaptation with preserved encoder and observable support."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

from amodal_benchmark.shape_v2 import observation_tensor, visible_box
from scripts.shape_robust_data import native_canvas, geometry


def freeze_encoder(model):
    for module in (model.stem, model.pool, model.layers):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return model


def eligible_improvement(result, baseline, best):
    """Require full-shape retention before ranking hidden-shape improvements."""
    keys = ('full_iou', 'hidden_iou')
    if any(not np.isfinite(row[key]) or not 0 <= row[key] <= 1 for row in (result, baseline) for key in keys):
        raise ValueError('Finite bounded native geometry required')
    score = result['hidden_iou'], result['full_iou']
    return result['full_iou'] >= baseline['full_iou'] and score > best


def evaluate_native(model, data, selected, sam_root, device, maximum_images=None):
    """Match the frozen diagnostic's cuDNN policy without changing training."""
    import torch
    previous = torch.backends.cudnn.deterministic
    try:
        torch.backends.cudnn.deterministic = False
        return _evaluate_native(model, data, selected, sam_root, device, maximum_images)
    finally:
        torch.backends.cudnn.deterministic = previous


def _evaluate_native(model, data, selected, sam_root, device, maximum_images=None):
    """Restore only input SAM support; truth is used exclusively for scoring."""
    import torch
    groups = defaultdict(list)
    for row in selected:
        groups[row['image_id']].append(row)
    model.eval()
    records = []
    with torch.inference_mode():
        for index, (image_id, objects) in enumerate(sorted(groups.items())):
            if maximum_images is not None and index >= maximum_images:
                break
            with Image.open(Path(data)/objects[0]['image']) as image:
                rgb = np.array(image.convert('RGB'))
            for row in objects:
                with Image.open(Path(data)/row['mask']) as image:
                    encoded = np.array(image)
                truth, visible = (encoded & 1) > 0, (encoded & 2) > 0
                with Image.open(Path(sam_root)/'objects'/row['id']/'sam-visible.png') as image:
                    support = np.array(image) > 0
                if support.shape != truth.shape:
                    raise ValueError('SAM input and scoring frames differ')
                if support.any():
                    box = visible_box(support, 2.)
                    observed = np.array(Image.fromarray(rgb).crop(box))
                    cropped = np.array(Image.fromarray(support).crop(box))
                    inputs = torch.from_numpy(observation_tensor(observed, cropped)).unsqueeze(0).to(device)
                    logits = model(inputs)
                    probability = torch.nn.functional.interpolate(logits, cropped.shape, mode='bilinear', align_corners=False).sigmoid()
                    prediction = (native_canvas(probability[0, 0].cpu().numpy(), box, truth.shape) >= .5) | support
                else:
                    prediction = np.zeros_like(truth)
                records.append(dict(id=row['id'], image_id=image_id, **geometry(prediction, truth, visible)))
    return records
