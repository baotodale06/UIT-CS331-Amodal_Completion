"""Bounded low-frequency surface consensus; retain each candidate's mask and detail."""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class EnsembleConfig:
    strength: float = 0.5
    sigma: float = 12.0
    max_shift: float = 32.0
    minimum_agreement: float = 0.5


def refine_surface_pool(candidates, masks, visible, config=EnsembleConfig()):
    """Transport smooth consensus residuals only where object masks agree.

    Inputs have already received the same observation-anchored calibration.
    Median RGB uses only object predictions, not changing background support.
    No candidate masks, visible pixels or generated high-frequency residuals
    are replaced. Outputs keep the identity and texture of their own candidate.
    """
    from scipy.ndimage import gaussian_filter
    images = np.asarray(candidates)
    masks, visible = np.asarray(masks, bool), np.asarray(visible, bool)
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError('Expected K,H,W,3 uint8 candidate images')
    if masks.shape != images.shape[:3] or visible.shape != images.shape[1:3]:
        raise ValueError('Consensus inputs use different frames')
    if not 0 <= config.strength <= 1 or config.sigma <= 0 or config.max_shift < 0:
        raise ValueError('Invalid consensus strength, scale or bound')
    if not 0 < config.minimum_agreement <= 1:
        raise ValueError('Mask agreement must lie in (0,1]')
    if not config.strength:
        return images.copy(), dict(config=asdict(config), consensus_pixels=0)
    count = masks.sum(0)
    supported = count >= max(2, int(np.ceil(config.minimum_agreement * len(images))))
    # Sort masked samples with a finite sentinel, then gather each pixel's median.
    values = np.sort(np.where(masks[..., None], images.astype(np.float32), 1000.), axis=0)
    index = np.maximum(count - 1, 0)
    low, high = index // 2, (index + 1) // 2
    median = (np.take_along_axis(values, np.broadcast_to(low[None,...,None], (1,*visible.shape,3)), axis=0)[0] +
              np.take_along_axis(values, np.broadcast_to(high[None,...,None], (1,*visible.shape,3)), axis=0)[0]) / 2
    output = []
    for candidate, mask in zip(images, masks):
        support = supported & mask
        denominator = gaussian_filter(support.astype(np.float32), config.sigma)
        residual = median - candidate.astype(np.float32)
        smooth = np.stack([gaussian_filter(residual[..., c] * support, config.sigma) /
                           np.maximum(denominator, 1e-6) for c in range(3)], axis=-1)
        generated = support & ~visible
        result = candidate.astype(np.float32)
        result[generated] += config.strength * np.clip(smooth[generated], -config.max_shift, config.max_shift)
        output.append(np.clip(np.rint(result), 0, 255).astype(np.uint8))
    return np.stack(output), dict(config=asdict(config), consensus_pixels=int(supported.sum()))
