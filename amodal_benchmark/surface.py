"""Observable-only surface compatibility; hidden truth is never an argument.

Scores are compatibility priors, not estimators of unseen RGB correctness.
All RGB inputs must be uint8 in one shared coordinate frame. Original visible
pixels anchor both sides; repainting the candidate's visible part cannot game
the surface terms. Square morphological bands are explicit for portability.
"""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageFilter

from .appearance import rgb_to_lab


@dataclass(frozen=True)
class SurfaceConfig:
    radius: int = 5
    min_region_pixels: int = 64
    min_visible_recall: float = 0.5
    histogram_bins: int = 8


def dilate(mask, radius):
    return np.asarray(Image.fromarray(np.asarray(mask, np.uint8) * 255).filter(
        ImageFilter.MaxFilter(2 * radius + 1))) > 0


def erode(mask, radius=1):
    return np.asarray(Image.fromarray(np.asarray(mask, np.uint8) * 255).filter(
        ImageFilter.MinFilter(2 * radius + 1))) > 0


def hellinger(first, second):
    first, second = np.asarray(first, float), np.asarray(second, float)
    if first.sum() <= 0 or second.sum() <= 0:
        return None
    first, second = first / first.sum(), second / second.sum()
    return float(np.sqrt(np.maximum(0, 1 - np.sqrt(first * second).sum())))


def color_histogram(lab, mask, bins=8):
    return np.histogramdd(lab[mask], bins=(bins,) * 3,
                          range=((0, 100.0001), (-128, 128), (-128, 128)))[0].ravel()


def texture_histograms(rgb, mask):
    core = erode(mask)
    # np.gradient uses one-sided image-edge differences; exclude those pixels.
    core[[0, -1], :] = False
    core[:, [0, -1]] = False
    if int(core.sum()) < 16:
        return None
    gray = np.asarray(rgb, np.float32) @ np.array([.2126, .7152, .0722], np.float32) / 255
    gy, gx = np.gradient(gray)
    magnitude = np.hypot(gx, gy)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    orientation = np.histogram(angle[core], bins=8, range=(0, np.pi),
                               weights=magnitude[core])[0].astype(float)
    # A separate flat bin keeps a genuinely flat surface well-defined.
    orientation = np.append(orientation, np.maximum(0, .01 - magnitude[core]).sum())
    strength = np.histogram(magnitude[core], bins=[0, .01, .025, .05, .1, .2, .4, 1.5])[0]
    return orientation, strength


def texture_distance(first, second):
    if first is None or second is None:
        return None
    values = [hellinger(a, b) for a, b in zip(first, second)]
    return None if any(v is None for v in values) else float(np.mean(values))


def paired_boundary_loss(observed_lab, candidate_lab, visible, hidden, radius):
    """Local axis-aligned pixel pairs across the interface, weighted by 1/d."""
    numerator = denominator = 0.
    for distance in range(1, radius + 1):
        for axis in (0, 1):
            first, second = [slice(None)] * 2, [slice(None)] * 2
            first[axis], second[axis] = slice(None, -distance), slice(distance, None)
            first, second = tuple(first), tuple(second)
            for v_slice, h_slice in ((first, second), (second, first)):
                pairs = visible[v_slice] & hidden[h_slice]
                count = int(pairs.sum())
                if count:
                    difference = observed_lab[v_slice][pairs] - candidate_lab[h_slice][pairs]
                    numerator += float(np.linalg.norm(difference, axis=1).sum()) / distance
                    denominator += count / distance
    return numerator / denominator / 100 if denominator else None


def score_surface(observed_rgb, visible, occlusion, candidate_rgb, candidate_mask,
                  config=SurfaceConfig()):
    """Return interpretable raw terms, supports, validity and preservation error."""
    observed, candidate = np.asarray(observed_rgb), np.asarray(candidate_rgb)
    if observed.dtype != np.uint8 or candidate.dtype != np.uint8:
        raise ValueError('RGB must be uint8, not ambiguously scaled floating point')
    if observed.ndim != 3 or observed.shape[-1] != 3 or candidate.shape != observed.shape:
        raise ValueError('RGB images must share an H,W,3 frame')
    masks = [np.asarray(mask, bool) for mask in (visible, occlusion, candidate_mask)]
    if any(mask.shape != observed.shape[:2] for mask in masks):
        raise ValueError('All masks must share the RGB frame')
    visible, occlusion, predicted = masks
    if (visible & occlusion).any():
        raise ValueError('Observed visible support must exclude occluded pixels')
    if config.radius < 1 or config.min_region_pixels < 1 or not 0 <= config.min_visible_recall <= 1:
        raise ValueError('Invalid surface configuration')
    hidden = predicted & occlusion & ~visible
    v_count, h_count = int(visible.sum()), int(hidden.sum())
    recall = float((visible & predicted).sum() / v_count) if v_count else 0.
    result = dict(valid=False, invalid_reasons=[], visible_pixels=v_count, hidden_pixels=h_count,
                  predicted_pixels=int(predicted.sum()), visible_recall=recall,
                  boundary=None, color=None, texture=None, local_boundary=None,
                  local_texture=None, visible_mae=None)
    if v_count:
        result['visible_mae'] = float(np.abs(observed[visible].astype(float) - candidate[visible]).mean() / 255)
    if v_count < config.min_region_pixels:
        result['invalid_reasons'].append('insufficient_visible_support')
    if h_count < config.min_region_pixels:
        result['invalid_reasons'].append('insufficient_generated_support')
    if recall < config.min_visible_recall:
        result['invalid_reasons'].append('lost_visible_target')
    if result['invalid_reasons']:
        return result
    bv, bh = dilate(hidden, config.radius) & visible, dilate(visible, config.radius) & hidden
    result.update(visible_band_pixels=int(bv.sum()), hidden_band_pixels=int(bh.sum()))
    observed_lab, candidate_lab = rgb_to_lab(observed), rgb_to_lab(candidate)
    if min(int(bv.sum()), int(bh.sum())) < 16:
        result['invalid_reasons'].append('missing_boundary_support')
        return result
    # Float64 accumulation prevents row-order roundoff from spuriously breaking
    # equal-distribution ties (e.g. phase-shifted stripes with identical means).
    result['boundary'] = float(np.linalg.norm(observed_lab[bv].mean(0, dtype=np.float64) -
                                            candidate_lab[bh].mean(0, dtype=np.float64)) / 100)
    result['local_boundary'] = paired_boundary_loss(observed_lab, candidate_lab, visible, hidden, config.radius)
    result['color'] = hellinger(color_histogram(observed_lab, visible, config.histogram_bins),
                                color_histogram(candidate_lab, hidden, config.histogram_bins))
    result['texture'] = texture_distance(texture_histograms(observed, visible),
                                        texture_histograms(candidate, hidden))
    wide_v, wide_h = dilate(hidden, 16) & visible, dilate(visible, 16) & hidden
    result['local_texture'] = texture_distance(texture_histograms(observed, wide_v),
                                              texture_histograms(candidate, wide_h))
    result['valid'] = True
    return result


METHODS = {
    'boundary': {'boundary': 1.},
    'color': {'color': 1.},
    'texture': {'texture': 1.},
    'boundary_color': {'boundary': .4 / .65, 'color': .25 / .65},
    'boundary_texture': {'boundary': .4 / .75, 'texture': .35 / .75},
    'surface_full': {'boundary': .4, 'color': .25, 'texture': .35},
    'local_surface': {'local_boundary': .6, 'local_texture': .4},
}


def select_candidate(scores, weights, scales=None):
    """Stable input-order tie breaks; explicit fixed-first fallback for invalid pools."""
    if not scores:
        raise ValueError('Cannot select from an empty candidate pool')
    scales = scales or {}
    if not weights or any(not math.isfinite(w) or w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError('Weights must be finite, nonnegative and nonzero')
    losses = []
    for score in scores:
        loss = 0.
        for key, weight in weights.items():
            if weight == 0:
                continue
            value, scale = score.get(key), scales.get(key, 1.)
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError('Normalization scales must be positive and finite')
            if not score.get('valid') or value is None or not math.isfinite(value):
                loss = math.inf
                break
            loss += weight * value / scale
        losses.append(loss)
    fallback = not any(math.isfinite(value) for value in losses)
    index = 0 if fallback else min(range(len(losses)), key=lambda i: losses[i])
    return dict(index=index, fallback=fallback, losses=[v if math.isfinite(v) else None for v in losses])
