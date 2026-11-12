"""Observation-only surface calibration and regularized candidate ranking.

Calibration estimates generator color drift on observed pixels. It never sees
hidden RGB or ground-truth support and does not replace generated texture with
an average of candidates. All inputs share one explicitly aligned RGB frame.
"""
from dataclasses import asdict, dataclass

import numpy as np

from .surface import erode
from .surface_features import DEEP_FEATURE_NAMES


@dataclass(frozen=True)
class RefinementConfig:
    mode: str = 'shift'
    strength: float = 0.5
    max_shift: float = 32.0
    sigma: float = 32.0
    decay: float = 128.0
    preserve_visible: bool = True


def refine_surface(observed, visible, candidate, predicted, config=RefinementConfig()):
    """Carry a bounded observed RGB residual into predicted generated pixels."""
    observed, candidate = np.asarray(observed), np.asarray(candidate)
    visible, predicted = np.asarray(visible, bool), np.asarray(predicted, bool)
    if observed.dtype != np.uint8 or candidate.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB')
    if observed.shape != candidate.shape or observed.ndim != 3 or observed.shape[-1] != 3:
        raise ValueError('RGB frame mismatch')
    if visible.shape != observed.shape[:2] or predicted.shape != visible.shape:
        raise ValueError('Mask frame mismatch')
    if config.mode not in ('none', 'shift', 'affine', 'field'):
        raise ValueError('Unknown correction mode')
    if not 0 <= config.strength <= 1 or config.max_shift < 0 or config.sigma <= 0 or config.decay <= 0:
        raise ValueError('Invalid correction parameters')
    support = erode(visible & predicted, 2)
    if support.sum() < 64:
        support = erode(visible, 1)
    result = candidate.astype(np.float32)
    details = dict(config=asdict(config), support_pixels=int(support.sum()), applied=False)
    if support.sum() >= 64 and config.mode != 'none' and config.strength:
        residual = observed.astype(np.float32) - result
        shift = np.clip(np.median(residual[support], axis=0), -config.max_shift, config.max_shift)
        details['shift_rgb'] = shift.tolist()
        if config.mode == 'shift':
            correction = np.broadcast_to(shift, result.shape)
        elif config.mode == 'affine':
            # Ridge toward unit slope, with robust residual clipping. A minimum
            # denominator prevents unstable fits on near-constant surfaces.
            x, y = result[support].astype(np.float64), observed[support].astype(np.float64)
            mx, my = np.median(x, axis=0), np.median(y, axis=0)
            xc, yc = x - mx, y - my
            slope = np.clip((np.mean(xc * yc, axis=0) + 400) /
                            (np.mean(xc * xc, axis=0) + 400), .8, 1.25)
            correction = np.clip((result - mx) * (slope - 1) + my - mx,
                                 -config.max_shift, config.max_shift)
            details['slope_rgb'] = slope.tolist()
        else:
            from scipy.ndimage import distance_transform_edt, gaussian_filter
            # Smooth residuals, not the generated image. Nearest support extends
            # the estimated field beyond the Gaussian's observed neighborhood.
            denominator = gaussian_filter(support.astype(np.float32), config.sigma)
            field = np.stack([gaussian_filter(np.clip(residual[..., c],
                -config.max_shift, config.max_shift) * support, config.sigma) /
                np.maximum(denominator, 1e-6) for c in range(3)], axis=-1)
            distance, indices = distance_transform_edt(~support, return_indices=True)
            nearest = field[indices[0], indices[1]]
            mix = np.clip(denominator / .1, 0, 1)[..., None]
            correction = (mix * field + (1 - mix) * nearest) * np.exp(
                -distance[..., None] / config.decay)
        generated = predicted & ~visible
        result[generated] += config.strength * correction[generated]
        details['applied'] = bool(generated.any())
        details['generated_pixels'] = int(generated.sum())
    if config.preserve_visible:
        result[visible] = observed[visible]
    return np.clip(np.rint(result), 0, 255).astype(np.uint8), details


FEATURE_NAMES = ('boundary', 'color', 'texture', 'local_boundary', 'local_texture',
                 'visible_mae', 'missing_visible', 'log_expansion', 'boundary_color',
                 'preservation_local', 'color_texture', 'invalid')
POOL_FEATURE_NAMES = ('consensus_mae', 'consensus_local_mae', 'mask_disagreement',
                      'visible_shift', 'visible_residual_spread', 'texture_scale_mismatch')


def rank_features(score, feature_names=None):
    """Observable features only; missing values are fitted using training medians."""
    values = [score.get(k) for k in FEATURE_NAMES[:6]]
    values = [float(v) if v is not None and np.isfinite(v) else np.nan for v in values]
    boundary, color, texture, local, local_texture, visible_mae = values
    values += [1 - score.get('visible_recall', 0),
               np.log1p(score.get('hidden_pixels', 0) / max(1, score.get('visible_pixels', 0))),
               boundary * color, visible_mae * local, color * texture,
               float(not score.get('valid'))]
    if feature_names is None:
        return np.asarray(values, float)
    by_name = dict(zip(FEATURE_NAMES, values))
    by_name.update({k: score.get(k, np.nan) for k in POOL_FEATURE_NAMES + DEEP_FEATURE_NAMES})
    return np.asarray([by_name[k] for k in feature_names], float)


def predict_rank(scores, model):
    features = np.stack([rank_features(s, model.get('feature_names')) for s in scores])
    medians = np.asarray(model['medians'])
    features = np.where(np.isfinite(features), features, medians)
    features = (features - np.asarray(model['center'])) / np.asarray(model['scale'])
    if model.get('kind') == 'gradient_boosting':
        features = (features - features.mean(0)).astype(np.float32)
        values = np.full(len(features), model['initial_prediction'], float)
        for tree in model['trees']:
            for i, vector in enumerate(features):
                node = 0
                while tree['children_left'][node] != -1:
                    node = (tree['children_left'][node] if vector[tree['feature'][node]] <= tree['threshold'][node]
                            else tree['children_right'][node])
                values[i] += model['learning_rate'] * tree['value'][node]
    else:
        values = features @ np.asarray(model['coefficients'])
    if model.get('valid_only', True):
        valid = np.array([s.get('valid', False) for s in scores])
        if valid.any():
            values[~valid] = np.inf
        else:
            return 0, [None] * len(scores)
    index = int(np.argmin(values))
    rule = model.get('selection_rule')
    if rule:
        if rule.get('kind') != 'semantic_within_surface_topk' or int(rule['top_k']) < 1:
            raise ValueError('Unknown or invalid surface-constrained selection rule')
        order = np.argsort(values, kind='stable')
        eligible = [int(i) for i in order if np.isfinite(values[i])][:int(rule['top_k'])]
        semantic = [scores[i].get('candidate_text_cosine') for i in eligible]
        if all(v is not None and np.isfinite(v) for v in semantic):
            index = eligible[int(np.argmax(semantic))]
    return index, [float(v) if np.isfinite(v) else None for v in values]


def observation_in_completion_frame(source_rgb, source_visible, completion_shape, offset_rc):
    """Map observed pixels into a completion crop using the saved offset, no resize."""
    source_rgb, source_visible = np.asarray(source_rgb), np.asarray(source_visible, bool)
    if source_rgb.shape[:2] != source_visible.shape:
        raise ValueError('Source mask frame mismatch')
    height, width = completion_shape[:2]
    row, col = offset_rc
    if not all(isinstance(v, (int, np.integer)) for v in (row, col)):
        raise ValueError('Offset must contain integral row and column coordinates')
    observed = np.zeros((height, width, 3), np.uint8)
    visible = np.zeros((height, width), bool)
    sr0, sc0 = max(0, row), max(0, col)
    sr1, sc1 = min(source_rgb.shape[0], row + height), min(source_rgb.shape[1], col + width)
    if sr1 > sr0 and sc1 > sc0:
        dest = (slice(sr0 - row, sr1 - row), slice(sc0 - col, sc1 - col))
        observed[dest] = source_rgb[sr0:sr1, sc0:sc1]
        visible[dest] = source_visible[sr0:sr1, sc0:sc1]
    return observed, visible


def pool_surface_features(observed, visible, occlusion, candidates, masks):
    """Estimate sampling agreement and observed color drift without hidden truth."""
    from .surface import dilate
    observed = np.asarray(observed, np.float32)
    visible, occlusion = np.asarray(visible, bool), np.asarray(occlusion, bool)
    images = np.stack(candidates).astype(np.float32)
    masks = np.asarray(masks, bool)
    if images.shape[1:] != observed.shape or masks.shape != images.shape[:3]:
        raise ValueError('Pool frame mismatch')
    support = masks.any(0) & occlusion & ~visible
    local = support & dilate(visible, 16)
    median = np.median(images, axis=0)
    values = []
    for i, (candidate, mask) in enumerate(zip(images, masks)):
        hidden = mask & occlusion & ~visible
        residual = observed[visible] - candidate[visible]
        shift = np.median(residual, axis=0) if len(residual) else np.zeros(3)
        disagreement = []
        for j, other in enumerate(masks):
            if j == i:
                continue
            a, b = mask & occlusion, other & occlusion
            disagreement.append(1 - float((a & b).sum() / max(1, (a | b).sum())))
        obs_std = np.std(observed[visible], axis=0) if visible.any() else np.zeros(3)
        hid_std = np.std(candidate[hidden], axis=0) if hidden.any() else np.zeros(3)
        values.append(dict(
            consensus_mae=float(np.abs(candidate[support] - median[support]).mean()/255) if support.any() else 1.,
            consensus_local_mae=float(np.abs(candidate[local] - median[local]).mean()/255) if local.any() else 1.,
            mask_disagreement=float(np.mean(disagreement)) if disagreement else 0.,
            visible_shift=float(np.abs(shift).mean()/255),
            visible_residual_spread=float(np.abs(residual - shift).mean()/255) if len(residual) else 1.,
            texture_scale_mismatch=float(np.abs(np.log((hid_std + 4)/(obs_std + 4))).mean())))
    return values
