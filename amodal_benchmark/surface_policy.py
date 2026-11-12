"""Deployable Surface-PMC policy; this module has no evaluation data interface."""
import numpy as np
from pathlib import Path

from .surface import score_surface
from .surface_refinement import pool_surface_features, predict_rank, refine_surface, RefinementConfig


def verify_frozen_surface_sources(freeze, root=None):
    """Reject stale deployments before publishing choices or accessing test RGB."""
    from .dataset import sha256
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    if freeze.get('status') != 'frozen_before_fresh_test' or not freeze.get('source_hashes'):
        raise ValueError('A final method freeze with implementation hashes is required')
    for relative, expected in freeze['source_hashes'].items():
        path = (root / relative.replace('\\', '/')).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256(path) != expected:
            raise ValueError('Frozen implementation differs from the deployment: ' + relative)


def apply_frozen_surface(observed, visible, occlusion, candidates, masks, freeze,
                         scores=None, deep_features=None):
    """Select and calibrate one generated image using observation-only evidence."""
    if len(candidates) != len(freeze['candidate_seeds']) or len(masks) != len(candidates):
        raise ValueError('Candidate inventory differs from the frozen method')
    if scores is None:
        scores = [score_surface(observed,visible,occlusion,rgb,mask)
                  for rgb,mask in zip(candidates,masks)]
    if len(scores) != len(candidates):
        raise ValueError('One score record per candidate is required')
    extra = pool_surface_features(observed,visible,occlusion,candidates,masks)
    scores = [dict(score,**features) for score,features in zip(scores,extra)]
    if freeze.get('pretrained_visible_features'):
        if deep_features is None or len(deep_features) != len(candidates):
            raise ValueError('The frozen policy requires pretrained visible features')
        scores = [dict(score,**features) for score,features in zip(scores,deep_features)]
    index, losses = predict_rank(scores,freeze['ranker'])
    config = RefinementConfig(**freeze['refinement'])
    if freeze.get('ensemble'):
        from .surface_ensemble import EnsembleConfig, refine_surface_pool
        calibrated = [refine_surface(observed,visible,rgb,mask,config)[0]
                      for rgb,mask in zip(candidates,masks)]
        refined, details = refine_surface_pool(calibrated,masks,visible,EnsembleConfig(**freeze['ensemble']))
        selected = refined[index]
    else:
        selected, details = refine_surface(observed,visible,candidates[index],masks[index],config)
    return selected, np.asarray(masks[index],bool).copy(), dict(
        index=index,seed=freeze['candidate_seeds'][index],losses=losses,
        fallback=not any(score.get('valid') for score in scores),
        observable_scores=scores,calibration=details)
