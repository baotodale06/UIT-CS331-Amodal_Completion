"""Pretrained surface evidence computed without evaluation RGB or support."""
import numpy as np
from PIL import Image


DEEP_FEATURE_NAMES = ('visible_lpips', 'visible_deep_distance',
                      'partial_object_deep_distance', 'candidate_text_cosine')


class ObservedSurfaceFeatures:
    """Compare generated appearance to the genuinely observed target pixels."""
    def __init__(self, metrics):
        self.metrics = metrics

    def __call__(self, observed, visible, candidates, masks, label):
        import torch
        import torch.nn.functional as functional
        observed, visible = np.asarray(observed), np.asarray(visible, bool)
        if observed.dtype != np.uint8 or visible.shape != observed.shape[:2]:
            raise ValueError('Invalid observation frame')
        model = self.metrics
        neutral = observed.copy()
        neutral[~visible] = 127
        partial = observed.copy()
        partial[~visible] = 0
        with torch.inference_mode():
            tensor = model.tensor(neutral)
            visible_index = torch.from_numpy(visible).to(tensor.device)
            deep = model.dreamsim.embed(model.dream_tensor(neutral))
            partial_deep = model.dreamsim.embed(model.dream_tensor(partial))
            text_inputs = model.models.processor(text=[label], return_tensors='pt', padding=True).to(tensor.device)
            text = model.models.clip.get_text_features(**text_inputs)
            result = []
            for candidate, mask in zip(candidates, masks):
                candidate, mask = np.asarray(candidate), np.asarray(mask, bool)
                if candidate.shape != observed.shape or mask.shape != visible.shape:
                    raise ValueError('Candidate frame differs from observation')
                generated_visible = candidate.copy()
                generated_visible[~visible] = 127
                obj = candidate.copy()
                obj[~mask] = 0
                spatial = model.spatial(tensor, model.tensor(generated_visible))[0, 0]
                candidate_deep = model.dreamsim.embed(model.dream_tensor(generated_visible))
                object_deep = model.dreamsim.embed(model.dream_tensor(obj))
                clip_inputs = model.models.processor(images=Image.fromarray(obj), return_tensors='pt').to(tensor.device)
                clip_image = model.models.clip.get_image_features(**clip_inputs)
                result.append(dict(
                    visible_lpips=float(spatial[visible_index].mean()) if visible.any() else 1.,
                    visible_deep_distance=float(1 - functional.cosine_similarity(deep, candidate_deep)),
                    partial_object_deep_distance=float(1 - functional.cosine_similarity(partial_deep, object_deep)),
                    candidate_text_cosine=float(functional.cosine_similarity(text, clip_image))))
        return result
