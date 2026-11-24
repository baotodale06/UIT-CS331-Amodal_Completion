"""Proper binary probability supervision outside the observable shape support."""


def probability_loss(logits, truth, support, total_objects=None):
    """Mean per-object Bernoulli NLL over pixels the prior is allowed to predict.

    SAM support is restored at inference, so those logits cannot influence the
    output. Targets remain the true amodal annotation. No class weighting, Dice
    term or hidden-positive bonus shifts the conditional probability optimum.
    ``total_objects`` permits equivalent accumulation over uneven microbatches.
    """
    import torch
    import torch.nn.functional as functional
    if logits.ndim!=4 or logits.shape[1]!=1 or logits.shape!=truth.shape or truth.shape!=support.shape:
        raise ValueError('Matching NCHW single-channel tensors are required')
    total = len(logits) if total_objects is None else total_objects
    if not isinstance(total,int) or total<len(logits) or total<=0:
        raise ValueError('A valid global object count is required')
    if (not torch.isfinite(truth).all() or not torch.isfinite(support).all()
            or torch.any((truth!=0)&(truth!=1)) or torch.any((support!=0)&(support!=1))):
        raise ValueError('Truth and input support must be binary masks')
    unknown = 1-support.float()
    per_pixel = functional.binary_cross_entropy_with_logits(logits.float(),truth.float(),reduction='none')
    per_object = (per_pixel*unknown).sum((1,2,3))/unknown.sum((1,2,3)).clamp_min(1)
    return per_object.sum()/total
