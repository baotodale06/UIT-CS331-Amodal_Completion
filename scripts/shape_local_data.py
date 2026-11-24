"""Native local arithmetic and memory-limited shape-loss accumulation."""
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from amodal_benchmark.shape_v2 import observation_tensor, visible_box
from scripts.shape_robust_data import geometry, native_canvas


def accumulated_loss(logits, truth, visible, total_objects, total_hidden_pixels):
    """One microbatch contribution to the original global-batch shape loss.

    The positive-hidden BCE uses a pixel denominator across the entire batch;
    averaging independently normalized microbatch losses would change the loss.
    """
    import torch.nn.functional as functional
    if total_objects < len(logits) or total_objects <= 0 or total_hidden_pixels < 0:
        raise ValueError('Invalid global batch normalization')
    probability = logits.sigmoid()
    hidden = truth*(1-visible)
    predicted_hidden = probability*(1-visible)
    axes = (1,2,3)

    def dice(prediction, target):
        return (1-(2*(prediction*target).sum(axes)+1)/(prediction.sum(axes)+target.sum(axes)+1)).sum()/total_objects

    bce = functional.binary_cross_entropy_with_logits(logits,truth)*len(logits)/total_objects
    positive = (functional.softplus(-logits)*hidden).sum()/max(1,total_hidden_pixels)
    return bce+dice(probability,truth)+dice(predicted_hidden,hidden)+.5*positive


def selection_floor(local, original):
    keys = ('full_iou','hidden_iou')
    if any(not np.isfinite(row[key]) or not 0 <= row[key] <= 1 for row in (local,original) for key in keys):
        raise ValueError('Finite bounded baseline geometry is required')
    return {key:max(local[key],original[key]) for key in keys}


def eligible(result, floor, best):
    if any(not np.isfinite(result[key]) or not 0 <= result[key] <= 1 for key in ('full_iou','hidden_iou')):
        raise ValueError('Finite bounded validation geometry is required')
    return (result['full_iou'] >= floor['full_iou'] and result['hidden_iou'] >= floor['hidden_iou']
            and (result['hidden_iou'],result['full_iou']) > tuple(best))


@contextmanager
def native_arithmetic():
    import torch
    settings = ((torch.backends.cudnn,'deterministic',False),(torch.backends.cudnn,'benchmark',False),
                (torch.backends.cudnn,'allow_tf32',False),(torch.backends.cuda.matmul,'allow_tf32',False))
    previous = [(owner,key,getattr(owner,key)) for owner,key,_ in settings]
    try:
        for owner,key,value in settings:
            setattr(owner,key,value)
        with torch.autocast('cuda',enabled=False):
            yield
    finally:
        for owner,key,value in previous:
            setattr(owner,key,value)


def evaluate_native(model, data, selected, sam_root, device, mask_archive=None):
    """Return exact mask hashes as well as native per-object geometry."""
    import torch
    groups = defaultdict(list)
    for row in selected:
        groups[row['image_id']].append(row)
    model.eval()
    records = []
    archive = zipfile.ZipFile(mask_archive,'x',zipfile.ZIP_DEFLATED) if mask_archive else None
    try:
        with native_arithmetic(),torch.inference_mode():
            for image_id,objects in sorted(groups.items()):
                with Image.open(Path(data)/objects[0]['image']) as image:
                    rgb = np.array(image.convert('RGB'))
                for row in objects:
                    with Image.open(Path(data)/row['mask']) as image:
                        encoded = np.array(image)
                    truth,visible = (encoded&1)>0,(encoded&2)>0
                    with Image.open(Path(sam_root)/'objects'/row['id']/'sam-visible.png') as image:
                        support = np.array(image)>0
                    if support.shape != truth.shape or rgb.shape[:2] != support.shape:
                        raise ValueError('Native input frames differ')
                    if support.any():
                        box = visible_box(support,2.)
                        observed = np.array(Image.fromarray(rgb).crop(box))
                        cropped = np.array(Image.fromarray(support).crop(box))
                        inputs = torch.from_numpy(observation_tensor(observed,cropped)).unsqueeze(0).to(device)
                        logits = model(inputs)
                        probability = torch.nn.functional.interpolate(logits,cropped.shape,mode='bilinear',align_corners=False).sigmoid()
                        prediction = (native_canvas(probability[0,0].cpu().numpy(),box,truth.shape)>=.5)|support
                    else:
                        box,prediction = None,np.zeros_like(truth)
                    record = dict(id=row['id'],image_id=image_id,box=list(box) if box else None,
                        shape=list(truth.shape),prediction_sha256=hashlib.sha256(prediction.astype(np.uint8).tobytes()).hexdigest(),
                        **geometry(prediction,truth,visible))
                    records.append(record)
                    if archive:
                        stream = io.BytesIO()
                        Image.fromarray(prediction.astype(np.uint8)*255).save(stream,format='PNG')
                        archive.writestr(row['id']+'.png',stream.getvalue())
        return sorted(records,key=lambda row:row['id'])
    finally:
        if archive:
            archive.close()
