"""Frozen probability-correction candidate; validation only until GPU admission."""
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import sha256
from .shape_v2 import observation_tensor, visible_box


def candidate_policy(folder):
    folder = Path(folder)
    policy = json.loads((folder / "candidate.json").read_bytes())
    if (policy.get("schema") != 1 or policy.get("method") != "shape-calibrated-probability"
            or policy.get("checkpoint_epoch") != 7 or policy.get("logit_bias") != -2.
            or policy.get("benchmark_admitted") is not False or policy.get("benchmark_success") is not False
            or policy.get("holdout_access") is not False or policy.get("benchmark_access") is not False):
        raise ValueError("Frozen shape-correction candidate policy changed")
    expected_files = {"best.pt", "training.json", "diagnostic-replay.json", "selected-native.json", "selected-masks.zip"}
    if set(policy.get("files", {})) != expected_files:
        raise ValueError("Candidate evidence inventory changed")
    for name, digest in policy["files"].items():
        if sha256(folder / name) != digest:
            raise ValueError("Candidate artifact changed: " + name)
    if policy["files"]["best.pt"] != policy["checkpoint_sha256"]:
        raise ValueError("Candidate checkpoint binding changed")
    root = Path(__file__).resolve().parents[1]
    for name, digest in policy["sources"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256(path) != digest:
            raise ValueError("Candidate source changed: " + name)
    return policy


@contextmanager
def arithmetic():
    import torch
    settings = ((torch.backends.cudnn, "deterministic", True),
                (torch.backends.cudnn, "benchmark", False),
                (torch.backends.cudnn, "allow_tf32", False),
                (torch.backends.cuda.matmul, "allow_tf32", False))
    previous = [(owner, name, getattr(owner, name)) for owner, name, _ in settings]
    try:
        for owner, name, value in settings:
            setattr(owner, name, value)
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            yield
    finally:
        for owner, name, value in previous:
            setattr(owner, name, value)


def original_prior(model, rgb, support, bias=-2.):
    """Correct logits in the native crop; never consume generated candidate RGB."""
    import torch
    rgb, support = np.asarray(rgb, np.uint8), np.asarray(support, bool)
    if rgb.shape != (*support.shape, 3) or bias != -2.:
        raise ValueError("Frozen observation frame and logit correction required")
    canvas = np.zeros(support.shape, np.float32)
    if not support.any():
        return canvas, None, None, None
    box = visible_box(support, 2.)
    observed = np.array(Image.fromarray(rgb).crop(box))
    cropped = np.array(Image.fromarray(support).crop(box))
    observed[~cropped] = 127
    device = next(model.parameters()).device
    with arithmetic():
        inputs = torch.from_numpy(observation_tensor(observed, cropped)).unsqueeze(0).to(device)
        logits = model(inputs)
        native_logits = torch.nn.functional.interpolate(logits, cropped.shape, mode="bilinear", align_corners=False)
        probability = (native_logits + bias).sigmoid()[0, 0].cpu().numpy()
    if not np.isfinite(probability).all():
        raise ValueError("Nonfinite candidate probability")
    x0, y0, x1, y1 = box
    height, width = support.shape
    left, top, right, bottom = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    canvas[top:bottom, left:right] = probability[top-y0:bottom-y0, left-x0:right-x0]
    canvas[support] = 1
    return canvas, box, observed, cropped
