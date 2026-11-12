"""CPU-only protocol decisions shared by inference, scoring, and tests."""
import json
from collections import Counter
from pathlib import Path
import re

import numpy as np

METRICS = ("CLIP", "LPIPS", "Feature Similarity", "SSIM")


def white_rgb(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Expected RGB or RGBA image")
    if image.shape[2] == 3:
        return image.copy()
    alpha = image[..., 3:4].astype(np.float64) / 255.
    return (image[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)


def align_rgba(reference, prediction, geometry):
    """Put both objects on one pixel-aligned canvas, including outpainting."""
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    width, height = geometry["original_size"]
    if reference.shape != (height, width, 4):
        raise ValueError("Reference RGBA must have original-image dimensions; cropped reference geometry is unknown")
    row, col = geometry["offset_rc"]
    ph, pw = prediction.shape[:2]
    top, left = min(0, row), min(0, col)
    bottom, right = max(height, row + ph), max(width, col + pw)
    shape = (bottom - top, right - left, 4)
    ref_canvas = np.zeros(shape, dtype=np.uint8)
    pred_canvas = np.zeros(shape, dtype=np.uint8)
    ref_canvas[-top:height-top, -left:width-left] = reference
    pred_canvas[row-top:row-top+ph, col-left:col-left+pw] = prediction
    return ref_canvas, pred_canvas


def failure_metrics():
    # Supplement p.8 specifies appearance penalties. It does not specify CLIP
    # for failures, so report its actual denominator instead of inventing a score.
    return {"CLIP": None, "LPIPS": 1., "Feature Similarity": 0., "SSIM": 0.}


def _optional_bool(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if not normalized:
            return None
        if normalized in ("true", "1"):
            return True
        if normalized in ("false", "0"):
            return False
    raise ValueError(f"Invalid iteration-limit flag: {value!r}")


def failure_stage(row):
    """Identify the failing stage without conflating adapter and PD-MC gates."""
    status, reason = row["status"], row.get("failure_reason")
    if status == "success":
        return ""
    if status == "missing_image":
        return "input_data"
    if status == "reference_failed":
        return "reference"
    if reason in ("no_detections", "segmentation_unavailable"):
        return "initial_segmentation"
    if status in ("not_detected", "ambiguous_target"):
        return "target_selection"
    if reason in ("query_mask_too_small", "detector_score_below_threshold", "unknown_detected_class", "invalid_query"):
        return "query_filter"
    if reason == "no_occlusion_detected":
        return "occlusion_check"
    if reason == "completion_not_detected":
        return "completion_segmentation"
    return "unclassified"


def aggregate(rows, *, detailed=False):
    output = {}
    sources = list(dict.fromkeys(r["source"] for r in rows))
    for source in sources + ["Overall"]:
        group = [r for r in rows if source == "Overall" or r["source"] == source]
        entry = {"targets": len(group), "successes": sum(r["status"] == "success" for r in group)}
        entry["failures"] = sum(r["status"] not in ("success", "missing_image", "reference_failed") for r in group)
        entry["reference_failures"] = sum(r["status"] == "reference_failed" for r in group)
        entry["missing_images"] = sum(r["status"] == "missing_image" for r in group)
        evaluated = entry["successes"] + entry["failures"]
        entry["failure_rate"] = entry["failures"] / evaluated if evaluated else None
        entry["status_counts"] = dict(Counter(r["status"] for r in group))
        entry["failure_reasons"] = dict(Counter(
            r.get("failure_reason", "unrecorded") for r in group
            if r["status"] not in ("success", "missing_image", "reference_failed")))
        if detailed:
            flags = [_optional_bool(r.get("iteration_limit_reached")) for r in group if r["status"] == "success"]
            entry.update(completed_outputs=sum(flag is False for flag in flags),
                         iteration_limited_outputs=sum(flag is True for flag in flags),
                         completion_state_unknown=sum(flag is None for flag in flags))
            stages = Counter(failure_stage(r) for r in group
                             if r["status"] not in ("success", "missing_image", "reference_failed"))
            entry.update(failure_stage_counts=dict(stages),
                         target_selection_failures=stages["target_selection"],
                         initial_segmentation_failures=stages["initial_segmentation"],
                         pdmc_failures=entry["failures"] - stages["target_selection"] - stages["initial_segmentation"])
        for metric in METRICS:
            values = [float(r[metric]) for r in group if r.get(metric) is not None]
            entry[metric] = sum(values) / len(values) if values else None
            entry[metric + "_n"] = len(values)
        output[source] = entry
    return output


def choose_target(masks, classes, label, reference_alpha=None, *,
                  reference_policy="strict", diagnostics=None):
    if reference_policy not in ("strict", "best"):
        raise ValueError("Reference matching policy must be strict or best")
    masks = np.asarray(masks)
    if len(masks) == 0:
        if diagnostics is not None:
            diagnostics.update(reason="no_detections", candidates=[])
        return None, "not_detected"
    if reference_alpha is not None:
        ref = np.asarray(reference_alpha) > 0
        if ref.shape != masks.shape[1:]:
            raise ValueError("Target-selection reference must match original image dimensions")
        scores = []
        for mask in masks:
            union = np.logical_or(mask, ref).sum()
            scores.append(np.logical_and(mask, ref).sum() / max(union, 1))
        # Stable first maximum, as in the released filter_out_amodal_segmentation.
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        if diagnostics is not None:
            diagnostics.update(policy=reference_policy, minimum_iou=.5, best_iou=float(scores[order[0]]),
                candidates=[{"index": int(i), "class": classes[i], "iou": float(scores[i])}
                            for i in order[:5]])
        if scores[order[0]] < .5:
            if diagnostics is not None:
                diagnostics["reason"] = "reference_iou_below_threshold"
            return None, "not_detected"
        if reference_policy == "strict" and len(order) > 1 and scores[order[0]] - scores[order[1]] < .05:
            if diagnostics is not None:
                diagnostics["reason"] = "reference_iou_near_tie"
            return None, "ambiguous_target"
        if diagnostics is not None:
            diagnostics["selected_index"] = int(order[0])
        return int(order[0]), "reference_iou"
    normalized = " ".join(label.casefold().split())
    matches = []
    for i, name in enumerate(classes):
        name = " ".join(name.casefold().split())
        if name and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", normalized):
            matches.append(i)
    if len(matches) == 1:
        return matches[0], "label_match"
    return None, "ambiguous_target" if matches else "not_detected"


def ensure_run_config(path, config):
    from .dataset import write_json
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != config:
            raise ValueError(f"Output directory has a different run configuration: {path}. "
                             "Use a new output directory for changed settings.")
    else:
        write_json(path, config)


def read_cached_result(path, row, image_hash, selection_reference_hash=None):
    from .dataset import sha256
    path = Path(path)
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["id"] != row["id"] or record["label"] != row["label"]:
        raise ValueError(f"Saved result does not match annotation: {row['id']}")
    if record["status"] == "missing_image" and record["input_sha256"] is None and image_hash is not None:
        return None  # Newly installed inputs are pending work, not changed inputs.
    if record["input_sha256"] != image_hash:
        raise ValueError(f"Saved result does not match current input: {row['id']}")
    if selection_reference_hash is not None and record.get("selection_reference_sha256") != selection_reference_hash:
        raise ValueError(f"Target-selection reference changed for {row['id']}; use a new output directory")
    if record["status"] == "success":
        completion = path.parent / "completion.png"
        if not completion.is_file() or sha256(completion) != record["completion_sha256"]:
            raise ValueError(f"Saved completion is missing or changed: {completion}")
    return record
