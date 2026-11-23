"""Scoring A.6 trên CPU; dữ liệu scorer chỉ gồm dự đoán và quan sát ban đầu."""
from dataclasses import dataclass
import numpy as np


@dataclass
class ScoreContext:
    shape_prior: np.ndarray
    visible_mask: np.ndarray
    prior_valid: np.ndarray

    def __post_init__(self):
        arrays = (self.shape_prior, self.visible_mask, self.prior_valid)
        if any(a.ndim != 2 or a.shape != arrays[0].shape for a in arrays):
            raise ValueError("Prior, visible và valid support phải cùng tọa độ/kích thước 2D")


def iou(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if a.shape != b.shape:
        raise ValueError("Hai mask phải cùng kích thước")
    return float((a & b).sum() / max(int((a | b).sum()), 1))


def target_mask(masks, visible, min_visible_recall=0.3):
    """Giữ quy tắc max IoU của PMC, nhưng không giả mask visible khi detector lỗi."""
    if masks is None or len(masks) == 0:
        return None, None, "no_detection"
    if not np.asarray(visible, bool).any():
        return None, None, "empty_visible_anchor"
    index = max(range(len(masks)), key=lambda k: iou(masks[k], visible))
    mask = np.asarray(masks[index], bool)
    recall = float((mask & visible).sum() / np.asarray(visible, bool).sum())
    if not mask.any() or recall < min_visible_recall:
        return None, None, "insufficient_visible_overlap"
    return index, mask, None


def shape_score(mask, ctx, weights=(0.70, 0.20, 0.10)):
    """Trả component để debug; None nghĩa là không có điểm hợp lệ, không phải 0."""
    if mask is None:
        return {"score": None, "reason": "invalid_target_mask"}
    if mask.shape != ctx.shape_prior.shape:
        raise ValueError("Candidate và prior không cùng crop")
    r = np.asarray(ctx.prior_valid, bool)
    m = np.asarray(mask, bool) & r
    s = np.asarray(ctx.shape_prior, bool) & r
    v = np.asarray(ctx.visible_mask, bool) & r
    if not m.any() or not v.any() or not s.any():
        return {"score": None, "reason": "empty_valid_support"}
    overlap = iou(m, s)
    recall = float((m & v).sum() / v.sum())
    leak = float((m & ~s).sum() / m.sum())
    wi, wv, wl = weights
    return {"score": float(wi * overlap + wv * recall - wl * leak),
            "iou": overlap, "visible_recall": recall, "leak": leak, "reason": None}


def select_candidate(scores):
    """Hòa điểm giữ seed đầu; tất cả invalid thì fallback candidate 0 có ghi lý do."""
    valid = [i for i, item in enumerate(scores)
             if item["score"] is not None and np.isfinite(item["score"])]
    if not scores:
        raise ValueError("Candidate pool rỗng")
    if not valid:
        return 0, "all_candidates_invalid"
    return max(valid, key=lambda i: scores[i]["score"]), None
