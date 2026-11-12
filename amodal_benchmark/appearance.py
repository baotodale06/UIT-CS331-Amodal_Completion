"""Read-only appearance diagnostics; predicted expansion is not true occlusion.

No learned metrics are recomputed here. Taxonomy labels require a recorded visual
review; numerical color differences alone are never converted into failure labels.
"""
from collections import defaultdict
import csv
import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .protocol import align_rgba

FAILURES = ("wrong_texture", "color_inconsistency", "pattern_inconsistency",
            "semantic_drift", "hidden_region_inconsistency", "cooccurring_object")
BIN_LABELS = ("[0,20)", "[20,30)", "[30,40)", "[40,50)", "[50,60)",
              "[60,70)", "[70,80)", "[80,100]")


def ratio_bin(ratio):
    if ratio is None:
        return "unavailable"
    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("Ratio must be finite and in [0,1]")
    for upper, label in zip((.2, .3, .4, .5, .6, .7, .8, math.inf), BIN_LABELS):
        if ratio < upper:
            return label


def mask_diagnostics(visible, predicted):
    visible, predicted = np.asarray(visible, bool), np.asarray(predicted, bool)
    if visible.shape != predicted.shape or visible.ndim != 2:
        raise ValueError("Masks must share a two-dimensional pixel frame")
    v, a = int(visible.sum()), int(predicted.sum())
    overlap = int((visible & predicted).sum())
    return {"visible_pixels": v, "predicted_pixels": a, "overlap_pixels": overlap,
            "generated_pixels": a - overlap, "lost_visible_pixels": v - overlap,
            "predicted_expansion_ratio": (a - overlap) / a if a and v else None,
            "visible_retention": overlap / v if v else None}


def rgb_to_lab(rgb):
    """sRGB / D65 to CIELAB, for a descriptive Delta E 1976 diagnostic."""
    rgb = np.asarray(rgb, dtype=np.float32) / 255
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    xyz = linear @ np.array([[.4124564, .3575761, .1804375],
                             [.2126729, .7151522, .0721750],
                             [.0193339, .1191920, .9503041]], dtype=np.float32).T
    xyz /= np.array([.95047, 1, 1.08883], dtype=np.float32)
    delta = 6 / 29
    f = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4 / 29)
    return np.stack((116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])), axis=-1)


def erode(mask, size=5):
    return np.asarray(Image.fromarray(np.asarray(mask, np.uint8) * 255)
                      .filter(ImageFilter.MinFilter(size))) > 0


def aligned_artifacts(run_dir, record):
    run_dir = Path(run_dir)
    with Image.open(run_dir / "visible_references" / record["reference"]) as image:
        reference = np.array(image.convert("RGBA"))
    with Image.open(run_dir / "visible_references/masks" / record["reference"]) as image:
        visible = np.array(image.convert("L")) > 0
    # Geometry comes from saved integer crop/outpaint offsets; never resize masks.
    raw = reference.copy()
    raw[..., 3] = visible.astype(np.uint8) * 255
    with Image.open(run_dir / "targets" / record["id"] / "completion.png") as image:
        completion = np.array(image.convert("RGBA"))
    return align_rgba(raw, completion, record["geometry"])


def appearance_diagnostics(reference, completion):
    v, a = reference[..., 3] > 0, completion[..., 3] > 0
    values = mask_diagnostics(v, a)
    core = erode(v & a)
    values["visible_color_pixels"] = int(core.sum())
    values["visible_delta_e76_median"] = None
    values["visible_delta_e76_p90"] = None
    if core.any():
        difference = np.linalg.norm(rgb_to_lab(reference[..., :3][core]) -
                                    rgb_to_lab(completion[..., :3][core]), axis=1)
        values["visible_delta_e76_median"] = float(np.median(difference))
        values["visible_delta_e76_p90"] = float(np.percentile(difference, 90))
    values["proxy_bin"] = ratio_bin(values["predicted_expansion_ratio"])
    return values


def stable_sample(rows, per_bin=15, seed=20261107):
    """Stratified hash sample, selected without looking at scores or image quality."""
    if per_bin < 1:
        raise ValueError("per_bin must be positive")
    groups = defaultdict(list)
    for row in rows:
        if row["status"] == "success" and row.get("proxy_bin") in BIN_LABELS:
            groups[row["proxy_bin"]].append(row)
    output = []
    for label in BIN_LABELS:
        group = sorted(groups[label], key=lambda r: hashlib.sha256(
            f"{seed}:{r['id']}".encode()).hexdigest())
        for row in group[:per_bin]:
            output.append({"id": row["id"], "proxy_bin": label,
                           "stratum_population": len(group),
                           "stratum_sample": min(per_bin, len(group)),
                           "inclusion_probability": min(per_bin, len(group)) / len(group)})
    return output


def bootstrap_mean(values, seed=20261107, repetitions=2000):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite observations")
    if len(values) == 1:
        return {"n": 1, "mean": float(values[0]), "ci_low": None, "ci_high": None}
    means = np.random.default_rng(seed).choice(values, (repetitions, len(values))).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"n": len(values), "mean": float(values.mean()),
            "ci_low": float(low), "ci_high": float(high)}


def wilson_interval(positive, total):
    if total == 0:
        return None, None
    p, z = positive / total, 1.959963984540054
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0, center - half), min(1, center + half)


def read_review(path, sampled_ids):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(sampled_ids):
        raise ValueError("Review must contain every sampled ID exactly once")
    for row in rows:
        for key in FAILURES:
            if row.get(key) not in ("0", "1", "2", "3", "U", "NA"):
                raise ValueError(f"Invalid or unreviewed {key} for {row['id']}")
        if not row.get("evidence", "").strip():
            raise ValueError("Every review needs an evidence note")
        if not row.get("reviewer", "").strip():
            raise ValueError("Every review needs an identified reviewer")
    return rows


def taxonomy_counts(review):
    result = []
    for key in FAILURES:
        assessed = [int(r[key]) for r in review if r[key] not in ("NA", "U")]
        n, k = len(assessed), sum(v > 0 for v in assessed)
        low, high = wilson_interval(k, n)
        result.append({"failure": key, "positive": k, "assessable": n,
                       "sample_total": len(review), "unassessable": sum(r[key] == "U" for r in review),
                       "not_applicable": sum(r[key] == "NA" for r in review),
                       "frequency": k / n if n else None,
                       "wilson_low": low, "wilson_high": high,
                       "mild": assessed.count(1), "moderate": assessed.count(2),
                       "severe": assessed.count(3)})
    return result


def weighted_taxonomy_counts(review):
    """Reweight the stratified audit; unknown/NA labels stay outside denominators.

    This estimates frequencies among assessable saved outputs, conditional on
    one reviewer's judgments. It is not a whole-benchmark failure rate, and no
    annotation reliability or missing-label uncertainty is implied.
    """
    counts = taxonomy_counts(review)
    for stat in counts:
        key = stat["failure"]
        positive = eligible = unknown = inapplicable = 0.
        for row in review:
            population, sample = int(row["stratum_population"]), int(row["stratum_sample"])
            if not 0 < sample <= population:
                raise ValueError("Invalid sampling denominator")
            weight = population / sample
            label = row[key]
            if label == "U":
                unknown += weight
            elif label == "NA":
                inapplicable += weight
            else:
                eligible += weight
                positive += weight * (int(label) > 0)
        stat.update(weighted_positive=positive, weighted_assessable=eligible,
                    weighted_unknown=unknown, weighted_not_applicable=inapplicable,
                    weighted_frequency=positive / eligible if eligible else None)
    return counts


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
