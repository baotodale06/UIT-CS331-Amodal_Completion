"""Independently verify native pixels and geometry before shape-only admission."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256
from amodal_benchmark.shape_calibrated_prior import candidate_policy
from amodal_benchmark.shape_calibrated_runtime import read, verify_native_records
from scripts.recover_shape_sam_validation import geometry
from scripts.validate_shape_external_adapter import binary_mask, pixel_hash, safe_asset, index_rows


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["image_id"]].append(row)
    result = dict(objects=len(rows), images=len(groups))
    for metric in ("full_iou", "hidden_iou", "brier", "nll"):
        means = [np.mean([row[metric] for row in group if row[metric] is not None])
                 for group in groups.values() if any(row[metric] is not None for row in group)]
        result[metric] = float(np.mean(means)) if means else None
        result[metric + "_images"] = len(means)
    return result


def replay(package, data=None, sam=None):
    package = Path(package)
    candidate = package / "candidate"
    policy = candidate_policy(candidate)
    native = read(candidate / "selected-native.json")
    for family in ("local", "quadro"):
        contract, result = (read(package / (family + suffix)) for suffix in ("-contract.json", "-result.json"))
        if result.get("contract_sha256") != sha256(package / (family + "-contract.json")):
            raise ValueError("Native contract binding changed")
        verify_native_records(candidate, contract, result, family)
    data = Path(data) if data is not None else ROOT / "output/shape-v2-20261123/data"
    sam = Path(sam) if sam is not None else ROOT / "output/shape-validation-sam-20261123/native/result"
    packed = (data / "manifest.json").read_bytes()
    if hashlib.sha256(packed).hexdigest() != policy["data_sha256"]:
        raise ValueError("Native validation manifest changed")
    manifest = json.loads(packed)
    selected = {row["id"]: row for row in manifest["rows"] if row["split"] == "validation"}
    index_rows(native["rows"], selected)
    if sha256(sam / "result.json") != policy["baseline_result_sha256"]:
        raise ValueError("Frozen SAM baseline changed")
    baseline = index_rows(read(sam / "result.json")["rows"], selected)
    rows = []
    with zipfile.ZipFile(candidate / "selected-masks.zip") as masks:
        if sorted(masks.namelist()) != sorted(key + ".png" for key in selected):
            raise ValueError("Complete selected-mask inventory required")
        for row in native["rows"]:
            source = selected[row["id"]]
            path = safe_asset(data, source["mask"])
            if sha256(path) != manifest["artifacts"][source["mask"]]:
                raise ValueError("True amodal validation label changed")
            with Image.open(path) as image:
                encoded = np.array(image)
            truth, visible = (encoded & 1) > 0, (encoded & 2) > 0
            prediction = binary_mask(masks.read(row["id"] + ".png"), truth.shape)
            if pixel_hash(prediction) != row["prediction_sha256"]:
                raise ValueError("Selected native pixels changed")
            support_path = safe_asset(sam, "objects/" + row["id"] + "/sam-visible.png")
            if sha256(support_path) != baseline[row["id"]]["artifacts"]["sam-visible.png"]:
                raise ValueError("Original SAM support changed")
            support = binary_mask(support_path.read_bytes(), truth.shape)
            if np.any(support & ~prediction) or (not support.any() and prediction.any()):
                raise ValueError("Input support policy changed")
            metrics = geometry(prediction, truth, visible)
            if any(row[key] != value for key, value in metrics.items()):
                raise ValueError("Independent native geometry differs")
            rows.append(dict(row, **metrics))
    computed = aggregate(rows)
    if computed != native["summary"] or computed != policy["native_selected"]:
        raise ValueError("Selected image-mean geometry or aggregation changed")
    if any(computed[metric] < policy["native_baseline"][metric] for metric in ("full_iou", "hidden_iou")):
        raise ValueError("Native geometry floor failed")
    return dict(verified=True, objects=1264, images=217,
        candidate_sha256=sha256(candidate / "candidate.json"),
        local_result_sha256=sha256(package / "local-result.json"),
        quadro_result_sha256=sha256(package / "quadro-result.json"), selected_geometry=computed,
        replay_source_sha256=sha256(Path(__file__)), benchmark_success=False,
        probability_scores_independently_recomputed=False,
        scope="Native masks/geometry independently replayed; GPU pixel hashes matched these masks; not a benchmark result")
