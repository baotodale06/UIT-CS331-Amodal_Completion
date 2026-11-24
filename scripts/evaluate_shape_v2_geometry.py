"""Evaluate natural amodal geometry in the full image, with no crop-label leakage."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.shape_v2_runtime import load_prior, original_prior, shape_v2_contract


def geometry_scores(probability, visible, truth):
    """Native-frame metrics include truth outside the model's visible-based crop."""
    if probability.shape != truth.shape or visible.shape != truth.shape:
        raise ValueError("Native geometry frames differ")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Invalid native probability")
    prediction = (probability >= .5) | visible
    hidden = truth & ~visible
    predicted_hidden = prediction & ~visible
    def iou(first, second):
        return float((first & second).sum() / max(1, (first | second).sum()))
    return dict(full_iou=iou(prediction, truth), visible_only_full_iou=iou(visible, truth),
        hidden_iou=iou(predicted_hidden, hidden) if hidden.any() else None,
        hidden_fraction=float(hidden.sum()/max(1, truth.sum())),
        predicted_pixels=int(prediction.sum()), true_pixels=int(truth.sum()))


def grouped_means(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["image_id"]].append(row)
    result = dict(objects=len(rows), images=len(groups))
    for metric in ("full_iou", "visible_only_full_iou", "hidden_iou"):
        means = [np.mean([r[metric] for r in group if r[metric] is not None])
                 for group in groups.values() if any(r[metric] is not None for r in group)]
        result[metric] = float(np.mean(means)) if means else None
        result[metric + "_images"] = len(means)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_args = parser.add_mutually_exclusive_group(required=True)
    model_args.add_argument("--checkpoint", type=Path)
    model_args.add_argument("--shape-calibrated-package", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "holdout"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing geometry evaluations; use a new output path")
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if args.shape_calibrated_package:
        from amodal_benchmark.shape_calibrated_runtime import load_prior as load_model, shape_calibrated_contract
        from amodal_benchmark.shape_calibrated_prior import original_prior as predict
        policy = shape_calibrated_contract(args.shape_calibrated_package)
        model_location = args.shape_calibrated_package
    else:
        load_model, predict = load_prior, original_prior
        policy = shape_v2_contract(args.checkpoint)
        model_location = args.checkpoint
    freeze = json.loads(args.freeze.read_bytes()) if args.freeze else None
    if args.split == "holdout":
        if (freeze is None or freeze.get("shape_policy") != policy
                or freeze.get("holdout_manifest_sha256") != sha256(manifest_path)
                or freeze.get("geometry_source_sha256") != sha256(Path(__file__))):
            raise ValueError("Holdout evaluation requires the exact prior frozen model, policy and manifest")
        if set(row["split"] for row in manifest["rows"]) != {"holdout"}:
            raise ValueError("Holdout must be packaged separately from training")
    selected = [r for r in manifest["rows"] if r["split"] == args.split]
    if not selected:
        raise ValueError("No geometry objects in requested split")
    model = load_model(model_location)
    verified, results = set(), []
    for index, row in enumerate(selected):
        for name in (row["image"], row["mask"]):
            path = (args.data / name).resolve()
            if not path.is_relative_to(args.data.resolve()):
                raise ValueError("Geometry asset path escapes its data directory")
            if name not in verified:
                if sha256(path) != manifest["artifacts"][name]:
                    raise ValueError("Geometry asset hash mismatch: " + name)
                verified.add(name)
        with Image.open(args.data / row["image"]) as image:
            rgb = np.array(image.convert("RGB"))
        with Image.open(args.data / row["mask"]) as image:
            encoded = np.array(image)
        truth, visible = (encoded & 1) > 0, (encoded & 2) > 0
        probability, box, _, _ = predict(model, rgb, visible)
        results.append(dict(id=row["id"], image_id=row["image_id"], box_xyxy=box,
                            **geometry_scores(probability, visible, truth)))
        if index % 100 == 0:
            print(f"{args.split}: {index+1}/{len(selected)} objects", flush=True)
    report = dict(schema=1, split=args.split, at=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=sha256(manifest_path), shape_policy=policy,
        geometry_source_sha256=sha256(Path(__file__)),
        freeze_sha256=sha256(args.freeze) if args.freeze else None,
        input_policy="Only original visible RGB/mask; no hidden RGB, amodal mask or truth-based crop",
        metric_frame="full native source image including truth outside visible-based model crop",
        aggregate=grouped_means(results), rows=results)
    write_json(args.output, report)
    print(json.dumps(report["aggregate"]), flush=True)


if __name__ == "__main__":
    main()
