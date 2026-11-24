"""Measure what the learned hidden silhouette adds to visible-mask selection on frozen PMC pools."""
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.shape import choose, overlap, score_masks
from scripts.analyze_surface_followup import paired_interval
from scripts.audit_shape_controlled_cocoa import check_file, read


def mask(path):
    with Image.open(path) as image:
        return np.array(image.convert("L")) > 0


def main():
    study = ROOT / "output/shape-20261121"
    source = ROOT / "output/surface-confirmation128-20261110"
    cases_path = source / "cases.json"
    check_file(cases_path, read(study / "training.json")["confirmation_sha256"])
    cocoa_path = study / "controlled-cocoa.json"
    outlines = read(cocoa_path)
    check_file(cases_path, outlines["cases_sha256"])
    polygons = {r["id"]: r for r in outlines["rows"]}
    cases = read(cases_path)
    if len(cases) != 128 or len({r["image_identity"] for r in cases}) != len(cases):
        raise ValueError("Expected 128 independent confirmation images")
    rows = []
    for case in cases:
        cid, tid = case["case_id"], case["target_id"]
        inputs = source / "inputs" / cid
        visible_file = inputs / "visible.png"
        check_file(visible_file, case["input_hashes"]["visible.png"])
        visible = mask(visible_file)
        decisions = read(source / "inference/decisions" / (cid + ".json"))
        pool = source / "full/candidates" / cid
        masks, hashes = [], []
        for seed in range(11, 89, 11):
            record = pool / f"seed-{seed}.json"
            check_file(record, decisions["candidate_record_sha256"][record.name])
            meta = read(record)
            if (meta["seed"], meta["case_id"], meta["input_hashes"]) != (seed, cid, case["input_hashes"]):
                raise ValueError(f"Candidate inputs differ for {cid} seed {seed}")
            path = pool / f"seed-{seed}-mask.png"
            digest = meta["artifacts"][path.name]
            check_file(path, digest)
            hashes.append(digest)
            masks.append(mask(path))
        visible_scores = score_masks(visible, masks)
        prior_file = study / "controlled" / cid / "prior.png"
        shape_scores = score_masks(mask(prior_file), masks)
        slots = {str(k): {"visible": choose(visible_scores[:k])[0], "shape": choose(shape_scores[:k])[0]}
                 for k in (4, 8)}
        result_file = prior_file.with_name("result.json")
        result = read(result_file)
        if (result["target_id"] != tid or shape_scores != result["scores"] or
                any(idx["shape"] != result["indices"]["shape" + k] for k, idx in slots.items())):
            raise ValueError(f"Learned-prior replay disagrees with the frozen decision for {tid}")

        truth_file = source / "evaluation_only" / cid / "full.png"
        check_file(truth_file, case["evaluation_hashes"]["full.png"])
        check_file(inputs / "occlusion.png", case["input_hashes"]["occlusion.png"])
        truth, occ = mask(truth_file), mask(inputs / "occlusion.png")
        metric_file = source / "metrics" / (cid + ".json")
        check_file(metric_file, result["reference_metrics_sha256"])
        raw = read(metric_file)["raw"]
        metrics = {"iou": [overlap(m, truth) for m in masks],
                   "hidden_iou": [overlap(m & occ, truth & occ) for m in masks],
                   "object_lpips": [r["object_lpips"] for r in raw]}
        if tid in polygons:
            polygon = polygons[tid]
            check_file(result_file, polygon["controlled_result_sha256"])
            if hashes != polygon["candidate_mask_sha256"]:
                raise ValueError(f"COCOA audit scored a different pool for {tid}")
            metrics["amodal_iou"] = polygon["candidate_amodal_iou"]
        rows.append(dict(id=tid, case_id=cid, source=case["source"], slots=slots, metrics=metrics,
                         visible_scores=visible_scores, mask_sha256=hashes, prior_sha256=sha256(prior_file),
                         result_sha256=sha256(result_file)))
    summary = {}
    for k in ("4", "8"):
        table = {"changed": sum(r["slots"][k]["visible"] != r["slots"][k]["shape"] for r in rows)}
        for metric in ("iou", "hidden_iou", "object_lpips", "amodal_iou"):
            group = [r for r in rows if metric in r["metrics"]]
            if len(group) != (32 if metric == "amodal_iou" else 128):
                raise ValueError(f"Incomplete panel for {metric}")
            visible = [r["metrics"][metric][r["slots"][k]["visible"]] for r in group]
            shape = [r["metrics"][metric][r["slots"][k]["shape"]] for r in group]
            table[metric] = paired_interval(visible, shape, [r["source"] for r in group],
                                            higher=metric != "object_lpips")
        summary[k] = table
    write_json(study / "controlled-visible-ablation.json", dict(cases_sha256=sha256(cases_path),
               cocoa_sha256=sha256(cocoa_path), analysis_sha256=sha256(Path(__file__)),
               comparisons=summary, rows=rows))


if __name__ == "__main__":
    main()
