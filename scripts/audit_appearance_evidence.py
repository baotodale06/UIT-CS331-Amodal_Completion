#!/usr/bin/env python3
"""Independently check external COCOA ratios and quantify proxy limitations."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from scipy.stats import spearmanr

from amodal_benchmark.appearance import ratio_bin, read_review, write_csv
from amodal_benchmark.dataset import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.analysis_dir
    manifest = json.loads((output / "analysis_manifest.json").read_text(encoding="utf-8"))
    run = Path(manifest["run_dir"])
    rows = json.loads((output / "per_target_diagnostics.json").read_text(encoding="utf-8"))
    for name, digest in manifest["inputs"].items():
        if sha256(run / name) != digest:
            raise ValueError(f"Run input changed: {name}")
    index = {}
    for filename, digest in manifest["external_annotation_hashes"].items():
        path = output / "external" / filename
        if sha256(path) != digest:
            raise ValueError(f"External annotation changed: {filename}")
        data = json.loads(path.read_text(encoding="utf-8"))
        index[filename] = {annotation["id"]: annotation for annotation in data["annotations"]}
    audit = []
    for row in rows:
        if row["gt_occlusion_ratio"] is None:
            continue
        annotation = index[row["gt_annotation_file"]][row["gt_annotation_id"]]
        region = next(r for r in annotation["regions"] if r["order"] == row["gt_region_order"])
        height, width = row["gt_image_height"], row["gt_image_width"]
        amodal = coco_mask.decode(coco_mask.frPyObjects([region["segmentation"]], height, width)).any(axis=2)
        visible = coco_mask.decode(region["visible_mask"]).astype(bool) if "visible_mask" in region else amodal.copy()
        hidden = coco_mask.decode(region["invisible_mask"]).astype(bool) if "invisible_mask" in region else np.zeros_like(amodal)
        ratio = float(hidden.sum() / max(1, (visible | hidden).sum()))
        geometric_ratio = float((amodal & ~visible).sum() / max(1, amodal.sum()))
        with Image.open(run / "visible_references/masks" / row["reference"]) as image:
            sam = np.array(image) > 0
        iou = float((visible & sam).sum() / max(1, (visible | sam).sum())) if sam.any() else None
        audit.append({"id": row["id"], "status": row["status"], "source_ratio": row["gt_occlusion_ratio"],
                      "decoded_invisible_union_ratio": ratio, "polygon_minus_visible_ratio": geometric_ratio,
                      "absolute_difference": abs(ratio - row["gt_occlusion_ratio"]),
                      "sam_vs_gt_visible_iou": iou, "sam_mask_present": bool(sam.any()),
                      "ratio_bin_matches": ratio_bin(ratio) == row["gt_bin"]})
    if any(r["absolute_difference"] > 1e-6 or not r["ratio_bin_matches"] for r in audit):
        raise ValueError("External mask ratios do not reproduce the declared bins")
    write_csv(output / "cocoa_mask_audit.csv", audit)
    info = {"targets": len(audit), "max_ratio_difference": max(r["absolute_difference"] for r in audit),
            "median_ratio_difference": float(np.median([r["absolute_difference"] for r in audit])),
            "sam_nonempty": sum(r["sam_mask_present"] for r in audit),
            "sam_iou_below_05": sum(r["sam_vs_gt_visible_iou"] is not None and r["sam_vs_gt_visible_iou"] < .5 for r in audit),
            "success_sam_iou_below_05": sum(r["status"] == "success" and r["sam_vs_gt_visible_iou"] is not None
                                            and r["sam_vs_gt_visible_iou"] < .5 for r in audit)}
    write_json(output / "cocoa_mask_audit.json", info)

    paired = [r for r in rows if r["gt_occlusion_ratio"] is not None and r["predicted_expansion_ratio"] is not None]
    gt = np.array([r["gt_occlusion_ratio"] for r in paired])
    proxy = np.array([r["predicted_expansion_ratio"] for r in paired])
    agreement = {"paired_saved_outputs": len(paired), "pearson_r": float(np.corrcoef(gt, proxy)[0, 1]),
                 "spearman_r": float(spearmanr(gt, proxy).statistic),
                 "median_absolute_error_pp": float(np.median(abs(gt - proxy)) * 100),
                 "same_bin": sum(r["gt_bin"] == r["proxy_bin"] for r in paired),
                 "note": "Descriptive comparison on the matched saved-output subset; not a calibrated occlusion estimator."}
    write_json(output / "proxy_vs_ground_truth.json", agreement)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    scatter = ax.scatter(gt * 100, proxy * 100, c=[r["visible_retention"] for r in paired],
                         cmap="viridis", vmin=0, vmax=1, alpha=.75, edgecolors="white", linewidths=.5)
    ax.plot([0, 100], [0, 100], linestyle="--", color="gray", label="Equality")
    ax.set(xlim=(-2, 102), ylim=(-2, 102), xlabel="COCOA annotated occlusion (%)",
           ylabel="Generated-area proxy (%)", title=f"The generated-area proxy is not true occlusion\n"
           f"n={len(paired)}; Spearman r={agreement['spearman_r']:.2f}; median error={agreement['median_absolute_error_pp']:.1f} pp")
    fig.colorbar(scatter, ax=ax, label="Fraction of initial visible mask retained")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output / "figures" / f"proxy_vs_ground_truth.{ext}", dpi=180)
    plt.close(fig)

    with (output / "review/sample.csv").open(encoding="utf-8", newline="") as stream:
        sample = list(csv.DictReader(stream))
    review = read_review(output / "review/annotations.csv", [r["id"] for r in sample])
    by_id = {r["id"]: r for r in rows}
    review_summary = {"sample_size": len(review), "source_counts": dict(Counter(r["source"] for r in review)),
        "category_counts": dict(Counter(r["category"] for r in review)),
        "confounds": dict(Counter(r["identity_or_mask_confound"] or "none" for r in review)),
        "at_least_one_observed_failure": sum(any(r[k].isdigit() and int(r[k]) > 0 for k in
            ("wrong_texture", "color_inconsistency", "pattern_inconsistency", "semantic_drift",
             "hidden_region_inconsistency", "cooccurring_object")) for r in review),
        "cooccurrence_subtypes": dict(Counter(r["cooccurrence_subtype"] for r in review if r["cooccurrence_subtype"])),
        "cooccurrence_scope": dict(Counter(r["cooccurrence_output_scope"] for r in review if r["cooccurrence_output_scope"])),
        "sample_gt_available": sum(by_id[r["id"]]["gt_occlusion_ratio"] is not None for r in review),
        "target_switches": sum("output_target_switch" in r["identity_or_mask_confound"].split(";") for r in review)}
    write_json(output / "review_summary.json", review_summary)
    retained = np.array([r["visible_retention"] for r in rows if r.get("visible_retention") is not None])
    write_json(output / "mask_diagnostics_summary.json", {
        "outputs": len(retained), "visible_retention_below_50_percent": int((retained < .5).sum()),
        "visible_retention_below_10_percent": int((retained < .1).sum()),
        "median_visible_retention": float(np.median(retained)),
        "median_visible_delta_e76": float(np.median([r["visible_delta_e76_median"] for r in rows
                                      if r.get("visible_delta_e76_median") is not None]))})
    print(json.dumps({"mask_audit": info, "proxy_agreement": agreement, "review": review_summary}, indent=2))


if __name__ == "__main__":
    main()
