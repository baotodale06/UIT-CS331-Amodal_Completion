#!/usr/bin/env python3
"""Build reproducible CPU diagnostics and a visual audit of an immutable run."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from amodal_benchmark.appearance import (BIN_LABELS, FAILURES, aligned_artifacts,
    appearance_diagnostics, bootstrap_mean, ratio_bin, read_review, stable_sample,
    taxonomy_counts, weighted_taxonomy_counts, wilson_interval, write_csv)
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.protocol import METRICS, white_rgb
from amodal_benchmark.reproduction import load_run


def cocoa_index(directory):
    index, hashes = {}, {}
    if directory is None:
        return index, hashes
    for path in sorted(Path(directory).glob("COCO_amodal_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        images = {r["id"]: r for r in data["images"]}
        hashes[path.name] = sha256(path)
        for annotation in data["annotations"]:
            image = images[annotation["image_id"]]
            key = Path(image["file_name"]).stem
            if key in index:
                raise ValueError(f"Duplicate external image {key}")
            index[key] = (image, annotation, path.name)
    return index, hashes


def cocoa_match(record, index):
    result = {"gt_occlusion_ratio": None, "gt_bin": "unavailable",
              "gt_join_status": "source_not_cocoa"}
    if record["source"] != "COCOA":
        return result
    source = index.get(record["original_id"])
    if source is None:
        result["gt_join_status"] = "external_image_unavailable"
        return result
    image, annotation, filename = source
    candidates = [region for region in annotation["regions"]
                  if region.get("name", "").strip().casefold() == record["label"].strip().casefold()]
    result["gt_candidate_count"] = len(candidates)
    if len(candidates) != 1:
        result["gt_join_status"] = "no_exact_name" if not candidates else "ambiguous_exact_name"
        return result
    region = candidates[0]
    rate = region.get("occlude_rate")
    if not isinstance(rate, (int, float)) or not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError(f"Invalid annotated occlusion ratio for {record['id']}")
    result.update(gt_occlusion_ratio=rate, gt_bin=ratio_bin(rate),
                  gt_join_status="unique_exact_name", gt_annotation_file=filename,
                  gt_annotation_id=annotation["id"], gt_region_order=region["order"],
                  gt_region_name=region["name"], gt_image_width=image["width"],
                  gt_image_height=image["height"])
    return result


def collect(run, dataset, index, workers):
    with (run / "per_target.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    def process(row):
        for key in METRICS:
            row[key] = float(row[key]) if row[key] else None
        record_path = run / "targets" / row["id"] / "result.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["status"] != row["status"]:
            raise ValueError(f"Status mismatch for {row['id']}")
        selection = record.get("selection_diagnostics", {})
        row.update(category=selection.get("annotation_class") or selection.get("annotation_head") or row["label"],
                   image=record["image"], reference=record["reference"],
                   proxy_bin="unavailable", predicted_expansion_ratio=None,
                   record_sha256=sha256(record_path), input_sha256=record["input_sha256"])
        row.update(cocoa_match(record, index))
        if row["gt_occlusion_ratio"] is not None:
            with Image.open(dataset / row["image"]) as image:
                if image.size != (row["gt_image_width"], row["gt_image_height"]):
                    raise ValueError(f"COCOA frame mismatch for {row['id']}")
        if row["status"] == "success":
            completion = run / "targets" / row["id"] / "completion.png"
            row["completion_sha256"] = sha256(completion)
            if row["completion_sha256"] != record["completion_sha256"]:
                raise ValueError(f"Completion changed for {row['id']}")
            for name, path in (("full", run / "visible_references" / row["reference"]),
                               ("mask", run / "visible_references/masks" / row["reference"])):
                row[f"reference_{name}_sha256"] = sha256(path)
                if row[f"reference_{name}_sha256"] != record["evaluation_reference_artifacts"][name]:
                    raise ValueError(f"Reference changed for {row['id']}")
            reference, prediction = aligned_artifacts(run, record)
            row.update(appearance_diagnostics(reference, prediction))
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(process, rows))


def summarize_bins(rows, key):
    result = []
    for label in (*BIN_LABELS, "unavailable"):
        group = [r for r in rows if r[key] == label]
        saved = [r for r in group if r["status"] == "success"]
        n, k = len(group), len(saved)
        low, high = wilson_interval(k, n)
        entry = {"bin": label, "targets": n, "saved": k, "failures": n - k,
                 "saved_rate": k / n if n else None, "saved_rate_low": low, "saved_rate_high": high}
        for metric in (*METRICS, "visible_delta_e76_median", "visible_retention"):
            for scope, subset in (("all", group), ("saved", saved)):
                for stat, value in bootstrap_mean([r.get(metric) for r in subset]).items():
                    entry[f"{metric}_{scope}_{stat}"] = value
        result.append(entry)
    return result


def font(size):
    for name in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


def panel(run, dataset, row, number):
    record = json.loads((run / "targets" / row["id"] / "result.json").read_text(encoding="utf-8"))
    ref, pred = aligned_artifacts(run, record)
    union = (ref[..., 3] > 0) | (pred[..., 3] > 0)
    rr, cc = np.where(union)
    box = (max(0, int(cc.min()) - 15), max(0, int(rr.min()) - 15),
           min(union.shape[1], int(cc.max()) + 16), min(union.shape[0], int(rr.max()) + 16))
    # The original context, visible support and completion use the same crop/scale.
    with Image.open(dataset / row["image"]) as image:
        rgb = np.array(image.convert("RGB"))
    context = np.full((*union.shape, 3), 238, dtype=np.uint8)
    off_r, off_c = record["geometry"]["offset_rc"]
    r0, c0 = -min(0, off_r), -min(0, off_c)
    context[r0:r0 + rgb.shape[0], c0:c0 + rgb.shape[1]] = rgb
    region = np.full((*union.shape, 3), 245, np.uint8)
    region[ref[..., 3] > 0] = [50, 130, 210]
    region[(pred[..., 3] > 0) & (ref[..., 3] == 0)] = [232, 143, 38]
    region[(ref[..., 3] > 0) & (pred[..., 3] == 0)] = [195, 65, 95]
    arrays = (context, white_rgb(ref), white_rgb(pred), region)
    result = Image.new("RGB", (1600, 345), "white")
    draw = ImageDraw.Draw(result)
    name = f"{number:03d}  {row['id']} | {row['label']}"
    draw.text((8, 5), name[:135], font=font(17), fill="#111827")
    draw.text((8, 28), f"Expansion proxy {100 * row['predicted_expansion_ratio']:.1f}% | "
              f"visible retained {100 * row['visible_retention']:.1f}% | "
              f"iterations {row['iterations']} | annotated occlusion "
              + (f"{100 * row['gt_occlusion_ratio']:.1f}%" if row.get('gt_occlusion_ratio') is not None else "unavailable"),
              font=font(14), fill="#374151")
    for i, (array, title) in enumerate(zip(arrays, ("Original context", "Visible target (SAM)",
                                                    "Saved completion", "Blue visible / orange new / pink lost"))):
        draw.text((i * 320 + 5, 53), title, font=font(13), fill="#374151")
        crop = Image.fromarray(array).crop(box)
        crop.thumbnail((312, 265), Image.Resampling.LANCZOS)
        result.paste(crop, (i * 320 + (320 - crop.width) // 2, 76 + (265 - crop.height) // 2))
    draw.text((1285, 53), "Raw generated RGB (entire crop)", font=font(13), fill="#374151")
    with Image.open(run / "targets" / row["id"] / "completion.jpg") as image:
        raw = image.convert("RGB")
    raw.thumbnail((312, 265), Image.Resampling.LANCZOS)
    result.paste(raw, (1280 + (320 - raw.width) // 2, 76 + (265 - raw.height) // 2))
    return result


def make_review(run, dataset, rows, output, per_bin):
    sample = stable_sample(rows, per_bin=per_bin)
    by_id = {r["id"]: r for r in rows}
    (output / "review/panels").mkdir(parents=True, exist_ok=True)
    (output / "review/sheets").mkdir(parents=True, exist_ok=True)
    templates, panels = [], []
    for i, sampled in enumerate(sample, 1):
        row = by_id[sampled["id"]]
        item = {"review_number": i, **sampled, "source": row["source"], "label": row["label"],
                "category": row["category"], "reviewer": "", **{k: "" for k in FAILURES},
                "identity_or_mask_confound": "", "evidence": ""}
        templates.append(item)
        rendered = panel(run, dataset, row, i)
        rendered.save(output / f"review/panels/{i:03d}.jpg", quality=94)
        panels.append(rendered)
        if len(panels) == 4 or i == len(sample):
            sheet = Image.new("RGB", (1600, len(panels) * 345), "#dddddd")
            for j, image in enumerate(panels):
                sheet.paste(image, (0, j * 345))
            sheet.save(output / f"review/sheets/{(i - 1) // 4 + 1:02d}.jpg", quality=94)
            panels = []
    write_csv(output / "review/sample.csv", sample)
    template_path = output / "review/annotations_template.csv"
    write_csv(template_path, templates)
    return sample


def plots(output, gt, proxy, categories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figures = output / "figures"
    figures.mkdir(exist_ok=True)

    def curve(ax, bins, field, label, color):
        x = np.arange(len(bins))
        means = np.array([np.nan if b[field + "mean"] is None else b[field + "mean"] for b in bins])
        lo = np.array([np.nan if b[field + "ci_low"] is None else b[field + "ci_low"] for b in bins])
        hi = np.array([np.nan if b[field + "ci_high"] is None else b[field + "ci_high"] for b in bins])
        ax.errorbar(x, means, yerr=np.vstack((means - lo, hi - means)), marker="o", capsize=3,
                    color=color, label=label)
        ax.set_xticks(x, [b["bin"] for b in bins])
        ax.grid(alpha=.18)

    bins = gt[1:7]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    ax = axes[0, 0]
    means = np.array([b["saved_rate"] for b in bins], float)
    lows = np.array([b["saved_rate_low"] for b in bins], float)
    highs = np.array([b["saved_rate_high"] for b in bins], float)
    ax.errorbar(np.arange(6), means * 100, yerr=np.vstack((means-lows, highs-means)) * 100,
                marker="o", capsize=4, color="#245d91")
    ax.set_xticks(np.arange(6), [f"{b['bin']}\nn={b['targets']}" for b in bins])
    ax.set(ylabel="Saved completions (%)", ylim=(-3, 103), title="Output availability (95% Wilson interval)")
    for ax, metric, direction in zip((axes[0, 1], axes[1, 0], axes[1, 1]),
                                      ("LPIPS", "SSIM", "CLIP"), ("lower is better", "higher is better", "higher is better")):
        if metric != "CLIP":
            curve(ax, bins, metric + "_all_", "All targets, with failure penalties", "#a34e29")
        curve(ax, bins, metric + "_saved_", "Saved completions only", "#245d91")
        ax.set_title(f"{metric} ({direction})")
        ax.set_ylabel(metric)
        ax.legend(fontsize=8)
    fig.suptitle("Performance vs annotated COCOA occlusion\nUnique exact-name joins; "
                 f"{sum(b['targets'] for b in bins)} targets in 20-80% bins", fontsize=15)
    fig.supxlabel("COCOA annotated occlusion (%) | Metric bars: target bootstrap 95% intervals; saved n = "
                  + ", ".join(str(b['saved']) for b in bins))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(figures / f"performance_vs_occlusion.{ext}", dpi=180)
    plt.close(fig)

    bins = proxy[1:7]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, metric, direction in zip(axes.flat, ("CLIP", "LPIPS", "SSIM", "visible_delta_e76_median"),
                                     ("higher", "lower", "higher", "lower")):
        curve(ax, bins, metric + "_saved_", "Saved completions", "#245d91")
        ax.set_title(f"{metric} ({direction} is better)")
        ax.set_xticks(np.arange(6), [f"{b['bin']}\nn={b['saved']}" for b in bins])
    fig.suptitle("Appearance vs generated-area proxy (saved outputs only)\nThe x-axis depends on the model output; it is NOT ground-truth occlusion", fontsize=14)
    fig.supxlabel("Predicted expansion (%) | 95% target bootstrap intervals; no failure-rate estimate is possible")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(figures / f"performance_vs_expansion_proxy.{ext}", dpi=180)
    plt.close(fig)

    top = sorted([r for r in categories if r["targets"] >= 20], key=lambda r: -r["targets"])
    fig, ax = plt.subplots(figsize=(11, 8), layout="constrained")
    ax.barh([f"{r['category']} (n={r['targets']})" for r in top],
            [100 * r["saved_rate"] for r in top], color="#245d91")
    ax.invert_yaxis()
    ax.set(xlabel="Saved completions (%)", xlim=(0, 100),
           title="Category coverage: groups with at least 20 targets\nCanonical annotation category; coverage is not visual quality")
    fig.savefig(figures / "category_coverage.png", dpi=180)
    fig.savefig(figures / "category_coverage.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cocoa-annotations", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--per-bin", type=int, default=15)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--reuse-derived", action="store_true")
    parser.add_argument("--rebuild-panels", action="store_true")
    args = parser.parse_args()
    run, output = args.run_dir.resolve(), args.output_dir.resolve()
    if run == output or run in output.parents:
        raise ValueError("Analysis must be outside the immutable run")
    output.mkdir(parents=True, exist_ok=True)
    validated = load_run(run)
    index, external_hashes = cocoa_index(args.cocoa_annotations)
    if args.reuse_derived:
        rows = json.loads((output / "per_target_diagnostics.json").read_text(encoding="utf-8"))
        prior = json.loads((output / "analysis_manifest.json").read_text(encoding="utf-8"))
        if prior["inputs"]["per_target.csv"] != sha256(run / "per_target.csv"):
            raise ValueError("Cached analysis input changed")
    else:
        rows = collect(run, args.dataset_root, index, args.workers)
        write_json(output / "per_target_diagnostics.json", rows)
        write_csv(output / "per_target_diagnostics.csv", rows)
    gt, proxy = summarize_bins(rows, "gt_bin"), summarize_bins(rows, "proxy_bin")
    write_csv(output / "occlusion_bins.csv", gt)
    write_csv(output / "expansion_proxy_bins.csv", proxy)
    categories = []
    for category in sorted({r["category"] for r in rows}):
        group = [r for r in rows if r["category"] == category]
        saved = [r for r in group if r["status"] == "success"]
        categories.append({"category": category, "targets": len(group), "saved": len(saved),
                           "failures": len(group) - len(saved), "saved_rate": len(saved) / len(group),
                           **{k + "_saved": bootstrap_mean([r[k] for r in saved])["mean"] for k in METRICS}})
    write_csv(output / "categories.csv", categories)
    write_csv(output / "pipeline_failures.csv", [{"reason": k, "count": v, "fraction_of_all_targets": v / len(rows)}
              for k, v in Counter(r["failure_reason"] for r in rows if r["status"] != "success").most_common()])
    if not args.reuse_derived or args.rebuild_panels:
        sample = make_review(run, args.dataset_root, rows, output, args.per_bin)
    else:
        with (output / "review/sample.csv").open(encoding="utf-8", newline="") as stream:
            sample = list(csv.DictReader(stream))
    if args.review:
        reviewed = read_review(args.review, [r["id"] for r in sample])
        counts = weighted_taxonomy_counts(reviewed)
        write_csv(output / "taxonomy.csv", counts)
        write_json(output / "taxonomy.json", counts)
        grouped = []
        for field in ("proxy_bin", "category", "source"):
            for value in sorted({r[field] for r in reviewed}):
                group = [r for r in reviewed if r[field] == value]
                grouped.extend({"group_field": field, "group": value, **stat}
                               for stat in taxonomy_counts(group))
        write_csv(output / "taxonomy_groups.csv", grouped)
        unconfounded = [r for r in reviewed if not any(token in r.get("identity_or_mask_confound", "")
                                                      for token in ("mask", "target_switch", "low_resolution"))]
        write_csv(output / "taxonomy_without_mask_confounds.csv", taxonomy_counts(unconfounded))
    plots(output, gt, proxy, categories)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(run),
                "dataset_root": str(args.dataset_root.resolve()), "command": sys.argv,
                "targets": len(rows), "saved_completions": sum(r["status"] == "success" for r in rows),
                "external_annotation_hashes": external_hashes,
                "gt_join_counts": dict(Counter(r["gt_join_status"] for r in rows)),
                "sample_size": len(sample), "sample_seed": 20261107,
                "review_sha256": sha256(args.review) if args.review else None,
                "inputs": {name: sha256(run / name) for name in
                           ("per_target.csv", "run_config.json", "metric_config.json", "summary.json")},
                "analysis_source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in
                                            (Path(__file__), ROOT / "amodal_benchmark/appearance.py")},
                "ground_truth": "COCOA supplied occlude_rate; unique exact region-name and image-ID join",
                "proxy_definition": "|predicted mask minus visible SAM mask| / |predicted mask|; original-frame alignment",
                "baseline_statistics": validated["summary"]["statistics"],
                "limitations": ["No hidden appearance ground truth", "Single model seed",
                    "COCOA joins are a subset, not the complete benchmark",
                    "Generated-area proxy is output-dependent and unavailable for failed completions",
                    "Existing learned metrics retain released alignment and failure-penalty conventions",
                    "Visual labels are a single assistant review, not human consensus"]}
    write_json(output / "analysis_manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("targets", "saved_completions", "sample_size", "gt_join_counts")}, indent=2))


if __name__ == "__main__":
    main()
