"""Prepare disjoint, true-amodal COCOA training data without benchmark labels."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.request

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json


def image_split(image_id):
    value = int(hashlib.sha256(f"shape-v2-20261123:{image_id}".encode()).hexdigest()[:8], 16) % 10
    return "validation" if value == 0 else "holdout" if value == 1 else "train"


def fingerprint(path):
    with Image.open(path) as original:
        rgb = original.convert("RGB")
        return dict(decoded=hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest(),
                    aspect=rgb.width / rgb.height,
                    thumbnail=np.asarray(rgb.resize((64, 64), Image.Resampling.BOX), np.float32) / 255)


def excluded_coco_ids(targets):
    return {int(re.search(r"(\d+)$", row["original_id"]).group(1))
            for row in targets if row["source"] == "COCOA"}


def download_image(item):
    row, path = item
    if path.exists():
        with Image.open(path) as image:
            image.verify()
        return None
    # The image host's HTTPS certificate does not cover its dotted bucket name.
    # The official S3 path has a valid certificate and serves the same COCO data.
    urls = ["https://s3.amazonaws.com/images.cocodataset.org/train2014/" + row["file_name"]]
    error = None
    for attempt in range(3):
        for url in urls:
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "shape-amodal-research/2"})
                with urllib.request.urlopen(request, timeout=25) as response:
                    content = response.read()
                temp = path.with_suffix(".part")
                temp.write_bytes(content)
                with Image.open(temp) as image:
                    image.verify()
                temp.replace(path)
                return None
            except Exception as exc:
                error = str(exc)
        time.sleep(attempt + 1)
    return dict(image_id=row["id"], error=error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=ROOT / "output/appearance-failures-20261107/external/COCO_amodal_train2014.json")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/open_world_amodal")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/shape-v2-20261123/data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise ValueError("Completed data is immutable; use a new output directory")
    annotation = json.loads(args.annotations.read_bytes())
    benchmark_path = args.dataset_root / "manifest.json"
    checksum_path = args.dataset_root / "image_checksums.json"
    benchmark = json.loads(benchmark_path.read_bytes())["targets"]
    hashes = json.loads(checksum_path.read_bytes())
    blocked = excluded_coco_ids(benchmark)
    images = [row for row in annotation["images"] if row["id"] not in blocked]
    plan = dict(annotation_sha256=sha256(args.annotations), benchmark_sha256=sha256(benchmark_path),
                checksums_sha256=sha256(checksum_path), images_before=len(annotation["images"]),
                excluded_by_coco_id=len(annotation["images"]) - len(images),
                eligible_images=len(images), split_images=dict(Counter(image_split(r["id"]) for r in images)),
                status="identity exclusion only; decoded and near duplicate audit pending")
    write_json(out / "plan.json", plan)
    print(json.dumps(plan), flush=True)
    if not args.download:
        return
    from pycocotools import mask as coco_mask
    (out / "images").mkdir(exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)
    work = [(row, out / "images" / row["file_name"]) for row in images]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i, result in enumerate(executor.map(download_image, work), 1):
            if result:
                failures.append(result)
            if i % 100 == 0:
                print(f"downloaded/checked {i}/{len(images)}; failures {len(failures)}", flush=True)
    write_json(out / "download_failures.json", failures)
    if failures:
        raise RuntimeError("Downloads incomplete; rerun unchanged to resume")
    reference = []
    for name in sorted({row["image"] for row in benchmark}):
        path = args.dataset_root / name
        if sha256(path) != hashes[name]:
            raise ValueError(f"Benchmark source changed: {name}")
        reference.append(fingerprint(path))
    decoded = {r["decoded"] for r in reference}
    aspects = np.array([r["aspect"] for r in reference])
    thumbnails = np.stack([r["thumbnail"] for r in reference])
    blocked_hashes = set(hashes.values())
    annotations = {row["image_id"]: row for row in annotation["annotations"]}
    excluded, rows, seen = [], [], set()
    artifact_hashes = {}
    for i, row in enumerate(images):
        path = out / "images" / row["file_name"]
        digest = sha256(path)
        fp = fingerprint(path)
        comparable = np.abs(aspects / fp["aspect"] - 1) <= .01
        near = comparable.any() and np.min(np.abs(thumbnails[comparable] - fp["thumbnail"]).mean(axis=(1, 2, 3))) <= .005
        reason = ("benchmark_file" if digest in blocked_hashes else "benchmark_decoded" if fp["decoded"] in decoded
                  else "benchmark_near" if near else "duplicate_training_image" if fp["decoded"] in seen else None)
        if reason:
            excluded.append(dict(image_id=row["id"], reason=reason))
            continue
        seen.add(fp["decoded"])
        image_name = path.relative_to(out).as_posix()
        artifact_hashes[image_name] = digest
        h, w = row["height"], row["width"]
        with Image.open(path) as image:
            if image.size != (w, h):
                raise ValueError("Annotation/image dimensions differ")
        for region in annotations[row["id"]]["regions"]:
            if region["isStuff"] or not region.get("segmentation"):
                continue
            amodal = coco_mask.decode(coco_mask.frPyObjects([region["segmentation"]], h, w)).any(axis=2)
            if "visible_mask" in region:
                visible = coco_mask.decode(region["visible_mask"]).astype(bool) & amodal
            elif region["occlude_rate"] == 0:
                visible = amodal.copy()
            else:
                continue
            if visible.sum() < 64 or amodal.sum() < 128:
                continue
            key = f"{row['id']:012d}-{region['order']:03d}"
            mask_name = f"masks/{key}.png"
            Image.fromarray(amodal.astype(np.uint8) + 2 * visible.astype(np.uint8)).save(out / mask_name)
            artifact_hashes[mask_name] = sha256(out / mask_name)
            rows.append(dict(id=key, image_id=row["id"], split=image_split(row["id"]), image=image_name,
                             mask=mask_name, label=region["name"], visible_pixels=int(visible.sum()),
                             amodal_pixels=int(amodal.sum()), decoded_sha256=fp["decoded"]))
        if (i + 1) % 100 == 0:
            print(f"screened {i+1}/{len(images)}; objects {len(rows)}", flush=True)
    # Excluded images remain in the download cache but are never packaged or loaded.
    manifest = dict(schema=2, **plan, objects=len(rows), counts=dict(Counter(r["split"] for r in rows)),
                    excluded_images=excluded, artifacts=artifact_hashes, rows=rows,
                    duplicate_screen="all benchmark sources; exact bytes/decoded pixels and aspect-within-1%, 64px RGB MAE<=0.005",
                    limitation="Cannot certify absence of cropped or transformed duplicates; ImageNet pretraining is separate",
                    implementation_sha256=sha256(Path(__file__)))
    manifest["status"] = "complete"
    write_json(out / "manifest.json", manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ("rows", "artifacts", "excluded_images")}), flush=True)


if __name__ == "__main__":
    main()
