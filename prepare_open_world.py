#!/usr/bin/env python3
"""Install the exact Open-World benchmark subset; never download all of COCO/VG."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time
import zipfile

from amodal_benchmark.dataset import SOURCES, audit_files, read_official, sha256, write_json

# Public release linked from https://github.com/saraao/amodal . Annotation hashes
# pin the downloaded subset, including the authors' target-specific filenames.
FILES = {
    "readme.txt": ("1RIjEy2_3qjNsggLQYG1LGFEPapZTW8RI", "3651d6762fbc34a8ff05c82ac4764f205013b29083b1ab9a197d85f3087234eb"),
    "COCOA/cocoa_annotation.json": ("1ZcnADIPnbYKPKkV2ecurbYDWZdO68hh2", SOURCES["COCOA"][4]),
    "COCOA/img_filenames_cocoa.txt": ("19XTZqIvFCV5q1w8W-uUC_q49bxlBy2aj", "bb5622ca45377ae5138380c1753ffd20f5003bc92632c766441483ee676f079f"),
    "VG/VG_annotation.json": ("1lyXwzua9ilZ_JFadgpr_I24a6yHwD3qZ", SOURCES["VG"][4]),
    "VG/img_filenames_VG.txt": ("1Y09Mk1he2NE4s8bXfomkaV-2T9UQQeYc", "84a6468bcb14912712cd7dfa48c92bc07dfa452ce9839be1a76389fa75e4213f"),
    "laion/laion_00000_annotation.json": ("19U99hlEIPSoclOH8_lWKjp-Q3UpfChIA", SOURCES["laion"][4]),
    "laion/img_filenames_laion.txt": ("199O5e-fe1n40-Q6s39W4ZCk4Jwgdn5as", "64aa6edcc2f8b3f91ef17932648c984827edc49bac8293ff524be70fb99cff06"),
    "free_images/website_annotations.json": ("1UDhYlc7vBy2VvbzTUo7Fd7EqklrovW5J", SOURCES["free_images"][4]),
    "free_images/img_filenames_website.txt": ("1RVRiMTyjoDXww8guyTm1vir3WorwKdxQ", "ad1b9212632a98e9c6dc69d8da7053a3ef91f81e7511a248b9b9860ff9777751"),
    "free_images/website.zip": ("1nmHcjrwdHwbrOrKz3Z8cz8Th8mFAaJG9", "1c60dbe1f109686e54783903f6fd8a9cf745b5788fa3434c2cf608e4c1520827"),
}


def checked_write(path, data, expected=None):
    if expected and hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Checksum mismatch: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def unpack_free_images(root):
    archive = root / "official/free_images/website.zip"
    if not archive.exists():
        return
    if sha256(archive) != FILES["free_images/website.zip"][1]:
        raise ValueError("Free-image archive checksum mismatch")
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if item.is_dir() or not item.filename.lower().endswith(".jpg"):
                continue
            checked_write(root / "images/free_images" / Path(item.filename).name, z.read(item))


def import_release(root, archive):
    with zipfile.ZipFile(archive) as z:
        for relative, (_, expected) in FILES.items():
            matches = [n for n in z.namelist() if n.endswith("/" + relative) or n == relative]
            if len(matches) != 1:
                raise ValueError(f"Release archive does not contain exactly one {relative}")
            checked_write(root / "official" / relative, z.read(matches[0]), expected)
    unpack_free_images(root)


def download_metadata(root, include_images=False):
    import requests
    records = []
    for relative, (file_id, expected) in FILES.items():
        if relative.endswith(".zip") and not include_images:
            continue
        dest = root / "official" / relative
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        if not dest.exists() or sha256(dest) != expected:
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            checked_write(dest, response.content, expected)
        records.append({"path": relative, "url": url, "sha256": expected,
                        "bytes": dest.stat().st_size})
    write_json(root / "official/download_manifest.json", records)
    unpack_free_images(root)


def image_urls(row):
    key = row["original_id"]
    if row["source"] == "VG":
        return [f"https://cs.stanford.edu/people/rak248/{folder}/{key}.jpg"
                for folder in ("VG_100K", "VG_100K_2")]
    if row["source"] == "COCOA":
        split = key.split("_")[1]
        # This is the COCO image bucket, using its valid HTTPS S3 endpoint.
        return [f"https://s3.amazonaws.com/images.cocodataset.org/{split}/{key}.jpg"]
    return []


def download_originals(root, rows, workers):
    import requests
    from PIL import Image
    by_path = {r["image"]: r for r in rows if image_urls(r)}

    def fetch(pair):
        relative, row = pair
        dest = root / relative
        if dest.is_file():
            with Image.open(dest) as im:
                im.load()
            return {"image": relative, "sha256": sha256(dest), "cached": True}
        errors = []
        for attempt in range(3):
            for url in image_urls(row):
                try:
                    response = requests.get(url, timeout=40)
                    response.raise_for_status()
                    with Image.open(io.BytesIO(response.content)) as im:
                        im.load()
                    checked_write(dest, response.content)
                    return {"image": relative, "url": url, "sha256": sha256(dest)}
                except (requests.RequestException, OSError, ValueError) as error:
                    errors.append(str(error))
            time.sleep(attempt + 1)
        return {"image": relative, "errors": errors}

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, item) for item in by_path.items()]
        for f in as_completed(futures):
            records.append(f.result())
            if len(records) % 100 == 0:
                print(f"Original images checked: {len(records)}/{len(futures)}", flush=True)
    write_json(root / "original_downloads.json", records)


def import_laion(root, archive, rows):
    from PIL import Image
    wanted = {Path(r["image"]).name for r in rows if r["source"] == "laion"}
    found = []
    with tarfile.open(archive, "r:*") as tar:
        for item in tar:
            # Write only exact requested basenames; never extract archive paths.
            if not item.isfile() or Path(item.name).name not in wanted:
                continue
            name = Path(item.name).name
            if name in found:
                raise ValueError(f"Duplicate LAION image in supplied shard: {name}")
            data = tar.extractfile(item).read()
            with Image.open(io.BytesIO(data)) as im:
                im.load()
            checked_write(root / "images/laion" / name, data)
            found.append(name)
    record = {"archive": str(Path(archive).resolve()), "archive_sha256": sha256(archive),
              "imported": sorted(found), "missing": sorted(wanted - set(found)),
              "identity_note": "User-supplied shard. The authors publish filenames but no shard checksum; "
                               "a filename match alone does not establish the shard's provenance."}
    write_json(root / "laion_import.json", record)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("datasets/open_world_amodal"))
    p.add_argument("--release-zip", type=Path, help="Import the authors' downloaded Google Drive release")
    p.add_argument("--download-metadata", action="store_true")
    p.add_argument("--download-images", action="store_true", help="Download only selected VG/COCO and free images")
    p.add_argument("--laion-tar", type=Path, help="Authors' original LAION-400M 00000.tar (not an arbitrary shard)")
    p.add_argument("--reference-dir", type=Path, help="Authors' visible-object RGBA references, SOURCE/TARGET.png")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--allow-incomplete", action="store_true", help="Write reports even if images are missing; label partial installation")
    args = p.parse_args(argv)
    if args.workers < 1:
        p.error("--workers must be positive")
    root = args.root.resolve()
    if args.release_zip:
        import_release(root, args.release_zip)
    if args.download_metadata or args.download_images:
        download_metadata(root, args.download_images)
    rows, counts = read_official(root)
    if args.download_images:
        download_originals(root, rows, args.workers)
    if args.laion_tar:
        import_laion(root, args.laion_tar, rows)
    report = audit_files(root, rows, args.reference_dir, decode=True)
    absent = set(report["missing_images"] + report["corrupt_images"])
    report["missing_targets_by_source"] = dict(Counter(r["source"] for r in rows if r["image"] in absent))
    report["images_complete"] = not absent
    report["reference_images_complete"] = bool(args.reference_dir) and not (
        report["missing_references"] or report["corrupt_references"])
    report["paper_identity"] = {"title": "Open-World Amodal Appearance Completion",
                                "images": 2379, "targets": 2565, "classes": 553}
    report["source_counts"] = counts
    checksums = {r["image"]: sha256(root / r["image"]) for r in rows if r["image"] not in absent}
    write_json(root / "image_checksums.json", checksums)
    write_json(root / "manifest.json", {"schema": 1, "source_counts": counts, "targets": rows})
    write_json(root / "verification.json", report)
    print(json.dumps({k: report[k] for k in ("total_targets", "unique_images", "images_complete",
                                            "reference_images_complete", "missing_targets_by_source")}, indent=2))
    print(f"Full verification report: {root / 'verification.json'}")
    return 0 if args.allow_incomplete or report["images_complete"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
