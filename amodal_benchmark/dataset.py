"""Read and verify the authors' exact evaluation subset, without re-encoding images."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

SOURCES = {
    "VG": ("VG_annotation.json", 1234, 1294, 269,
           "ae34ea4799229aef0581899147acb807835f763bca65f9bd3cbbd822d57d9cf1"),
    "COCOA": ("cocoa_annotation.json", 751, 845, 256,
              "b6b83727442f5c6f5f544821910ee10a23e65e0f67330b22df6b3ebf5d12fa89"),
    "free_images": ("website_annotations.json", 228, 249, 147,
                    "391870fd203c02ae7500a41b4cf360d6d06414187018437eb4b4dfe27cdf1ff6"),
    "laion": ("laion_00000_annotation.json", 166, 177, 97,
              "dbb0f7512256e2988bb9fb1c5253b111f9e6db503b4e6ca5b6134bcdd9bdd935"),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def target_record(source, annotation):
    name = Path(annotation["filename"]).name
    labels = annotation["labels"]
    if source not in SOURCES or len(labels) != 1 or not labels[0].strip():
        raise ValueError(f"Invalid target annotation: {annotation}")
    stem = Path(name).stem
    if source == "VG":
        match = re.fullmatch(r"(\d+)_.+", stem)
        original = match[1] if match else None
    elif source == "COCOA":
        match = re.fullmatch(r"(COCO_(?:train|val|test)2014_\d+)_.+", stem)
        original = match[1] if match else None
    elif source == "laion":
        match = re.fullmatch(r"(\d{9})_.+", stem)
        original = match[1] if match else None
    else:
        original = stem.rsplit("_", 1)[0] if "_" in stem else None
    if original is None:
        raise ValueError(f"Unrecognized {source} filename: {name}")
    image = name if source == "free_images" else original + ".jpg"
    return {"id": source + "/" + stem, "source": source, "label": labels[0],
            "annotation_filename": annotation["filename"], "original_id": original,
            "image": f"images/{source}/{image}", "reference": f"{source}/{stem}.png"}


def read_official(root):
    root = Path(root)
    rows, counts = [], {}
    for source, (filename, images, instances, classes, expected_hash) in SOURCES.items():
        path = root / "official" / source / filename
        if not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(f"Official annotation missing or checksum mismatch: {path}. "
                             "Run prepare_open_world.py --download-metadata.")
        annotations = json.loads(path.read_text(encoding="utf-8"))["annotations"]
        group = [target_record(source, a) for a in annotations]
        observed = (len({r["original_id"] for r in group}), len(group),
                    len({r["label"] for r in group}))
        if observed != (images, instances, classes):
            raise ValueError(f"Dataset identity mismatch for {source}: {observed}")
        counts[source] = dict(zip(("images", "instances", "classes"), observed))
        rows.extend(group)
    if len({r["id"] for r in rows}) != 2565 or len({r["label"] for r in rows}) != 553:
        raise ValueError("Dataset does not match the 2565-target, 553-class benchmark")
    return rows, counts


def audit_files(root, rows, references=None, decode=False):
    from PIL import Image
    root = Path(root)
    missing, corrupt, missing_refs, corrupt_refs = set(), set(), [], []
    seen, sizes = set(), {}
    for row in rows:
        relative = row["image"]
        if relative not in seen:
            seen.add(relative)
            path = root / relative
            if not path.is_file():
                missing.add(relative)
            elif decode:
                try:
                    with Image.open(path) as image:
                        image.load()
                        sizes[relative] = image.size
                except (OSError, ValueError):
                    corrupt.add(relative)
        if references is not None and relative not in missing and relative not in corrupt:
            path = Path(references) / row["reference"]
            if not path.is_file():
                missing_refs.append(row["reference"])
            elif decode:
                try:
                    with Image.open(path) as image:
                        image.load()
                        if image.mode != "RGBA":
                            raise ValueError("Reference must be RGBA")
                        if image.getchannel("A").getextrema()[1] == 0:
                            from .references import cached_reference
                            record = cached_reference(references, row, sha256(root / relative))
                            if record is None or record["status"] not in ("empty_mask", "eroded_empty"):
                                raise ValueError("Empty alpha needs a recorded LISA segmentation failure")
                        if relative in sizes and image.size != sizes[relative]:
                            raise ValueError("Reference must retain original-image coordinates")
                except (OSError, ValueError):
                    corrupt_refs.append(row["reference"])
    return {"total_targets": len(rows), "unique_images": len({(r['source'], r['original_id']) for r in rows}),
            "missing_images": sorted(missing), "corrupt_images": sorted(corrupt),
            "reference_directory": str(references) if references else None,
            "missing_references": missing_refs, "corrupt_references": corrupt_refs,
            "reference_status": "checked" if references else "not_provided"}
