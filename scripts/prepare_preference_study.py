"""Freeze a blinded, full-run preference packet without changing experiment outputs.

Only Pillow and Python's standard library are needed. Run at the project root on
the cluster when native completion PNGs have not been recovered locally.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import shutil

from PIL import Image

VERSION = "amodal-preference-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check_file(path, expected):
    if not expected or digest(path) != expected:
        raise ValueError(f"Artifact hash mismatch: {path}")


def load_run(root, method, expected):
    run = root / method["run"]
    execution = json.loads((run / "execution.json").read_bytes())
    config = json.loads((run / "run_config.json").read_bytes())
    if execution["status"] != "complete":
        raise ValueError(f"Run not complete: {run}")
    if config.get("detector_vocabulary", {}).get("mode") != "annotated":
        raise ValueError(f"Expected annotated vocabulary: {run}")
    records = {}
    for path in sorted((run / "targets").glob("*/*/result.json")):
        row = json.loads(path.read_bytes())
        if row["id"] in records or path.parent != run / "targets" / row["id"]:
            raise ValueError("Duplicate or misplaced target record")
        records[row["id"]] = row
    if len(records) != expected or set(execution["target_ids"]) != set(records):
        raise ValueError(f"Incomplete target inventory: {run}")
    for filename, sha in execution["artifacts"].items():
        check_file(run / filename, sha)
    with (run / "per_target.csv").open(encoding="utf-8-sig", newline="") as stream:
        scored = list(csv.DictReader(stream))
    if len(scored) != expected or len({r["id"] for r in scored}) != expected:
        raise ValueError("Scored inventory is incomplete or duplicated")
    for row in scored:
        if row["id"] not in records or row["status"] != records[row["id"]]["status"]:
            raise ValueError("Scored status differs from inference record")
    return records, {"run": method["run"], "execution_id": execution["id"],
                     "execution_sha256": digest(run / "execution.json"),
                     "run_config_sha256": digest(run / "run_config.json")}


def common_bounds(images):
    """Union of nonempty alpha supports, in original coordinates, plus equal padding."""
    boxes = []
    for image, (row, col) in images:
        box = image.getchannel("A").getbbox()
        if box is None:
            raise ValueError("Successful output has empty alpha support")
        boxes.append((box[0] + col, box[1] + row, box[2] + col, box[3] + row))
    left, top = min(b[0] for b in boxes), min(b[1] for b in boxes)
    right, bottom = max(b[2] for b in boxes), max(b[3] for b in boxes)
    pad = max(8, round(max(right-left, bottom-top)*.06))
    return left-pad, top-pad, right+pad, bottom+pad


def align(image, offset, bounds):
    left, top, right, bottom = bounds
    frame = Image.new("RGBA", (right-left, bottom-top), (0, 0, 0, 0))
    row, col = offset
    frame.paste(image, (col-left, row-top))
    return frame


def build(root, config_path, destination, workers=4):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_bytes())
    methods = config["methods"]
    if not 2 <= len(methods) <= 4 or len({m["id"] for m in methods}) != len(methods):
        raise ValueError("Use two to four distinct methods")
    if destination.exists():
        raise ValueError("Use a new destination; frozen packets must not be overwritten")
    records, runs = {}, {}
    for method in methods:
        records[method["id"]], runs[method["id"]] = load_run(root, method, config["expected_targets"])
    ids = set(next(iter(records.values())))
    if any(set(rows) != ids for rows in records.values()):
        raise ValueError("Run target sets differ")
    coverage = {m: dict(Counter(r["status"] for r in rows.values())) for m, rows in records.items()}
    eligible, excluded = [], []
    for target in sorted(ids):
        rows = [records[m["id"]][target] for m in methods]
        if len({(r["image"], r["input_sha256"], r["label"]) for r in rows}) != 1:
            raise ValueError(f"Input identity differs: {target}")
        statuses = {m["id"]: r["status"] for m, r in zip(methods, rows)}
        if all(r["status"] == "success" for r in rows):
            eligible.append(target)
        else:
            excluded.append({"target_id": target, "statuses": statuses})
    if not eligible:
        raise ValueError("No shared successful outputs")
    seed = config["seed"]
    order_hash = lambda s: hashlib.sha256(f"{VERSION}:{seed}:{s}".encode()).hexdigest()
    eligible.sort(key=lambda x: (order_hash(x), x))
    permutations = list(itertools.permutations(m["id"] for m in methods))
    permutations.sort(key=lambda x: order_hash("order:" + ":".join(x)))
    public, private = destination / "dist", destination / "COORDINATOR_ONLY"
    (public / "assets").mkdir(parents=True)
    private.mkdir()
    slots = list("ABCD"[:len(methods)])

    def prepare(item):
        index, target = item
        case_id = f"P{index+1:04d}"
        assignment = dict(zip(slots, permutations[index % len(permutations)]))
        row = records[methods[0]["id"]][target]
        source_path = root / config["dataset_root"] / row["image"]
        check_file(source_path, row["input_sha256"])
        with Image.open(source_path) as im:
            source = im.convert("RGBA")
        images, provenance = {}, {}
        for method in methods:
            name = method["id"]
            record = records[name][target]
            folder = root / method["run"] / "targets" / target
            path = folder / "completion.png"
            check_file(path, record["completion_sha256"])
            if list(source.size) != record["geometry"]["original_size"]:
                raise ValueError("Geometry does not match source frame")
            with Image.open(path) as im:
                if im.mode != "RGBA":
                    raise ValueError("Expected native RGBA completion")
                images[name] = (im.copy(), record["geometry"]["offset_rc"])
            provenance[name] = {"completion_sha256": record["completion_sha256"],
                                "record_sha256": digest(folder / "result.json"),
                                "geometry": record["geometry"]}
        # A single existing SAM visible reference identifies the target. It is not
        # hidden-region truth. All options use the same context and frame.
        ref_path = root / methods[0]["run"] / "visible_references" / row["reference"]
        check_file(ref_path, row["evaluation_reference_artifacts"]["full"])
        with Image.open(ref_path) as im:
            visible = im.convert("RGBA")
        if visible.size != source.size:
            raise ValueError("Visible reference is not in the original frame")
        bounds = common_bounds([*images.values(), (visible, (0, 0))])
        files = {}
        def save(name, image):
            relative = f"assets/{case_id}-{name}.webp"
            image.save(public / relative, "WEBP", lossless=True, method=4, exact=True)
            files[name] = relative
        save("context", align(source, (0, 0), bounds))
        save("visible", align(visible, (0, 0), bounds))
        # Full scene retained for ambiguous context, also lossless at native scale.
        save("scene", source)
        for slot, method in assignment.items():
            save(slot, align(*images[method], bounds))
        case = {"id": case_id, "label": row["label"], "files": files}
        key = {"case_id": case_id, "target_id": target, "source": row["source"],
               "image": row["image"], "input_sha256": row["input_sha256"],
               "reference_sha256": row["evaluation_reference_artifacts"]["full"],
               "assignment": assignment, "bounds_xyxy": bounds, "artifacts": provenance,
               "display_sha256": {k: digest(public / v) for k, v in files.items()}}
        return case, key

    with ThreadPoolExecutor(max_workers=workers) as pool:
        prepared = list(pool.map(prepare, enumerate(eligible)))
    cases, key = map(list, zip(*prepared))
    identity = {"version": VERSION, "round": config["round"], "seed": seed,
                "config_sha256": digest(config_path), "cases": key, "runs": runs}
    study_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    study = {"schema": VERSION, "study_id": study_id, "round": config["round"],
             "title": config["title"], "slots": slots, "total_targets": len(ids),
             "eligible": len(cases), "excluded": len(excluded), "cases": cases,
             "pending": config.get("pending", []), "batch_size": 20,
             "display": "Native-resolution, lossless WebP; common original-coordinate crop, no resizing."}
    write_json(public / "study.json", study)
    (public / "study.js").write_text("window.PREFERENCE_STUDY = " + json.dumps(study, ensure_ascii=True) + ";\n", encoding="utf-8")
    image_hashes = {case["files"][name]: sha for case, mapping in zip(cases, key)
                    for name, sha in mapping["display_sha256"].items()}
    (public / "integrity.js").write_text("window.PREFERENCE_HASHES = " + json.dumps(image_hashes) + ";\n", encoding="utf-8")
    write_json(private / "key.json", {**identity, "study_id": study_id, "methods": methods,
                                     "coverage": coverage, "excluded": excluded})
    write_json(private / "build.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
                                       "builder_sha256": digest(__file__), "config": config,
                                       "study_sha256": digest(public / "study.json"),
                                       "counts": {"population": len(ids), "eligible": len(cases), "excluded": len(excluded)}})
    print(json.dumps({"study_id": study_id, "eligible": len(cases), "excluded": len(excluded),
                      "coverage": coverage, "destination": str(destination)}))
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    build(args.root, args.config, args.output_dir, args.workers)
