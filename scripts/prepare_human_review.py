#!/usr/bin/env python3
"""Build an offline, blinded rating packet from frozen appearance artifacts.

No inference, automatic visual labels, or edits to existing research artifacts.
The public packet is an allowlist; the identity key stays with the coordinator.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.appearance import FAILURES
from amodal_benchmark.dataset import sha256, write_json

TEMPLATE = ROOT / "scripts/human_review"
VERSION = "appearance-human-v1"


def ordered_cases(rows, seed):
    saved = [r for r in rows if r["status"] == "success"]
    if len({r["id"] for r in saved}) != len(saved):
        raise ValueError("Duplicate target identity")
    return sorted(saved, key=lambda r: (hashlib.sha256(
        f"{VERSION}:{seed}:{r['id']}".encode()).hexdigest(), r["id"]))


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build(analysis, destination, count, seed=20261108):
    analysis, destination = analysis.resolve(), destination.resolve()
    if destination.exists():
        raise ValueError("Use a new destination; existing packets are immutable")
    manifest = json.loads((analysis / "analysis_manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((analysis / "per_target_diagnostics.json").read_text(encoding="utf-8"))
    ordered = ordered_cases(rows, seed)
    count = len(ordered) if count == 0 else count
    if not 1 <= count <= len(ordered):
        raise ValueError("Requested count is outside the saved-output population")
    selected = ordered[:count]
    # Read identity numbers from the blank template, never from prior ratings.
    with (analysis / "review/annotations_template.csv").open(encoding="utf-8-sig", newline="") as stream:
        panel_numbers = {r["id"]: int(r["review_number"]) for r in csv.DictReader(stream)}
    run, dataset = Path(manifest["run_dir"]), Path(manifest["dataset_root"])
    if sha256(run / "per_target.csv") != manifest["inputs"]["per_target.csv"]:
        raise ValueError("Run inventory changed")
    identity = {"version": VERSION, "seed": seed, "ids": [r["id"] for r in selected],
                "diagnostics_sha256": sha256(analysis / "per_target_diagnostics.json"),
                "template_sha256": sha256(TEMPLATE / "index.html"),
                "guide_sha256": sha256(TEMPLATE / "RATER_GUIDE.md")}
    pack_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    public = destination / "FOR_RATER"
    private = destination / "COORDINATOR_ONLY"
    (public / "assets").mkdir(parents=True)
    private.mkdir()

    def prepare(item):
        index, row = item
        case_id = f"H{index:04d}"
        target = run / "targets" / row["id"]
        paths = {"source": dataset / row["image"],
                 "visible": run / "visible_references" / row["reference"],
                 "completion": target / "completion.png", "raw": target / "completion.jpg"}
        expected = {"source": row["input_sha256"], "visible": row["reference_full_sha256"],
                    "completion": row["completion_sha256"]}
        hashes = {name: sha256(path) for name, path in paths.items()}
        if any(hashes[k] != v for k, v in expected.items()):
            raise ValueError(f"Changed source artifact: {row['id']}")
        if sha256(target / "result.json") != row["record_sha256"]:
            raise ValueError(f"Changed geometry/record: {row['id']}")
        files = {}
        for name, path in paths.items():
            relative = f"assets/{case_id}_{name}{path.suffix.lower()}"
            shutil.copyfile(path, public / relative)
            files[name] = relative
        panel_path = analysis / f"review/panels/{panel_numbers[row['id']]:03d}.jpg"
        relative = f"assets/{case_id}_overview.jpg"
        # Existing panel body has fixed neutral column labels, no ratings.
        # Remove both metadata header rows (case ID, scores and occlusion).
        with Image.open(panel_path) as panel:
            if panel.size != (1600, 345):
                raise ValueError("Unexpected panel layout; inspect before cropping")
            panel.crop((0, 53, 1600, 345)).save(public / relative, quality=97)
        files["overview"] = relative
        item = {"case_id": case_id, "label": row["label"], "batch": (index - 1) // 20 + 1,
                **files}
        mapping = {"case_id": case_id, "id": row["id"], "order": index,
                   "source": row["source"], "category": row["category"],
                   "proxy_bin": row["proxy_bin"], "gt_bin": row["gt_bin"],
                   "inclusion_probability": count / len(ordered),
                   "original_review_number": panel_numbers[row["id"]],
                   "source_hashes": hashes, "panel_sha256": sha256(panel_path)}
        return item, mapping

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(prepare, enumerate(selected, 1)))
    cases, mapping = [r[0] for r in results], [r[1] for r in results]
    config = {"version": VERSION, "pack_id": pack_id, "cases": cases,
              "failure_keys": list(FAILURES), "batch_size": 20}
    (public / "data.js").write_text("window.REVIEW_DATA = " + json.dumps(
        config, ensure_ascii=True).replace("<", "\\u003c") + ";\n", encoding="utf-8")
    for name in ("index.html", "RATER_GUIDE.md"):
        shutil.copyfile(TEMPLATE / name, public / name)
    (public / "START_HERE.txt").write_text(
        f"INDEPENDENT VISUAL REVIEW - {count} cases / {(count + 19) // 20} batches\n\n"
        "1. Extract the entire ZIP to a normal folder.\n"
        "2. Open index.html in Chrome, Edge or Firefox. No internet/account needed.\n"
        "3. Read the guide, enter a reviewer code and disclose prior involvement/exposure.\n"
        "4. Rate each case independently. Blank ratings are not zero.\n"
        "5. Download a JSON backup after each batch and before closing the browser.\n"
        "6. Return the final JSON and CSV to the study coordinator.\n\n"
        "Local autosave is a convenience. It can be lost if the folder moves, the browser\n"
        "changes or browser storage is cleared. Import your JSON backup to resume.\n"
        "If the browser form cannot be used, fill ratings_blank.csv using RATER_GUIDE.md,\n"
        "record your reviewer code and independence disclosures in the accompanying email.\n"
        "Do not consult the project report, model scores, prior labels, AI or other raters.\n"
        "Aim for 1-2 minutes per case, with breaks every 20 cases. Timing is an estimate;\n"
        "take longer when needed. U is appropriate when the evidence is insufficient.\n",
        encoding="utf-8")
    fields = ["case_id", "label", "batch", "reviewer_code", *FAILURES,
              "confounds", "evidence", "cooccurrence_subtype", "cooccurrence_output_scope"]
    write_csv(public / "ratings_blank.csv", cases, fields)
    write_json(private / "identity_key.json", {"pack_id": pack_id, "cases": mapping})
    write_csv(private / "identity_key.csv", mapping,
              ["case_id", "id", "order", "source", "category", "proxy_bin", "gt_bin",
               "inclusion_probability", "original_review_number"])
    private_manifest = {**identity, "pack_id": pack_id, "created_utc": datetime.now(timezone.utc).isoformat(),
                        "analysis_dir": str(analysis), "count": count, "population": len(ordered),
                        "all_targets": len(rows), "no_output": len(rows) - len(ordered),
                        "selection": "First n of SHA-256 permutation, without replacement; no rating/metric selection",
                        "prior_review_sha256": manifest["review_sha256"],
                        "responses_received": 0, "human_evaluation_completed": False}
    write_json(private / "protocol.json", private_manifest)
    inventory = []
    for path in sorted(public.rglob("*")):
        if path.is_file():
            inventory.append({"path": path.relative_to(public).as_posix(),
                              "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(private / "public_manifest.json", {"pack_id": pack_id, "files": inventory})
    archive = destination / f"human-review-{count}-cases.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for item in inventory:
            z.write(public / item["path"], "human-review/" + item["path"])
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP CRC verification failed")
        if len(z.namelist()) != len(inventory):
            raise ValueError("ZIP inventory mismatch")
    receipt = {"pack_id": pack_id, "cases": count, "files": len(inventory),
               "archive": archive.name, "bytes": archive.stat().st_size,
               "sha256": sha256(archive), "zip_crc_check": "passed",
               "source_hash_checks": count * 4,
               "blank_ratings": True, "coordinator_files_in_zip": False}
    write_json(private / "delivery_receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=160, help="0 = every saved output")
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args()
    print(json.dumps(build(args.analysis_dir, args.output_dir, args.count, args.seed), indent=2))


if __name__ == "__main__":
    main()
