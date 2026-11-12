#!/usr/bin/env python3
"""Prepare census contact sheets and compile explicitly recorded visual labels.

This never infers a visual label from a metric or fills missing reviews with zero.
Batch text: case_number | six comma-separated labels | confounds | evidence.
Label order: texture, color, pattern, semantic, hidden, co-occurrence.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image
from amodal_benchmark.appearance import FAILURES, read_review, write_csv
from amodal_benchmark.dataset import sha256, write_json


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def prepare(output):
    rows = read_csv(output / "review/annotations_template.csv")
    dest = output / "review/census_sheets"
    dest.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(rows), 8):
        group = rows[start:start + 8]
        sheet = Image.new("RGB", (1600, 345 * len(group)), "#ddd")
        for index, row in enumerate(group):
            with Image.open(output / f"review/panels/{int(row['review_number']):03d}.jpg") as panel:
                sheet.paste(panel, (0, 345 * index))
        sheet.save(dest / f"{start // 8 + 1:03d}.jpg", quality=93)
    (output / "review/batches").mkdir(exist_ok=True)
    print(json.dumps({"cases": len(rows), "sheets": (len(rows) + 7) // 8}))


def compile_review(output, allow_partial=False, census=False):
    templates = read_csv(output / "review/annotations_template.csv")
    selected_path = output / "review" / ("census_index.csv" if census else "sample.csv")
    selected_rows = read_csv(selected_path)
    selected = {r["id"]: r for r in selected_rows}
    if len(selected) != len(selected_rows):
        raise ValueError("Duplicate selected target IDs")
    if census and set(selected) != {r["id"] for r in templates}:
        raise ValueError("Census index must include all template targets")
    if census and any(float(r["inclusion_probability"]) != 1 or
                      int(r["stratum_population"]) != int(r["stratum_sample"]) for r in selected_rows):
        raise ValueError("Census index must use unit inclusion probabilities")
    templates = [{**r, **selected[r["id"]]} for r in templates if r["id"] in selected]
    by_number = {int(r["review_number"]): r for r in templates}
    if len(by_number) != len(templates):
        raise ValueError("Duplicate review numbers")
    seen = set()
    receipt = {}
    paths = list((output / "review/batches").glob("*.txt"))
    if census:
        paths += list((output / "review/additional_batches").glob("*.txt"))
    for path in sorted(paths):
        receipt[str(path.relative_to(output / "review"))] = sha256(path)
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            fields = [v.strip() for v in line.split("|", 3)]
            if len(fields) != 4:
                raise ValueError(f"Invalid review line in {path}: {line}")
            number = int(fields[0])
            if number in seen or number not in by_number:
                raise ValueError(f"Duplicate or unexpected review case {number}")
            values = [v.strip() for v in fields[1].split(",")]
            if len(values) != 6 or any(v not in ("0", "1", "2", "3", "U", "NA") for v in values):
                raise ValueError(f"Invalid labels for case {number}")
            row = by_number[number]
            row.update(zip(FAILURES, values))
            row.update(reviewer="Codex single-assistant visual census, 2026-11-07",
                       identity_or_mask_confound=fields[2], evidence=fields[3],
                       cooccurrence_subtype="", cooccurrence_output_scope="")
            if values[-1] in ("1", "2", "3"):
                # Scope is deliberately explicit in evidence; do not invent attribution.
                row["cooccurrence_subtype"] = "see_evidence"
                row["cooccurrence_output_scope"] = "see_evidence"
            seen.add(number)
    missing = sorted(set(by_number) - seen)
    if missing and not allow_partial:
        raise ValueError(f"Missing {len(missing)} reviews: {missing[:30]}")
    reviewed = [by_number[n] for n in sorted(seen)]
    if census:
        write_csv(output / "review/annotations_census_initial.csv", reviewed)
        revisions = output / "review/adjudications.json"
        if revisions.exists():
            for item in json.loads(revisions.read_text(encoding="utf-8-sig")):
                row = by_number[item["review_number"]]
                for key, value in item.get("labels", {}).items():
                    if key not in FAILURES or value not in ("0", "1", "2", "3", "U", "NA"):
                        raise ValueError(f"Invalid adjudication {item}")
                    row[key] = value
                row["evidence"] = item.get("evidence", row["evidence"])
                for key in ("identity_or_mask_confound", "cooccurrence_subtype", "cooccurrence_output_scope"):
                    if key in item:
                        row[key] = item[key]
                if row["cooccurring_object"] not in ("1", "2", "3"):
                    row["cooccurrence_subtype"] = ""
                    row["cooccurrence_output_scope"] = ""
            receipt["adjudications.json"] = sha256(revisions)
        for row in reviewed:
            if row["cooccurring_object"] in ("1", "2", "3") and row["cooccurrence_output_scope"] not in ("raw_rgb_only", "returned_mask", "both"):
                raise ValueError(f"Positive co-occurrence lacks checked scope: {row['review_number']}")
        write_csv(output / "review/sample.csv", selected.values())
        design = {
            "saved_population": len(templates), "reviewed": len(reviewed),
            "sample_size": len(reviewed),
            "all_saved_outputs": not missing, "inclusion_probability": 1,
            "method": "Single-assistant screen-level visual census; targeted full-source rechecks",
            "limitations": "Finite display resolution and one reviewer; no independent human adjudication"}
        write_json(output / "review/census_design.json", design)
        design_path = output / "review/sampling_design.json"
        archive_path = output / "review/sampling_design_interim_552.json"
        if design_path.exists() and not archive_path.exists():
            previous = json.loads(design_path.read_text(encoding="utf-8"))
            if previous.get("not_a_visual_census"):
                archive_path.write_bytes(design_path.read_bytes())
        write_json(design_path, design)
    path = output / "review" / ("annotations_partial.csv" if missing else "annotations.csv")
    write_csv(path, reviewed)
    if not missing:
        read_review(path, [r["id"] for r in templates])
    write_json(output / "review/review_receipt.json", {
        "expected": len(templates), "reviewed": len(seen), "missing": missing,
        "batch_sha256": receipt, "annotation_sha256": sha256(path),
        "label_order": list(FAILURES), "census": census,
        "method": "Single-assistant visual review; no automatic label propagation"})
    print(json.dumps({"reviewed": len(seen), "missing": len(missing)}))


def select(output):
    from collections import defaultdict
    from amodal_benchmark.appearance import BIN_LABELS
    rows = read_csv(output / "review/annotations_template.csv")
    counts = dict(zip(BIN_LABELS, (264, 48, 48, 48, 48, 32, 32, 32)))
    groups = defaultdict(list)
    for row in rows:
        groups[row["proxy_bin"]].append(row)
    selected = []
    for label in BIN_LABELS:
        group = groups[label]
        sample = group[:counts[label]]
        for row in sample:
            row.update(stratum_sample=len(sample), inclusion_probability=len(sample) / len(group))
        selected.extend(sample)
    original = output / "review/census_index.csv"
    if not original.exists():
        original.write_bytes((output / "review/sample.csv").read_bytes())
    write_csv(output / "review/sample.csv", selected)
    pending = selected[264:]
    dest = output / "review/audit_sheets"
    dest.mkdir(exist_ok=True)
    for start in range(0, len(pending), 8):
        group = pending[start:start + 8]
        sheet = Image.new("RGB", (1600, 345 * len(group)), "#ddd")
        for i, row in enumerate(group):
            with Image.open(output / f"review/panels/{int(row['review_number']):03d}.jpg") as panel:
                sheet.paste(panel, (0, 345 * i))
        sheet.save(dest / f"{start // 8 + 1:03d}.jpg", quality=94)
    write_json(output / "review/sampling_design.json", {
        "saved_population": len(rows), "sample_size": len(selected),
        "stratum_samples": counts, "seed": 20261107,
        "ordering": "SHA256(seed:target_id), ascending within expansion stratum",
        "allocation_reason": "Time-bounded audit: first stratum reviewed before reserving balanced coverage of remaining strata",
        "extrapolation": "Inverse-probability weights within stratum; unknown and NA excluded per failure",
        "not_a_visual_census": True})
    print(json.dumps({"selected": len(selected), "remaining_after_first_264": len(pending)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "select", "compile"))
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--census", action="store_true", help="Compile both batch directories and explicit adjudications")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.analysis_dir)
    elif args.command == "select":
        select(args.analysis_dir)
    else:
        compile_review(args.analysis_dir, args.allow_partial, args.census)


if __name__ == "__main__":
    main()
