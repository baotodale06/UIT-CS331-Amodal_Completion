#!/usr/bin/env python3
"""Verify packet coverage, blank ratings, allowlisted disclosure and ZIP bytes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.appearance import FAILURES
from amodal_benchmark.dataset import sha256, write_json


def validate(packet):
    public, private = packet / "FOR_RATER", packet / "COORDINATOR_ONLY"
    protocol = json.loads((private / "protocol.json").read_text(encoding="utf-8"))
    key = json.loads((private / "identity_key.json").read_text(encoding="utf-8"))
    inventory = json.loads((private / "public_manifest.json").read_text(encoding="utf-8"))["files"]
    data = (public / "data.js").read_text(encoding="utf-8")
    assert data.startswith("window.REVIEW_DATA = ") and data.endswith(";\n")
    config = json.loads(data[len("window.REVIEW_DATA = "):-2])
    assert config["pack_id"] == protocol["pack_id"] == key["pack_id"]
    cases = config["cases"]
    assert len(cases) == protocol["count"] == len(key["cases"])
    assert {c["case_id"] for c in cases} == {c["case_id"] for c in key["cases"]}
    assert len({c["case_id"] for c in cases}) == len(cases)
    assert len({c["id"] for c in key["cases"]}) == len(cases)
    expected_fields = {"case_id", "label", "batch", "source", "visible", "completion", "raw", "overview"}
    expected_files = {"index.html", "data.js", "RATER_GUIDE.md", "START_HERE.txt", "ratings_blank.csv"}
    decoded = 0
    for i, case in enumerate(cases, 1):
        assert set(case) == expected_fields
        assert case["case_id"] == f"H{i:04d}" and case["batch"] == (i - 1) // 20 + 1
        mapping = key["cases"][i - 1]
        for name in ("source", "visible", "completion", "raw", "overview"):
            path = public / case[name]
            assert path.resolve().is_relative_to((public / "assets").resolve())
            expected_files.add(case[name])
            with Image.open(path) as image:
                image.verify()
            decoded += 1
            if name != "overview":
                assert sha256(path) == mapping["source_hashes"][name]
    with (public / "ratings_blank.csv").open(encoding="utf-8-sig", newline="") as stream:
        blanks = list(csv.DictReader(stream))
    assert len(blanks) == len(cases)
    assert {r["case_id"] for r in blanks} == {c["case_id"] for c in cases}
    assert all(not row[field] for row in blanks for field in (*FAILURES, "reviewer_code", "evidence", "confounds"))
    actual_files = {p.relative_to(public).as_posix() for p in public.rglob("*") if p.is_file()}
    assert actual_files == expected_files == {item["path"] for item in inventory}
    receipt = json.loads((private / "delivery_receipt.json").read_text(encoding="utf-8"))
    archive = packet / receipt["archive"]
    assert sha256(archive) == receipt["sha256"]
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {"human-review/" + name for name in expected_files}
        for item in inventory:
            assert sha256(public / item["path"]) == item["sha256"]
            with z.open("human-review/" + item["path"]) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            assert digest == item["sha256"]
    result = {"passed": True, "pack_id": protocol["pack_id"], "cases": len(cases),
              "images_decoded": decoded, "native_artifact_hashes": len(cases) * 4,
              "public_and_zip_hashes": len(inventory) * 2,
              "blank_score_cells": len(cases) * len(FAILURES),
              "private_files_excluded": True, "archive_sha256": receipt["sha256"],
              "limitations": "Checks packaging and preserved evidence, not the validity of future human judgments."}
    write_json(private / "validation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.packet_dir), indent=2))
