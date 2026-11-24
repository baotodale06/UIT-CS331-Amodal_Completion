"""Verify frozen study coverage, blinding, image hashes, and common-frame geometry."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from PIL import Image


def validate(packet, workers=4):
    packet = Path(packet)
    public = packet / "dist"
    study = json.loads((public / "study.json").read_bytes())
    key = json.loads((packet / "COORDINATOR_ONLY/key.json").read_bytes())
    receipt = json.loads((packet / "COORDINATOR_ONLY/build.json").read_bytes())
    if hashlib.sha256((public / "study.json").read_bytes()).hexdigest() != receipt["study_sha256"]:
        raise ValueError("Study manifest changed")
    if study["study_id"] != key["study_id"] or len(study["cases"]) != study["eligible"]:
        raise ValueError("Study identity or count mismatch")
    if study["eligible"] + len(key["excluded"]) != study["total_targets"]:
        raise ValueError("Failure coverage mismatch")
    rows = {r["case_id"]: r for r in key["cases"]}
    if len(rows) != len(study["cases"]) or len({r["target_id"] for r in rows.values()}) != len(rows):
        raise ValueError("Duplicate cases or targets")
    if set(rows) != {c["id"] for c in study["cases"]}:
        raise ValueError("Key coverage mismatch")
    slots = study["slots"]
    methods = {m["id"] for m in key["methods"]}
    allowed_fields = {"id", "label", "files"}
    balance = {slot: Counter() for slot in slots}
    expected_files = set()
    hashes = {}
    for case in study["cases"]:
        if set(case) != allowed_fields or set(case["files"]) != {"context", "visible", "scene", *slots}:
            raise ValueError("Unexpected public case data")
        row = rows[case["id"]]
        if set(row["assignment"]) != set(slots) or set(row["assignment"].values()) != methods:
            raise ValueError("Invalid blinded permutation")
        for slot, method in row["assignment"].items():
            balance[slot][method] += 1
        for name, relative in case["files"].items():
            if relative != f"assets/{case['id']}-{name}.webp" or relative in expected_files:
                raise ValueError("Unblinded, duplicate, or unsafe asset path")
            expected_files.add(relative)
            hashes[relative] = row["display_sha256"][name]
    if any(max(c.values()) - min(c.values()) > len(methods) for c in balance.values()):
        raise ValueError("Position allocation is unbalanced")
    actual = {p.relative_to(public).as_posix() for p in (public / "assets").rglob("*") if p.is_file()}
    if actual != expected_files:
        raise ValueError(f"Image inventory differs: {len(expected_files-actual)} missing, {len(actual-expected_files)} extra")
    for path in public.rglob("*"):
        if path.is_file() and path.relative_to(public).as_posix() not in expected_files | {
                "index.html", "style.css", "app.js", "core.js", "study.json", "study.js", "integrity.js"}:
            raise ValueError(f"Unexpected public file: {path}")
    integrity = (public / "integrity.js").read_text(encoding="utf-8")
    if not integrity.startswith("window.PREFERENCE_HASHES = ") or json.loads(integrity.removeprefix("window.PREFERENCE_HASHES = ").strip().removesuffix(";")) != hashes:
        raise ValueError("Browser image hashes differ")
    data_js = (public / "study.js").read_text(encoding="utf-8")
    if json.loads(data_js.removeprefix("window.PREFERENCE_STUDY = ").strip().removesuffix(";")) != study:
        raise ValueError("Browser study data differs")

    def verify(case):
        dimensions = []
        for name, relative in case["files"].items():
            path = public / relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[relative]:
                raise ValueError(f"Image changed: {relative}")
            with Image.open(path) as im:
                im.load()
                if name != "scene":
                    dimensions.append(im.size)
        bounds = rows[case["id"]]["bounds_xyxy"]
        expected_size = (bounds[2] - bounds[0], bounds[3] - bounds[1])
        if any(size != expected_size for size in dimensions):
            raise ValueError(f"Comparison frames differ: {case['id']}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(verify, study["cases"]))
    return {"status": "passed", "study_id": study["study_id"], "cases": study["eligible"],
            "images": len(expected_files), "excluded": study["excluded"],
            "position_counts": {slot: dict(c) for slot, c in balance.items()}}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet-dir", required=True)
    p.add_argument("--receipt")
    a = p.parse_args()
    result = validate(a.packet_dir)
    if a.receipt:
        Path(a.receipt).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
