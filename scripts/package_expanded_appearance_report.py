#!/usr/bin/env python3
"""Package validated report evidence without datasets, models or environments."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.analysis_dir.resolve()
    validation = json.loads((output / "validation.json").read_text())
    if not validation["passed"] or validation["report_sha256"] != sha256(output / "REPORT.pdf"):
        raise ValueError("Validate the current PDF before packaging")
    if validation["review_sha256"] != sha256(output / "review/annotations.csv"):
        raise ValueError("Review changed after validation")
    paths = {p for p in output.iterdir() if p.is_file() and p.suffix in (".md", ".html", ".pdf", ".csv", ".json")
             and p.name not in ("delivery_manifest.json", "delivery_receipt.json")}
    paths.update(p for p in (output / "review").iterdir() if p.is_file())
    for directory in ("figures", "review/panels", "review/source_checks", "review/batches", "review/additional_batches", "reproducibility"):
        paths.update(p for p in (output / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    paths.update((output / "validation").glob("*.json"))
    paths.add(output / "external/download_receipt.json")
    items = [{"path": p.relative_to(output).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(paths)]
    manifest = output / "delivery_manifest.json"
    write_json(manifest, {"created_utc": datetime.now(timezone.utc).isoformat(), "files": items,
                         "excluded": ["models", "dataset", "external annotation JSONs", "CPU environment", "duplicate contact sheets"]})
    archive = output / "appearance-expanded-delivery.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(paths | {manifest}):
            z.write(p, p.relative_to(output).as_posix())
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise ValueError(f"Corrupt archive member: {bad}")
        panel_count = sum(n.startswith("review/panels/") and n.endswith(".jpg") for n in z.namelist())
        if panel_count != 1390:
            raise ValueError("Archive is missing census panels")
    receipt = {"archive": archive.name, "bytes": archive.stat().st_size, "sha256": sha256(archive),
               "files": len(items) + 1, "review_panels": panel_count, "zip_crc_check": "passed",
               "manifest_sha256": sha256(manifest)}
    write_json(output / "delivery_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
