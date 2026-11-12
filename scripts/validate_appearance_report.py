#!/usr/bin/env python3
"""Check report arithmetic, links, sample identities and frozen input hashes."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from html.parser import HTMLParser
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image
from pypdf import PdfReader
from amodal_benchmark.appearance import (
    BIN_LABELS, FAILURES, read_review, stable_sample, weighted_taxonomy_counts)
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.protocol import METRICS


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def check(condition, message):
    if not condition:
        raise ValueError(message)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.images = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "img":
            self.images.append(attrs["src"])
        for name in ("src", "href"):
            if name in attrs:
                self.links.append(attrs[name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.analysis_dir.resolve()
    manifest = json.loads((output / "analysis_manifest.json").read_text(encoding="utf-8"))
    run = Path(manifest["run_dir"])
    for name, digest in manifest["inputs"].items():
        check(sha256(run / name) == digest, f"Run input changed: {name}")
    for name, digest in manifest["external_annotation_hashes"].items():
        check(sha256(output / "external" / name) == digest, f"External annotation changed: {name}")
    for name, digest in manifest["analysis_source_hashes"].items():
        check(sha256(ROOT / name.replace("\\", "/")) == digest, f"Analysis source changed: {name}")
    rows = json.loads((output / "per_target_diagnostics.json").read_text(encoding="utf-8"))
    original = {r["id"]: r for r in read_csv(run / "per_target.csv")}
    check(len(rows) == 2565 and len({r["id"] for r in rows}) == 2565, "Target inventory mismatch")
    check(sum(r["status"] == "success" for r in rows) == 951, "Saved-output count mismatch")
    for row in rows:
        source = original[row["id"]]
        check(row["status"] == source["status"], "Status mismatch")
        for key in METRICS:
            expected = float(source[key]) if source[key] else None
            check(expected == row[key], f"Metric changed: {row['id']} {key}")
    checks = {}
    for filename, key in (("occlusion_bins.csv", "gt_bin"), ("expansion_proxy_bins.csv", "proxy_bin")):
        for stat in read_csv(output / filename):
            selected = [r for r in rows if r[key] == stat["bin"]]
            saved = [r for r in selected if r["status"] == "success"]
            check((len(selected), len(saved)) == (int(stat["targets"]), int(stat["saved"])), f"{filename}: count mismatch")
            for metric in (*METRICS, "visible_delta_e76_median", "visible_retention"):
                for scope, subset in (("all", selected), ("saved", saved)):
                    values = [r[metric] for r in subset if r.get(metric) is not None]
                    prefix = f"{metric}_{scope}_"
                    check(int(stat[prefix+"n"]) == len(values), f"{filename}: metric denominator mismatch")
                    if values:
                        check(math.isclose(float(stat[prefix+"mean"]), sum(values)/len(values), abs_tol=1e-12),
                              f"{filename}: metric mean mismatch")
                    else:
                        check(stat[prefix+"mean"] == "", f"{filename}: missing metric was filled")
        checks[filename] = "counts, all metric means and denominators agree with per-target data"
    sample = read_csv(output / "review/sample.csv")
    deterministic = stable_sample(rows)
    check(len(sample) == len(deterministic) == 120, "Sample size mismatch")
    for saved, expected in zip(sample, deterministic):
        check(all(saved[k] == str(v) for k, v in expected.items()), "Deterministic sample changed")
    review = read_review(output / "review/annotations.csv", [r["id"] for r in sample])
    check(sha256(output / "review/annotations.csv") == manifest["review_sha256"], "Review hash changed")
    for label, sampled in zip(review, sample):
        check(all(label[k] == v for k, v in sampled.items()), "Review sampling metadata mismatch")
    taxonomy = json.loads((output / "taxonomy.json").read_text(encoding="utf-8"))
    expected = weighted_taxonomy_counts(review)
    check(taxonomy == expected, "Taxonomy does not reproduce final labels")
    check(sum(any(r[k] in ("1", "2", "3") for k in FAILURES) for r in review) == 69, "Positive review count mismatch")
    check(sum(r["gt_occlusion_ratio"] is not None for r in rows) == 321, "Ground-truth subset mismatch")
    check(sum(r["gt_bin"] in BIN_LABELS[1:7] for r in rows) == 114, "Requested bin population mismatch")
    check(sum(r["gt_bin"] in BIN_LABELS[1:7] and r["status"] == "success" for r in rows) == 44,
          "Requested bin saved-output count mismatch")
    link_counts = {}
    for filename, expected_images in (("REPORT.html", 9), ("review/index.html", 120)):
        path = output / filename
        html = Links()
        html.feed(path.read_text(encoding="utf-8"))
        check(len(html.images) == expected_images, f"{filename}: missing rendered figures")
        for link in html.links:
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = path.parent / unquote(parsed.path)
            check(target.is_file(), f"Broken local link: {link}")
        for source in html.images:
            with Image.open(path.parent / source) as image:
                image.verify()
        link_counts[filename] = {"local_and_external_links": len(html.links), "verified_images": len(html.images)}
    pdf = PdfReader(output / "REPORT.pdf")
    check(len(pdf.pages) == 12, "PDF page count changed")
    text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    check(all(len(p.extract_text() or "") > 600 for p in pdf.pages), "Blank or orphan PDF page")
    check(all(t in text for t in ("54/112", "51/113", "25/45", "4/114", "59/100", "11/105")), "Taxonomy missing from PDF")
    check("\u25a0" not in text, "PDF has unsupported-glyph markers")
    result = {"validated_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
              "target_rows": len(rows), "sample_rows": len(review), "pdf_pages": len(pdf.pages),
              "input_hashes_unchanged": True, "sample_and_review_metadata": "reproduced",
              "taxonomy": "all raw and weighted fields reproduced", "bin_checks": checks,
              "html_checks": link_counts, "pdf_page_text_lengths": [len(p.extract_text() or "") for p in pdf.pages],
              "report_dependencies": {name: version(name) for name in ("numpy", "Pillow", "matplotlib", "markdown", "reportlab", "pypdf")},
              "notes": "CLI checks inspect the saved files. No GPU or new learned-metric evaluation."}
    browser_path = output / "browser_validation.json"
    if browser_path.is_file():
        browser = json.loads(browser_path.read_text(encoding="utf-8"))
        for name, digest in browser["html_sha256"].items():
            check(sha256(output / name) == digest, "Browser receipt belongs to a different HTML artifact")
        result["browser_receipt"] = browser
    write_json(output / "report_validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
