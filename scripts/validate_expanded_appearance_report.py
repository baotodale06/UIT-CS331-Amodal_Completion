#!/usr/bin/env python3
"""Validate census coverage, artifact identity, denominators and report links."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from amodal_benchmark.appearance import (FAILURES, read_review, stable_sample, taxonomy_counts,
    weighted_taxonomy_counts, ratio_bin)
from amodal_benchmark.dataset import sha256, write_json
from build_appearance_report import csv_rows
from analyze_appearance_failures import summarize_bins


def check(ok, message):
    if not ok:
        raise ValueError(message)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.images = []
        self.articles = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])
        if tag == "img":
            self.images.append(attrs["src"])
        if tag == "article":
            self.articles += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.analysis_dir.resolve()
    manifest = json.loads((output / "analysis_manifest.json").read_text())
    run, dataset = Path(manifest["run_dir"]), Path(manifest["dataset_root"])
    rows = json.loads((output / "per_target_diagnostics.json").read_text())
    by_id = {r["id"]: r for r in rows}
    check(len(by_id) == len(rows) == 2565, "Target census incomplete or duplicated")
    original = {r["id"]: r for r in csv_rows(run / "per_target.csv")}
    check(set(original) == set(by_id), "Original result identities differ")
    for row in rows:
        check(row["status"] == original[row["id"]]["status"], "Original status differs")
        for metric in ("CLIP", "LPIPS", "Feature Similarity", "SSIM"):
            value = original[row["id"]][metric]
            check(row[metric] is None if value == "" else row[metric] == float(value), "Original metric differs")
    for name, digest in manifest["inputs"].items():
        check(sha256(run / name) == digest, f"Changed immutable input {name}")
    config = json.loads((run / "run_config.json").read_text())
    check(config["detector_vocabulary"]["mode"] == "annotated", "Wrong vocabulary experiment")
    check(config["label_matching"]["version"] == 2, "Wrong matcher")
    check(config["reference_backend"] == "grounded-sam", "Wrong reference protocol")

    def artifacts(row):
        checks = [(run / "targets" / row["id"] / "result.json", row["record_sha256"]),
                  (dataset / row["image"], row["input_sha256"])]
        if row["status"] == "success":
            checks += [(run / "targets" / row["id"] / "completion.png", row["completion_sha256"]),
                       (run / "visible_references" / row["reference"], row["reference_full_sha256"]),
                       (run / "visible_references/masks" / row["reference"], row["reference_mask_sha256"])]
        for path, digest in checks:
            check(sha256(path) == digest, f"Changed source artifact {path}")
        record = json.loads(checks[0][0].read_text())
        check(record["status"] == row["status"], f"Changed target status {row['id']}")
        check(row["gt_bin"] == ratio_bin(row["gt_occlusion_ratio"]), f"Incorrect true-occlusion bin {row['id']}")
        check(row["proxy_bin"] == ratio_bin(row["predicted_expansion_ratio"]), f"Incorrect proxy bin {row['id']}")
        return len(checks)
    with ThreadPoolExecutor(max_workers=6) as pool:
        checked_artifacts = sum(pool.map(artifacts, rows))

    saved = [r for r in rows if r["status"] == "success"]
    review = read_review(output / "review/annotations.csv", [r["id"] for r in saved])
    check(len(review) == 1390, "Missing saved-output labels")
    check(sha256(output / "review/annotations.csv") == manifest["review_sha256"], "Changed label digest")
    expected_sample = stable_sample(rows, per_bin=10000)
    sample = csv_rows(output / "review/sample.csv")
    check([r["id"] for r in sample] == [r["id"] for r in expected_sample], "Census ordering differs")
    sample_by_id = {r["id"]: r for r in sample}
    for row in review:
        s = sample_by_id[row["id"]]
        check(float(row["inclusion_probability"]) == float(s["inclusion_probability"]) == 1, "Non-unit census weight")
        check(int(row["stratum_population"]) == int(row["stratum_sample"]) == int(s["stratum_population"]), "Incorrect census metadata")
        check(row["proxy_bin"] == by_id[row["id"]]["proxy_bin"], "Wrong label stratum")
        check(row["source"] == by_id[row["id"]]["source"] and row["category"] == by_id[row["id"]]["category"], "Wrong label metadata")
        if row["cooccurring_object"] in ("1", "2", "3"):
            check(row["cooccurrence_output_scope"] == "raw_rgb_only", "Co-occurrence attribution differs from narrative")
            check((output / f"review/source_checks/{int(row['review_number']):04d}.jpg").exists(), "Missing full-source evidence")
    receipt = json.loads((output / "review/review_receipt.json").read_text())
    check(receipt["census"] and not receipt["missing"] and receipt["reviewed"] == 1390, "Review receipt incomplete")
    for name, digest in receipt["batch_sha256"].items():
        check(sha256(output / "review" / name) == digest, f"Changed review batch {name}")
    check(json.loads((output / "taxonomy.json").read_text()) == weighted_taxonomy_counts(review), "Taxonomy does not reproduce labels")
    check(sum(any(r[k] in ("1", "2", "3") for k in FAILURES) for r in review) == 652, "Narrative positive count differs")
    for field, filename in (("gt_bin", "occlusion_bins.csv"), ("proxy_bin", "expansion_proxy_bins.csv")):
        expected = summarize_bins(rows, field)
        actual = csv_rows(output / filename)
        check(len(expected) == len(actual), "Bin row count differs")
        for e, a in zip(expected, actual):
            for key, value in e.items():
                check(a[key] == "" if value is None else (a[key] == value if isinstance(value, str) else math.isclose(float(a[key]), value, rel_tol=1e-12, abs_tol=1e-12)), f"Bin statistic differs: {field}/{e['bin']}/{key}")
    grouped = csv_rows(output / "taxonomy_by_occlusion.csv") + csv_rows(output / "taxonomy_groups.csv")
    for field in {r["group_field"] for r in grouped}:
        for group in {r["group"] for r in grouped if r["group_field"] == field}:
            stats = {r["failure"]: r for r in taxonomy_counts([r for r in review if by_id[r["id"]][field] == group])}
            for a in [r for r in grouped if r["group_field"] == field and r["group"] == group]:
                for key in ("positive", "assessable", "unassessable", "not_applicable", "mild", "moderate", "severe"):
                    check(int(a[key]) == stats[a["failure"]][key], "Grouped taxonomy differs")
    for filename, digest in manifest["external_annotation_hashes"].items():
        check(sha256(output / "external" / filename) == digest, "External COCOA data changed")
    mask_audit = json.loads((output / "cocoa_mask_audit.json").read_text())
    check(mask_audit["targets"] == 321 and mask_audit["max_ratio_difference"] < 1e-6, "Mask-decoded ratios disagree")
    link_counts = {}
    for filename in ("REPORT.html", "review/index.html"):
        path = output / filename
        parsed = Links()
        parsed.feed(path.read_text(encoding="utf-8"))
        for link in parsed.links + parsed.images:
            if not urlsplit(link).scheme:
                check((path.parent / unquote(urlsplit(link).path)).exists(), f"Broken local link {filename}: {link}")
        if filename.startswith("review"):
            check(parsed.articles == len(parsed.images) == len(review), "Viewer coverage differs")
        link_counts[filename] = {"images": len(parsed.images), "links": len(parsed.links), "articles": parsed.articles}
    from pypdf import PdfReader
    pdf = PdfReader(output / "REPORT.pdf")
    build = json.loads((output / "report_build.json").read_text())
    check(len(pdf.pages) == build["report_pages"] == 13, "Unexpected PDF page count")
    for page in pdf.pages:
        content = page.extract_text()
        check("Expanded-vocabulary PD-MC" in content, "Wrong PDF run footer")
        check(len(content) > 150, "Empty report page")
    check(max(build["pdf_page_scales"]) <= 1.1, "PDF content shrunk excessively")
    result = {"validated_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
        "target_records": len(rows), "saved_outputs_reviewed": len(review), "pipeline_failures": len(rows)-len(review),
        "source_artifact_hash_checks": checked_artifacts, "run_protocol": "annotated vocabulary / matcher v2 / grounded-sam",
        "taxonomy": "all severity, assessability, weighted and grouped fields reproduce explicit labels",
        "bin_metrics": "all means, denominators and 95% interval endpoints reproduced",
        "external_mask_ratios": "321 decoded ratios agree; max difference < 1e-6", "html": link_counts,
        "pdf_pages": len(pdf.pages), "report_sha256": sha256(output / "REPORT.pdf"),
        "review_sha256": sha256(output / "review/annotations.csv"),
        "limitations": "Does not independently validate the assistant's visual judgments or historical execution provenance"}
    write_json(output / "validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
