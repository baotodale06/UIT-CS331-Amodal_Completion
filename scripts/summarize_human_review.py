#!/usr/bin/env python3
"""Validate returned human JSON; retain missingness and compare frozen labels.

Writes new outputs only. Never upgrades assistant annotations into human ratings.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.appearance import FAILURES
from amodal_benchmark.dataset import sha256, write_json

CONFOUNDS = {"none_observed", "mask_contamination", "mask_undercoverage", "output_mask_loss",
             "target_switch", "low_resolution", "incomplete_geometry", "occluder_residue", "other"}
SUBTYPES = {"new_secondary_object", "same_category_duplication", "regenerated_occluder",
            "replacement_food", "unsupported_accessory", "other_unclear"}
SCOPES = {"inside_or_overlapping", "raw_rgb_only", "unclear"}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object)


def validate_response(response, protocol, mapping, allow_partial=False):
    if response.get("pack_id") != protocol["pack_id"] or response.get("version") != protocol["version"]:
        raise ValueError("Wrong packet identity/version")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", response.get("reviewer_code", "")):
        raise ValueError("Reviewer code is required (letters, digits, underscore, hyphen)")
    if response.get("understood") is not True:
        raise ValueError("Reviewer guide declaration is required")
    for key in ("involved", "exposed"):
        if response.get(key) not in ("yes", "no"):
            raise ValueError("Explicit involvement/exposure disclosures are required")
    if not isinstance(response.get("disclosure"), str):
        raise ValueError("Disclosure must be a string (empty if neither answer is yes)")
    if (response["involved"] == "yes" or response["exposed"] == "yes") and not response.get("disclosure", "").strip():
        raise ValueError("Explain the declared involvement/exposure")
    expected = {r["case_id"] for r in mapping}
    responses = response.get("responses")
    if not isinstance(responses, dict) or set(responses) - expected:
        raise ValueError("Unexpected response IDs")
    complete = {}
    for case_id, row in responses.items():
        if not isinstance(row, dict) or type(row.get("complete")) is not bool:
            raise ValueError(f"Invalid completion flag: {case_id}")
        for key in FAILURES:
            valid = {"0", "1", "2", "3", "U"} | ({"NA"} if key == "pattern_inconsistency" else set())
            if row.get(key, "") not in valid | ({""} if not row["complete"] else set()):
                raise ValueError(f"Invalid or missing {key}: {case_id}")
        if not row["complete"]:
            continue
        if not isinstance(row.get("evidence"), str) or not row["evidence"].strip():
            raise ValueError(f"Evidence note required: {case_id}")
        confounds = row.get("confounds")
        if (not isinstance(confounds, list) or not confounds or
                any(c not in CONFOUNDS for c in confounds) or len(set(confounds)) != len(confounds) or
                ("none_observed" in confounds and len(confounds) > 1)):
            raise ValueError(f"Invalid/conflicting confounds: {case_id}")
        if row["cooccurring_object"] in ("1", "2", "3"):
            if row.get("cooccurrence_subtype") not in SUBTYPES or row.get("cooccurrence_output_scope") not in SCOPES:
                raise ValueError(f"Secondary-object detail required: {case_id}")
        elif row.get("cooccurrence_subtype") or row.get("cooccurrence_output_scope"):
            raise ValueError(f"Secondary-object details conflict with rating: {case_id}")
        complete[case_id] = row
    if len(complete) != len(expected) and not allow_partial:
        raise ValueError(f"Incomplete assignment: {len(complete)}/{len(expected)}; use --allow-partial for a flagged interim summary")
    return complete


def label_summary(values):
    assessed = [int(v) for v in values if v in ("0", "1", "2", "3")]
    positive = sum(v > 0 for v in assessed)
    return {"completed": len(values), "assessable": len(assessed), "positive": positive,
            "positive_fraction": positive / len(assessed) if assessed else None,
            "none": assessed.count(0), "mild": assessed.count(1), "moderate": assessed.count(2),
            "severe": assessed.count(3), "unknown": values.count("U"), "not_applicable": values.count("NA")}


def agreement(pairs):
    comparable = [(int(a), int(b)) for a, b in pairs if a in ("0", "1", "2", "3") and b in ("0", "1", "2", "3")]
    n = len(comparable)
    if not n:
        return {"jointly_assessable": 0, "binary_agreement": None, "binary_cohen_kappa": None,
                "exact_severity_agreement": None}
    observed = sum((a > 0) == (b > 0) for a, b in comparable) / n
    p, q = sum(a > 0 for a, _ in comparable) / n, sum(b > 0 for _, b in comparable) / n
    chance = p * q + (1 - p) * (1 - q)
    return {"jointly_assessable": n, "binary_agreement": observed,
            "binary_cohen_kappa": (observed - chance) / (1 - chance) if chance < 1 else None,
            "exact_severity_agreement": sum(a == b for a, b in comparable) / n}


def summarize(packet, response_path, destination, allow_partial=False):
    if destination.exists():
        raise ValueError("Use a new output directory; prior responses are preserved")
    private = packet / "COORDINATOR_ONLY"
    protocol = load_json(private / "protocol.json")
    mapping = load_json(private / "identity_key.json")["cases"]
    response = load_json(response_path)
    complete = validate_response(response, protocol, mapping, allow_partial)
    annotation_path = Path(protocol["analysis_dir"]) / "review/annotations.csv"
    if sha256(annotation_path) != protocol["prior_review_sha256"]:
        raise ValueError("Prior assistant annotations changed after the protocol was frozen")
    with annotation_path.open(encoding="utf-8-sig", newline="") as stream:
        prior = {r["id"]: r for r in csv.DictReader(stream)}
    by_case = {r["case_id"]: r for r in mapping}
    rows, disagreements = [], []
    for case_id, ratings in complete.items():
        identity = by_case[case_id]
        rows.append({"case_id": case_id, "id": identity["id"], "reviewer": response["reviewer_code"],
                     **{k: ratings[k] for k in FAILURES}, "evidence": ratings["evidence"],
                     "identity_or_mask_confound": ";".join(ratings["confounds"]),
                     "cooccurrence_subtype": ratings.get("cooccurrence_subtype", ""),
                     "cooccurrence_output_scope": ratings.get("cooccurrence_output_scope", ""),
                     "inclusion_probability": identity["inclusion_probability"]})
        for key in FAILURES:
            if ratings[key] != prior[identity["id"]][key]:
                disagreements.append({"case_id": case_id, "id": identity["id"], "criterion": key,
                                      "human": ratings[key], "assistant": prior[identity["id"]][key]})
    summary = {"pack_id": protocol["pack_id"], "reviewer_code": response["reviewer_code"],
               "independence_self_report": {k: response[k] for k in ("involved", "exposed", "disclosure")},
               "independence_not_verified": True, "assigned": len(mapping), "complete": len(complete),
               "missing_or_draft": len(mapping) - len(complete), "interim": len(complete) != len(mapping),
               "population_saved_outputs": protocol["population"],
               "scope": "full saved-output census" if len(mapping) == protocol["population"] else "fixed random sample of saved outputs",
               "labels": {}, "human_assistant_agreement": {},
               "limitations": ["One human rater; no inter-human reliability or consensus estimate.",
                               "U and NA are excluded separately per criterion; blanks/drafts are not negatives.",
                               "No hidden RGB ground truth; ratings measure observed consistency.",
                               "No-output targets remain separate pipeline failures.",
                               "Human-assistant agreement is not independent human-human agreement.",
                               "Interim/nonresponse subsets cannot be treated as the full random sample."]}
    for key in FAILURES:
        summary["labels"][key] = label_summary([r[key] for r in rows])
        summary["human_assistant_agreement"][key] = agreement([(r[key], prior[r["id"]][key]) for r in rows])
    destination.mkdir(parents=True)
    write_json(destination / "summary.json", summary)
    # Preserve the exact submitted bytes as well as parsed/validated outputs.
    (destination / "response_original.json").write_bytes(response_path.read_bytes())
    for filename, data, fields in [
        ("human_annotations.csv", rows, ["case_id", "id", "reviewer", *FAILURES, "evidence", "identity_or_mask_confound", "cooccurrence_subtype", "cooccurrence_output_scope", "inclusion_probability"]),
        ("disagreements.csv", disagreements, ["case_id", "id", "criterion", "human", "assistant"])]:
        with (destination / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    text = ["# Human review results", "", f"Reviewer {response['reviewer_code']}: {len(complete)}/{len(mapping)} assigned cases completed.",
            "", "Interim results; assignment incomplete." if summary["interim"] else "Assignment complete.",
            "", "| Criterion | Positive / assessable | U | NA |", "| --- | --- | --- | --- |"]
    for key, value in summary["labels"].items():
        text.append(f"| {key} | {value['positive']}/{value['assessable']} | {value['unknown']} | {value['not_applicable']} |")
    text.extend(["", "Do not describe these judgments as multi-human consensus. Independence is self-reported and must be checked by the coordinator.",
                 "Keep original human ratings and assistant ratings separate. Disagreements are a later discussion aid; do not retrospectively replace either original set."])
    (destination / "RESULTS.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    write_json(destination / "receipt.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
               "response_sha256": sha256(response_path), "protocol_sha256": sha256(private / "protocol.json"),
               "identity_key_sha256": sha256(private / "identity_key.json"),
               "prior_review_sha256": protocol["prior_review_sha256"]})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    summary = summarize(args.packet_dir, args.response, args.output_dir, args.allow_partial)
    print(json.dumps({k: summary[k] for k in ("assigned", "complete", "missing_or_draft", "interim")}, indent=2))


if __name__ == "__main__":
    main()
