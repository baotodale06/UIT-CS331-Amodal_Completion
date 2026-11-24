"""Validate a human response and produce a separate final-report supplement."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import shutil

SCHEMA = "amodal-preference-response-v1"
CRITERIA = ("overall", "shape", "surface")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def validate(response, study, allow_partial=False):
    check(response.get("schema") == SCHEMA, "Wrong response schema")
    check(response.get("study_id") == study["study_id"] and response.get("round") == study["round"], "Wrong study or round")
    reviewer = response.get("reviewer", {})
    check(isinstance(reviewer.get("code"), str) and 0 < len(reviewer["code"].strip()) <= 80, "Missing rater code")
    check(reviewer.get("role") in ("researcher", "independent", "other"), "Missing rater involvement")
    check(reviewer.get("prior_exposure") in ("yes", "no"), "Missing prior-exposure disclosure")
    records = response.get("records")
    check(isinstance(records, list), "Missing records")
    cases = {c["id"] for c in study["cases"]}
    seen, completed = set(), 0
    for row in records:
        case = row.get("case_id")
        check(case in cases and case not in seen, "Unknown or duplicate case")
        seen.add(case)
        check(row.get("status") in ("rated", "skipped", "draft"), "Invalid status")
        check(isinstance(row.get("note"), str) and len(row["note"]) <= 4000, "Invalid note")
        check(row.get("confidence") in ("", "low", "medium", "high"), "Invalid confidence")
        check(isinstance(row.get("skip_reason"), str) and len(row["skip_reason"]) <= 1000, "Invalid skip reason")
        for field in ("created_at", "updated_at"):
            check(isinstance(row.get(field), str), "Missing timestamp")
            datetime.fromisoformat(row[field].replace("Z", "+00:00"))
        check(isinstance(row.get("judgements"), dict), "Missing judgements")
        for criterion in CRITERIA:
            choice = row["judgements"].get(criterion)
            if choice is None:
                check(not (criterion == "overall" and row["status"] == "rated"), "Missing overall judgement")
                continue
            check(isinstance(choice, dict) and choice.get("kind") in ("preference", "unassessable"), "Invalid judgement")
            winners = choice.get("winners")
            check(isinstance(winners, list) and all(isinstance(x, str) and x in study["slots"] for x in winners), "Invalid option")
            check(len(winners) == len(set(winners)), "Duplicate option")
            check(bool(winners) == (choice["kind"] == "preference"), "Invalid winner set")
        if row["status"] == "skipped":
            check(row["skip_reason"].strip() and all(row["judgements"].get(c) is None for c in CRITERIA), "Skip contains ratings or lacks a reason")
        completed += row["status"] != "draft"
    complete = completed == len(cases)
    check(isinstance(response.get("complete"), bool) and response["complete"] == complete, "Incorrect completion flag")
    check(complete or allow_partial, "Incomplete assignment; use --allow-partial for an explicit interim summary")
    return records


def summarize(response, study, key, allow_partial=False):
    records = validate(response, study, allow_partial)
    check(key["study_id"] == study["study_id"], "Coordinator key belongs to a different study")
    mappings = {r["case_id"]: r for r in key["cases"]}
    check(set(mappings) == {c["id"] for c in study["cases"]}, "Key coverage mismatch")
    methods = [m["id"] for m in key["methods"]]
    for mapping in mappings.values():
        check(set(mapping["assignment"]) == set(study["slots"]) and set(mapping["assignment"].values()) == set(methods), "Invalid blinded assignment")
    criteria, decoded = {}, []
    for criterion in CRITERIA:
        counts = {m: {"unique_wins": 0, "tied_top": 0, "fractional_wins": 0.0} for m in methods}
        pairs = {f"{a}__{b}": {"method_a": a, "method_b": b, "a_wins": 0, "b_wins": 0, "ties": 0, "not_compared": 0}
                 for a, b in itertools.combinations(methods, 2)}
        assessed, unknown, unanswered = 0, 0, 0
        for row in records:
            if row["status"] != "rated":
                continue
            choice = row["judgements"].get(criterion)
            if choice is None:
                unanswered += 1
                continue
            if choice["kind"] == "unassessable":
                unknown += 1
                continue
            winners = {mappings[row["case_id"]]["assignment"][slot] for slot in choice["winners"]}
            assessed += 1
            for method in winners:
                counts[method]["fractional_wins"] += 1 / len(winners)
                counts[method]["unique_wins" if len(winners) == 1 else "tied_top"] += 1
            for pair in pairs.values():
                a, b = pair["method_a"], pair["method_b"]
                label = "ties" if a in winners and b in winners else "a_wins" if a in winners else "b_wins" if b in winners else "not_compared"
                pair[label] += 1
            decoded.append({"case_id": row["case_id"], "target_id": mappings[row["case_id"]]["target_id"],
                            "image": mappings[row["case_id"]]["image"], "criterion": criterion, "top_methods": "|".join(sorted(winners))})
        for counts_for_method in counts.values():
            counts_for_method["preference_percent"] = 100 * counts_for_method["fractional_wins"] / assessed if assessed else None
        criteria[criterion] = {"assessable": assessed, "cannot_judge": unknown, "optional_unanswered": unanswered,
                               "methods": counts, "pairwise": list(pairs.values())}
    statuses = Counter(r["status"] for r in records)
    return {"study_id": study["study_id"], "round": study["round"], "complete": response["complete"],
            "reviewer": response["reviewer"], "assigned": len(study["cases"]),
            "rated": statuses["rated"], "skipped": statuses["skipped"],
            "unfinished": len(study["cases"]) - statuses["rated"] - statuses["skipped"],
            "population": study["total_targets"], "excluded_pipeline_targets": study["excluded"],
            "coverage": key["coverage"], "criteria": criteria}, decoded


def run(packet, response_path, destination, allow_partial=False):
    packet, response_path, destination = Path(packet), Path(response_path), Path(destination)
    check(not destination.exists(), "Use a new response output directory")
    study = json.loads((packet / "dist/study.json").read_bytes())
    key = json.loads((packet / "COORDINATOR_ONLY/key.json").read_bytes())
    response = json.loads(response_path.read_bytes())
    summary, decoded = summarize(response, study, key, allow_partial)
    summary["response_sha256"] = hashlib.sha256(response_path.read_bytes()).hexdigest()
    destination.mkdir(parents=True)
    shutil.copyfile(response_path, destination / "original-response.json")
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = ["method", "criterion", "preference_percent", "unique_wins", "tied_top", "fractional_wins", "assessable", "cannot_judge"]
    with (destination / "human-preference.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fields); writer.writeheader()
        for criterion, result in summary["criteria"].items():
            for method, numbers in result["methods"].items():
                writer.writerow({"method": method, "criterion": criterion, **numbers,
                                 "assessable": result["assessable"], "cannot_judge": result["cannot_judge"]})
    with (destination / "per-case-preferences.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, ["case_id", "target_id", "image", "criterion", "top_methods"]); writer.writeheader(); writer.writerows(decoded)
    names = {m["id"]: m["name"] for m in key["methods"]}
    lines = ["# Human preference supplement", "", f"Round `{study['round']}`; single rater `{response['reviewer']['code']}`. "
             f"Involvement: {response['reviewer']['role']}; previous exposure: {response['reviewer']['prior_exposure']}.", "",
             f"{summary['rated']} rated, {summary['skipped']} skipped, {summary['unfinished']} unfinished out of {summary['assigned']} assigned shared-success outputs. "
             f"The source population has {study['total_targets']} targets; {study['excluded']} lack an output from at least one method.", "",
             "| Method | Overall preference (%) | Unique wins | Tied top | Assessable |", "|---|---:|---:|---:|---:|"]
    overall = summary["criteria"]["overall"]
    for method, value in overall["methods"].items():
        percent = f"{value['preference_percent']:.2f}" if value["preference_percent"] is not None else "N/A"
        lines.append(f"| {names[method]} | {percent} | {value['unique_wins']} | {value['tied_top']} | {overall['assessable']} |")
    lines += ["", "Preference shares split one vote equally across the selected top options. Ties, inability to judge, skips, drafts, and pipeline failures are distinct. "
              "With three or more options, the relative order of two nonselected options is unknown; it is never counted as a tie.", "",
              "This is a descriptive single-rater result, conditional on shared successful outputs. It does not establish independent consensus, inter-rater reliability, "
              "hidden RGB accuracy, or an all-target improvement. Shared source images can be dependent. Add this table beside the defined metrics; do not average it into LPIPS, SSIM, CLIP, or IoU.", "",
              "Shape-aware and combined rounds must be reported separately. The two-method round cannot be pooled with a later three-method choice task."]
    if not summary["complete"]:
        lines.insert(2, "**INTERIM: incomplete, potentially selective responses.**\n")
    (destination / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "rated": summary["rated"], "complete": summary["complete"]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet-dir", required=True)
    p.add_argument("--response", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    run(args.packet_dir, args.response, args.output_dir, args.allow_partial)
