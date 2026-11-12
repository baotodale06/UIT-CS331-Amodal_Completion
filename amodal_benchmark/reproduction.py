"""CPU reports for the published open-world PD-MC baseline (Tables 3 and 5).

Published values are references, never scores measured by this module. The
authors' CLIP failure/aggregation convention and baseline target adapter are
unspecified, so numerical deltas alone do not establish protocol equivalence.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from .dataset import SOURCES
from .protocol import METRICS, aggregate

PAPER_URL = "https://openaccess.thecvf.com/content/CVPR2025/papers/Ao_Open-World_Amodal_Appearance_Completion_CVPR_2025_paper.pdf"
# CLIP, LPIPS, VGG similarity, SSIM, complete-failure proportion.
PAPER_PDMC = {
    "VG": (28.367, .578, .413, .463, .400),
    "COCOA": (27.383, .664, .328, .382, .485),
    "free_images": (28.333, .720, .279, .309, .582),
    "laion": (27.573, .692, .299, .346, .542),
    "Overall": (27.984, .628, .364, .413, .455),
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def data_identity(rows, audit):
    """Fingerprint inputs/references, excluding generated completion pixels."""
    ordered = sorted(rows, key=lambda r: r["id"])
    references = audit.get("reference_records", {})
    return {
        "targets_sha256": _digest([(r["id"], r.get("source"), r.get("label")) for r in ordered]),
        "inputs_sha256": (_digest([(r["id"], r["input_sha256"]) for r in ordered])
                          if ordered and all(r.get("input_sha256") for r in ordered) else None),
        "references_sha256": (_digest([(r["id"], references[r["id"]]["artifacts"]) for r in ordered])
                              if ordered and all(references.get(r["id"], {}).get("artifacts")
                                                 for r in ordered) else None),
    }


def paper_comparison(summary):
    sources = {}
    for source, stat in summary["statistics"].items():
        if source not in PAPER_PDMC:
            raise ValueError(f"Unknown benchmark source: {source}")
        expected = 2565 if source == "Overall" else SOURCES[source][2]
        complete = stat["targets"] == expected
        metrics = {}
        for index, metric in enumerate(METRICS):
            observed = stat.get(metric) if summary.get("evaluation_performed") else None
            count = stat.get(metric + "_n", 0) if observed is not None else 0
            if observed is not None and not math.isfinite(observed):
                raise ValueError(f"Non-finite {metric} in {source}")
            if metric == "CLIP":
                status = "paper_denominator_unknown"
            elif not summary.get("evaluation_performed"):
                status = "not_scored"
            elif summary.get("reference_backend", "lisa") != "lisa":
                status = "different_reference_backend"
            elif summary.get("metric_alignment") != "released":
                status = "different_alignment"
            elif not complete:
                status = "subset"
            elif count != expected:
                status = "excluded_targets"
            else:
                status = "diagnostic_comparison"
            metrics["CLIP_success" if metric == "CLIP" else metric] = {
                "paper": PAPER_PDMC[source][index], "observed": observed,
                "delta": observed - PAPER_PDMC[source][index] if observed is not None else None,
                "n": count, "comparison_status": status,
            }
        evaluated = stat["successes"] + stat["failures"]
        failure = stat["failures"] / evaluated if evaluated else None
        sources[source] = {
            "paper_targets": expected, "selected_targets": stat["targets"],
            "complete_source": complete, "reference_failures": stat.get("reference_failures", 0),
            "missing_images": stat.get("missing_images", 0), "metrics": metrics,
            "failure_rate": {"paper": PAPER_PDMC[source][4], "observed": failure,
                             "delta_pp": 100 * (failure - PAPER_PDMC[source][4]) if failure is not None else None,
                             "n": evaluated},
            "failure_reasons": stat.get("failure_reasons", {}),
        }
    return {"schema": 1, "reference": {"paper": PAPER_URL, "metric_table": 3, "failure_table": 5},
            "exact_table_reproduction": False, "sources": sources,
            "notes": ["Deltas are observed minus published, not improvements in reproduction fidelity.",
                      "CLIP_success uses successful completions; the paper's CLIP denominator is unknown.",
                      "Failure-rate deltas use percentage points; missing images and reference failures are excluded.",
                      "Visible references are not amodal ground truth; local target matching remains an adapter assumption."]}


def load_run(folder):
    """Read a saved run and reject stale summaries or duplicate target results."""
    folder = Path(folder).resolve()
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
    metric_path = folder / "metric_config.json"
    metric_config = json.loads(metric_path.read_text(encoding="utf-8")) if metric_path.exists() else None
    with (folder / "per_target.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        columns = reader.fieldnames or []
    identifiers = [r["id"] for r in rows]
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Duplicate or empty target ID in {folder / 'per_target.csv'}")
    for row in rows:
        if row["source"] not in SOURCES:
            raise ValueError(f"Unknown source in {folder / 'per_target.csv'}")
        for metric in METRICS:
            value = row.get(metric)
            row[metric] = float(value) if value not in (None, "") else None
            if row[metric] is not None and not math.isfinite(row[metric]):
                raise ValueError(f"Non-finite {metric} for {row['id']}")
    optional = ("completed_outputs", "iteration_limited_outputs", "completion_state_unknown",
                "target_selection_failures", "initial_segmentation_failures", "pdmc_failures", "failure_stage_counts")
    detailed = ("iteration_limit_reached" in columns or
                any(key in stat for stat in summary["statistics"].values() for key in optional))
    computed = aggregate(rows, detailed=detailed)
    keys = ("targets", "successes", "failures", "missing_images", "reference_failures",
            *METRICS, *(m + "_n" for m in METRICS))
    if summary.get("selected_targets") != len(rows) or set(summary["statistics"]) != set(computed):
        raise ValueError(f"Saved summary does not match per_target.csv: {folder}")
    for source, stat in computed.items():
        checked = (*keys, *(key for key in optional if key in summary["statistics"][source]))
        for key in checked:
            actual, saved = stat[key], summary["statistics"][source].get(key)
            if actual != saved and not (isinstance(actual, (int, float)) and
                                       isinstance(saved, (int, float)) and
                                       math.isclose(actual, saved, rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError(f"Saved summary differs from per_target.csv for {source}/{key}: {folder}")
    # Recompute older reports' missing failure-rate/reason fields from the CSV.
    summary["statistics"] = computed
    return {"run_dir": str(folder), "run_config": config, "metric_config": metric_config,
            "summary": summary, "target_ids_sha256": _digest(sorted(identifiers))}


def seed_statistics(runs):
    """Group only identical protocols and verified input/reference identities."""
    grouped = {}
    for index, run in enumerate(runs):
        config = {k: v for k, v in run["run_config"].items() if k != "seed"}
        identity = run["summary"].get("data_identity", {})
        verified = all(identity.get(k) for k in ("targets_sha256", "inputs_sha256", "references_sha256"))
        verified = bool(verified and run.get("metric_config") and run["summary"].get("evaluation_performed"))
        key = _digest({"config": config, "metric_config": run.get("metric_config"), "identity": identity,
                       "target_ids": run.get("target_ids_sha256"),
                       "unverified_run": index if not verified else None})
        group = grouped.setdefault(key, {"seeds": [], "runs": [], "members": [],
                                         "verified_data_identity": verified})
        seed = run["run_config"].get("seed")
        if seed in group["seeds"]:
            raise ValueError(f"Repeated seed {seed} for the same protocol; these are not independent trials")
        group["seeds"].append(seed)
        group["runs"].append(run["run_dir"])
        group["members"].append(run)
    result = []
    for group in grouped.values():
        members = group.pop("members")
        stats = {}
        for source in members[0]["summary"]["statistics"]:
            stats[source] = {}
            for metric in (*METRICS, "failure_rate"):
                values, denominators = [], []
                for member in members:
                    stat = member["summary"]["statistics"][source]
                    if metric == "failure_rate":
                        denominator = stat["successes"] + stat["failures"]
                        value = stat["failures"] / denominator if denominator else None
                    else:
                        denominator, value = stat.get(metric + "_n", 0), stat.get(metric)
                    if value is not None:
                        values.append(value)
                        denominators.append(denominator)
                stats[source]["CLIP_success" if metric == "CLIP" else metric] = {
                    "mean": statistics.mean(values) if values else None,
                    "std": statistics.stdev(values) if len(values) > 1 else None,
                    "n": len(values), "denominators": denominators}
        group["statistics"] = stats
        result.append(group)
    return result


def render_comparison(report):
    lines = ["# PD-MC reproduction comparison", "",
             f"Published reference: [Open-World Amodal Appearance Completion, Tables 3 and 5]({PAPER_URL}).", "",
             "CLIP_success is conditional on successful completion; its paper denominator is unknown.",
             "Delta = observed - paper. Failure deltas are percentage points (pp).", "",
             "Numerical agreement does not certify exact reproduction. Inspect source coverage, excluded targets, and protocol.", ""]
    for run in report["runs"]:
        lines += [f"## {Path(run['run_dir']).name}", ""]
        config = run.get("run_config", {})
        lines += [f"Boundary: {config.get('boundary_check', 'released')}; "
                  f"matching: {config.get('target_selection', 'unknown')}; seed: {config.get('seed', 'unknown')}.", "",
                  "| Source | Metric | Paper | Observed | Delta | n | Status |",
                  "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
        for source, entry in run["comparison"]["sources"].items():
            for metric, value in entry["metrics"].items():
                observed = "unavailable" if value["observed"] is None else f"{value['observed']:.4f}"
                delta = "unavailable" if value["delta"] is None else f"{value['delta']:+.4f}"
                lines.append(f"| {source} | {metric} | {value['paper']:.4f} | {observed} | {delta} | "
                             f"{value['n']} | {value['comparison_status']} |")
            failure = entry["failure_rate"]
            observed = "unavailable" if failure["observed"] is None else f"{100 * failure['observed']:.2f}%"
            delta = "unavailable" if failure["delta_pp"] is None else f"{failure['delta_pp']:+.2f} pp"
            status = "full source" if entry["complete_source"] else "subset"
            if entry["missing_images"] or entry["reference_failures"]:
                status += f"; missing={entry['missing_images']}, reference failures={entry['reference_failures']}"
            lines.append(f"| {source} | Failure rate | {100 * failure['paper']:.1f}% | {observed} | {delta} | "
                         f"{failure['n']} | {status} |")
        lines.append("")
    for index, group in enumerate(report.get("seed_groups", []), 1):
        lines += [f"## Seed group {index}", "", f"Seeds: {group['seeds']}. Runs: {', '.join(group['runs'])}.", ""]
        if not group["verified_data_identity"]:
            lines += ["Input/reference identities are unavailable; this run is not pooled with other seeds.", ""]
        lines += ["Sample standard deviation (ddof=1); one trial has no estimated standard deviation.", "",
                  "| Source | Metric | Mean | Sample std | Seeds with values | Instance denominators |",
                  "| --- | --- | ---: | ---: | ---: | --- |"]
        for source, metrics in group["statistics"].items():
            for metric, value in metrics.items():
                mean = "unavailable" if value["mean"] is None else f"{value['mean']:.4f}"
                std = "unavailable" if value["std"] is None else f"{value['std']:.4f}"
                lines.append(f"| {source} | {metric} | {mean} | {std} | {value['n']} | {value['denominators']} |")
        lines.append("")
    return "\n".join(lines) + "\n"
