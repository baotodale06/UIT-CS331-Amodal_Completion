"""Apply the user's four-metric 3% gate to full scored tables; never refit."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json

METRICS = {"CLIP": 1, "LPIPS": -1, "Feature Similarity": 1, "SSIM": 1}


def table(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {row["id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate target IDs in score table")
    return indexed


def assess(control, method, expected_ids, threshold=3.):
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Duplicate target IDs in expected cohort")
    if set(control) != set(expected_ids) or set(method) != set(expected_ids):
        raise ValueError("All expected targets must be present in both score tables")
    result = {}
    for metric, direction in METRICS.items():
        values = []
        for arm in (control, method):
            current = {}
            for tid in expected_ids:
                value = arm[tid].get(metric)
                if value in (None, ""):
                    if metric != "CLIP":
                        raise ValueError(f"Missing failure-inclusive metric: {tid}/{metric}")
                    continue
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError("Nonfinite score")
                current[tid] = value
            values.append(current)
        if not all(values):
            raise ValueError(f"No scored targets for {metric}")
        baseline = sum(values[0].values())/len(values[0])
        score = sum(values[1].values())/len(values[1])
        if baseline <= 0:
            raise ValueError("Relative improvement requires a positive baseline")
        gain = direction*(score-baseline)/baseline*100
        shared = values[0].keys() & values[1].keys()
        result[metric] = dict(control=baseline, method=score, relative_gain_percent=gain,
                              control_n=len(values[0]), method_n=len(values[1]), shared_n=len(shared),
                              shared_control=sum(values[0][k] for k in shared)/len(shared) if shared else None,
                              shared_method=sum(values[1][k] for k in shared)/len(shared) if shared else None,
                              pass_point_threshold=gain >= threshold)
    clip_coverage = result["CLIP"]["method_n"] >= result["CLIP"]["control_n"]
    return dict(targets=len(expected_ids), required_relative_gain_percent=threshold, metrics=result,
                clip_coverage_not_reduced=clip_coverage,
                all_four_point_thresholds_pass=all(v["pass_point_threshold"] for v in result.values()),
                eligible_for_success_claim=clip_coverage and all(v["pass_point_threshold"] for v in result.values()),
                caveat="Arithmetic gate only. Full provenance, matching scoring references, verification, registration and paired intervals remain mandatory.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/open_world_amodal/manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research/shape-v2-20261123.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_bytes())
    if sha256(args.control) != config["success"]["control_scores_sha256"]:
        raise ValueError("Control differs from the frozen original PMC score table")
    ids = [row["id"] for row in json.loads(args.manifest.read_bytes())["targets"]]
    if len(ids) != 2565:
        raise ValueError("The frozen full benchmark must contain 2565 targets")
    result = assess(table(args.control), table(args.method), ids)
    result["input_sha256"] = {name:sha256(path) for name,path in (("control",args.control),("method",args.method),("manifest",args.manifest))}
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
