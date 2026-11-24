"""Freeze and verify the admitted shape-only benchmark and its original references."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json
from amodal_benchmark.shape_calibrated_runtime import METHOD, read, shape_calibrated_contract
from amodal_benchmark.verify import verify_run
from scripts.assess_shape_v2 import table
from scripts.audit_shape_v2 import audit_decisions

HOLDOUT_SHA256 = "da69e45fcbd28f0e798f6ce88906fcd81c250513378262d728a0ed03f26be19a"


def verify_source(study):
    manifest = read(study / "evaluation-source-manifest.json")
    source = (study / "evaluation-source").resolve()
    for name, expected in manifest["files"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or path.name == "AGENTS.md" or sha256(path) != expected:
            raise ValueError("Frozen calibrated source changed: " + name)
    if source != ROOT:
        raise ValueError("Execute from the isolated frozen evaluation source")
    return manifest


def verify_freeze(study, control):
    freeze = read(study / "policy-freeze.json")
    verify_source(study)
    if (sha256(study / "evaluation-source-manifest.json") != freeze["evaluation_source_manifest_sha256"]
            or shape_calibrated_contract(study / "admitted") != freeze["shape_policy"]
            or sha256(control / "per_target.csv") != freeze["control_scores_sha256"]):
        raise ValueError("Frozen calibrated benchmark inputs changed")
    return freeze


def freeze_policy(study, control, holdout):
    destination = study / "policy-freeze.json"
    if destination.exists():
        return verify_freeze(study, control)
    verify_source(study)
    policy = shape_calibrated_contract(study / "admitted")
    success = read(ROOT / "configs/research/shape-v2-slurm-20261123.json")["success"]
    if sha256(control / "per_target.csv") != success["control_scores_sha256"]:
        raise ValueError("Comparator is not the pinned original PMC")
    manifest = read(holdout / "manifest.json")
    if (sha256(holdout / "manifest.json") != HOLDOUT_SHA256
            or len(manifest["rows"]) != 1183 or len({row["image_id"] for row in manifest["rows"]}) != 195
            or {row["split"] for row in manifest["rows"]} != {"holdout"}):
        raise ValueError("Reserved geometry inventory changed")
    report = dict(schema=1, at=datetime.now(timezone.utc).isoformat(), shape_policy=policy,
        scope="Shape-only calibrated prior; no Surface or RGB correction", success=success,
        control_scores_sha256=success["control_scores_sha256"], holdout_manifest_sha256=HOLDOUT_SHA256,
        geometry_source_sha256=sha256(ROOT / "scripts/evaluate_shape_v2_geometry.py"),
        evaluation_source_manifest_sha256=sha256(study / "evaluation-source-manifest.json"),
        benchmark_success=False, validation_reused=True, holdout_previously_observed=True,
        benchmark_previously_observed=True,
        policy_selected_using="Validation geometry floors; preset native logit bias -2; frozen robust epoch 7",
        interpretation="Iterative research; neither reserved geometry nor benchmark is untouched confirmation")
    write_json(destination, report)
    return report


def audit_target_links(run, policy):
    if policy.get("method") != METHOD:
        raise ValueError("Calibrated audit requires its saved policy")
    pools, targets = 0, 0
    for path in sorted((run / "targets").glob("**/result.json")):
        record = read(path)
        decisions = sorted((path.parent / METHOD).glob("iteration-*/decision.json"))
        diagnostics = record.get("diagnostics", {}).get("shape_calibrated_iterations", [])
        if len(decisions) != len(diagnostics):
            raise ValueError("Calibrated decision inventory differs from the closed target")
        if record["status"] == "success" and not decisions:
            raise ValueError("Successful calibrated completion has no shape decisions")
        for decision_path, closed in zip(decisions, diagnostics):
            decision = read(decision_path)
            if (decision.get("method") != METHOD or decision.get("logit_bias") != -2.
                    or decision.get("target_seed") != record.get("seed")
                    or decision_path.parent.name != f'iteration-{decision["iteration"]:02d}'
                    or closed != dict(index=decision["index"], fallback=decision["fallback"],
                                      decision_sha256=sha256(decision_path))):
                raise ValueError("Decision is not bound to its closed target, seed and iteration")
        pools += len(decisions)
        targets += 1
    if not pools:
        raise ValueError("No calibrated shape decisions to verify")
    return dict(targets=targets, pools=pools, verified=True)


def prepare(study, control, dataset):
    freeze = verify_freeze(study, control)
    holdout = read(study / "holdout-native.json")
    if (holdout["freeze_sha256"] != sha256(study / "policy-freeze.json")
            or holdout["shape_policy"] != freeze["shape_policy"]
            or holdout["manifest_sha256"] != HOLDOUT_SHA256
            or holdout["aggregate"]["objects"] != 1183 or holdout["aggregate"]["images"] != 195):
        raise ValueError("Complete frozen reserved geometry report required")
    checks = read(study / "pipeline-validation.json")
    if (checks.get("verified") is not True or checks.get("freeze_sha256") != sha256(study / "policy-freeze.json")
            or checks.get("source_manifest_sha256") != sha256(study / "evaluation-source-manifest.json")):
        raise ValueError("Complete matching CUDA and pipeline checks required")
    for name, expected in checks["logs"].items():
        if Path(name).name != name or sha256(study / name) != expected:
            raise ValueError("Pipeline validation log changed")
    for path in study.glob("preserved-before-resume-*.json"):
        retained = read(path)
        if (retained["freeze_sha256"] != sha256(study / "policy-freeze.json")
                or retained["run_config_sha256"] != sha256(study / "benchmark/run_config.json")):
            raise ValueError("Interrupted benchmark configuration changed")
        for name, expected in retained["files"].items():
            if sha256(study / "benchmark" / name) != expected:
                raise ValueError("Completed benchmark artifact changed before continuation")
    receipt_path = study / "references.json"
    source, destination = control / "visible_references", study / "benchmark/visible_references"
    if not receipt_path.exists():
        verified = verify_run(control, dataset, require_full=True, require_provenance=True)
        inventory = {p.relative_to(source).as_posix(): sha256(p) for p in sorted(source.rglob("*")) if p.is_file()}
        for name, expected in inventory.items():
            target = destination / name
            if target.exists() and sha256(target) != expected:
                raise ValueError("Existing scoring reference differs from original PMC")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / name, target)
        write_json(receipt_path, dict(files=inventory, source_verification=verified))
    for name, expected in read(receipt_path)["files"].items():
        if sha256(source / name) != expected or sha256(destination / name) != expected:
            raise ValueError("Original and calibrated scoring references differ")
    manifest_path = study / "smoke-manifest.json"
    official, _ = read_official(dataset)
    scores, seen, selected = table(control / "per_target.csv"), set(), []
    for row in official:
        if row["source"] not in seen and scores[row["id"]]["status"] == "success":
            seen.add(row["source"])
            selected.append(row["id"])
    if len(selected) != 4:
        raise ValueError("Four-source engineering smoke requires four targets")
    manifest = dict(target_ids=selected,
        selection="First original-PMC success in official order per source; engineering check only")
    if manifest_path.exists() and read(manifest_path) != manifest:
        raise ValueError("Engineering smoke target selection changed")
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    return freeze


def verify_smoke(study, control, dataset):
    freeze = prepare(study, control, dataset)
    ids = read(study / "smoke-manifest.json")["target_ids"]
    run = study / "smoke"
    verified = verify_run(run, dataset, require_provenance=True)
    if verified["selected_targets"] != 4 or set(table(run / "per_target.csv")) != set(ids):
        raise ValueError("Four scored smoke targets required")
    config = read(run / "run_config.json")
    if config.pop("shape_policy") != freeze["shape_policy"] or config != read(control / "run_config.json"):
        raise ValueError("Smoke changes more than the frozen shape-only policy")
    if (run / "metric_config.json").read_bytes() != (control / "metric_config.json").read_bytes():
        raise ValueError("Smoke metric configuration differs from original PMC")
    records = [read(run / "targets" / tid / "result.json") for tid in ids]
    for record in records:
        original = read(control / "targets" / record["id"] / "result.json")
        if record["evaluation_reference_artifacts"] != original["evaluation_reference_artifacts"]:
            raise ValueError("Smoke scoring reference differs from original PMC")
    links = audit_target_links(run, freeze["shape_policy"])
    decisions = audit_decisions(run / "targets", METHOD)
    if decisions["pools"] != links["pools"] or not any(row["status"] == "success" for row in records):
        raise ValueError("Smoke produced no usable shape-selected completion")
    report = dict(verified=True, scored=True, freeze_sha256=sha256(study / "policy-freeze.json"),
        targets=4, statuses={row["id"]: row["status"] for row in records}, selector=decisions,
        links=links, verification=verified, benchmark_success=False)
    write_json(study / "smoke-validation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--holdout-data", type=Path)
    parser.add_argument("--mode", choices=("freeze", "prepare", "verify-smoke"), required=True)
    args = parser.parse_args()
    if args.mode == "freeze":
        if args.holdout_data is None:
            parser.error("--holdout-data is required for freezing")
        report = freeze_policy(args.study, args.control, args.holdout_data)
    elif args.mode == "verify-smoke":
        report = verify_smoke(args.study, args.control, args.dataset_root)
    else:
        report = prepare(args.study, args.control, args.dataset_root)
    print(json.dumps(dict(mode=args.mode, verified=True, benchmark_success=False)), flush=True)


if __name__ == "__main__":
    main()
