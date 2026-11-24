"""Verify a frozen policy and copy the exact original scoring references once."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json
from amodal_benchmark.shape_v2_runtime import shape_v2_contract
from amodal_benchmark.verify import verify_run
from scripts.assess_shape_v2 import table
from scripts.audit_shape_v2 import audit_decisions
from scripts.continue_shape_v2 import verify_frozen_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--verify-smoke", action="store_true")
    args = parser.parse_args()
    freeze = json.loads((args.study/"policy-freeze.json").read_bytes())
    verify_frozen_source(args.study)
    if freeze["shape_policy"] != shape_v2_contract(args.study/"training/best.pt"):
        raise ValueError("The selected model or inference policy changed after freezing")
    holdout = json.loads((args.study/"holdout-native.json").read_bytes())
    if (holdout["freeze_sha256"] != sha256(args.study/"policy-freeze.json")
            or holdout["shape_policy"] != freeze["shape_policy"]):
        raise ValueError("Reserved geometry evaluation does not match this frozen policy")
    if sha256(args.control/"per_target.csv") != freeze["control_scores_sha256"]:
        raise ValueError("Original PMC score table changed")
    for path in args.study.glob("preserved-before-resume-*.json"):
        previous = json.loads(path.read_bytes())
        if (previous["freeze_sha256"] != sha256(args.study/"policy-freeze.json")
                or previous["run_config_sha256"] != sha256(args.study/"benchmark/run_config.json")):
            raise ValueError("Interrupted benchmark configuration changed")
        for name, expected in previous["files"].items():
            if sha256(args.study/"benchmark"/name) != expected:
                raise ValueError("Previously completed result changed before continuation")
    receipt_path = args.study/"references.json"
    source, destination = args.control/"visible_references", args.study/"benchmark/visible_references"
    if not receipt_path.exists():
        verified = verify_run(args.control, args.dataset_root, require_full=True, require_provenance=True)
        inventory = {p.relative_to(source).as_posix():sha256(p) for p in sorted(source.rglob("*")) if p.is_file()}
        for name, expected in inventory.items():
            target = destination/name
            if target.exists() and sha256(target) != expected:
                raise ValueError("Existing scoring reference differs from original PMC")
        for name in inventory:
            target = destination/name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source/name, target)
        write_json(receipt_path, dict(files=inventory, source_verification=verified))
    receipt = json.loads(receipt_path.read_bytes())
    for name, expected in receipt["files"].items():
        if sha256(source/name) != expected or sha256(destination/name) != expected:
            raise ValueError("Original and method scoring-reference bytes differ")
    manifest_path = args.study/"smoke-manifest.json"
    if not manifest_path.exists():
        official, _ = read_official(args.dataset_root)
        scores = table(args.control/"per_target.csv")
        seen, selected = set(), []
        for row in official:
            if row["source"] not in seen and scores[row["id"]]["status"] == "success":
                seen.add(row["source"])
                selected.append(row["id"])
        if len(selected) != 4:
            raise ValueError("Cannot construct the four-source engineering smoke")
        write_json(manifest_path, dict(target_ids=selected,
            selection="First historical successful target in official order per source; engineering check, no model selection"))
    if args.verify_smoke:
        ids = json.loads(manifest_path.read_bytes())["target_ids"]
        records = [json.loads((args.study/"smoke/targets"/tid/"result.json").read_bytes()) for tid in ids]
        decisions = audit_decisions(args.study/"smoke/targets")
        if not decisions["pools"] or not any(r["status"] == "success" for r in records):
            raise ValueError("Smoke produced no usable shape-selected completion")
        for record in records:
            original = json.loads((args.control/"targets"/record["id"]/"result.json").read_bytes())
            if record["evaluation_reference_artifacts"] != original["evaluation_reference_artifacts"]:
                raise ValueError("Smoke references differ from original PMC")
        write_json(args.study/"smoke-validation.json", dict(verified=True,
            freeze_sha256=sha256(args.study/"policy-freeze.json"), targets=len(records),
            successful_outputs=sum(r["status"] == "success" for r in records),
            decisions=decisions, statuses={r["id"]:r["status"] for r in records}, benchmark_success=False))
    print("Frozen model, geometry holdout and original scoring references verified", flush=True)


if __name__ == "__main__":
    main()
