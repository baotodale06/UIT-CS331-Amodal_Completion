"""Freeze the completed shape-only model before holdout or benchmark evaluation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.shape_v2_runtime import shape_v2_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("An existing policy freeze cannot be overwritten")
    policy = shape_v2_contract(args.checkpoint)
    validation = json.loads(args.validation.read_bytes())
    config = json.loads((ROOT / "configs/research/shape-v2-slurm-20261123.json").read_bytes())
    if validation.get("split") != "validation" or validation.get("shape_policy") != policy:
        raise ValueError("Full-image development geometry must use the selected model")
    if sha256(args.control / "per_target.csv") != config["success"]["control_scores_sha256"]:
        raise ValueError("The comparator is not the pinned original PMC control")
    holdout = json.loads(args.holdout_manifest.read_bytes())
    if (sha256(args.holdout_manifest) != "da69e45fcbd28f0e798f6ce88906fcd81c250513378262d728a0ed03f26be19a"
            or set(row["split"] for row in holdout["rows"]) != {"holdout"}
            or len(holdout["rows"]) != 1183 or len({r["image_id"] for r in holdout["rows"]}) != 195):
        raise ValueError("The reserved natural-amodal geometry holdout changed")
    write_json(args.output, dict(schema=1, at=datetime.now(timezone.utc).isoformat(),
        scope="shape-only; no Surface or RGB correction", shape_policy=policy,
        validation_sha256=sha256(args.validation), holdout_manifest_sha256=sha256(args.holdout_manifest),
        geometry_source_sha256=sha256(ROOT / "scripts/evaluate_shape_v2_geometry.py"),
        evaluation_source_manifest_sha256=sha256(args.source_manifest),
        control_scores_sha256=sha256(args.control / "per_target.csv"), success=config["success"],
        policy_selected_using="Original development contract; checkpoint selected using training validation only",
        no_holdout_or_benchmark_selection=True))
    print("Frozen completed shape-only policy:", sha256(args.output), flush=True)


if __name__ == "__main__":
    main()
