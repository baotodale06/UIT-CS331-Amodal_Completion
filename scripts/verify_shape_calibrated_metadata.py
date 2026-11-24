"""Replay the complete calibrated benchmark, native admission and original control."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, write_json
from scripts.verify_shape_v2_metadata import reconcile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("method", "control", "snapshot", "dataset-root", "unseen-inventory", "native-data", "sam-validation", "geometry-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing independent replay")
    freeze = json.loads((args.snapshot / "policy-freeze.json").read_bytes())
    if freeze["shape_policy"]["method"] != "shape-calibrated-probability":
        raise ValueError("This replay requires the calibrated shape-only method")
    official, _ = read_official(args.dataset_root)
    report = reconcile(args.method, args.control, args.snapshot, official, args.unseen_inventory,
                       args.native_data, args.sam_validation, args.geometry_dir)
    write_json(args.output, report)
    print(json.dumps(dict(verified=True, targets=report["targets"], saved=report["saved"], gate=report["gate"])), flush=True)


if __name__ == "__main__":
    main()
