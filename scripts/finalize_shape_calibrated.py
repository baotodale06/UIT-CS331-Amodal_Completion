"""Export the completed calibrated chain without modifying or repeating inference."""
import argparse
import json
from scripts.finalize_shape_v2 import finalize


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--controller-id", required=True)
    args = parser.parse_args()
    if not args.job_id.isdigit() or not args.controller_id.isdigit():
        raise ValueError("Numeric job IDs required")
    print(json.dumps(finalize(args.study, args.job_id, args.controller_id, method="shape-calibrated")), flush=True)
