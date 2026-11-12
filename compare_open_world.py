#!/usr/bin/env python3
"""Compare saved PD-MC runs with the paper, without loading models or changing runs."""
import argparse
from pathlib import Path
import sys

from amodal_benchmark.dataset import write_json
from amodal_benchmark.reproduction import load_run, paper_comparison, render_comparison, seed_statistics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True,
                        help="Repeat for each saved run/seed; summary, config and per_target.csv are required")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = [p.resolve() for p in args.run_dir]
    if len(set(paths)) != len(paths):
        parser.error("--run-dir must list distinct runs")
    output = args.output_dir.resolve()
    if output in paths:
        parser.error("Use a separate --output-dir to preserve the saved run")
    runs = [load_run(path) for path in paths]
    report = {"schema": 1, "runs": [{**run, "comparison": paper_comparison(run["summary"])} for run in runs],
              "seed_groups": seed_statistics(runs)}
    write_json(output / "comparison.json", report)
    (output / "comparison.md").write_text(render_comparison(report), encoding="utf-8")
    print(f"Compared {len(runs)} saved run(s): {output / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
