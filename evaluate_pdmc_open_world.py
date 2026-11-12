#!/usr/bin/env python3
"""Evaluate released PD-MC with Grounding DINO + SAM and no LISA dependency.

Uses the shared evaluator's released-sam profile, including its validated
dataset, seeds, worker scheduling, result cache and appearance metrics.
"""
import sys

from evaluate_open_world import main as evaluate


def main(argv=None):
    return evaluate(["--target-selection", "label-semantic", "--background-sizing", "auto",
                     *(sys.argv[1:] if argv is None else argv),
                     "--pipeline-profile", "released-sam"])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
