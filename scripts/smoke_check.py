#!/usr/bin/env python3
"""Compatibility convenience; install the project with pip install -e . first."""
from amodal_benchmark.cli import main

if __name__ == "__main__":
    import sys
    raise SystemExit(main(["smoke", *sys.argv[1:]]))
