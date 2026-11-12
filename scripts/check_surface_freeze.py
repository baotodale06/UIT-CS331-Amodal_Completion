#!/usr/bin/env python3
"""Verify the deployed frozen implementation without opening evaluation data."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amodal_benchmark.dataset import sha256
from amodal_benchmark.surface_policy import verify_frozen_surface_sources

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('freeze', type=Path)
args = parser.parse_args()
freeze = json.loads(args.freeze.read_text())
verify_frozen_surface_sources(freeze)
print(json.dumps(dict(passed=True, freeze_sha256=sha256(args.freeze),
                      source_files=len(freeze['source_hashes']), evaluation_data_access=False)))
