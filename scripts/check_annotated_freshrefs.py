#!/usr/bin/env python3
"""Verify the documented fresh-reference amendment before each stage."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256

design = ROOT / 'output/surface-annotated-20261112-design'
amendment = json.loads((design / 'fresh-reference-amendment.json').read_text())
for relative, expected in amendment['files'].items():
    if sha256(ROOT / relative) != expected:
        raise ValueError('Fresh-reference amendment drift: ' + relative)
print('PASS: fresh-reference amendment, unchanged cohort and trained method')
