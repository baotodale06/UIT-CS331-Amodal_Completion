#!/usr/bin/env python3
"""Fail before inference if the frozen annotated study's sources or inputs drift."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256
from amodal_benchmark.surface_policy import verify_frozen_surface_sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--design', type=Path, required=True)
    a = p.parse_args()
    freeze = json.loads((a.design / 'study-freeze.json').read_text())
    for relative, expected in freeze['files'].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError('Study source/input drift: ' + relative)
    method = json.loads((a.design / 'method-freeze.json').read_text())
    verify_frozen_surface_sources(method)
    cohort = json.loads((a.design / 'cohort.json').read_text())
    for case in cohort['cases']:
        if sha256(ROOT / 'datasets/open_world_amodal' / case['source_image']) != case['source_sha256']:
            raise ValueError('Cohort source image changed: ' + case['image_identity'])
    print('PASS: frozen annotated study inputs and implementation')


if __name__ == '__main__':
    main()
