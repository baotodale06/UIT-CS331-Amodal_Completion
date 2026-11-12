#!/usr/bin/env python3
"""Audit and compare the same observation/seed smoke across GPU runtimes."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--previous-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows = []
    for path in sorted((a.run_dir / 'candidates').glob('*/*.json')):
        r = json.loads(path.read_text())
        assert r['status'] == 'generated', r
        assert r.get('surface', {}).get('valid'), r
        for name, digest in r['artifacts'].items():
            assert sha256(path.parent / name) == digest
        name = f"seed-{r['seed']}.png"
        previous = a.previous_dir / 'candidates' / r['case_id'] / name
        first = np.asarray(Image.open(path.parent / name).convert('RGB'))
        second = np.asarray(Image.open(previous).convert('RGB'))
        rows.append(dict(case_id=r['case_id'], seed=r['seed'], seconds=r['total_seconds'],
                         identical_rgb=bool(np.array_equal(first, second)),
                         cross_runtime_rgb_mae=float(np.abs(first.astype(float)-second).mean()/255)))
    assert len(rows) == 8, len(rows)
    write_json(a.output, dict(status='pass', candidates=rows,
        mean_seconds=float(np.mean([r['seconds'] for r in rows])),
        note='Cross-runtime pixel changes are disclosed; fresh policies share identical generated pools.'))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
