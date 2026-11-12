#!/usr/bin/env python3
"""Download the original DreamSim ensemble weights into an isolated cache."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir', type=Path, required=True)
    a = p.parse_args()
    from dreamsim.model import download_weights
    download_weights(str(a.cache_dir), 'ensemble')
    from dreamsim import dreamsim
    model, preprocess = dreamsim(pretrained=True, device='cpu', cache_dir=str(a.cache_dir))
    import torch
    torch.set_num_threads(2)
    with torch.no_grad():
        x = torch.zeros(1, 3, 224, 224)
        value = float(model(x, x))
    assert abs(value) < 1e-5, value
    result = dict(model='DreamSim original ensemble CLS', identity_distance=value,
                  files={str(p.relative_to(a.cache_dir)): sha256(p) for p in a.cache_dir.rglob('*')
                         if p.is_file() and p.name != 'manifest.json'})
    write_json(a.cache_dir / 'manifest.json', result)
    print(json.dumps(dict(identity_distance=value, files=len(result['files']))))


if __name__ == '__main__':
    main()
