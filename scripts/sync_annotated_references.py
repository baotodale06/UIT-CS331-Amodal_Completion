#!/usr/bin/env python3
"""Copy newly generated control references without replacing any existing bytes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.verify import verify_run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    verification = verify_run(a.source, ROOT / 'datasets/open_world_amodal', require_provenance=True)
    source = a.source / 'visible_references'
    destination = a.destination / 'visible_references'
    manifest = {path.relative_to(source).as_posix(): sha256(path)
                for path in sorted(source.rglob('*')) if path.is_file()}
    # Preflight the entire existing destination before copying any new file.
    for relative, expected in manifest.items():
        target = destination / relative
        if target.exists() and sha256(target) != expected:
            raise ValueError('Existing reference differs: ' + relative)
    copied = 0
    for relative, expected in manifest.items():
        target = destination / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, target)
            copied += 1
        if sha256(target) != expected:
            raise ValueError('Reference transfer differs: ' + relative)
    write_json(a.receipt, dict(verified=True, at=datetime.now(timezone.utc).isoformat(),
        source=str(a.source), destination=str(a.destination), files=manifest,
        new_files=copied, source_targets=verification['selected_targets']))
    print(json.dumps(dict(verified=True, files=len(manifest), copied=copied)))


if __name__ == '__main__':
    main()
