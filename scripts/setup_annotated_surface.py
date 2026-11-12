#!/usr/bin/env python3
"""Copy canonical annotated reference bytes into fresh experiment directories."""
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
    historical = ROOT / 'output/pdmc-open-world-sam-annotated'
    design = ROOT / 'output/surface-annotated-20261112-design'
    study = ROOT / 'output/surface-annotated-20261112'
    report = verify_run(historical, ROOT / 'datasets/open_world_amodal', require_full=True)
    write_json(design / 'historical-annotated-verification.json', report)
    source = historical / 'visible_references'
    inventory = {path.relative_to(source).as_posix(): sha256(path)
                 for path in sorted(source.rglob('*')) if path.is_file()}
    for name in ('baseline', 'surface'):
        destination = study / name / 'visible_references'
        if destination.exists():
            raise ValueError('Setup destination already exists; inspect before continuing')
        shutil.copytree(source, destination)
        for relative, expected in inventory.items():
            if sha256(destination / relative) != expected:
                raise ValueError('Reference copy mismatch: ' + name + '/' + relative)
    write_json(design / 'reference-copy-receipt.json', dict(
        completed_at=datetime.now(timezone.utc).isoformat(),
        source=str(source), files=inventory, copies=['baseline', 'surface'], verified=True,
        historical_config_sha256=sha256(historical / 'run_config.json')))
    print(json.dumps(dict(verified=True, reference_files=len(inventory), targets=report['selected_targets'])))


if __name__ == '__main__':
    main()
