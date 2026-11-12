#!/usr/bin/env python3
"""Read-only source/checkpoint identity capture for the surface experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    project = args.project_dir.resolve()
    cache, repo = project / '.amodal-cache', project / '.amodal-cache/amodal'
    model_paths = [repo / relative for relative in (
        'Grounded-Segment-Anything/groundingdino_swint_ogc.pth',
        'Grounded-Segment-Anything/sam_vit_h_4b8939.pth', 'lama/big-lama/models/best.ckpt',
        'lama/big-lama/config.yaml', 'images/gray_wallpaper.jpeg')]
    hub = cache / 'huggingface/hub'
    if not hub.exists():
        raise ValueError('Expected shared HF hub cache')
    for model in ('models--sd2-community--stable-diffusion-2-inpainting', 'models--bert-base-uncased'):
        directory = hub / model / 'snapshots'
        if not directory.exists():
            # Transformers' legacy cache may own the actual snapshot directory.
            directory = cache / 'huggingface/transformers' / model / 'snapshots'
        if not directory.exists():
            raise ValueError('Model cache not found: ' + model)
        for snapshot in directory.iterdir():
            if model.startswith('models--sd2') and snapshot.name != '5f74973cbb64c8568780732c17f43eb269d63a0d':
                continue
            model_paths.extend(path for path in snapshot.rglob('*') if path.is_file())
    files = []
    for path in sorted(set(model_paths)):
        files.append(dict(path=str(path.relative_to(project)), bytes=path.stat().st_size, sha256=digest(path)))
        print('HASHED', path.name, path.stat().st_size, flush=True)
    sources = [path for path in repo.rglob('*.py') if '.git' not in path.parts and '__pycache__' not in path.parts]
    source_hashes = {str(path.relative_to(repo)): digest(path) for path in sorted(sources)}
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), node=os.uname().nodename,
        job_id=os.environ.get('SLURM_JOB_ID'), model_files=files, source_files=source_hashes,
        upstream_commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        upstream_status=subprocess.check_output(['git', '-C', str(repo), 'status', '--short'], text=True),
        source_sha256=digest(Path(__file__)), purpose='Actual cached checkpoint and runtime source hashes; read-only capture')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('COMPLETE', len(files), 'checkpoint/config files,', len(sources), 'Python source files')


if __name__ == '__main__':
    main()
