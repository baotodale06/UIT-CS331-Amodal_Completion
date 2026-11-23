#!/usr/bin/env python3
"""Tải source/weights COCOA chính thức; ghi SHA256 để lưu provenance trên server."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = '817d656281b92f7f3d457b0937501b5ad162ffcc'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'third_party/c2f-seg')
    parser.add_argument('--destination', type=Path, default=ROOT/'checkpoints/c2f')
    parser.add_argument('--code-only', action='store_true')
    args = parser.parse_args()
    if not args.source.exists():
        args.source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', 'https://github.com/amazon-science/c2f-seg.git', str(args.source)], check=True)
    current = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    if current != REVISION:
        # Không ghi đè source đã sửa trên server.
        dirty = subprocess.check_output(['git', '-C', str(args.source), 'status', '--porcelain'], text=True)
        if dirty:
            raise RuntimeError('Source C2F có thay đổi; hãy dùng --source thư mục mới')
        subprocess.run(['git', '-C', str(args.source), 'checkout', REVISION], check=True)
    if args.code_only:
        return
    args.destination.mkdir(parents=True, exist_ok=True)
    manifest = {'repository': 'https://github.com/amazon-science/c2f-seg', 'revision': REVISION, 'files': {}}
    for filename in ('c2f_seg_COCOA.pth', 'vqgan_COCOA.pth', 'c2f_seg_COCOA.yml', 'vqgan_COCOA.yml'):
        url = 'https://data.dgl.ai/dataset/C2F-Seg/'+filename
        path = args.destination/filename
        if not path.exists():
            print('Đang tải', url, flush=True)
            temporary = path.with_suffix(path.suffix+'.part')
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as output:
                while chunk := response.read(1024*1024):
                    output.write(chunk)
            temporary.replace(path)
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b''):
                digest.update(chunk)
        manifest['files'][filename] = {'url': url, 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}
        print(filename, manifest['files'][filename], flush=True)
    (args.destination/'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
