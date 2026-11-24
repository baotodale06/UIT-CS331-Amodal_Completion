#!/usr/bin/env python3
"""Export small all-target review images only after the complete native audit passes."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tarfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json
from amodal_benchmark.references import reference_paths


def main():
    design = ROOT/'output/surface-annotated-20261112-design'
    study = ROOT/json.loads((design/'active-study.json').read_text())['study']
    report = study/'report'
    audit = json.loads((report/'audit.json').read_text())
    pairs = json.loads((report/'paired_records.json').read_text())
    if not audit['passed'] or audit['targets'] != 204 or len(pairs) != 204:
        raise ValueError('Require a complete successful native cohort audit')
    destination = study/'review-assets'
    if destination.exists():
        raise ValueError('Preserve the existing immutable review export')
    images = destination/'images'
    images.mkdir(parents=True)
    official = {row['id']:row for row in read_official(ROOT/'datasets/open_world_amodal')[0]}
    panels = {}
    for index,row in enumerate(pairs):
        source_row = official[row['id']]
        paths = [ROOT/'datasets/open_world_amodal'/source_row['image'],
                 reference_paths(study/'baseline-cohort/visible_references',source_row)['crop']]
        paths += [study/(side+'-cohort')/'targets'/row['id']/'completion.png'
                  if row[side+'_status']=='success' else None for side in ('baseline','surface')]
        for position,path in enumerate(paths):
            if path is None:
                continue
            name=f'case-{index+1:03d}-{position}.jpg'
            with Image.open(path) as image:
                image=image.convert('RGBA')
                image.thumbnail((340,280),Image.Resampling.LANCZOS)
                canvas=Image.new('RGB',(340,280),(247,248,250))
                canvas.paste(image,((340-image.width)//2,(280-image.height)//2),image)
                canvas.save(images/name,quality=92)
            panels['images/'+name]=dict(source=path.relative_to(ROOT).as_posix(),
                source_sha256=sha256(path),thumbnail_sha256=sha256(images/name))
    write_json(destination/'native-sources.json',dict(cases=len(pairs),panels=panels,
        audit_sha256=sha256(report/'audit.json'),source_sha256=sha256(Path(__file__)),
        scope='Thumbnails derived on the cluster from independently audited native images'))
    # Preserve the audit exactly as produced, apart from its location in this export.
    import shutil
    for name in ('audit.json','paired_records.json','decision_replay.json'):
        shutil.copy2(report/name,destination/name)
    manifest={path.relative_to(ROOT).as_posix():sha256(path)
              for path in sorted(destination.rglob('*')) if path.is_file()}
    archive=design/'review-assets.tar.gz'
    if archive.exists():
        raise ValueError('Preserve the existing review archive')
    with tarfile.open(archive,'w:gz',compresslevel=6) as bundle:
        for relative in manifest:
            bundle.add(ROOT/relative,arcname=relative,recursive=False)
    manifest_path=design/'review-assets-manifest.json'
    write_json(manifest_path,manifest)
    receipt=dict(created_at=datetime.now(timezone.utc).isoformat(),members=len(manifest),
        bytes=archive.stat().st_size,archive=archive.relative_to(ROOT).as_posix(),
        archive_sha256=sha256(archive),manifest=manifest_path.relative_to(ROOT).as_posix(),
        manifest_sha256=sha256(manifest_path),audit_sha256=sha256(report/'audit.json'))
    write_json(design/'review-assets-receipt.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
