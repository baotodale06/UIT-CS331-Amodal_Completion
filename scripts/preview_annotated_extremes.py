#!/usr/bin/env python3
"""Recover manifest-verified extreme examples while the full native export downloads."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json
from amodal_benchmark.references import reference_paths
from build_annotated_surface_report import qualitative_examples, thumbnail


def main():
    design = ROOT/'output/surface-annotated-20261112-design'
    study = ROOT/json.loads((design/'active-study.json').read_text())['study']
    recovery = ROOT/'output/surface-annotated-20261112-recovery'
    native = recovery/(study.name+'-cohort-native')
    receipt = json.loads((native/'remote-receipt.json').read_text())
    manifest_path = native/'remote-manifest.json'
    if sha256(manifest_path) != receipt['manifest_sha256']:
        raise ValueError('Native export manifest is not verified')
    manifest = json.loads(manifest_path.read_text())
    pairs = json.loads((recovery/'cluster-audits/cohort/paired_records.json').read_text())
    items = [dict(id=row['id'], key=f'case-{index+1:03d}', transition='shared',
                  delta=row['baseline']['LPIPS']-row['surface']['LPIPS'])
             for index,row in enumerate(pairs) if row['baseline_status'] == row['surface_status'] == 'success']
    selected = sorted((row for row in items if row['delta'] > 1e-12), key=lambda row: (-row['delta'],row['id']))[:2]
    selected += sorted((row for row in items if row['delta'] < -1e-12), key=lambda row: (row['delta'],row['id']))[:2]
    official = {row['id']:row for row in read_official(ROOT/'datasets/open_world_amodal')[0]}
    paths = {}
    for row in selected:
        source = official[row['id']]
        paths[row['id']] = [ROOT/'datasets/open_world_amodal'/source['image'],
            reference_paths(study/'baseline-cohort/visible_references',source)['crop'],
            study/'baseline-cohort/targets'/row['id']/'completion.png',
            study/'surface-cohort/targets'/row['id']/'completion.png']
    def recover(path):
        relative = path.relative_to(ROOT).as_posix()
        expected = manifest[relative]
        if not path.is_file():
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary = path.with_suffix('.preview-partial')
            command = 'cat '+shlex.quote('/home/daoqm/Vu_Hung/cv_nangcao/'+relative)
            with temporary.open('wb') as stream:
                subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
                    'daoqm@slurm.csse.rose-hulman.edu',command],stdout=stream,check=True)
            if sha256(temporary) != expected:
                raise ValueError('Preview native image checksum mismatch: '+relative)
            temporary.replace(path)
        if sha256(path) != expected:
            raise ValueError('Existing native image differs from the export: '+relative)
        return relative, expected
    with ThreadPoolExecutor(max_workers=4) as workers:
        verified = dict(workers.map(recover,[path for group in paths.values() for path in group[1:]]))
    output = design/'qualitative-preview'
    (output/'images').mkdir(parents=True,exist_ok=True)
    for row in selected:
        for position,path in enumerate(paths[row['id']]):
            thumbnail(path,output/'images'/f'{row["key"]}-{position}.jpg')
    qualitative_examples(output,selected)
    write_json(output/'native-recovery-receipt.json',dict(verified=True,files=verified,
        manifest_sha256=sha256(manifest_path),source_sha256=sha256(Path(__file__))))
    print(json.dumps(dict(verified=True,examples=selected,native_files=len(verified))))


if __name__ == '__main__':
    main()
