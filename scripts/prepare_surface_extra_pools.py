#!/usr/bin/env python3
"""Prepare additional common seeds with byte-identical observation inputs."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json


def main():
    design = ROOT / 'output/surface-followup-20261110-design'
    for cohort, name in ((96, 'surface-consistency-20261108'),
                         (87, 'surface-consistency-replication87-20261108'),
                         (166, 'surface-followup-20261110')):
        source = ROOT / 'output' / name
        destination = design / f'extra-pools-{cohort}'
        if destination.exists():
            raise ValueError('Refuse to overwrite an existing extra-seed cohort')
        destination.mkdir()
        protocol = json.loads((source / 'protocol.json').read_text())
        protocol['candidate_seeds'] = [55, 66, 77, 88]
        protocol['additional_seed_ancestry'] = dict(source_experiment=str(source),
            protocol_sha256=sha256(source / 'protocol.json'),
            observation_manifest_sha256=sha256(source / 'generation_manifest.json'))
        write_json(destination / 'protocol.json', protocol)
        shutil.copyfile(source / 'generation_manifest.json', destination / 'generation_manifest.json')
        (destination / 'inputs').symlink_to(source / 'inputs', target_is_directory=True)
        print(cohort, destination, flush=True)
    provisional = json.loads((design / 'provisional-development-v1.json').read_text())
    provisional['candidate_seeds'] = [11,22,33,44,55,66,77,88]
    provisional['status'] = 'provisional_development_review'
    provisional['pretrained_visible_features'] = False
    provisional['note'] = 'Eight-candidate development review; not a final model and never authorizes test metrics.'
    write_json(design / 'provisional-development-eight.json', provisional)


if __name__ == '__main__':
    main()
