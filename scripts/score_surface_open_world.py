#!/usr/bin/env python3
"""Score derived and baseline artifacts on the same GPU and metric protocol."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json
from amodal_benchmark.metrics import MetricModels
from amodal_benchmark.references import reference_paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--worker-index', type=int, default=0)
    p.add_argument('--worker-count', type=int, default=4)
    a = p.parse_args()
    import torch
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rows, _ = read_official(a.dataset_root)
    rows = rows[a.worker_index::a.worker_count]
    baseline, run = a.baseline.resolve(), a.run_dir.resolve()
    metric_config = json.loads((baseline / 'metric_config.json').read_text())
    runtime = dict(started_at=datetime.now(timezone.utc).isoformat(), job_id=os.environ.get('SLURM_JOB_ID'),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(0), worker=a.worker_index,
        metric_config=metric_config, run_config_sha256=sha256(run / 'run_config.json'),
        source_sha256=sha256(Path(__file__)), metric_source_sha256=sha256(ROOT / 'amodal_benchmark/metrics.py'))
    write_json(run / 'paired_scoring' / f'runtime-worker-{a.worker_index}.json', runtime)
    models = MetricModels()
    for row in rows:
        folder, old_folder = run / 'targets' / row['id'], baseline / 'targets' / row['id']
        record, old = [json.loads((d / 'result.json').read_text()) for d in (folder, old_folder)]
        if record['status'] != old['status']:
            raise ValueError('Postprocessing changed success/failure status')
        if record['status'] != 'success':
            continue
        paths = reference_paths(run / 'visible_references', row)
        reference = paths['crop'] if metric_config['alignment'] == 'released' else paths['full']
        paired = run / 'paired_scoring' / row['id'] / 'metrics.json'
        expected = dict(reference_sha256=sha256(reference), completion_sha256=record['completion_sha256'],
                        baseline_completion_sha256=old['completion_sha256'])
        if paired.exists():
            if json.loads(paired.read_text())['inputs'] != expected:
                raise ValueError('Changed artifacts on metric resume')
            continue
        for d, r in ((folder, record), (old_folder, old)):
            if sha256(d / 'completion.png') != r['completion_sha256']:
                raise ValueError('Completion checksum mismatch')
        first = models.score(reference, old_folder / 'completion.png', row['label'], old['geometry'], metric_config['alignment'])
        second = models.score(reference, folder / 'completion.png', row['label'], record['geometry'], metric_config['alignment'])
        write_json(folder / 'metrics.json', dict(inputs={k: expected[k] for k in
                   ('reference_sha256', 'completion_sha256')}, metrics=second))
        write_json(paired, dict(id=row['id'], inputs=expected, baseline=first, surface=second))
        print('PAIRED', row['id'], first['LPIPS'], second['LPIPS'], flush=True)
    runtime['finished_at'] = datetime.now(timezone.utc).isoformat()
    write_json(run / 'paired_scoring' / f'runtime-worker-{a.worker_index}.json', runtime)


if __name__ == '__main__':
    main()
