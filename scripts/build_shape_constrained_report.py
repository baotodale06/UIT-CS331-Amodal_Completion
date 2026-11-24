"""Report the completed constrained shape experiment after independent replay."""
import json
from pathlib import Path
import sys
import zipfile
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json


def main():
    study = ROOT/'output/shape-sam-constrained-v2-20261123'
    replay = json.loads((study/'native-replay.json').read_bytes())
    cost = json.loads((study/'allocation-cost.json').read_bytes())
    evidence = study/'recovered'
    training = json.loads((evidence/'training/training.json').read_bytes())
    if (replay.get('verified') is not True or replay.get('benchmark_success') is not False
            or cost.get('final') is not True or training.get('complete') is not True
            or sha256(study/'training/best.pt') != replay['checkpoint_sha256']
            or sha256(evidence/'training/training.json') != replay['training_sha256']
            or sha256(evidence/'native-validation/result.json') != replay['result_sha256']
            or training['new_checkpoint_eligible'] != replay['new_checkpoint_eligible']):
        raise ValueError('Complete verified recovery required')
    for stem in ('training-source', 'native-source'):
        name = stem+'-manifest.json'
        if sha256(evidence/name) != sha256(study/name):
            raise ValueError('Deployed source manifest differs')
        manifest = json.loads((evidence/name).read_bytes())
        if sha256(evidence/(stem+'.zip')) != manifest['zip_sha256']:
            raise ValueError('Recovered source archive differs')
        with zipfile.ZipFile(evidence/(stem+'.zip')) as bundle:
            if set(bundle.namelist()) != set(manifest['files']):
                raise ValueError('Source inventory differs')
            for path, expected in manifest['files'].items():
                if hashlib.sha256(bundle.read(path)).hexdigest() != expected:
                    raise ValueError('Source bytes differ')
    if any(row['state'] != 'COMPLETED' or row['exit_code'] != '0:0' for row in cost['jobs'].values()):
        raise ValueError('Review unsuccessful jobs before reporting')
    output = study/'report'
    if output.exists():
        raise FileExistsError('Preserve previous report')
    rows = []
    for metric, value in replay['geometry'].items():
        lo, hi = value['absolute_ci']
        rows.append(f"| {metric} | {value['v2_sam']:.6f} | {value['constrained_sam']:.6f} | {100*value['absolute_change']:+.3f} | [{100*lo:+.3f}, {100*hi:+.3f}] |")
    eligibility = ('A new checkpoint passed the validation geometry constraint; benchmark improvement remains unassessed.'
        if replay['new_checkpoint_eligible'] else 'No new checkpoint passed the improvement rule. Epoch zero retains the original model; do not promote this as a new method.')
    text = f"""# Constrained SAM adaptation: native validation

All 20 epochs completed. Selected epoch: **{replay['selected_epoch']}**.
{eligibility}

| Metric | Frozen v2 with SAM | Selected constrained prior | Change (IoU points) | Paired 95% interval (points) |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Native replay covers 1,264 validation objects / 217 images; hidden IoU covers
680 objects / 177 images. Intervals are nominal paired image-bootstrap intervals
after validation selection. This is iterative development, not untouched confirmation.
The previously observed partial benchmark is not used for training or selection.

The fixed v2 epoch-47 encoder and its BatchNorm buffers remain frozen. Only the
decoder trains, with AdamW learning rate 0.00001, 20 epochs and global batch 64.
Inputs mix SAM and annotated visible support; only shape supervision is used.
Native inference restores input SAM support. Selection includes epoch zero,
requires full IoU at least the replayed original, then maximizes hidden IoU.
Exact initial and selected-checkpoint replay checks remain mandatory. Native
validation matches the original cuDNN arithmetic; training restores its own setting.

No RGB correction, appearance loss, Surface ranker or combined method is used.
No new holdout or full-benchmark evaluation is claimed. The >=3% goal on CLIP,
LPIPS, feature similarity and SSIM remains open until a complete verified benchmark.

This chain used {cost['total_reserved_gpu_hours']:.6f} allocated GPU-hours.
The earlier failed preflight 45671 and diagnostic 45676 add 0.074167 GPU-hours;
inherited v2 training, SAM mask preparation and prior robustness work are further
separate costs. Failed attempt records remain in `../../shape-sam-constrained-20261123/`.

Checkpoint SHA-256: `{replay['checkpoint_sha256']}`.
See `../native-replay.json`, `../native-export.json`, `../allocation-cost.json`
and the recovered immutable source archives and native masks.
"""
    output.mkdir()
    (output/'REPORT.md').write_text(text, encoding='utf8')
    write_json(output/'receipt.json', dict(native_replay_sha256=sha256(study/'native-replay.json'),
        allocation_sha256=sha256(study/'allocation-cost.json'), report_sha256=sha256(output/'REPORT.md'),
        source_sha256=sha256(Path(__file__)), benchmark_success=False))
    print(str(output/'REPORT.md'))


if __name__ == '__main__':
    main()
