"""Report a versioned Kaggle result only after independent native replay."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from scripts.collect_shape_kaggle import replay, verify_sources
from scripts.verify_shape_v2_metadata import equal


def build(evidence, receipt_path, output):
    if output.exists():
        raise FileExistsError('Preserve previous reports')
    receipt = json.loads(receipt_path.read_bytes())
    if receipt.get('disposition') != 'verified' or receipt.get('state') != 'COMPLETE':
        raise ValueError('Complete verified private Kaggle recovery is required')
    verify_sources(evidence)
    verified = replay(evidence)
    equal({key:receipt[key] for key in verified}, verified, 'fresh local replay')
    training = json.loads((evidence/'training/training.json').read_bytes())
    completion = json.loads((evidence/'completion.json').read_bytes())
    equal(completion['training_receipt_sha256'], verified['training_sha256'], 'completed training')
    equal(completion['selected_checkpoint_sha256'], verified['checkpoint_sha256'], 'completed checkpoint')
    equal(completion['native_result_sha256'], verified['result_sha256'], 'completed native replay')
    rows = []
    for metric, values in verified['geometry'].items():
        lo, hi = values['absolute_ci']
        rows.append(f"| {metric} | {values['v2_sam']:.9f} | {values['constrained_sam']:.9f} | "
                    f"{100*values['absolute_change']:+.4f} | [{100*lo:+.4f}, {100*hi:+.4f}] |")
    interpretation = ('A new checkpoint passed the declared full-IoU retention and hidden-IoU selection rule.'
        if verified['new_checkpoint_eligible'] else
        'Epoch zero won. No newly trained checkpoint passed the rule; do not promote this as an improved method.')
    run = training['runs'][-1]
    elapsed = run['elapsed_seconds']
    gpu_hours = sum(item['elapsed_seconds']*item['world_size']/3600 for item in training['runs'])
    text = f'''# Shape-only Kaggle native validation

All 20 training epochs completed. Selected epoch: **{verified['selected_epoch']}**.
{interpretation}

| Metric | Original v2 with SAM | Selected prior | Change (IoU points) | Paired 95% interval (points) |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Independent replay covers all 1,264 objects / 217 images. The intervals are
nominal image-bootstrap intervals after validation selection. This is iterative
development, not an untouched confirmation. The reserved holdout and full
benchmark were absent from the training package.

Training used two Tesla T4s, global batch64, frozen v2 encoder and encoder
BatchNorm buffers, decoder GroupNorm, AdamW LR1e-5 and 20 epochs. Only shape
supervision was used. The selected native checkpoint replay is exact; local
recovery independently verified every saved mask, object identity, selection
decision and paired statistic. The latest training process took {elapsed/60:.2f}
minutes. Its recorded processes account for {gpu_hours:.4f} GPU-hours; that
excludes provisioning, environment installation, preflight and final native
evaluation and is not a complete allocation or Kaggle quota measurement.

No benchmark adapter admission or new full-benchmark result is established by
this report. The >=3% relative improvement target for CLIP, LPIPS, VGG feature
similarity and SSIM across all 2,565 benchmark targets remains unachieved.
Failures, CLIP coverage, historical unseen reporting and native amodal geometry
remain required. No Surface ranker, RGB correction or combined method is used.

Kernel version: `{receipt['kernel']}`.
Selected checkpoint SHA256: `{verified['checkpoint_sha256']}`.
Training receipt SHA256: `{verified['training_sha256']}`.
Native result SHA256: `{verified['result_sha256']}`.
'''
    output.mkdir(parents=True)
    (output/'REPORT.md').write_text(text, encoding='utf8')
    write_json(output/'receipt.json', dict(at=datetime.now(timezone.utc).isoformat(),
        recovery_sha256=sha256(receipt_path), source_sha256=sha256(Path(__file__)),
        report_sha256=sha256(output/'REPORT.md'), benchmark_success=False))
    return output/'REPORT.md'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(build(args.evidence, args.replay, args.output))


if __name__ == '__main__':
    main()
