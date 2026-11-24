"""Build a native validation report after fresh independent local mask replay."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from amodal_benchmark.dataset import sha256,write_json
from scripts.verify_shape_local import STUDY,replay


def build(output):
    if output.exists():
        raise FileExistsError('Preserve earlier local result reports')
    evidence = replay(STUDY/'run-v2')
    if evidence!=json.loads((STUDY/'native-replay.json').read_bytes()):
        raise ValueError('Independent native replay changed')
    training = json.loads((STUDY/'run-v2/training/training.json').read_bytes())
    lines = []
    for comparison,key in (('Original Quadro','geometry_vs_quadro'),('Original local GPU','geometry_vs_local')):
        for metric,values in evidence[key].items():
            low,high = values['absolute_ci']
            lines.append(f"| {comparison} | {metric} | {values['annotated_input']:.9f} | {values['sam_input']:.9f} | "
                f"{100*values['sam_minus_annotated']:+.4f} | [{100*low:+.4f}, {100*high:+.4f}] |")
    interpretation = ('A newly trained checkpoint passed both original Quadro and local full/hidden IoU floors.'
        if evidence['new_checkpoint_eligible'] else
        'Epoch zero won. No newly trained checkpoint is eligible; do not promote this run as an improved model.')
    elapsed = sum(row['elapsed_seconds'] for row in training['runs'])
    text = f'''# Shape-only local native validation

All 20 epochs completed. Selected epoch: **{evidence['selected_epoch']}**.
{interpretation}

| Reference | Metric | Reference prior | Selected prior | Change (IoU points) | Paired 95% interval (points) |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(lines)}

Independent replay covers all 1,264 validation objects / 217 images, native
binary masks, input hashes, checkpoint selection and image-bootstrap statistics.
The intervals are nominal after validation selection. This is iterative research,
not an untouched confirmation or a full-benchmark result.

The initial RTX3060 prediction differed from original Quadro masks in
{evidence['original_quadro_different_pixels']} pixels across
{evidence['original_quadro_different_mask_objects']} objects. Initial and selected
local checkpoints each passed exact same-device native pixel replay. Both original
Quadro and local geometry floors remained in force throughout selection.

Training used one 6GB RTX3060 Laptop GPU in isolated WSL, exact cluster conda
packages, PyTorch1.13.1/CUDA11.7, FP16 training, and FP32 native evaluation with
TF32 disabled. The v2 encoder and its BatchNorm buffers were frozen; only the
GroupNorm decoder was optimized. Effective batch64 was accumulated in microbatches8
with the original global loss normalization, verified by loss and gradient tests.
AdamW LR1e-5, seed20261125, 20 epochs and the data/augmentation policy were frozen
before training. Recorded training processes took {elapsed/60:.2f} minutes on one
GPU; this excludes setup, calibration, preflight and final native replay.

No Surface ranker, appearance loss, RGB correction or combined method was used.
Reserved holdout and benchmark assets were absent from the training inputs.
No benchmark adapter admission is established by this report. The >=3% relative
improvement goal on CLIP, LPIPS, VGG feature similarity and SSIM across all 2,565
targets remains unachieved. Failure penalties, CLIP coverage, historical unseen
reporting and native amodal geometry remain required.

Checkpoint SHA256: `{evidence['checkpoint_sha256']}`.
Training SHA256: `{evidence['training_sha256']}`.
Native result SHA256: `{evidence['native_sha256']}`.
'''
    output.mkdir(parents=True)
    (output/'REPORT.md').write_text(text,encoding='utf8')
    write_json(output/'receipt.json',dict(at=datetime.now(timezone.utc).isoformat(),
        replay_sha256=sha256(STUDY/'native-replay.json'),report_sha256=sha256(output/'REPORT.md'),
        source_sha256=sha256(Path(__file__)),benchmark_success=False))
    return output/'REPORT.md'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=STUDY/'report')
    args = parser.parse_args()
    print(build(args.output))


if __name__=='__main__':
    main()
