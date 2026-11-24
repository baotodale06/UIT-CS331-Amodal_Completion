# Shape-only local probability-loss native validation

All 20 epochs completed. Selected epoch: **0**.
Epoch zero won. No newly trained checkpoint is eligible; do not promote this run as an improved model.

| Reference | Metric | Reference prior | Selected prior | Change (IoU points) | Paired 95% interval (points) |
| --- | --- | ---: | ---: | ---: | ---: |
| Original Quadro | full_iou | 0.811373590 | 0.811373590 | +0.0000 | [+0.0000, +0.0000] |
| Original Quadro | hidden_iou | 0.370333476 | 0.370333476 | +0.0000 | [+0.0000, +0.0000] |
| Original local GPU | full_iou | 0.811373590 | 0.811373590 | +0.0000 | [+0.0000, +0.0000] |
| Original local GPU | hidden_iou | 0.370333476 | 0.370333476 | +0.0000 | [+0.0000, +0.0000] |

Independent replay covers all 1,264 validation objects / 217 images, native
binary masks, input hashes, checkpoint selection and image-bootstrap statistics.
The intervals are nominal after validation selection. This is iterative research,
not an untouched confirmation or a full-benchmark result.

The initial RTX3060 prediction differed from original Quadro masks in
0 pixels across
0 objects. Initial and selected
local checkpoints each passed exact same-device native pixel replay. Both original
Quadro and local geometry floors remained in force throughout selection.

Training used one 6GB RTX3060 Laptop GPU in isolated WSL, exact cluster conda
packages, PyTorch1.13.1/CUDA11.7, FP16 training, and FP32 native evaluation with
TF32 disabled. The v2 encoder and its BatchNorm buffers were frozen; only the
GroupNorm decoder was optimized. Effective batch64 was accumulated in microbatches8
with equivalent per-object loss normalization, verified by loss and gradient tests.
The loss is binary cross entropy outside the input support; no class weights, Dice
term or hidden-positive bonus are used. This is distinct from the constrained run.
AdamW LR1e-5, seed20261125, 20 epochs and the data/augmentation policy were frozen
before training. Recorded training processes took 149.73 minutes on one
GPU; this excludes setup, calibration, preflight and final native replay.

No Surface ranker, appearance loss, RGB correction or combined method was used.
Reserved holdout and benchmark assets were absent from the training inputs.
No benchmark adapter admission is established by this report. The >=3% relative
improvement goal on CLIP, LPIPS, VGG feature similarity and SSIM across all 2,565
targets remains unachieved. Failure penalties, CLIP coverage, historical unseen
reporting and native amodal geometry remain required.

Checkpoint SHA256: `a52493d7d78cb1c6fc4f9765c6c818e706d22270c58dc6fb95caebabedb44fa8`.
Training SHA256: `5c3dade6cdf7b4e5a6ef7ff9db18625f3685713c11d1bdcd39b52e267ed61baf`.
Native result SHA256: `2c6bb8f679de71e85b634b84cb2807bf1d2dc704239b2c3db78e584a98748f1d`.
