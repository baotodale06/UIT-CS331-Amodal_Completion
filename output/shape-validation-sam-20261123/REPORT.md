# Validation-only SAM input diagnostic

Job 45645 completed on one Quadro RTX 6000 in 4m40s, costing 0.077778 allocated
GPU-hours. All 1,264 validation objects across 217 images were evaluated. There
were zero empty SAM masks. No training, checkpoint change, holdout evaluation or
benchmark scoring occurred in this diagnostic.

| Native image-mean metric | Annotated input | SAM input | SAM minus annotated | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Full IoU | 0.888731 | 0.811374 | -0.077358 | [-0.087837, -0.067444] |
| Hidden IoU | 0.436647 | 0.370333 | -0.066313 | [-0.090397, -0.043154] |

Full IoU includes all 217 images/1,264 objects. Hidden IoU includes the 177
images/680 objects with annotated hidden pixels. Both conditions use exactly the
same amodal truth and annotated hidden region. The bootstrap samples images,
with 5,000 draws and seed 20261123. These are absolute IoU changes, not relative
percentage gains. The mean input SAM/annotated-visible IoU is 0.841347.

SAM receives tight boxes derived only from annotated visible support, with
exclusive maximum coordinates and one returned mask. The frozen shape prior
then receives the original image restricted to that input mask. Its native
visible-based crop and probability projection are unchanged. Predictions from
SAM inputs are never repaired with annotated-visible pixels for scoring.

Independent local recovery replayed every native mask, object identity, score,
image denominator and bootstrap interval. The annotated-input condition matches
the original frozen validation evaluation exactly (maximum difference zero).
The archive contains 5,063 files and 4,204,111 bytes; SHA-256 is
`07b5da85957043523a6c3490e2211cc840617004689e88164fce56e4f8e4e8f3`.
See `native-replay.json`, `native-export.json`, and the recovered `native/` tree.

This exposes sensitivity to input-mask errors and motivates separate development
with SAM training inputs and a prior allowed to correct its input support.
It does not reproduce GroundingDINO selection errors, establish a full benchmark
gain, or satisfy any of the active four 3% gates. The separately frozen v2
benchmark and its evaluated holdout remain unchanged.
