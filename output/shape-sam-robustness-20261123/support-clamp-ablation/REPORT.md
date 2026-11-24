# Fixed-checkpoint SAM-support ablation

Restoring the unchanged SAM support does not recover the full-shape regression.
The same epoch-7 checkpoint is used; no training, checkpoint reselection, RGB
correction, holdout access or benchmark evaluation occurs in this ablation.

| Native metric | Frozen v2 + SAM | Fine-tuned prior, SAM support restored | Relative change | Paired 95% interval (IoU points) |
| --- | ---: | ---: | ---: | ---: |
| full_iou | 0.811374 | 0.795808 | -1.918% | [-2.127, -1.010] |
| hidden_iou | 0.370333 | 0.389935 | +5.293% | [+0.511, +3.528] |

Full IoU covers 217 images / 1,264 objects; hidden IoU covers 177 images /
680 objects. Predicted support is recovered raw native prediction OR unchanged
SAM input. Annotated visible pixels enter scoring only. The native mask hashes,
per-object scores and paired image-bootstrap intervals are in `result.json`.
These are exploratory validation comparisons; they cannot establish the four
required full-benchmark gains. This simple clamp is not promoted as a solution
to the validation tradeoff.
