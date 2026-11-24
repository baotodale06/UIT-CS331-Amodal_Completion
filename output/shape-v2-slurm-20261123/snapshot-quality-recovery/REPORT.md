# Frozen v2: partial quality diagnostic

This is the existing **193-target runtime-completed VG snapshot**, fixed by
engineering audit 45665 before quality scoring. It is not a random sample, a
full benchmark or evidence that the >=3% goal is achieved. Frozen v2 continues
unchanged. The diagnostic has no policy tuning or early-stopping rule.

| Metric | Registered control | Frozen v2 | Relative improvement | Paired 95% interval | Control / method n |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLIP | 27.960169 | 28.385203 | +1.520% | [+0.389%, +2.746%] | 96 / 96 |
| LPIPS | 0.653222 | 0.647460 | +0.882% | [-0.127%, +1.918%] | 193 / 193 |
| Feature Similarity | 0.327768 | 0.338066 | +3.142% | [+1.084%, +5.523%] | 193 / 193 |
| SSIM | 0.403876 | 0.406321 | +0.605% | [-0.321%, +1.551%] | 193 / 193 |

Every terminal failure is retained with the released appearance penalties;
CLIP uses its actual scored denominator. Method statuses: `{'not_detected': 13, 'no_completion': 32, 'success': 96, 'ambiguous_target': 52}`.
Control statuses: `{'not_detected': 13, 'no_completion': 32, 'success': 96, 'ambiguous_target': 52}`. CLIP coverage is not reduced:
`True`. Paired intervals resample image clusters
within this prefix; they do not address its nonrandom selection.

The registered control was also rescored on the same Quadro and unchanged
metric implementation. Maximum absolute differences: `{'CLIP': 0.0, 'LPIPS': 1.1920928955078125e-07, 'Feature Similarity': 1.1920928955078125e-07, 'SSIM': 0.0}`.
The primary table above retains the original registered control scores.

Job 45668 completed with exit 0:0 in 132 seconds,
adding 0.036667 allocated GPU-hours. It reads inference
outputs and writes only its separate diagnostic directory. This is an additional
diagnostic cost, not part of the frozen inference allocation.

Input/reference/completion hashes were checked on the cluster. Local recovery
verified the fixed inventory, registered control and all arithmetic/bootstrap
results. Native RGB remains on the cluster. See `recovery.json`, `replay.json`,
`snapshot-quality-45665/contract.json` and `snapshot-quality-45665/result.json`.
