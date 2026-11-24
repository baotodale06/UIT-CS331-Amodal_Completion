# PD-MC reproduction comparison

Published reference: [Open-World Amodal Appearance Completion, Tables 3 and 5](https://openaccess.thecvf.com/content/CVPR2025/papers/Ao_Open-World_Amodal_Appearance_Completion_CVPR_2025_paper.pdf).

CLIP_success is conditional on successful completion; its paper denominator is unknown.
Delta = observed - paper. Failure deltas are percentage points (pp).

Numerical agreement does not certify exact reproduction. Inspect source coverage, excluded targets, and protocol.

## pdmc-baseline-provenance-20261105

Boundary: released; matching: label-semantic; seed: 0.

| Source | Metric | Paper | Observed | Delta | n | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| VG | CLIP_success | 28.3670 | 28.6665 | +0.2995 | 588 | paper_denominator_unknown |
| VG | LPIPS | 0.5780 | 0.6689 | +0.0909 | 1294 | different_reference_backend |
| VG | Feature Similarity | 0.4130 | 0.3050 | -0.1080 | 1294 | different_reference_backend |
| VG | SSIM | 0.4630 | 0.3743 | -0.0887 | 1294 | different_reference_backend |
| VG | Failure rate | 40.0% | 54.56% | +14.56 pp | 1294 | full source |
| COCOA | CLIP_success | 27.3830 | 28.0002 | +0.6172 | 249 | paper_denominator_unknown |
| COCOA | LPIPS | 0.6640 | 0.8015 | +0.1375 | 845 | different_reference_backend |
| COCOA | Feature Similarity | 0.3280 | 0.1778 | -0.1502 | 845 | different_reference_backend |
| COCOA | SSIM | 0.3820 | 0.2353 | -0.1467 | 845 | different_reference_backend |
| COCOA | Failure rate | 48.5% | 70.53% | +22.03 pp | 845 | full source |
| free_images | CLIP_success | 28.3330 | 27.9045 | -0.4285 | 62 | paper_denominator_unknown |
| free_images | LPIPS | 0.7200 | 0.8245 | +0.1045 | 249 | different_reference_backend |
| free_images | Feature Similarity | 0.2790 | 0.1675 | -0.1115 | 249 | different_reference_backend |
| free_images | SSIM | 0.3090 | 0.2004 | -0.1086 | 249 | different_reference_backend |
| free_images | Failure rate | 58.2% | 75.10% | +16.90 pp | 249 | full source |
| laion | CLIP_success | 27.5730 | 27.7652 | +0.1922 | 52 | paper_denominator_unknown |
| laion | LPIPS | 0.6920 | 0.7747 | +0.0827 | 177 | different_reference_backend |
| laion | Feature Similarity | 0.2990 | 0.2023 | -0.0967 | 177 | different_reference_backend |
| laion | SSIM | 0.3460 | 0.2553 | -0.0907 | 177 | different_reference_backend |
| laion | Failure rate | 54.2% | 70.62% | +16.42 pp | 177 | full source |
| Overall | CLIP_success | 27.9840 | 28.3931 | +0.4091 | 951 | paper_denominator_unknown |
| Overall | LPIPS | 0.6280 | 0.7350 | +0.1070 | 2565 | different_reference_backend |
| Overall | Feature Similarity | 0.3640 | 0.2426 | -0.1214 | 2565 | different_reference_backend |
| Overall | SSIM | 0.4130 | 0.3034 | -0.1096 | 2565 | different_reference_backend |
| Overall | Failure rate | 45.5% | 62.92% | +17.42 pp | 2565 | full source |

