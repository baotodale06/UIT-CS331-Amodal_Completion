# PD-MC reproduction comparison

Published reference: [Open-World Amodal Appearance Completion, Tables 3 and 5](https://openaccess.thecvf.com/content/CVPR2025/papers/Ao_Open-World_Amodal_Appearance_Completion_CVPR_2025_paper.pdf).

CLIP_success is conditional on successful completion; its paper denominator is unknown.
Delta = observed - paper. Failure deltas are percentage points (pp).

Numerical agreement does not certify exact reproduction. Inspect source coverage, excluded targets, and protocol.

## baseline

Boundary: released; matching: label-semantic; seed: 0.

| Source | Metric | Paper | Observed | Delta | n | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| VG | CLIP_success | 28.3670 | 28.5328 | +0.1658 | 49 | paper_denominator_unknown |
| VG | LPIPS | 0.5780 | 0.6567 | +0.0787 | 109 | different_reference_backend |
| VG | Feature Similarity | 0.4130 | 0.3158 | -0.0972 | 109 | different_reference_backend |
| VG | SSIM | 0.4630 | 0.3773 | -0.0857 | 109 | different_reference_backend |
| VG | Failure rate | 40.0% | 55.05% | +15.05 pp | 109 | subset |
| COCOA | CLIP_success | 27.3830 | 26.8524 | -0.5306 | 28 | paper_denominator_unknown |
| COCOA | LPIPS | 0.6640 | 0.7065 | +0.0425 | 67 | different_reference_backend |
| COCOA | Feature Similarity | 0.3280 | 0.2513 | -0.0767 | 67 | different_reference_backend |
| COCOA | SSIM | 0.3820 | 0.3437 | -0.0383 | 67 | different_reference_backend |
| COCOA | Failure rate | 48.5% | 58.21% | +9.71 pp | 67 | subset |
| free_images | CLIP_success | 28.3330 | 29.4763 | +1.1433 | 6 | paper_denominator_unknown |
| free_images | LPIPS | 0.7200 | 0.6848 | -0.0352 | 14 | different_reference_backend |
| free_images | Feature Similarity | 0.2790 | 0.2627 | -0.0163 | 14 | different_reference_backend |
| free_images | SSIM | 0.3090 | 0.3549 | +0.0459 | 14 | different_reference_backend |
| free_images | Failure rate | 58.2% | 57.14% | -1.06 pp | 14 | subset |
| laion | CLIP_success | 27.5730 | 28.8731 | +1.3001 | 4 | paper_denominator_unknown |
| laion | LPIPS | 0.6920 | 0.8062 | +0.1142 | 14 | different_reference_backend |
| laion | Feature Similarity | 0.2990 | 0.1810 | -0.1180 | 14 | different_reference_backend |
| laion | SSIM | 0.3460 | 0.2354 | -0.1106 | 14 | different_reference_backend |
| laion | Failure rate | 54.2% | 71.43% | +17.23 pp | 14 | subset |
| Overall | CLIP_success | 27.9840 | 28.0727 | +0.0887 | 87 | paper_denominator_unknown |
| Overall | LPIPS | 0.6280 | 0.6853 | +0.0573 | 204 | different_reference_backend |
| Overall | Feature Similarity | 0.3640 | 0.2817 | -0.0823 | 204 | different_reference_backend |
| Overall | SSIM | 0.4130 | 0.3550 | -0.0580 | 204 | different_reference_backend |
| Overall | Failure rate | 45.5% | 57.35% | +11.85 pp | 204 | subset |

