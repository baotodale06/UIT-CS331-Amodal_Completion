# PD-MC reproduction comparison

Published reference: [Open-World Amodal Appearance Completion, Tables 3 and 5](https://openaccess.thecvf.com/content/CVPR2025/papers/Ao_Open-World_Amodal_Appearance_Completion_CVPR_2025_paper.pdf).

CLIP_success is conditional on successful completion; its paper denominator is unknown.
Delta = observed - paper. Failure deltas are percentage points (pp).

Numerical agreement does not certify exact reproduction. Inspect source coverage, excluded targets, and protocol.

## surface

Boundary: released; matching: label-semantic; seed: 0.

| Source | Metric | Paper | Observed | Delta | n | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| VG | CLIP_success | 28.3670 | 28.8409 | +0.4739 | 49 | paper_denominator_unknown |
| VG | LPIPS | 0.5780 | 0.6535 | +0.0755 | 109 | different_reference_backend |
| VG | Feature Similarity | 0.4130 | 0.3195 | -0.0935 | 109 | different_reference_backend |
| VG | SSIM | 0.4630 | 0.3805 | -0.0825 | 109 | different_reference_backend |
| VG | Failure rate | 40.0% | 55.05% | +15.05 pp | 109 | subset |
| COCOA | CLIP_success | 27.3830 | 27.5307 | +0.1477 | 28 | paper_denominator_unknown |
| COCOA | LPIPS | 0.6640 | 0.7095 | +0.0455 | 67 | different_reference_backend |
| COCOA | Feature Similarity | 0.3280 | 0.2504 | -0.0776 | 67 | different_reference_backend |
| COCOA | SSIM | 0.3820 | 0.3412 | -0.0408 | 67 | different_reference_backend |
| COCOA | Failure rate | 48.5% | 58.21% | +9.71 pp | 67 | subset |
| free_images | CLIP_success | 28.3330 | 29.2255 | +0.8925 | 6 | paper_denominator_unknown |
| free_images | LPIPS | 0.7200 | 0.6808 | -0.0392 | 14 | different_reference_backend |
| free_images | Feature Similarity | 0.2790 | 0.2662 | -0.0128 | 14 | different_reference_backend |
| free_images | SSIM | 0.3090 | 0.3592 | +0.0502 | 14 | different_reference_backend |
| free_images | Failure rate | 58.2% | 57.14% | -1.06 pp | 14 | subset |
| laion | CLIP_success | 27.5730 | 28.8625 | +1.2895 | 4 | paper_denominator_unknown |
| laion | LPIPS | 0.6920 | 0.7965 | +0.1045 | 14 | different_reference_backend |
| laion | Feature Similarity | 0.2990 | 0.1902 | -0.1088 | 14 | different_reference_backend |
| laion | SSIM | 0.3460 | 0.2392 | -0.1068 | 14 | different_reference_backend |
| laion | Failure rate | 54.2% | 71.43% | +17.23 pp | 14 | subset |
| Overall | CLIP_success | 27.9840 | 28.4468 | +0.4628 | 87 | paper_denominator_unknown |
| Overall | LPIPS | 0.6280 | 0.6836 | +0.0556 | 204 | different_reference_backend |
| Overall | Feature Similarity | 0.3640 | 0.2843 | -0.0797 | 204 | different_reference_backend |
| Overall | SSIM | 0.4130 | 0.3564 | -0.0566 | 204 | different_reference_backend |
| Overall | Failure rate | 45.5% | 57.35% | +11.85 pp | 204 | subset |

