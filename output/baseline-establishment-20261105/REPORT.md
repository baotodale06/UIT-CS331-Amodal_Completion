# Accepted provenance baseline — 2026-11-06

**The full 2,565-target baseline is scored, provenance-verified and registered.** Slurm job 31548 used four Quadro RTX 6000 GPUs with pinned PyTorch 1.13.1 / CUDA 11.7. The recovered copy independently passed full verification against the local dataset.

Saved 951 completions and recorded 1,614 failures (62.92% failure rate), with zero missing images and zero reference failures. The allocation completed in 4h 02m 38s with exit code 0; peak host RAM was 46.89 GiB of 64 GiB.

This evaluates pretrained PD-MC with the unchanged `configs/baseline.json`: SAM references, semantic matcher v2, released vocabulary, seed 0, timestep 35, five-iteration configuration, released metric alignment and four workers.

## Coverage and termination

| Source | Targets | Saved completions | Failures | Non-cap outputs | At iteration cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| VG | 1294 | 588 | 706 | 502 | 86 |
| COCOA | 845 | 249 | 596 | 205 | 44 |
| free_images | 249 | 62 | 187 | 45 | 17 |
| laion | 177 | 52 | 125 | 44 | 8 |
| Overall | 2565 | 951 | 1614 | 796 | 155 |

Non-cap outputs describe recorded termination states; they do not certify visual completeness.

## Metrics and denominators

| Source | CLIP ↑ (n) | LPIPS ↓ (n) | Feature similarity ↑ (n) | SSIM ↑ (n) |
| --- | ---: | ---: | ---: | ---: |
| VG | 28.666509 (588) | 0.668910 (1294) | 0.304972 (1294) | 0.374274 (1294) |
| COCOA | 28.000240 (249) | 0.801542 (845) | 0.177771 (845) | 0.235267 (845) |
| free_images | 27.904510 (62) | 0.824483 (249) | 0.167476 (249) | 0.200436 (249) |
| laion | 27.765186 (52) | 0.774741 (177) | 0.202314 (177) | 0.255333 (177) |
| Overall | 28.393098 (951) | 0.735009 (2565) | 0.242636 (2565) | 0.303397 (2565) |

CLIP excludes failed completions. Appearance metrics include the configured failure penalties; their denominators must accompany comparisons.

## Failure reasons

| Recorded reason | Targets |
| --- | ---: |
| label_not_matched | 877 |
| no_occlusion_detected | 344 |
| multiple_distinct_instances | 251 |
| query_mask_too_small | 69 |
| no_detections | 57 |
| completion_not_detected | 15 |
| unknown_detected_class | 1 |

## Validation and preserved evidence

- All four devices passed matrix and FFT round trips; the real CUDA extension build/forward/backward and complete SAM/Mixed Context/InstaOrder/LaMa import checks passed.
- Source-image preflight passed all 2,565 targets. The eight-target four-worker scored smoke passed provenance verification and exactly matched the one-worker target records.
- The launcher flag conflict that stopped job 30426 before inference was repaired. Eight focused regressions passed in WSL and on the cluster, including the actual Bash parser.
- Recovered 16,814 files; every file matches its remote SHA-256. The original failed attempts and historical outputs remain preserved.
- Recovery retained complete files from the initial partial transfer, reused independently rehashed identical local bytes, and downloaded missing metadata/masks/JPEGs. PNG encodings were restored from recorded RGB and masks only when the resulting bytes matched the independent remote SHA-256; no model inference or metric values were synthesized.
- All 36 captured project source files match the current maintained source. Runtime package versions and allocation details are retained in the execution snapshot.
- [Per-target CSV](../baseline-31548/output/pdmc-baseline-provenance-20261105/per_target.csv) and [complete summary](../baseline-31548/output/pdmc-baseline-provenance-20261105/summary.json)
- [Registered cluster ledger snapshot](../baseline-31548/experiments.csv)
- [Independent full verification](full-independent-verification.json)
- [Recovery receipt](accepted-recovery-receipt.json) and [fresh remote file hashes](completed-remote-manifest.json)
- [Runtime and source audit](accepted-source-runtime-audit.json) and [Slurm accounting](slurm-accounting-31548.json)
- [Launch and deployment evidence](LAUNCH.md)
- Execution ID: `20261106T053037.969259Z-fdce84ff`.
- Registered entry ID: `ec823094e31f226957ef0442f0480cf14ae2db91c79550a9701a9e0aef99b07c`.
- Recovered run: `C:\Users\daovu\Documents\Project\cv_nangcao\output\baseline-31548\output\pdmc-baseline-provenance-20261105`.

## Historical reproducibility comparison

The historical semantic-v2 run and this baseline have identical scientific/metric configurations, target/input identities, and all 2,565 recorded seeds. The historical run saved 946 completions; this run saved 951, with six gains and one loss. All 945 shared completion PNGs differ in bytes; 32 initial reference masks and 19 full reference PNGs differ. Shared-target metric means are close, but the historical run lacks execution/environment provenance, so the cause of the differences cannot be isolated. This is descriptive evidence, not an improvement or byte-exact historical reproduction claim. [Paired comparison and changed target IDs](historical-semantic-v2-comparison.json).

## Limits

SAM-derived references measure appearance agreement with generated visible-object references, not hidden-pixel ground truth. Reconstructed LAION bytes do not establish exact original-archive reproduction. Independent CPU verification checks saved neural-metric caches and their input hashes rather than recomputing those models. This establishes a reference baseline; it makes no improvement or best-model claim.
