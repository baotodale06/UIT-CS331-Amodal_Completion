# Independent Surface-PMC selection-only confirmation

## Completed and verified, 11 November 2026

Job **37843** completed successfully with all128 images and zero failures among
1,024 generated candidates. The four predeclared controlled endpoints exceed3%
versus expected uniform selection from the identical eight-candidate pools:

| Metric | Relative improvement | Individual nominal95% interval |
| --- | ---: | --- |
| Hidden RGB MAE | +7.644% | [+4.287%, +11.291%] |
| Object LPIPS | +11.328% | [+8.021%, +14.828%] |
| DreamSim | +16.766% | [+11.808%, +21.583%] |
| CLIP cosine | +4.068% | [+3.504%, +4.653%] |

Every individual lower bound is above3%; these are not simultaneous confidence
bounds. SSIM improves1.668%, below3%. CLIP is a selector input. Against CLIP-only
selection, reconstruction improves but CLIP declines0.632%.

The [report](../output/surface-confirmation128-20261110/report/REPORT.md),
[PDF](../output/surface-confirmation128-20261110/report/REPORT.pdf),
[all128 cases](../output/surface-confirmation128-20261110/report/gallery.html) and
[local audit](../output/surface-confirmation128-20261110/report/audit.json)
retain the full comparison and limitations. Audit verifies2,631 archive members,
all1,024 candidate identities and128 decisions, exact output identity and timing,
independent selected-image hidden MAE, and all recorded summary means/intervals.
Native all-candidate replay passed; local neural metrics were not rerun.

This confirms one-step raw selection on independent artificial occlusions. The
earlier corrected primary remains a missed-CLIP result. The full corrected run
subsequently completed as **39387**, with finalizer **37814**: CLIP/LPIPS/VGG/SSIM
gains are1.123/0.835/2.789/1.782%, below the3% full target. Final-RGB diagnostics
39404/39407/39411 also completed; see the
[follow-up report](../output/surface-lastpass-review-20261111/REPORT.md).
Those natural-occlusion comparisons do not change this controlled confirmation.

## Frozen design and historical execution notes

At 04:12:48 UTC on 10 November 2026, the selection-only policy and a new
128-image cohort were frozen before generating any outputs. Four-GPU Slurm job
**37843** was submitted at 04:15:55 UTC on `debug-gpu`. Results are pending.
The scheduler estimates 09:45:49 UTC, subject to change. The original five-hour
deadline has passed. Full progressive job **37812** and finalizer **37814** remain
separate.

At 05:07 UTC, 37843 remains pending with the same estimated start. A read-only
four-GPU scheduling probe on the alternative `gpu` partition estimated a later
21:48:49 UTC start and created no job. The local GPU has 6 GiB VRAM and does not
fit the validated configuration. Existing jobs remain unchanged. Full control
37812 has 963/2,565 records with four recently active worker logs and no recent
tracebacks; full surface generation has not started. See the
[capacity/status receipt](../output/surface-confirmation128-20261110-design/status-20261110T050720Z.json).

The preceding 166-image experiment's corrected primary missed the CLIP target.
Its pre-specified selection-only component improved MAE 5.845%, object LPIPS
6.754%, DreamSim 12.764%, and CLIP cosine 3.075%. That secondary result motivated
this confirmation. The earlier primary and secondary labels remain unchanged.

The ranker's coefficients, feature definitions, valid-candidate rule, and eight
seeds (11, 22, 33, 44, 55, 66, 77, 88) are unchanged. Selection returns the exact
generated RGB and mask: no color calibration, consensus correction, or visible
pixel restoration. The generator and pretrained metric networks remain frozen.
No ranker fitting or method search is performed on this cohort.

After excluding all 349 previous images by identity and source bytes, 861 images
pass the native-mask eligibility screen (629 VG, 232 COCOA). The fixed hash order
with selection seed 20261110128 selects 64 VG and 64 COCOA images. All 128 also
pass the resized-support checks. Decoded-image comparison against all 349 prior
images finds no exact duplicates, within-cohort duplicates, or aligned near-copy
flags. Minimum comparable thumbnail MAE is 0.118375 against a 0.005 flag threshold.
This does not rule out transformed or cropped copies.

The primary is `trained_surface_raw`, compared with expected uniform selection
from the same eight candidates. The four primary endpoints are hidden-core RGB
MAE, complete-object LPIPS, DreamSim, and category-text CLIP cosine. Object metrics
use the existing black-background paper-style protocol. CLIP is also an explicit
ranker input. Fixed seed, semantic-only, consensus-only, and other shared-pool
controls remain available in the complete comparison table.

Success requires all four relative point gains to be at least 3% across all 128
images; improvements above 5% also satisfy the target. Report paired,
source-stratified 95% bootstrap intervals with 5,000 resamples separately from
that point criterion. All lower bounds exceeding 3% is a stronger, separately
reported criterion. Keep all cases and explicit generation failures. Missing
CLIP comparisons cannot silently satisfy the all-image criterion. There is no
sequential stopping or selection of a favorable subset.

This evaluates artificial occlusion on VG and COCOA. It cannot establish the
full progressive benchmark result, performance on excluded source populations,
or reconstruction of naturally hidden RGB.

Artifacts:

- [Frozen policy](../output/surface-confirmation128-20261110-design/selection-only-freeze.json),
  SHA-256 `711f49db26e19d0655c64b59682d9ae50c0841d38071096ad9dc7f83d733d4c4`.
- [Split audit](../output/surface-confirmation128-20261110-design/split-audit.json),
  [cases](../output/surface-confirmation128-20261110/cases.json), and
  [submission receipt](../output/surface-confirmation128-20261110-design/submission.json).
- [Runner](../scripts/run_surface_confirmation_slurm.sh) and
  [verification/reporting helper](../scripts/finalize_surface_confirmation.py).
- [Validation log](../logs/surface-confirmation-check-20261110.log): 248 tests
  discovered, 242 passed, six optional SciPy/sklearn cases skipped in that CPU
  environment; shell regressions and synthetic smoke passed. The two new
  regressions also pass in the pinned replay environment.

The uploaded 65,917,776-byte inputs/source archive independently matches SHA-256
`c07874df475cece5ec04acf9ed7189ca2cafd4090f9f178c02f78e19dbd93038` on the cluster.
Remote preflight verifies all frozen implementation and cohort input hashes.
An initially missing split-audit helper was uploaded before preflight passed;
no scientific source or policy was revised.

The queued runner performs both required GPU integration checks, generates all
1,024 candidates, extracts observation-only features, publishes all decisions,
then measures outcomes. Exact CPU replay stays on the generation node. It writes
`output/surface-confirmation128-20261110/analysis/confirmation.json` and
`analysis/CONFIRMATION.md`, plus a hash-recorded evidence archive in the design
directory. Recover and verify those artifacts after job completion. Do not
resubmit while 37843 is pending or running.

```bash
cd ~/Vu_Hung/cv_nangcao
sacct -j 37812,37814,37843 --format=JobID,JobName,State,ExitCode,Elapsed,Start,End -X
squeue --start -j 37843
```
