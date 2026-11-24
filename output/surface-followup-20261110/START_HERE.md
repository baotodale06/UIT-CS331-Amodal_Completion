# Surface-PMC follow-up

The frozen method improves hidden RGB MAE, LPIPS and DreamSim substantially on
166 fresh controlled cases. **The requested 3% improvement on every metric is
not achieved: CLIP improves by 2.42%.** The original five-hour window elapsed
before final evaluation.

The **pre-specified selection-only component** reaches all four 3% point targets:
MAE **5.85%**, LPIPS **6.75%**, DreamSim **12.76%**, and CLIP **3.07%** improvement.
It keeps the same frozen candidate decisions and omits correction. This remains
a secondary component result; promoting it after inspecting this test requires
independent confirmation. Its MAE and CLIP confidence intervals extend below3%.
The primary corrected method is reported below without changing its freeze.

| Metric | Shared-pool baseline | Surface-PMC | Relative improvement |
| --- | ---: | ---: | ---: |
| Hidden RGB MAE | 0.172749 | 0.157053 | 9.09% |
| Object LPIPS | 0.172372 | 0.150723 | 12.56% |
| DreamSim | 0.165411 | 0.134722 | 18.55% |
| CLIP category-text cosine | 0.284143 | 0.291026 | 2.42% |

The baseline is expected uniform selection from the same eight candidates.
All four paired 95% intervals exclude zero improvement; CLIP's interval is
1.93–2.95%. The fixed-seed and equally corrected controls are also reported.
There are zero generation failures among 1,328 test candidates. Hidden MAE
improves on 120 images and worsens on 46; the gallery retains all cases.

- [Full report](report/REPORT.md) and [PDF](report/REPORT.pdf).
- [Download the verified reading package](surface-followup-reading-package.zip).
- [Search every case](report/gallery/index.html), including losses and fallbacks.
- [Complete numerical comparisons](analysis-eight/summary.json) and
  [CSV](analysis-eight/comparisons.csv).
- [Strict native replay](analysis-eight/replay-verification.json) and
  [report verification](report/validation.json).
- [Frozen method](../surface-followup-20261110-design/final-method-freeze.json)
  and [exact commands](../surface-followup-20261110-design/commands.sh).
- [Usage and reproduction](../../cookbooks/surface-followup.md).

These are artificially hidden, previously observed object pixels, with one
target per source image (64 VG, 64 COCOA, 38 free-images). They are not true
hidden amodal RGB or the paper's original COCO/Open Images evaluation set.
The previous 183 images were development only; fresh images are disjoint by
identity and image bytes. No parameters changed after fresh scoring.

The pretrained generator remains frozen. A small ridge ranker combines observed
surface continuity, preservation, sample agreement and pretrained appearance
features; bounded color calibration and smooth consensus correction follow.
CLIP is also an explicit selector feature. Against the equally corrected
semantic-only selector, the method improves MAE 4.92%, LPIPS 6.52% and
DreamSim 11.20%, while CLIP falls 0.71%.

The earlier full cached calibration attempt missed the objective (LPIPS 0.39%,
VGG 1.02%, SSIM 0.95%, CLIP −0.33%). It is preserved in the report. The full
progressive comparison is a separate long-running experiment: job 37812 runs
the matched 2,565-target control/method pair after native replay and a completed,
provenance-verified final smoke. At 03:25 UTC the control was running with 82
target records; the full method had not started. Job 37814 verifies, compares
and registers complete results. Check
[job records](../surface-followup-20261110-design/jobs.json) for the current
state. No full progressive gain is established by the controlled table above.

Strict primitive-score replay must use the recorded generation CPU/runtime.
Different CPU dispatch paths moved float32 Lab values across histogram bins;
the failed replay attempts are retained. Tolerances and frozen scientific source
were not changed to conceal this discrepancy. Native replay and report checks
must pass before this handoff is considered verified.

The reading ZIP includes reports, all display panels, metric records, decisions
and frozen source. Native candidates, original datasets, model checkpoints and
environments remain in the separately preserved project and cluster artifacts.
