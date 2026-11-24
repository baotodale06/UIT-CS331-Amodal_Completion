# Surface-PMC on annotated vocabulary: measured cohort results

The frozen 12 November transfer experiment is complete on **192 images / 204
targets**. It does **not establish an improvement on the primary LPIPS endpoint**.
The separate full 2,565-target continuation is now complete; see the
[14 November full results](SURFACE_ANNOTATED_FULL_RESULTS_20261114.md). Annotated vocabulary
is the team's baseline for both arms; historical released-vocabulary results are
not the control for this experiment.

Start with the [research report](../output/surface-annotated-20261112-freshrefs/report/REPORT.md),
[PDF](../output/surface-annotated-20261112-freshrefs/report/REPORT.pdf), and
[all-target gallery](../output/surface-annotated-20261112-freshrefs/report/gallery.html).
The [frozen protocol](SURFACE_ANNOTATED_20261112.md),
[reference amendment](SURFACE_ANNOTATED_REFERENCE_AMENDMENT_20261112.md), and
[mathematical methods](SURFACE_ANNOTATED_METHODS_20261112.md) retain the design,
implementation, training lineage, equations, and interpretation limits.

## Primary result and secondary endpoints

| Metric | Annotated PD-MC | Surface-PMC | Relative gain | Nominal 95% interval | Paired targets |
| --- | ---: | ---: | ---: | --- | ---: |
| LPIPS, primary, lower is better | 0.685276 | 0.683564 | +0.250% | [-0.269%, +0.810%] | 204 |
| CLIP image/text logit | 28.072684 | 28.446768 | +1.333% | [+0.258%, +2.496%] | 87 |
| VGG feature similarity | 0.281690 | 0.284273 | +0.917% | [-0.350%, +2.205%] | 204 |
| SSIM | 0.354982 | 0.356430 | +0.408% | [-0.426%, +1.247%] | 204 |

Positive gain is favorable in this table. The primary absolute LPIPS reduction
is **0.00171193**, with interval **[-0.00186716, +0.00550586]**. Intervals use the
frozen 5,000-resample source-stratified image-cluster bootstrap. CLIP is secondary,
uses shared successes, and has no invented failure penalty. Secondary intervals
are nominal and not adjusted for multiple comparisons. The primary result is
inconclusive, rather than evidence of equivalence or proof of no useful effect.

Both arms save **87 outputs and retain 117 failures**. There are no gained or lost
successes. Five control outputs and seven Surface-PMC outputs reach the iteration
limit; saved-output status does not certify complete or correct hidden geometry.
There are no missing input images or reference failures.

All 117 failed targets stop before surface sampling: 47 ambiguous instances,
27 unmatched labels, nine masks below the size gate, and 34 cases with no detected
occlusion. The entire absolute LPIPS difference therefore comes from shared
successes. Their diagnostic LPIPS gain is +1.532% [-1.599%, +4.870%], which also
crosses zero and does not replace the all-target denominator.

## Frozen final-pool diagnostic

The pre-specified secondary analysis evaluates all eight saved raw candidates
for each of 87 successful final passes. It fixes the corrected final mask and
preceding trajectory, preserves all 204 target records and failure penalties,
and compares exact uniform expectations with the frozen choice.

| Contrast | LPIPS relative gain | Nominal 95% interval | CLIP relative gain | Nominal 95% interval |
| --- | ---: | --- | ---: | --- |
| Frozen raw selection vs uniform raw choice | -0.0069% | [-0.0331%, +0.0196%] | +0.4514% | [+0.1048%, +0.7980%] |
| Corrected output vs selected raw output | +0.3057% | [+0.1761%, +0.4707%] | -0.4252% | [-0.8185%, -0.0176%] |

There is no clear final-pool LPIPS selection benefit. The final RGB correction
improves visible-reference LPIPS while slightly reducing CLIP. These are
conditional, exploratory observations, not an isolated estimate of the selector's
effect throughout progressive inference. Final masks were chosen by the corrected
trajectory, which can favor that trajectory. Different per-metric oracle choices
are reported as headroom bounds; they are not a deployable or jointly achievable
method. Corrected RGB lies outside the saved raw candidate set.

## Cost, training, and validation

The transferred method uses the existing 22-feature ridge selector and frozen
corrections without any new fitting. The diffusion generator and feature networks
remain pretrained. All 477 previously used controlled-study images were excluded
from this cohort by identity/source bytes, followed by the frozen near-copy screen.
The cohort was never used to tune the selector or change its endpoints.

Surface-PMC generated **1,152 candidate images in 144 pools**, with no all-invalid
fallbacks. Both arms requested four Quadro RTX 6000 GPUs and four workers:

| Arm | Job | Allocation wall time | Reserved GPU-hours |
| --- | --- | --- | ---: |
| Annotated control | 39486 | 24m 00s | 1.600 |
| Surface-PMC | 39487 | 1h 49m 34s | 7.304 |

The wall-time ratio is 4.565. Reserved GPU-hours include idle, scoring and
verification time; they are not utilization measurements. The failed historical
reference attempts, smoke, diagnostic and audits have additional costs. K=1 vs
K=8 measures the full method plus additional sampling, not equal-compute selection.

Independent cluster audits 39498 and 39499 passed. They replayed 144 selector
decisions, exact corrected RGB at candidate resolution, 2,736 pool artifact hashes,
and 696 native final-candidate derivations. Same-runtime neural-metric replay drift
was at most 1.79e-7. Local recovery, replay, report checks, and visual inspection
are recorded separately in the delivered report's validation and recovery receipts.
Feature extraction and neural metrics are not claimed to have been rerun locally.

The reference amendment preserves a strict matched comparison: an initial attempt
to reuse historical references failed on a one-pixel SAM difference. Fresh control
references were copied and hash-verified before Surface-PMC inference; paired input,
reference, model and environment checks passed. The numerical drift's exact cause
is unresolved. Historical runs remain preserved and are not relabeled as this control.

## Interpretation and continuation

The evidence supports retaining annotated PD-MC as the operational baseline while
reporting this Surface-PMC transfer as inconclusive on its primary endpoint. The
secondary CLIP improvement is measured, but does not rescue the primary test.
Natural hidden RGB is unavailable: visible-reference LPIPS and text similarity
cannot establish correct hidden surfaces, amodal shape, or human preference.
Example crop and mask differences also prevent attributing the progressive metric
change solely to appearance correction.

Post-hoc assistant inspection of the mechanically selected two largest positive
and negative shared-success LPIPS differences illustrates this limitation:

| Target | Absolute LPIPS gain | Control foreground pixels | Surface foreground pixels |
| --- | ---: | ---: | ---: |
| VG/2315974_pinkteddybear | +0.155772 | 59,688 | 35,646 |
| laion/000012713_saxophone | +0.111078 | 5,481 | 4,133 |
| COCOA/COCO_val2014_000000185941_phonebooth | -0.168987 | 144,334 | 194,128 |
| VG/1159293_table | -0.119059 | 2,120 | 24,581 |

Within each of these four pairs the native canvas dimensions agree, but the
foreground masks differ substantially. The teddy and saxophone gains accompany
smaller masks. The table control retains only a small fragment; Surface-PMC
retains more of the table yet receives worse LPIPS. Neither image establishes
the hidden truth. These selected extremes demonstrate an interpretive pitfall,
not the prevalence of a failure, a causal effect of mask size, or human preference.
Native dimensions, offsets and hashes are retained in the report evidence.

The full baseline **39488**, Surface-PMC **39490**, timeout continuation **39491**,
finalizer **39492**, independent full/unseen-image audit **39497**, and export
**39495** have now finished; 39490 timed out and its planned continuation 39491
completed. Both full runs are verified and registered. No new training occurred.
See the [completed full results](SURFACE_ANNOTATED_FULL_RESULTS_20261114.md) and
[continuation record](SURFACE_ANNOTATED_CONTINUATION_20261112.md).

Future mechanism experiments would need a new frozen design: equal-budget
progressive random selection, correction-only ablation, and independent geometry
or human assessment. These are recommendations informed by this result, not
submitted experiments or established improvements.
