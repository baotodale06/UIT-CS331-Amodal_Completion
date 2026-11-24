# Annotated Surface-PMC: completed full benchmark

The frozen 12 November continuation is complete, verified and registered.
No new model training, fitting, tuning or inference was launched on 14 November.
The full comparison measures a modest improvement, below the earlier 3% target.
The original 204-target primary cohort remains inconclusive.

Read the [four-page report](../output/surface-annotated-full-review-20261114/REPORT.pdf),
[report with evidence links](../output/surface-annotated-full-review-20261114/REPORT.md),
[all-target CSV](../output/surface-annotated-full-review-20261114/paired-records.csv),
and [independent local audit](../output/surface-annotated-full-review-20261114/audit.json).

## Full operational result

Both arms use annotated vocabulary and exactly matched fresh SAM references.
The unchanged transferred method uses K=8; the control uses K=1.
All 2,565 targets from 2,379 images are present.

| Metric | Control | Surface-PMC | Relative gain | Nominal 95% interval | n |
| --- | ---: | ---: | ---: | --- | ---: |
| LPIPS, lower is better | 0.614408 | 0.607088 | +1.191% | [+0.749%, +1.641%] | 2,565 |
| CLIP, shared successes | 28.168627 | 28.541179 | +1.323% | [+1.007%, +1.641%] | 1,387 |
| VGG feature similarity | 0.349608 | 0.357859 | +2.360% | [+1.644%, +3.100%] | 2,565 |
| SSIM | 0.445624 | 0.451793 | +1.384% | [+0.744%, +2.061%] | 2,565 |

The frozen 2,017-target / 1,900-image unseen sensitivity has LPIPS **+0.801%
[+0.396%, +1.179%]**, CLIP +1.337%, VGG +1.901%, and SSIM +0.938%.
The full comparison includes prior development/training images. The unseen subset
excludes image identities/source bytes; exhaustive near-copy screening applied
only to the original primary cohort. These overlapping analyses are not independent
replications and do not replace the inconclusive primary result of +0.250%
[-0.269%, +0.810%].

Intervals use 5,000 source-stratified image-cluster bootstrap draws. Failures retain
the original LPIPS/VGG/SSIM penalties; CLIP uses shared successes without a failure
penalty. Secondary, sensitivity and source intervals are nominal and unadjusted.

## Coverage and cost

Control: **1,394 saved / 1,171 failed**. Surface-PMC: **1,399 saved / 1,166 failed**.
There are 1,387 shared successes, 12 gained, seven lost and 1,159 failed in both.
Iteration-capped saved outputs decrease from 209 to 199; this is not proof of
correct hidden geometry. Of 1,166 Surface-PMC failures, 1,156 stop before sampling.

Shared-success LPIPS improves 4.044% [2.972%, 5.125%], but this conditional result
does not meet a 3% all-target goal. Most absolute LPIPS gain comes from shared
successes; the remaining net contribution comes from changed completion coverage.
The report retains the exact additive decomposition and source-level results.

The saved 2,847 pools contain 22,776 candidates, with nine all-invalid fallbacks.
Both arms used four Quadro RTX 6000 GPUs and four workers. Including the reused
cohort and timeout continuation, control cost is **24.597 reserved GPU-hours**;
Surface-PMC cost is **116.406 reserved GPU-hours**, a **4.733× allocation-time ratio**.
These include setup/scoring/verification and idle workers, not measured utilization.
Smoke, failed reference attempts, the pool diagnostic and CPU audits are excluded.

## Completion and evidence

Control 39488 completed. Surface 39490 timed out after 24 hours; the existing
unchanged continuation 39491 completed. Finalizer 39492 verified and registered
both arms; independent audit 39497 and metadata export 39495 completed. No study
job remains to resume. Exact commands and IDs remain in the design submission
receipts; raw accounting and the two registry entries were recovered separately.

All 16,197 metadata members match independent remote SHA-256 hashes. Local review
verifies both execution provenance records, 5,538 historical result-hash uses,
all 408 original cohort records, both registry entries, and all 2,847 selector
choices/losses/seeds using their saved observable features. Full and unseen means,
denominators, transitions and intervals reproduce the cluster audit within 1e-10.

Full native artifacts passed verification on the cluster and remain there. Local
review does not rerun neural metrics, feature extraction or full RGB corrections.
The earlier cohort has a separate native replay. Frozen method/source checks pass.

Keep annotated PD-MC as the operational baseline pending a cost/benefit decision.
A new mechanism study would require a frozen equal-budget random-selection arm,
a correction-only ablation, and independent geometry or blinded human assessment.
No such experiments were launched. Visible-reference scores do not establish
natural hidden RGB correctness, amodal shape accuracy, or human preference.
