# Surface consistency: controlled results

**The experiment does not establish a reliable full-surface improvement over equal-budget random selection.**

Frozen pretrained PMC, four candidates per image, 24 development images and 159 held-out image identities: 72 pilot test plus 87 prespecified replication. All selectors share the same seeds and generated images. No diffusion model was trained.

Pooled hidden RGB MAE: uniform selection 0.1699; full surface 0.1655. Paired gain +0.0043 (95% CI -0.0005 to +0.0091), equivalent to +1.10 intensity levels or +2.5% relative error reduction.

Pilot test, n=72: full-surface MAE 0.1685; gain over all-candidate uniform selection +0.00297, 95% CI [-0.00472, +0.01128].
Replication, n=87: full-surface MAE 0.1631; gain over all-candidate uniform selection +0.00542, 95% CI [+0.00004, +0.01119].

The primary result concerns recovery of artificially hidden, originally observed SAM-supported pixels. It does not prove correct reconstruction of naturally hidden surfaces, human preference, improved amodal shape or end-to-end progressive completion.

## Results on shared candidate pools

| Rule | Hidden MAE | Gain vs uniform [95% CI] | Fallbacks |
| --- | ---: | ---: | ---: |
| Fixed seed 11 | 0.1695 | +0.0004 [-0.0042, +0.0047] | 0 |
| Uniform selection (expected) | 0.1699 | +0.0000 [+0.0000, +0.0000] | 0 |
| Uniform among valid candidates | 0.1691 | +0.0007 [-0.0000, +0.0016] | 0 |
| Visible-pixel preservation | 0.1643 | +0.0056 [+0.0005, +0.0108] | 1 |
| Boundary | 0.1635 | +0.0064 [+0.0019, +0.0111] | 1 |
| Color | 0.1626 | +0.0073 [+0.0021, +0.0125] | 1 |
| Texture | 0.1666 | +0.0033 [-0.0014, +0.0081] | 1 |
| Boundary + color | 0.1633 | +0.0066 [+0.0016, +0.0114] | 1 |
| Boundary + texture | 0.1641 | +0.0058 [+0.0012, +0.0105] | 1 |
| Full surface | 0.1655 | +0.0043 [-0.0005, +0.0091] | 1 |
| Local surface | 0.1627 | +0.0072 [+0.0022, +0.0123] | 1 |
| Development-selected rule | 0.1666 | +0.0033 [-0.0014, +0.0081] | 1 |
| Hidden-MAE oracle (unavailable) | 0.1388 | +0.0311 [+0.0277, +0.0347] | 0 |

The fixed local-seam variant has pooled gain +0.0072 [+0.0022, +0.0123]. It is an exploratory ablation. The development-selected rule was Texture, so the local rule was not selected by development performance. Intervals are nominal and do not correct for comparisons across the ablation menu. Pilot test local-surface gain: +0.0111 [+0.0033, +0.0194]. Replication local-surface gain: +0.0040 [-0.0021, +0.0097].

![Paired gains](figures/paired_gains.png)

## What the theory means

Surface consistency is a compatibility prior: visible color, texture and seam continuity may help choose between candidates when they share one material or pattern. It is not hidden ground truth.
Identifiability limit: identical visible evidence can correspond to different hidden materials or parts. Any observable-only selector can therefore fail on a legitimate hidden transition.
Constructed examples show that global means/histograms cannot resolve stripe phase; a local seam term can. Both variants prefer an incorrect uniform fill over a true unseen material change. These examples test mechanisms, not natural-image success rates.

## Design and leakage controls

The pilot contains 96 image-disjoint cases, 24 per source. Six per source are development and 18 per source are test. Replication excludes all pilot identities and image bytes, with 24 VG, 24 COCOA, 24 free-images and 15 LAION cases. LAION eligibility was exhausted; the amendment was recorded before test errors were inspected.

Cached visible SAM support defines a square context crop at 512 pixels. A hash-selected half-plane hides approximately 35% of visible target pixels. RGB is removed after resizing. Hidden truth and support are kept in evaluation-only files; generation and scoring consume only observations.

The pinned Mixed Context sampler uses step 35 of 50 and explicit seeds 11,22,33,44. Grounding DINO plus SAM predict each candidate target mask. This is one shared completion step with known artificial occlusion; ordering, the progressive loop and released occluder dilation are outside this experiment.

Full surface combines original-visible anchored Lab boundary distance, Lab histogram Hellinger distance and masked gradient texture distance with weights .40/.25/.35. Local surface combines paired seam color and local texture. Positive development medians normalize terms; all settings are frozen before test analysis.

The separately reported development-selected rule is Texture. It was chosen on 24 development cases from the fixed menu including the fixed-seed control, then held unchanged for both test cohorts.

Primary MAE is evaluated on all held-out target-core pixels, independent of predicted mask size. Missing generations receive MAE 1. LPIPS masks both images outside that same core. Source-stratified paired bootstrap intervals use 5,000 resamples; ablations are exploratory.


## Explanation and independent diagnostics

![Within-pool association](figures/score_error_association.png)

Full surface has defined scores for 611/636 held-out candidates. All-invalid pools use the declared fixed-first fallback: laion-009-acb3bbb567 (pizza). Missing terms and rejected masks remain in the tables and their RGB errors remain in evaluation.

Available oracle gain is 0.0311; full-surface oracle regret is 0.0267. Selection gain equals available gain minus regret. The oracle uses hidden truth and is unavailable during inference.

Full surface improves on the pool mean in 84/159 cases, ties in 0, and loses in 75. It falls back to seed 11 in 1 pools. These counts include every test case.

Mean held-out mask recall is 0.731 for full surface versus 0.688 for uniform selection. Visible RGB MAE is 0.0313 versus 0.0315. These diagnostics do not certify full amodal geometry.

The simple visible-pixel-preservation control has hidden MAE 0.1643. It selects using known visible RGB only, under the same validity guards, and was declared before test errors. It is an additional diagnostic control, outside the frozen development selection menu.

Uniform selection restricted to exactly the candidates valid for full surface has MAE 0.1691. Full-surface gain beyond these guards alone is +0.0036 [-0.0011, +0.0082]. This separates the mask-validity filter from the appearance-ranking contribution.

The scatter removes between-image difficulty by centering errors and standardizing scores within candidate pools. Its correlation is descriptive. A low optimized compatibility loss alone is not evidence of better hidden reconstruction.

Post-hoc exact decomposition: pooled total gain +0.0043 = filter contribution +0.0007 + appearance-ranking contribution +0.0036. Remaining regret relative to the best available candidate is 0.0267, comprising ranking regret 0.0266 and guard-exclusion regret 0.0001.

Full surface orders pairs of valid candidates correctly with mean per-image concordance 0.532 [0.493, 0.569], n=157. Score ties receive half credit; error ties are omitted. Chance ordering is 0.5. This explanatory statistic never changes selection.

Local surface orders pairs of valid candidates correctly with mean per-image concordance 0.516 [0.475, 0.556], n=157. Score ties receive half credit; error ties are omitted. Chance ordering is 0.5. This explanatory statistic never changes selection.

On identical evaluation-only support for the source truth and selected RGB, the selected reconstruction scores more compatible than the actual source in 51/159 cases. This shows that compatibility can prefer a reconstruction with pixel error even after removing the changing-mask confound. It does not identify the visual cause without inspection.

The exploratory paired local-seam gain over visible-pixel preservation is +0.0016 [-0.0050, +0.0085]. A benefit over random selection alone does not establish superiority to this simpler observable control.

Directly comparing local and full rules gives paired MAE gain +0.0029 [-0.0024, +0.0085]. The current data do not establish local superiority to full surface. A positive interval for one method and an interval crossing zero for another are not themselves a significant difference between methods.


Masked spatial LPIPS provides a separate perceptual check on 159 complete pools: full-surface gain +0.0089, 95% CI [-0.0010, +0.0192]; local-surface gain +0.0187 [+0.0093, +0.0288]. Positive is better. This is not a human preference score. Both images are neutral outside the held-out core; spatial AlexNet v0.1 LPIPS is averaged on that core. Mask edges affect receptive fields.

Pilot test LPIPS gains, n=72: Full surface +0.0032 [-0.0139, +0.0207]; Local surface +0.0191 [+0.0036, +0.0355].
Replication LPIPS gains, n=87: Full surface +0.0137 [+0.0021, +0.0251]; Local surface +0.0184 [+0.0065, +0.0302].

The exploratory historical audit scores 1347/1390 outputs. For pre-existing assistant hidden-inconsistency labels, full-surface loss has descriptive AUC 0.522 on 1102 assessable scored cases. These are one-candidate historical outputs with assistant labels, not an independent human evaluation or a reranking test.

## Compute and failures

Generated records: {'generated': 732}; 732 candidates. Median candidate time including generation, segmentation and scoring 14.54s; median scorer time 1.402s; summed candidate-processing time 2.96 GPU-worker hours, excluding initial model loading. Peak allocated VRAM 8.90 GiB. Four candidates require four sampling calls; a lower error than K=1 alone cannot isolate the scorer.

## Examples and complete evidence

[Open all held-out cases](gallery/index.html). The gallery exposes every candidate, support masks and the deterministic choices. The first three examples are the largest loss, largest gain and middle-ranked result. Additional examples and visual descriptions are explicitly post-hoc assistant observations, not independent human ratings.

### Largest full-surface loss (selected by measured error)

bowl (Pilot test). Full surface selects seed 22, while the unavailable hidden-MAE oracle selects seed 33. Gain versus the pool mean is -0.0888. The gallery exposes every case, including all losses. The full rule selects seed 22, whose generated upper bowl interior is visibly darker than the source. It has the lowest global color-histogram loss, whereas seed 33 has the best mean boundary term and much lower held-out error. The local rule chooses seed 33. Seed 22 covers essentially all held-out target pixels, so this particular error is not explained by missing-mask coverage alone. The example shows why matching a global distribution can conflict with the appearance of a particular part.

![bowl](gallery/laion-023-3c05dee92a.jpg)

![Weighted score explanation](figures/explanation-laion-023-3c05dee92a.png)

### Largest full-surface gain (selected by measured error)

toilet (Replication). Full surface selects seed 44, while the unavailable hidden-MAE oracle selects seed 44. Gain versus the pool mean is +0.1324. The gallery exposes every case, including all losses. Seeds 11 and 33 retain large gray regions where the held-out source contains white ceramic. Every candidate has hidden-mask recall 1.00, so the validity guard accepts these RGB failures. The full score selects seed 44, which continues the white surface and has the smallest color-histogram loss. Seed 22 is another plausible white completion, selected by the local rule, but differs more from the original opening and interior. This is a concrete case where appearance ranking helps beyond accepting a mask.

![toilet](gallery/COCOA-017-7ea421952f.jpg)

![Weighted score explanation](figures/explanation-COCOA-017-7ea421952f.png)

### Middle-ranked full-surface outcome

giraffe (Replication). Full surface selects seed 11, while the unavailable hidden-MAE oracle selects seed 33. Gain versus the pool mean is +0.0011. The gallery exposes every case, including all losses. The candidates retain the recognizable brown-and-white patch pattern, but their generated neck and head positions differ from the source. All four hidden pixel errors remain high and close together (0.2485-0.2610). Full surface selects the smallest boundary loss, while the local rule finds the pixel-error oracle, yet even the oracle has little headroom over the pool mean. Surface compatibility alone does not identify the original pose, and this pool cannot provide a low-error reconstruction.

![giraffe](gallery/VG-017-7e03cf5392.jpg)

![Weighted score explanation](figures/explanation-VG-017-7e03cf5392.png)

### A pixel-error gain can retain a visible structural defect

horse (Pilot test). Full surface selects seed 33, while the unavailable hidden-MAE oracle selects seed 33. Gain versus the pool mean is +0.1185. The gallery exposes every case, including all losses. Seed 33 extends dark target structure into the hidden side; the other candidates mostly replace that area with water. Its held-out mask recall is 0.74, compared with 0.01-0.04 for the alternatives, and it has lower pixel error. However, it also introduces a conspicuous pale upright protrusion on the right. The numerical gain is real under the fixed metric, while the completed structure still has visible defects. This example is included to prevent equating a metric win with a correct completion.

![horse](gallery/free_images-020-f312566e68.jpg)

![Weighted score explanation](figures/explanation-free_images-020-f312566e68.png)

### All-invalid pool: an explicit fallback, not a scored success

pizza (Replication). Full surface selects seed 11, while the unavailable hidden-MAE oracle selects seed 11. Gain versus the pool mean is +0.0039. The gallery exposes every case, including all losses. Every candidate leaves the hidden side without meaningful predicted target support: the predicted hidden masks contain only 0-41 pixels, below the fixed 64-pixel guard. All surface losses are undefined. FULL* and LOCAL* therefore mean the declared seed-11 fallback, not a successful score-based choice. The fallback happens to have the lowest pixel error in this poor pool, but its error is still 0.3766. The candidate generator did not supply a satisfactory completion to select.

![pizza](gallery/laion-009-acb3bbb567.jpg)

![Weighted score explanation](figures/explanation-laion-009-acb3bbb567.png)

## Limits and next decision

- The experiment uses pseudo-occlusion of observed pixels. Original SAM masks can contain errors; the original target need not be fully visible. Crop construction uses pre-occlusion support. Real hidden appearance remains unobserved.
- Pixel error penalizes plausible alternative geometry and pose. LPIPS is a supporting metric with an explicit masked protocol; no independent human preference ratings were collected. The historical census uses prior assistant judgments.
- Candidate masks vary. Guards reject missing/tiny hidden regions and severe loss of visible support, but do not prevent every mask-based shortcut or semantic error. Error is therefore measured on fixed held-out support.
- These results apply to a fixed four-seed menu and one sampling step. They do not establish a new full benchmark score, full progressive-loop behavior, a trained method or superiority to current research methods.
- Preserve the frozen test sets. Any revised scorer should be tuned on development data and evaluated on a new independent cohort. The next useful question is whether local evidence and an explicit material-change uncertainty model improve generalization without selecting smaller or smoother objects.

## Reproducibility checks

Pilot pinned-runtime replay: 384 candidate metrics, 1032 policy decisions and 1440 artifact checks; maximum raw-score difference 0. Preparation was replayed from original source pixels and SAM masks; transport hashes were independently checked.

Pilot numerical portability audit under NumPy 2.5.4, Pillow 12.3.0: changed decisions {'boundary_color': 1}. Largest raw-term difference 0.04718. Frozen decisions and original reported scores remain unchanged.

Replication pinned-runtime replay: 348 candidate metrics, 957 policy decisions and 1305 artifact checks; maximum raw-score difference 0. Preparation was replayed from original source pixels and SAM masks; transport hashes were independently checked.

Replication numerical portability audit under NumPy 2.5.4, Pillow 12.3.0: changed decisions none. Largest raw-term difference 0.03378. Frozen decisions and original reported scores remain unchanged.

Exact scorer replay used Python 3.10, NumPy 1.23.5 and Pillow 9.5.0. Do not loosen the replay tolerance or replace the frozen results after upgrading numerical packages. The portability audit changes one boundary-plus-color pilot decision; full and local decisions are unchanged in both cohorts. A CPU replay verifies saved generation artifacts; it does not guarantee regeneration on arbitrary hardware.


Tables: [candidate scores and metrics](tables/candidates.csv), [selection decisions](tables/selections.csv), [method summary](tables/pooled_methods.csv).

Additional tables: [separate cohorts](tables/cohort_methods.csv), [sources](tables/source_methods.csv), [perceptual diagnostics](tables/perceptual_methods.csv).

The [theory derivations](../../../docs/SURFACE_CONSISTENCY_THEORY.md) distinguish an observable compatibility prior, a ranking condition and an identifiability limit. Post-hoc explanation code does not alter the frozen experiment.

Reproduction and job records are in the parent experiment folder. Source, input and output checksums are retained; the original benchmark and historical outputs remain separate.
