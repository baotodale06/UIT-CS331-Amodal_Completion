# Annotated-vocabulary Surface-PMC: mathematical methods

This appendix describes the implementation and estimands frozen for the
12 November 2026 transfer experiment. It adds no experiment, tuning decision,
endpoint, or empirical claim. The original protocol and reference amendment
remain authoritative. Numerical results belong in the completed cohort report.

## 1. Inputs, annotation use and the comparison

For target i, let X_i be the observed image, V_i its initial SAM-visible mask,
q_i its category query, and Q_i the scene detector vocabulary. Annotated mode
extends the released vocabulary with missing categories from **all official
annotations for that image**. Worker assignment and cohort selection do not
change Q_i. Base categories are chunked within the detector's 256-token budget;
each added category is queried separately. This is category-annotation-assisted
detection, not a detector operating without knowledge of annotated categories.
Neither arm receives true hidden RGB or ground-truth amodal masks at inference.

Both arms share Q_i, the semantic matcher v2, model checkpoints, initial seed,
runtime, and exact visible-reference bytes. Control A is the single-candidate
progressive PD-MC baseline. Method B uses eight candidates per progressive pass,
a transferred selector, observation-anchored RGB correction, and the upstream
progressive stopping check. Both allow iteration indices 0 through 5.

The primary estimand is the **complete intervention's** mean paired change on
the frozen 204 targets. Its components include sampling budget, candidate seeds,
selection, correction, resegmented masks, and subsequent trajectories. It is
not an equal-compute estimate of the selector alone.

## 2. Progressive candidate selection

At pass t, the current crop and occlusion state determine a shared diffusion
input. Candidate k has RGB C_itk and a target-only segmented mask M_itk.
Its noise seed is deterministically derived from target seed s_i, pass t and
slot b_k in {11,22,33,44,55,66,77,88}:

$$s_{itk}=\operatorname{uint64}_{LE}\!\left(
\operatorname{SHA256}(\texttt{surface-progressive-v1:}s_i\texttt{:}t\texttt{:}b_k)_{0:8}
\right)\bmod(2^{63}-1).$$

Initial detection and progressive occlusion checks use Q_i. Candidate mask
extraction uses the original target-only segmenter captured before installing
the annotated-scene wrapper. The latter otherwise ignores the requested narrow
query, so capturing the original segmenter is necessary for this transfer.

The visible anchor always comes from X_i and V_i, aligned into the current crop.
For resizing, RGB is multiplied by V_i before bilinear interpolation and divided
by the bilinearly resized mask weight (floor 1e-6). The support mask uses nearest
neighbor resizing. Generated pixels from a previous pass never become measured
observations for feature extraction or correction.

Let f_itk contain the 22 frozen observable features. Missing/nonfinite entries
use the corresponding training median m_j; centering and scaling use the frozen
training parameters mu_j and sigma_j:

$$z_{itkj}=\frac{\operatorname{impute}(f_{itkj},m_j)-\mu_j}{\sigma_j},
\qquad r_{itk}=\sum_{j=1}^{22}w_jz_{itkj}.$$

$$k^*_{it}=\begin{cases}
\arg\min_{k\in\mathcal V_{it}}r_{itk},&\mathcal V_{it}\ne\varnothing,\\
0,&\mathcal V_{it}=\varnothing.
\end{cases}$$

Here V_it denotes candidates passing the frozen validity checks. Ties select
the first index. All-invalid pools retain the logged candidate-zero fallback.
At candidate scoring resolution, validity requires at least 64 visible pixels,
64 predicted generated pixels, visible-mask recall at least 0.5, and at least
16 pixels on each side of the boundary band formed with dilation radius 5.
Texture features can still be missing and then use frozen median imputation.
The features cover boundary/color/texture agreement, local agreement, visible
preservation, expansion, mask disagreement, pool consensus, observed perceptual
features, and category-text cosine. Exact feature order, weights, imputation,
centering, and scaling are in `method-freeze.json` and `ranker-parameters.csv`.

## 3. What was trained, and what transfers unchanged

The selector was previously fitted on 183 development images with artificial
occlusion. The generator, detector and feature networks were pretrained. This
annotated transfer does not update any of their parameters or the selector.
Excluding all 477 previously used controlled-study images is broader than
excluding only the 183 fitting images.

For clarity, the historical fitting operation in
`scripts/develop_surface_multimetric.py` is within-pool-centered ridge regression.
The target for each previously corrected development candidate was

$$y_{ik}=\frac{e^{MAE}_{ik}}{a_{MAE}}
+\frac{e^{spatial\ LPIPS}_{ik}}{a_{sLPIPS}}
+\frac{e^{object\ LPIPS}_{ik}}{a_{oLPIPS}}
+\frac{e^{DreamSim}_{ik}}{a_D}
-16\frac{c^{CLIP}_{ik}}{a_C}.$$

The a values are development candidate means stored in the frozen model.
Centering both features and targets within each pool removes its common offset:

$$\tilde z_{ik}=z_{ik}-K^{-1}\sum_lz_{il},\qquad
\tilde y_{ik}=y_{ik}-K^{-1}\sum_ly_{il},$$

$$\hat w=\arg\min_w\sum_{i,k}(\tilde y_{ik}-\tilde z_{ik}^{\mathsf T}w)^2
+100\lVert w\rVert_2^2
=(\tilde Z^{\mathsf T}\tilde Z+100I)^{-1}\tilde Z^{\mathsf T}\tilde y.$$

The implementation solves the linear system. It does not explicitly invert
the matrix. At inference, subtracting a common pool mean would not change the
linear ranking, so the exported ranker uses standardized features directly.

The development target contains supervised hidden-region errors, available
under artificial occlusion. These are training labels, absent from inference.
Natural-occlusion LPIPS is a different evaluation quantity. CLIP is both a
strongly weighted historical training target and an inference feature; observed
LPIPS features also share the LPIPS family with the primary evaluation metric.
Neither is a fully independent perceptual assessor of the learned method.
No generalization theorem follows from the ridge objective or its regularizer.

## 4. Frozen RGB corrections

Selection uses raw-candidate features. The output RGB is then corrected. For
each candidate, the affine fit uses an eroded overlap of visible and predicted
support. If that support has fewer than 64 pixels, the implementation falls
back to a one-pixel erosion of visible support; if fewer than 64 remain, it skips
the affine adjustment. Visible restoration still applies.

For each color channel, write x=C_k on the fitting support and y=X there, with
channel medians m_x,m_y. The regularized slope and bounded correction are

$$a=\operatorname{clip}\!\left(
\frac{\operatorname{mean}[(x-m_x)(y-m_y)]+400}
{\operatorname{mean}[(x-m_x)^2]+400},0.8,1.25\right),$$

$$\delta_k(p)=\operatorname{clip}((C_k(p)-m_x)(a-1)+m_y-m_x,-32,32).$$

The frozen strength is 1. The correction applies to M_k outside original
visible support. Values are rounded and clipped to uint8; visible pixels are
restored from X. These bounded changes are a prior for transferring observed
color drift into generated areas, not evidence that hidden appearance matches.

Next, the calibrated pool supplies a channelwise median T(p) using only masks
that cover p. Consensus support requires at least four of eight masks. For
candidate k, let S_k be that support intersected with M_k. Gaussian smoothing
has sigma 4 and uses a normalized masked residual:

$$u_k(p)=\frac{G_4*(S_k(T-C'_k))(p)}{\max(G_4*S_k(p),10^{-6})},$$

$$C''_k(p)=C'_k(p)+0.5\operatorname{clip}(u_k(p),-32,32)
\quad\text{for }p\in S_k\setminus V.$$

Other pixels retain C'_k, followed by rounding/clipping. The selected corrected
image is upsampled, measured native visible RGB is restored, and the progressive
pipeline resegments it. Original visible equality is checked at saved candidate
resolution. JPEG serialization can change native RGB afterward. The final mask
and trajectory are therefore not assumed equal to the raw candidate's mask.

## 5. Evaluation quantity and failure accounting

The benchmark reference is a SAM-visible RGBA crop, not the true complete
object. Transparent pixels become white. The released metric alignment takes
the bounding box of reference alpha < 255, with each output's recorded crop
geometry and the released resizing operations. LPIPS uses AlexNet features;
VGG similarity uses VGG16 features; SSIM retains the released dynamic data range.
CLIP is the model's image/text **logit** for the completion and category label,
not a cosine and not a comparison to hidden truth. Exact-copy metric sentinels
are retained: LPIPS 0 becomes 1, VGG similarity 1 becomes 0, and SSIM 1 becomes 0.

For all 204 targets, failure contributes LPIPS 1, VGG similarity 0, SSIM 0. No
CLIP score is invented for a failure. The primary LPIPS mean includes upstream
detection, matching, and completion failures. Saved-output status includes
iteration-limited outputs and is not a claim of full visual completion.

For a metric with direction h=-1 for LPIPS and h=+1 for similarities,

$$d_i=h(B_i-A_i),\qquad \hat\Delta=N^{-1}\sum_i d_i,\qquad
\hat g=100\hat\Delta/|\bar A|.$$

CLIP's paired N is the number of shared successes; other primary-table metrics
use N=204. Separate success-only CLIP means can have different denominators and
do not estimate the paired effect. Failure-inclusive LPIPS gain decomposes
exactly into shared-success, recovered-success, lost-success and both-failed
contributions, each divided by the same 204-target denominator.

## 6. Uncertainty and scope of inference

The sole primary test is failure-inclusive LPIPS. Each of 5,000 bootstrap
replicates samples source images with replacement within each source, retaining
all targets from every sampled image. Each source retains its original number
of sampled images; target counts may vary because images have different numbers
of targets. The statistic remains a target-weighted ratio of sums. Percentiles
2.5 and 97.5 form the nominal two-sided 95% interval. RNG seed is 20261110.

Evidence for the predeclared improvement requires the absolute LPIPS interval
to lie above zero. A relative interval is undefined if any bootstrap control
mean is zero; such draws are counted, not discarded. Secondary/source intervals
are exploratory, without multiplicity correction. There is no 3% pass threshold.

The bootstrap conditions on this trained method, these frozen candidate seeds
and the selected source mixture. It does not measure uncertainty from retraining
or different diffusion seeds. Source-stratified sampling improves the stated
sample coverage; it is not a guarantee for arbitrary future image populations.
The exclusion/near-copy screen also cannot rule out every transformed duplicate.

## 7. Interpreting the saved-pool diagnostic

For a final saved pool and a lower-is-better metric, let e_k be each raw
candidate's score under the fixed corrected final mask and preceding trajectory.
The uniform policy score is the exact expectation K^-1 sum e_k, not an average
RGB image. The valid-only expectation uses the frozen valid set and candidate
zero if the set is empty. The semantic-only choice maximizes category-text
cosine within that same set, breaking ties by index.

At that fixed state,

$$\bar e-e_{k^*}=(\bar e-\bar e_{valid})+(\bar e_{valid}-e_{k^*})$$

separates filtering from ranking, and

$$\bar e-e_{k^*}=(\bar e-\min_ke_k)-(e_{k^*}-\min_ke_k)$$

separates available pool headroom from selected regret. Similarity signs reverse.
Corrected versus selected raw scores quantify the last RGB correction under
that fixed mask/trajectory. Reference-informed oracle choices are per-metric
upper bounds, not deployable selectors or a jointly achievable score vector.

Earlier progressive paths and final masks were chosen by the corrected method.
This diagnostic conditions on them and can favor that trajectory. It does not
simulate an equal-budget progressive random policy. Keep it secondary to the
complete frozen-cohort comparison and do not tune on its outcomes.

## 8. Implementation references

The mathematical operations above map to `amodal_benchmark/detection.py`,
`surface_runtime.py`, `surface_policy.py`, `surface_refinement.py`,
`surface_ensemble.py`, and `metrics.py`; the historical fit is in
`scripts/develop_surface_multimetric.py`. Frozen hashes and parameters are in
`method-freeze.json`. The paired analysis and independent replay use
`compare_surface_progressive.py`, `audit_annotated_surface.py`, and
`audit_annotated_surface_pools.py`. Native candidate/reference images, execution
snapshots, and independent archive/member checksums are retained separately
from the compact report package.
