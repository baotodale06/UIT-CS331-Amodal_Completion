# Surface-Consistent PMC: durable research reference

Recorded 2026-11-05 at the user's request. This note preserves the idea for future
questions without depending on conversational recall or rereading the HTML.

**2026-11-11 measured update:** the independent
[128-image confirmation](SURFACE_CONFIRMATION_20261110.md) now establishes a
controlled one-step raw-selection result: MAE +7.644%, object LPIPS +11.328%,
DreamSim +16.766%, CLIP cosine +4.068% versus expected uniform selection from the
same eight candidates. Each individual nominal 95% lower bound exceeds 3%; SSIM
improves only 1.668%. All decisions and selected-image MAE were independently
audited. This is artificial occlusion on VG/COCOA. The separate full corrected
progressive comparison is now complete: CLIP +1.123%, LPIPS +0.835%, VGG +2.789%,
SSIM +1.782%, below the requested 3% full-benchmark threshold. Read the
[full report](../output/surface-progressive-review-20261111/REPORT.md).
A frozen diagnostic of saved final raw RGB is queued; its masks and preceding
trajectory remain those of the corrected run. Historical proposal/status
statements below describe their original dates, not the current measured result.

**2026-11-08 continuation:** the retained proposal below is now being tested in a
separate [controlled experiment](SURFACE_CONSISTENCY_EXPERIMENT.md), with an
implemented observation-anchored scorer, shared candidate pools and a frozen
development/test protocol. The original status statements describe the note's
2026-11-05 recording; consult the continuation for subsequent implementation and
measured evidence. The [completed results](SURFACE_CONSISTENCY_RESULTS.md) report
the inconclusive full rule, exploratory local benefit and explicit failure cases.
Historical source wording remains unchanged below.

Source: `ChatGPT - Assess Reproduction Feasibility.html`, primarily section 3,
with the PMC integration, experiment design and feasibility context from sections
1 and 4-8. Original SHA-256:
`a398b6d068bd7abb2e7aee9453c6796f2c486d9b5143624b2c0112f624eeed3f`.
The [focused source extract](research/SURFACE_CONSISTENT_PMC_SOURCE.md) preserves
the Vietnamese wording, all equations and code text in section 3, and the relevant
shared context. The original HTML was retained unchanged.

**Status:** an unvalidated research proposal from the supplied conversation. It
is not a method published by the original PMC authors, an implemented feature of
this repository, or a demonstrated improvement. No model, inference code,
configuration or experiment result was changed while recording this knowledge.
Sections 1-9 explain the supplied proposal; section 10 explicitly records
additional reasoning and unresolved implementation choices; section 11 connects
the proposal to the current repository.

## 1. Core idea and research question

Keep the pretrained Progressive Mixed Context Diffusion (PMC, called PD-MC in
this project's evaluation) generator. Generate several possible completions of
the same partially occluded object, and select the candidate whose generated
surface is most compatible with the object's originally visible appearance.
The practical first version changes candidate selection, without retraining
diffusion or changing its denoising algorithm.

The source's research question is:

> Can appearance features extracted from the visible target object guide diffusion
> candidate selection toward more visually consistent hidden-region reconstruction?

A red-and-white striped shirt can have a plausible completed silhouette while
its generated part becomes solid red or its stripes stop at the seam. Surface
consistency concerns color, texture, pattern and continuity across that seam.
Correct geometry does not imply correct appearance.

The original PMC already conditions on visible object pixels. The proposed
contribution is an explicit appearance-based selection criterion over multiple
generated candidates. Shape-Guided PMC is a separate proposal that evaluates
silhouette/geometry against an amodal shape prior. Surface-Consistent PMC does
not require that predictor. The two scores may later be combined.

## 2. Where it fits in PMC

PMC does not require an independently predicted complete object mask before
generating RGB. It analyzes occluders, prepares a crop and masks, pads boundaries
when necessary, performs Mixed Context inpainting, segments the generated result,
and checks whether progressive completion should continue.

The proposal places selection inside a progressive pass:

```text
Input image + visible target mask + occluder mask
  -> original run_iteration() preparation: crop, padding, masks, prompt
  -> run_mixed_context_diffusion(): K candidates with different seeds
  -> segment the target instance in every candidate
  -> compute surface losses and select the lowest-loss valid candidate
  -> check_occlusion(selected candidate): update/check progressive state
  -> next pass if needed, otherwise completed RGB and object mask
```

All candidates in a pool use the same input, masks, prompt and diffusion settings;
only their seed/noise differs. The source suggests K=4 as an experimental target
and seeds `[11, 22, 33, 44]` as an example, not required or paper-established
values. Start with K=2 when checking resource use. Generation is sequential at
first. The source mentions released `mc_timestep=35`; that is not a new
surface-specific hyperparameter.

Generate wrappers such as `generate_pmc_candidates()` and target-segmentation
wrappers in the source are architectural pseudocode, not existing APIs. The seed
must actually reach the custom diffusion pipeline. Segment the query instance,
not every detected object indiscriminately. Reusing its mask in the subsequent
occlusion check is a possible optimization after behavior is established.

## 3. Regions, coordinates and notation

| Symbol | Meaning |
| --- | --- |
| I / I_occ | Original observed image with occlusion |
| V | Originally observed visible target-object mask |
| O | Occlusion region relevant to completion |
| I_k | Generated candidate k |
| M_k | Segmented target-object mask in candidate k |
| H_k | Candidate's generated hidden target region |
| B_v, B_h | Narrow visible-side and hidden-side bands near their interface |
| K | Number of candidates generated for a selection step |

For in-image occlusion, the source defines:

\[
H_k=(M_k\cap O)\setminus V.
\]

The whole occluder mask is not the hidden target: some of it covers background.
For outpainting, newly generated target pixels outside the original image must
also be included after aligning the canvas. The source does not fully specify
that extension; it must be defined explicitly in an implementation.

All images and masks must share coordinates and resolution before scoring.
PMC crops, pads and resizes; an original-image mask cannot be compared directly
with a 512-by-512 candidate mask. Preserve the transformations, including the
original visible support, throughout the progressive loop.

## 4. The surface scorer

The proposal combines three losses:

\[
\mathcal L_{surface}(I_k)=
\lambda_b L_{boundary}(I_k)+
\lambda_c L_{color}(I_k)+
\lambda_t L_{texture}(I_k).
\]

\[
Q_{surface}=-\mathcal L_{surface},\qquad
k^*=\arg\min_k\mathcal L_{surface}(I_k).
\]

Higher score and lower loss mean the same selection. The example weights are
`lambda_b=0.40`, `lambda_c=0.25`, `lambda_t=0.35`. These are starting suggestions,
not validated settings. Normalize the loss scales using validation data before
combining them; freeze normalization and weights before test evaluation.

### 4.1 Boundary continuity: implement this first

Using a dilation radius r:
\[
B_v=\operatorname{Dilate}(H_k,r)\cap V,\qquad
B_h=\operatorname{Dilate}(V,r)\cap H_k.
\]

These bands lie on the two sides of the visible/generated target interface, not
the object's entire outer silhouette. The source suggests r=3-8 pixels for a
512-by-512 crop; its example uses r=5 and an elliptical `(2r+1)` kernel.

Convert the candidate RGB image to Lab and compare the bands' mean colors:

\[
L_{boundary}=
\frac{\|\mu_{Lab}(B_v)-\mu_{Lab}(B_h)\|_2}{C}.
\]

The source's minimal implementation:

```python
import cv2
import numpy as np

def boundary_color_loss(candidate_rgb, visible_mask, hidden_mask, radius=5):
    v = visible_mask.astype(np.uint8)
    h = hidden_mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    visible_band = (cv2.dilate(h, kernel) > 0) & (v > 0)
    hidden_band = (cv2.dilate(v, kernel) > 0) & (h > 0)
    if not visible_band.any() or not hidden_band.any():
        return None
    lab = cv2.cvtColor(
        candidate_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB
    ).astype(np.float32)
    return float(
        np.linalg.norm(
            lab[visible_band].mean(axis=0)
            - lab[hidden_band].mean(axis=0)
        ) / (np.sqrt(3) * 255.0)
    )
```

This is a simple baseline, not a complete surface metric. The denominator matches
the example's OpenCV 8-bit Lab representation. The code assumes RGB values on a
0-255 scale; blindly casting an image on a 0-1 scale to uint8 is incorrect. It is
not a calibrated Delta-E-2000 metric.

A mean-color mismatch can be legitimate: different materials, shirt versus
trousers, shading, or a stripe crossing the boundary. The source suggests local
patch correspondence and restricting the constraint to regions with evidence
that they belong to the same surface.

### 4.2 Color distribution

Build normalized color histograms from visible and generated target pixels:

\[
h_v=\operatorname{Hist}(I,V),\qquad
h_h=\operatorname{Hist}(I_k,H_k),\qquad
L_{color}=D_{hist}(h_v,h_h).
\]

Suggested color spaces are Lab or HSV. Suggested distances are Bhattacharyya,
Jensen-Shannon or Wasserstein. The example uses
`cv2.compareHist(hist_visible, hist_hidden, cv2.HISTCMP_BHATTACHARYYA)`.
The source does not specify bins or a final distance choice. Histograms are a
supporting signal: different object parts need not have identical colors.

### 4.3 Texture and pattern

Color alone cannot distinguish a red striped surface from a plain red surface.
The source describes three levels:

| Level | Features/comparison | Proposed role |
| --- | --- | --- |
| Handcrafted | Sobel gradients, LBP, Gabor filters, gradient-orientation histograms; distance between feature distributions | Lightweight, debuggable first baseline |
| Pretrained deep features | DINOv2/DINOv3 or VGG patch features; cosine distance, nearest-neighbor patch similarity or feature statistics | Add selected features after the simple scorer works |
| Learned compatibility scorer | Small network on visible/hidden patch pairs | Optional; requires training data and negative-pair design |

The abstract formulation is
`L_texture = D(T(I_visible), T(I_hidden))`, or features
`F_v=E(I_visible)` and `F_h=E(I_hidden)` for a pretrained extractor.
Use masked patch features and appropriate spatial alignment. Whole-crop embedding
cosine similarity alone can confuse different semantic parts and background.
The eight-week recommendation is level 1 plus part of level 2; a new training
pipeline is not required for the main proposal.

### 4.4 Selection and diagnostics

The source skeleton constructs H_k for each candidate, computes boundary loss,
then color and texture losses, combines the example weights, and uses argmin.
It assigns infinity when boundary bands are unavailable. Its
`color_histogram_loss()` and `texture_feature_loss()` are placeholders to implement.

Save the source's proposed CSV fields separately:

```text
sample_id, seed, boundary_loss, color_loss, texture_loss, total_loss, selected
```

Individual terms make it possible to diagnose which signal chose a worse image.
Invalid masks, missing bands and all-invalid pools need an explicit policy;
the source skeleton does not provide a complete one (see section 10).

## 5. Experiments and what would establish success

The main claim to test is better hidden-region appearance than a control using
the same number of candidates, without materially degrading shape plausibility.
The scorer cannot create a good candidate if the generator never produces one.

| Experiment | Purpose |
| --- | --- |
| Original PMC, K=1 | Reproduction baseline |
| PMC, K=4, random candidate | Same sampling-budget control described in section 3.5 of the source |
| PMC, K=4 generated, predetermined fixed seed selected | Alternative control described in section 5 of the source |
| Boundary only | Isolate seam continuity |
| Boundary + color | Measure added color information |
| Boundary + texture | Measure added texture information |
| Full boundary + color + texture | Main Surface-Consistent PMC proposal |
| Shape only / joint shape and surface | Optional complementary comparison |

Predeclare which control is used. Random and predetermined selection are different
protocols; neither means oracle selection using hidden ground truth. Compare
methods on the same candidate pools when evaluating the selection rule itself.
The source calls for the same four seeds for each image across the improved
methods. Progressive-state caveats to this requirement are in section 10.

Do not attribute a K=4-versus-K=1 gain solely to the surface score. Report sampling
cost and scorer overhead. Add qualitative examples, failure cases, and human
preference evaluation rather than relying on a single aggregate score.

### Ground truth and leakage

For controlled pseudo-occlusion, start with an actually fully visible target,
retain its original RGB and full mask S_GT, and overlay an occluder O:
\[
V=S_{GT}\cap(1-O),\qquad H_{GT}=S_{GT}\cap O.
\]

A COCO instance mask is normally a visible mask, not inherently an amodal mask.
Exclude already occluded or frame-truncated targets when constructing this
complete-object ground truth. Otherwise the alleged hidden ground truth is wrong.

Test-time surface scoring receives only observed appearance, legitimate inference
masks and generated candidates. It must not inspect the original unoccluded test
RGB, full ground-truth shape or hidden target pixels. Full RGB and masks are for
evaluation, validation and any separately declared training. Ground-truth shape
selection is an oracle diagnostic, not a deployable method's headline result.

### Metrics

| Metric | What it evaluates |
| --- | --- |
| Hidden-region MAE, lower is better | RGB error where the target was synthetically hidden |
| Hidden-region LPIPS / DreamSim distance, lower is better | Perceptual reconstruction, with a declared aligned crop/masking protocol |
| Masked color/texture discrepancy | Appearance agreement under a defined region rule |
| Boundary discrepancy / seam-artifact rate | Continuity at the visible/generated interface |
| Human preference | Plausibility when multiple completions can be valid |
| Amodal IoU, higher is better | Geometry against an available complete target mask |
| CLIP similarity, higher is better | Semantic agreement; not an exact surface metric |
| Inference time and peak VRAM | Cost of generation and selection |

The source's hidden-region MAE is:

\[
\operatorname{MAE}_{hidden}=
\frac{\sum_{x,y,c}H_{GT}(x,y)
|\hat I(x,y,c)-I_{GT}(x,y,c)|}{3|H_{GT}|+\epsilon}.
\]

LPIPS and DreamSim need an explicit, consistent masked/aligned evaluation design;
the source does not provide one finished implementation. Whole-image background
can dominate perceptual scores. Real occlusions generally do not reveal the true
hidden appearance, and a plausible completion need not exactly match the one
ground-truth image available in a synthetic test.

## 6. Optional joint shape-and-surface selection

The proposed joint name is **Shape-and-Surface-Aware Progressive Mixed Context
Diffusion (SS-PMC)**. This also is a proposed group name, not a published method.

\[
Q_{joint}(I_k)=
\lambda_s\widetilde Q_{shape}(I_k)
-\lambda_a\widetilde{\mathcal L}_{surface}(I_k).
\]

Select the largest joint score. Tildes mean validation-normalized quantities.
Shape rewards geometry; surface rewards appearance. Both use the same candidate
pool. Complementarity is a hypothesis: the scores may conflict, and combined
selection can be worse than either separately. Surface-only work does not depend
on implementing the other person's shape predictor.

## 7. Optional extension: guidance during diffusion

Reranking uses appearance only after candidate generation. A harder extension
would influence denoising itself. The source gives this conceptual expression:

\[
\epsilon_{guided}=
\epsilon_\theta(z_t,t,c)-\eta_t\nabla_{z_t}\mathcal L_{surface}.
\]

This is an illustration of gradient guidance, not a validated scheduler update or
a drop-in patch. The actual sign, scale and conversion between image-space loss,
latent gradients and scheduler/noise predictions must be derived for the pipeline.
It needs a differentiable loss, gradients restricted to generatable regions,
preservation of visible pixels, compatibility with Mixed Context compositing,
and VRAM/stability checks. OpenCV histograms and hard mask operations from the
simple scorer do not directly supply that differentiable loss.

Reference-image conditioning through an image adapter is another possible avenue;
the custom pipeline may need UNet attention-processor changes. Neither this nor
gradient guidance is required to complete the initial surface-reranking project.

## 8. Feasibility and implementation order

The source assumed eight weeks, two people and a 24 GB GPU. Those are its planning
assumptions, not a measured feasibility result or a newly imposed project budget.
One 512-by-512 FP16 SD2 inpainting pass may fit, but PMC also uses Grounding DINO,
SAM, InstaOrder and LaMa. Do not assume all components and an extra feature
extractor fit simultaneously.

1. Reproduce the pinned PMC baseline and evaluation first.
2. Inspect surface failures, establish aligned masks, and try boundary reranking.
3. Add normalized color histograms, then handcrafted texture.
4. Add selected pretrained patch features only when validation supports them.
5. Run component ablations and compare against the same sampling-budget control.
6. Consider joint shape selection after the independent surface method works.
7. Reserve diffusion guidance or learned scorers for a later extension.

Start with K=2, measure runtime/peak VRAM, then try K=4 sequentially. Cache inputs,
masks, segmentation and features where valid; move auxiliary models between
stages or free GPU memory when necessary. Use `torch.inference_mode()` for frozen
inference in the reranking version. Gradient-guided sampling would instead need
autograd in its guidance path. Preserve the original environment before changing
Diffusers or its scheduler.

Expect roughly K diffusion generations per selection step plus segmentation and
scoring, not a measured exact K-times total runtime. Progressive pass counts and
caching can change end-to-end cost.

## 9. Source's surface-work schedule and decision points

| Week | Proposed surface work |
| --- | --- |
| 1-2 | Environment, PMC inference and evaluation; reproducible baseline |
| 3 | Inspect failures and boundary/color features; quantitative failure analysis |
| 4 | Boundary, color and texture scorers; working prototype |
| 5 | Refined scorer or deep texture features |
| 6 | Surface ablations and optional joint integration; comparison table |
| 7 | Qualitative examples, human evaluation and runtime; freeze results |
| 8 | Explain surface contribution and prepare the presentation |

If the scorer favors smooth but semantically wrong images by the end of week 4,
reduce global color matching and try local boundary/texture patch matching.
Retain only components that improve validation. An informative negative result,
including conflict with shape scoring, is valid research evidence. The source
sets selection methods as the main deliverables and deeper generation changes
as optional; it reports no actual experimental success.

## 10. Additional reasoning and unresolved choices

The points below are consequences and implementation cautions identified while
reading, not extra results claimed by the source.

- **Anchor to observations.** The histogram equation uses original visible pixels,
  but the supplied skeleton and boundary example read visible pixels from the
  candidate. These agree only if visible appearance is preserved. Otherwise a
  candidate could lower its loss by changing both sides. A future implementation
  must explicitly choose original aligned visible RGB as its anchor or enforce
  and verify preservation; the source leaves this discrepancy unresolved.
- **Keep original evidence separate from generated evidence.** In later passes,
  a newly completed region is not a newly observed truth. Track its provenance
  rather than silently treating it as the original visible reference.
- **Define outpaint support.** One possible explicit extension is
  `H_k = M_k & (O | P_outside_original) & ~V` on the aligned canvas. This is a
  proposed formalization here; masks for cumulative versus current-pass generated
  support need a deliberate policy.
- **Handle invalid pools explicitly.** The source's `argmin([inf, inf, ...])`
  selects index 0 without reporting the absence of a valid score, and an empty
  pool fails. Predeclare and log fallback to an existing baseline candidate or
  an explicit failure; do not treat an undefined term as a perfect zero loss.
  Empty H_k or absent seam bands can also represent a genuinely different case
  such as no newly hidden target, so distinguish reasons before penalizing it.
- **Prevent trivial solutions.** A tiny or missing segmented object, blurred
  texture or a nearly uniform fill can look color-consistent while being a bad
  completion. Track target validity, generated-region area, geometry, visible
  preservation and semantic plausibility. These checks must not peek at test GT.
- **Allow genuine appearance variation.** Hidden and visible parts need not share
  exact colors or texture because of material changes, lighting, pose and
  perspective. Patch matching can help but does not guarantee correct stripe
  phase, scale or orientation. Low histogram loss alone cannot prove continuity.
- **Equal seeds are not sufficient for equal progressive pools.** Identical
  inputs, state and seeds can define a shared one-step pool. After two selectors
  choose different images, later progressive inputs may diverge. Use cached,
  identical states/pools to isolate the scorer, and distinguish that experiment
  from end-to-end runs with a matched generation budget and seed schedule.
- **Separate selection from evaluation.** Reporting only the same boundary/color
  score optimized by selection would be circular evidence. Use independent
  hidden-GT perceptual measures where available and blinded human assessment.
- **Record all choices.** Specify histogram bins, RGB/Lab ranges, texture model
  and layer, patch-mask inclusion, normalization, weights, radius/resolution,
  tie-breaking, invalid-term handling and deterministic seeds. None has an
  established best setting in the source. Add iteration ID, validity reasons,
  selected seed, score components, timing and geometry to provenance.

## 11. Relationship to this repository

The [paper-to-code map](PAPER_CODE_MAP.md) is the reference for actual execution.
`amodal_benchmark/inference.py:complete_target()` calls the pinned upstream
`U/main.py:run_iteration()`. `U/` denotes the runtime checkout under the model/cache
directory, not this project's root. The archived
[pdmc_main.py](../datasets/open_world_amodal/references/pdmc_main.py) is a baseline
snapshot; editing that archive does not modify the active generator.

The likely experiment insertion point is between `run_mixed_context_diffusion()`
and `check_occlusion()` inside upstream iteration preparation. Existing runtime
hooks, target matching, crop geometry, completion guards, pin/hash checks and
resume provenance must be respected. The existing code map also notes that
upstream reuses original query inputs in places even after canvas updates; inspect
actual state propagation before assuming the source's conceptual progressive
pseudocode describes every released detail. `check_occlusion()` is the stopping
check, not the paper's separate counterfactual completion-curation classifier.

The [current Open-World protocol](../OPEN_WORLD_REPRODUCTION.md) measures visible
appearance preservation with SAM or LISA references, plus semantic scores and
failures. Those protocols are distinct, and neither supplies hidden-pixel ground
truth just because a reference mask exists. The proposed hidden-region MAE,
LPIPS/DreamSim and full-mask IoU experiments require appropriate additional
ground truth/evaluation design. They are not already provided by running the
current visible-reference benchmark unchanged.

Preserve the baseline, report failures and denominators, and use new output
directories for changed scientific settings. This knowledge-capture task neither
changes the active baseline/annotated experiment plan nor authorizes a remote
submission. No surface improvement has been measured here.

## 12. How to answer future questions

Read this reference before answering detailed Surface-Consistent PMC questions.
Use the focused source extract for exact wording or pseudocode and the current
code map for implementation claims. Explain which statements are source
proposals, repository observations, additional reasoning or measured results.
If a requested detail is not specified, identify it as an open design choice
rather than inventing a setting or result.

The source HTML can be absent without losing the core idea, formulas, example
scorer, experiments, schedule and limitations recorded here. The notes persist
in this workspace; they are not a guarantee of recall in a future conversation
that cannot access these files.
