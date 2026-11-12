# Appearance-failure analysis

The completed Person 2 deliverable is [the 2026-11-07 report](../output/appearance-failures-20261107/REPORT.md), with a shareable PDF, six-bin performance figures, all per-target measurements, and a 120-image visual audit. This analysis reads the accepted baseline; it does not change or rerun inference.

## What is measured

- Full baseline: 2,565 targets, 951 saved completions, 1,614 pipeline failures. A saved completion is not a certification of visual correctness.
- Annotated occlusion: the original COCOA release provides `occlude_rate`, verified independently from its visible/invisible masks. Benchmark targets are joined using original image identity and a unique exact region name, with matching image dimensions. No synonyms, best-score choices or ambiguous names are used. This gives 321 joined targets, including 114 in the requested `[20,30), ..., [70,80)` percent bins. Other targets retain unavailable ratios; they are not assigned zero.
- The COCOA ratio measures in-frame occlusion. It does not measure unknown extent beyond the image border, and there is no hidden-RGB ground truth.
- The full saved-output diagnostic is `|predicted_mask \\ visible_SAM_mask| / |predicted_mask|`. Both masks are put into original-image coordinates using recorded integer outpainting offsets. This is an output-dependent expansion proxy, not a ground-truth occlusion ratio. Visible-pixel retention is reported separately.
- Existing CLIP, LPIPS, VGG feature similarity and SSIM are read from verified caches. Their released alignment/reference protocol is unchanged. CLIP is conditional on saved outputs; the other three also have separate aggregates including failure penalties. These metrics do not measure hidden-pixel correctness.
- Median CIELAB Delta E 1976 is measured on the eroded intersection of visible and predicted masks. It describes change to known visible pixels, not color accuracy of hidden regions. No automatic threshold turns it into a taxonomy label.

## Visual-review protocol

The audit uses 15 deterministic hash-selected saved outputs in each of eight expansion strata: `[0,20)`, six requested ten-point intervals, and `[80,100]`. The seed is 20261107. Sampling uses only ID and stratum, not metric score or inspection. The 120 selected outputs span 36 canonical categories and all four sources. It is not a simple random sample of the 2,565 targets.

Each review compares original context, the visible SAM target, final masked completion, aligned mask regions and the entire raw generated RGB crop. Potential co-occurrence cases receive an additional full-source comparison. Labels were assigned by one Codex assistant from these images, without a second human reviewer. They are exploratory visual judgments, not human-consensus ground truth.

| Code | Definition | Do not automatically count |
| --- | --- | --- |
| `wrong_texture` | Material, grain, fur, gloss or surface detail conflicts with visible evidence, including substantial smoothing | Shape defects alone; plausible unseen material differences |
| `color_inconsistency` | Obvious incompatible hue/brightness or color leakage | Plausible lighting, reflection and ordinary shading |
| `pattern_inconsistency` | Surface print, markings, lettering, screen/keyboard graphics or repeated motif fails to continue consistently | Unknown hidden designs; physical slat/limb geometry by itself |
| `semantic_drift` | Generated part/content loses the recognizable identity of the intended object, including hybrid or duplicated identity cues | A pose or shape change alone; a final mask selecting an existing different object |
| `hidden_region_inconsistency` | Newly filled/output regions conflict with the object's visible surface or part content | Exact hidden-pixel truth, which is unavailable; merely omitted geometry |
| `cooccurring_object` | A new or reconstructed recognizable secondary object/accessory appears in generated RGB without visible support | An unchanged object already present in the scene; ambiguous occluder residue |

Co-occurrence subtypes distinguish new secondary objects, same-category duplication, regenerated occluders, replacement food, and unsupported accessories. Raw-RGB-only cases are distinguished from cases entering the returned target region. Semantic labels denote interpreted part/identity drift, not verified class-label changes. A changed classification score is not used as a semantic label.

Severity is `0` (no clear observed failure), `1` (mild/local), `2` (moderate/obvious), or `3` (severe/dominant). `U` means uncertain or unassessable; `NA` means no applicable visible pattern. Unknown and inapplicable cases are never treated as negatives. Multiple failure labels may apply to one image, so frequencies must not be added together. Severity describes visual impact, not a measured causal mechanism.

The separate confound field records initial-mask contamination/undercoverage, output-mask loss, target switching, low resolution, incomplete geometry, and ambiguous occluder residue. The report also recomputes taxonomy counts after removing explicit mask/identity and low-resolution confounds. Heavy occlusion is a condition of the task; shape failure is outside this appearance taxonomy.

Raw frequency is positive / assessable for each label. A second estimate applies weight `N_stratum / n_stratum` to correct the deliberate oversampling of high-expansion outputs, excluding `U` and `NA` separately for each label. It estimates a frequency among assessable saved outputs under this review, not among all targets. It does not correct reviewer errors or unknown-label bias. Wilson columns in the raw taxonomy CSV are descriptive unweighted binomial intervals; they are not design-correct population intervals and are not used for population claims. The main performance plot uses Wilson intervals for output availability and target-bootstrap intervals for metric means. Those intervals do not estimate inference-seed variability.

## Reproduce

Use a separate CPU analysis environment. `requirements-analysis.txt` records the exact Python 3.14 reporting dependencies used for this deliverable; do not install them into the pinned GPU runtime. The core analysis module needs only the project's NumPy/Pillow dependencies; plots need Matplotlib, external-mask auditing needs pycocotools/SciPy, and PDF export needs ReportLab.

```bash
python -m pip install -e .
python -m pip install -r requirements-analysis.txt
python scripts/analyze_appearance_failures.py \
  --run-dir output/baseline-31548/output/pdmc-baseline-provenance-20261105 \
  --dataset-root datasets/open_world_amodal \
  --output-dir output/appearance-failures-20261107 \
  --cocoa-annotations output/appearance-failures-20261107/external \
  --review output/appearance-failures-20261107/review/annotations.csv
python scripts/audit_appearance_evidence.py --analysis-dir output/appearance-failures-20261107
python scripts/render_appearance_report.py --analysis-dir output/appearance-failures-20261107
```

The original external annotation archive and download receipt are preserved under `external/`. Its source is the [COCOA authors' API repository](https://github.com/Wakeupbuddy/amodalAPI), citing Zhu et al., *Semantic Amodal Segmentation*, CVPR 2017. Only the three named COCO JSON files were read from the archive. Their hashes, exact target-region joins and independently decoded mask ratios are recorded; the benchmark dataset remains unchanged.

`--reuse-derived` regenerates summaries from saved per-target diagnostics. `--rebuild-panels` additionally rebuilds image sheets. Reproducing figures and counts from recorded labels does not constitute an independent visual reannotation. Analysis must be written outside the immutable baseline run.
