#!/usr/bin/env python3
"""Package the recorded appearance audit as a report, figures and review index.

Consumes existing measurements and labels; never generates or changes labels.
Optional reporting dependencies: matplotlib, markdown, reportlab, pypdf.
The narrative is specific to the frozen baseline and review identities below.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amodal_benchmark.appearance import FAILURES, read_review, taxonomy_counts, write_csv
from amodal_benchmark.dataset import sha256, write_json

EXPECTED_RESULTS_SHA256 = "885f446c0dc00266f117a5b59c0fb59a69696369f3f142688f3f052146755e2a"
EXPECTED_REVIEW_SHA256 = "22cb572cc0484e74d0035ecf3f62df29dbecbb073f9f8dba68b6c817a607f457"

NAMES = {
    "wrong_texture": "Wrong texture",
    "color_inconsistency": "Color inconsistency",
    "pattern_inconsistency": "Pattern inconsistency",
    "semantic_drift": "Semantic drift",
    "hidden_region_inconsistency": "Hidden-region inconsistency",
    "cooccurring_object": "Co-occurring object regeneration",
}
CAUSES = {
    "wrong_texture": "Material priors override local surface evidence; no explicit material-continuity constraint.",
    "color_inconsistency": "Context color leakage or diffusion repainting changes the visible object's palette.",
    "pattern_inconsistency": "No explicit phase, spacing or orientation constraint for repeated structures.",
    "semantic_drift": "Sparse visible evidence and progressive synthesis permit identity or part substitution.",
    "hidden_region_inconsistency": "Completion is weakly anchored to visible material and boundary appearance.",
    "cooccurring_object": "Scene/context priors regenerate an occluder, accessory or additional object.",
}
DEFINITIONS = {
    "wrong_texture": "Material or fine-scale surface character changes, such as fur becoming a smooth or ridged surface.",
    "color_inconsistency": "An unsupported hue, brightness or saturation discontinuity appears relative to visible object evidence.",
    "pattern_inconsistency": "A repeated motif, stripe, grid, graphic or key layout changes spacing, direction, scale or continuity.",
    "semantic_drift": "The target becomes a different object/part identity or an incoherent hybrid. A distorted shape alone is insufficient.",
    "hidden_region_inconsistency": "A synthesized region conflicts with the visible object's material or appearance. This is consistency, not hidden-pixel accuracy.",
    "cooccurring_object": "A recognizable unsupported secondary object appears in generated RGB. Original scene context and surviving occluder residue are excluded when distinguishable.",
}


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value, digits=3):
    return "--" if value is None or value == "" else f"{float(value):.{digits}f}"


def table_markdown(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join("| " + " | ".join(map(cell, row)) + " |" for row in
                     [headers, ["---"] * len(headers), *rows])


def make_figures(output, taxonomy, census=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    positions = np.arange(len(taxonomy))
    fig, ax = plt.subplots(figsize=(11, 5.8), layout="constrained")
    left = np.zeros(len(taxonomy))
    for key, label, color in [("mild", "Mild (1)", "#8bb8c9"),
                               ("moderate", "Moderate (2)", "#306d88"),
                               ("severe", "Severe (3)", "#bc633f")]:
        widths = np.array([100 * r[key] / r["assessable"] for r in taxonomy])
        ax.barh(positions, widths, left=left, label=label, color=color)
        left += widths
    for i, r in enumerate(taxonomy):
        ax.text(left[i] + 1, i, f"{r['positive']}/{r['assessable']} ({left[i]:.1f}%)", va="center")
    ax.set_yticks(positions, [NAMES[r["failure"]].replace("Co-occurring object regeneration", "Co-occurrence") for r in taxonomy])
    ax.invert_yaxis()
    scope = f"{taxonomy[0]['sample_total']:,} " + ("saved outputs (census)" if census else "sampled saved outputs")
    ax.set(xlim=(0, 80), xlabel="Observed failures / assessable audited outputs (%)",
           title=f"Appearance failure taxonomy: {scope}\nSingle assistant review; labels overlap; unknown and NA excluded per type")
    ax.legend(loc="lower right", ncol=3, fontsize=10)
    ax.grid(axis="x", alpha=.15)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output / "figures" / f"taxonomy_severity.{ext}", dpi=180)
    plt.close(fig)


def report_blocks(output, manifest, rows, review, taxonomy):
    blocks = []

    def heading(text, level=2):
        blocks.append(("heading", level, text))

    def paragraph(text):
        blocks.append(("paragraph", text))

    def table(headers, data, widths=None):
        blocks.append(("table", headers, data, widths))

    def figure(path, caption, width=1.0, max_height=365):
        blocks.append(("figure", path, caption, width, max_height))

    def page():
        blocks.append(("page",))

    heading("Person 2 - Appearance Failures", 1)
    paragraph("**PD-MC / Open-World Amodal Appearance Completion | 7 November 2026**")
    paragraph("The accepted baseline saves 951 outputs from 2,565 targets. In a reproducible, stratified audit of 120 saved outputs, hidden-region inconsistency, texture mismatch and color inconsistency are the most frequent appearance labels. Pattern failures are also common where a pattern can be assessed. Semantic drift and co-occurring object regeneration occur less often, but can substantially change the returned object.")
    paragraph("The requested 20-80% occlusion analysis is supported by **114 targets with matched COCOA annotations**, including 44 saved outputs. Results fluctuate across bins; this subset does not establish a smooth deterioration curve or a causal effect of occlusion. The generated-area proxy is reported separately because it has only weak rank correlation with annotated occlusion (Spearman rho = 0.133, n = 142).")
    table(["Evidence", "Population / denominator", "What it supports"], [
        ["Pipeline results", "2,565 targets; 951 saved; 1,614 failed", "Full-benchmark output availability and existing metrics"],
        ["Matched true occlusion", "321 COCOA targets; 114 in [20,80)%", "Subset performance versus annotated hidden area"],
        ["Visual appearance audit", "120 saved outputs; 36 categories; four sources", "Observed defect counts and severity, conditional on assessability"],
        ["Generated-area diagnostic", "951 saved outputs", "Model expansion and retained visible support, not true occlusion"],
    ], [1.4, 2.8, 3.9])
    heading("Baseline and metric scope", 3)
    paragraph("Run: `pdmc-baseline-provenance-20261105`, Slurm 31548; released PD-MC, semantic matcher v2, released detector vocabulary, SAM visible references and seed 0. The run was independently verified with full/provenance requirements. No new model inference or learned-metric recomputation was performed for this analysis.")
    table(["Metric", "All targets", "Saved outputs", "Direction"], [
        ["CLIP", "28.393 (n=951; failures omitted)", "28.393 (n=951)", "Higher"],
        ["LPIPS", "0.7350 (n=2,565)", "0.2853 (n=951)", "Lower"],
        ["Feature similarity", "0.2426 (n=2,565)", "0.6544 (n=951)", "Higher"],
        ["SSIM", "0.3034 (n=2,565)", "0.8183 (n=951)", "Higher"],
    ], [1.4, 3.0, 2.2, 1.0])
    paragraph("Failed targets receive LPIPS=1, feature similarity=0 and SSIM=0. CLIP has no failure score. Existing appearance metrics use the released visible-reference preprocessing, not hidden-image ground truth. A saved output, including one that reaches the iteration cap, is not certified visually complete.")
    page()

    heading("Failure taxonomy and severity")
    paragraph("Frequency is positive labels divided by assessable audited outputs for that failure. One image can have several labels. The sample is balanced across model-expansion strata, so these raw fractions describe the audit and must not be read as whole-benchmark prevalence. Causes below are hypotheses suggested by the visible evidence and pipeline behavior; no causal ablation was run.")
    table(["Failure", "Frequency", "Mild / moderate / severe", "Possible cause"], [
        [NAMES[r["failure"]], f"{r['positive']}/{r['assessable']} ({100*r['frequency']:.1f}%)",
         f"{r['mild']} / {r['moderate']} / {r['severe']}", CAUSES[r["failure"]]] for r in taxonomy
    ], [2.1, 1.4, 1.3, 4.0])
    paragraph("**Severity rubric:** 0 = no observed defect; 1 = mild/local artifact with overall surface identity preserved; 2 = clear, consequential inconsistency while the main target remains recognizable; 3 = extensive or strongly misleading appearance corruption. U = unassessable/ambiguous evidence; NA = no applicable pattern. U and NA are excluded from the relevant frequency denominator, never counted as passes.")
    paragraph("**69/120 audited outputs have at least one positive appearance label.** The maximum recorded severity is mild for 14, moderate for 42 and severe for 13. Another 47 have no positive observed label, and four have no assessable appearance label at all. These counts do not certify the remaining outputs as correct: geometry can be incomplete and some individual labels can still be unknown.")
    table(["Failure", "Operational distinction"], [[NAMES[k], DEFINITIONS[k]] for k in FAILURES], [2.2, 6.6])
    page()

    heading("Performance versus annotated occlusion")
    figure("figures/performance_vs_occlusion.png", "Figure 1. Required bins [20,30), ..., [70,80)%. Output availability uses all matched targets in each bin. Appearance curves show both failure-penalized and saved-output means. Bars are 95% Wilson intervals for availability and 2,000-resample target bootstrap intervals for metric means. Empty saved-output bins are left empty.", max_height=408)
    paragraph("CLIP evaluates agreement with the annotated text label. LPIPS, feature similarity and SSIM compare with the detected visible reference under the existing evaluator. They do not measure the unobserved true texture or color behind an occluder. Target-level intervals do not capture reviewer uncertainty, run-to-run variation, annotation-join uncertainty or dependence between targets in the same image.")
    page()

    heading("Occlusion bins, coverage and interpretation")
    bins = csv_rows(output / "occlusion_bins.csv")
    table(["Occlusion (%)", "Targets", "Saved", "Saved %", "LPIPS all", "LPIPS saved", "SSIM saved", "CLIP saved"], [
        [r["bin"], r["targets"], r["saved"], number(float(r["saved_rate"])*100, 1),
         number(r["LPIPS_all_mean"]), number(r["LPIPS_saved_mean"]),
         number(r["SSIM_saved_mean"]), number(r["CLIP_saved_mean"], 2)] for r in bins[1:7]
    ], [1.3, .65, .65, .85, 1, 1.1, 1, 1])
    paragraph("The [20,30)% group saves 17/30 outputs (56.7%); the [50,60)% group saves 2/15 (13.3%); [60,70)% rebounds to 7/16 (43.8%); and [70,80)% has 0/5. Its LPIPS=1 is entirely the failure penalty, not a measured completion score. Conditional appearance estimates are particularly unstable with two saved outputs at 50-60%. Differences in category, visible area and detector/matcher success remain confounded with occlusion.")
    heading("Ground-truth definition and join", 3)
    paragraph("For the external COCOA annotation, `r_occ = |H_gt| / |V_gt union H_gt|`, where H_gt and V_gt are the annotated invisible and visible masks. The supplied `occlude_rate` was independently reconstructed by decoding those masks for all 321 matched targets; the maximum absolute difference is 3.66e-10 and every bin agrees. This supplies hidden **area**, not hidden RGB appearance.")
    paragraph("A benchmark target joins only when its original COCO image identifier and trimmed, case-insensitive region name match exactly one COCOA region. Annotation dimensions must also match the source image. The matching uses no completion score or output mask. Among 845 COCOA benchmark targets, 321 join uniquely, 462 have no exact name and 62 remain ambiguous. The other 1,720 targets have no ratio from this source. Name-based joins are conservative identity matches, not independently adjudicated object-instance ground truth.")
    table(["Occlusion availability", "Targets", "Saved outputs"], [
        ["Below 20%", "203", "97"], ["Requested [20,80)%", "114", "44"],
        ["80-100%", "4", "1"], ["Unavailable", "2,244", "809"],
        ["Total", "2,565", "951"],
    ], [4.5, 1.5, 2])
    paragraph("The report therefore fulfills the requested bins on the defensible ground-truth subset, while preserving missing-ratio counts. It does not assign fabricated occlusion values to the remaining targets. The full bin CSV also includes feature similarity, all metric denominators and interval bounds.")
    page()

    heading("Audit sampling and severity distribution")
    figure("figures/taxonomy_severity.png", "Figure 2. Each bar uses that failure type's assessable denominator; segments show the severity distribution. The failures overlap and cannot be summed into a total failure probability.", max_height=290)
    paragraph("Fifteen saved outputs were selected from each of eight expansion-proxy strata: [0,20), [20,30), ..., [70,80), [80,100]%. Within each stratum, IDs were ranked by SHA-256 of seed `20261107` and target ID, without consulting scores or image quality. Stratum populations are 494, 172, 99, 75, 49, 29, 15 and 18; all 15 outputs in the 70-80% proxy stratum are reviewed. This intentionally oversamples large expansions, which are not necessarily heavy true occlusions.")
    paragraph("The review includes 70 VG, 33 COCOA, eight free-images and nine LAION outputs. Only 17 of the 120 have matched true occlusion. A single assistant inspected aligned original context, SAM visible target, returned completion, support masks and full generated RGB. A same-reviewer source-context check refined co-occurrence attributions and removed two unsupported semantic-drift labels; it is not an independent second rater. Initial and final annotations are retained.")
    paragraph("Source-context checks distinguish newly synthesized objects from objects already in the scene. Among 11 positive co-occurrence labels, seven affect the returned region and four appear only in raw generated RGB. Six additional final-mask target switches are tracked as identity/mask confounds, not automatically counted as semantic drift. Hidden-region labels refer to visual compatibility with observed evidence, not correctness against inaccessible hidden pixels.")
    page()

    heading("Representative failures: texture, color and pattern")
    paragraph("Examples are selected from the recorded audit to illustrate the taxonomy, not to estimate frequencies. In each panel, the first four views share spatial alignment and crop. The last view shows the entire raw generated crop at its own display scale. Blue = visible support, orange = predicted support outside it, pink = visible support lost. Orange can include segmentation undercoverage, so it is not a ground-truth hidden-region mask.")
    for n, caption in [
        (35, "Case 035, horse: brown coat shifts toward saturated red/black with patchy ridged texture. Texture, color and hidden-region labels are severe; horse identity remains recognizable."),
        (91, "Case 091, zebra: broad, flowing bands fail to continue the visible fine striping in scale and direction. Pattern and hidden-region labels are severe; zebra identity remains intact."),
        (53, "Case 053, white bowl: dark rough multicolored material conflicts with the smooth white ceramic rim. Co-occurrence is unknown because fruit remnants cannot be confidently distinguished from a new object."),
    ]:
        figure(f"review/panels/{n:03d}.jpg", caption, max_height=125)
    page()

    heading("Representative failures: identity and co-occurrence")
    for n, caption in [
        (99, "Case 099, cat: a second cat-like head/body appears where the pot was removed and the original face changes color. Semantic drift is moderate; co-occurrence, texture, color and hidden-region inconsistency are severe. Annotated occlusion is 68.9%, whereas expansion proxy is 79.6%."),
        (62, "Case 062, knife: source-context inspection confirms a newly synthesized second knife. Target material can remain plausible while an additional object is generated. This is moderate co-occurrence, not a texture failure."),
        (117, "Case 117, keyboard: the flat key surface grows into a deep vertical panel with repeated marks. Pattern, texture and hidden-region defects are severe; semantic identity drifts into a keyboard/panel hybrid."),
    ]:
        figure(f"review/panels/{n:03d}.jpg", caption, max_height=125)
    paragraph("All 120 cases, including no-positive-label and ambiguous cases, are available in [the offline audit viewer](review/index.html). Per-case target IDs, evidence notes, confounds, severity labels and sampling probabilities are in [annotations.csv](review/annotations.csv). The six cases above are illustrations, not an additional evaluation sample.")
    page()

    heading("Category-wise analysis")
    categories = csv_rows(output / "categories.csv")
    selected = ["cat", "dog", "couch", "bus", "laptop", "plate", "bed", "refrigerator"]
    by_category = {r["category"]: r for r in categories}
    table(["Category", "Targets", "Saved / targets", "LPIPS saved", "SSIM saved", "CLIP saved"], [
        [c, by_category[c]["targets"], f"{by_category[c]['saved']}/{by_category[c]['targets']} ({100*float(by_category[c]['saved_rate']):.1f}%)",
         number(by_category[c]["LPIPS_saved"]), number(by_category[c]["SSIM_saved"]), number(by_category[c]["CLIP_saved"], 2)] for c in selected
    ], [1.3, .8, 2, 1.2, 1.2, 1.2])
    paragraph("These illustrative categories show the distinction between coverage and appearance: laptop has 50/62 saved outputs, dog 21/105 and plate 0/62. The zero-output category has no conditional appearance score. Bed has higher saved-output LPIPS than laptop (0.419 vs 0.303), but category size, source, occlusion and output selection are uncontrolled. Category keys use the recorded annotation class/head when available and otherwise the original label; they are not a comprehensive ontology (for example, plate and white plate remain separate).")
    heading("Observed audit patterns by category", 3)
    groups = []
    for category, count in Counter(r["category"] for r in review).most_common():
        if count < 6:
            continue
        counts = {r["failure"]: r for r in taxonomy_counts([r for r in review if r["category"] == category])}
        groups.append([category, count, *[f"{counts[k]['positive']}/{counts[k]['assessable']}" for k in
                       ("wrong_texture", "color_inconsistency", "pattern_inconsistency", "hidden_region_inconsistency", "cooccurring_object")]])
    table(["Category", "Audited", "Texture", "Color", "Pattern", "Hidden region", "Co-occurrence"], groups,
          [1.5, .8, 1, 1, 1, 1.3, 1.3])
    paragraph("Each entry is positive/assessable, with 0/0 meaning no assessable examples. Fur smoothing/repainting is visible in cats; textile/material changes occur in beds and couches; laptop/keyboard examples expose repeated-structure failures. These are exploratory sample observations. The audit was stratified by expansion, not category, and these small counts cannot support a category ranking or significance claim. Full category coverage is plotted in `figures/category_coverage.png`; all categories and metric means are in `categories.csv`.")
    page()

    heading("Why generated area cannot stand in for true occlusion")
    figure("figures/proxy_vs_ground_truth.png", "Figure 3. On 142 matched saved outputs, generated-area proxy versus annotated occlusion has Pearson r=0.263, Spearman rho=0.133 and median absolute discrepancy 11.54 percentage points. Only 76/142 fall in the same reporting bin; the broad [0,20)% bin contributes to agreement.", width=.66, max_height=335)
    paragraph("The proxy is `q = |M_pred minus V_SAM| / |M_pred|` after aligning integer crop/outpaint offsets to the original frame. It measures expansion beyond the initial SAM support. It changes with under-segmentation, overgrowth, dropped visible pixels and final-mask target switching, and it exists only when an output is saved. It therefore cannot estimate pipeline failure rates by true occlusion. The separately labelled `performance_vs_expansion_proxy` figure is a diagnostic, not a replacement for Figure 1.")
    paragraph("Across 951 saved outputs, median retained visible support is 98.9%; 18 retain under 50% and nine under 10%. Median per-target visible-core Delta E76 is 4.60, measured after 5x5 erosion of the overlapping support. It is descriptive, with no automatic pass/fail threshold. Among matched COCOA targets, nine saved outputs have SAM-versus-annotated-visible IoU below 0.5. These mask discrepancies can affect appearance metrics and apparent hidden-region size.")
    page()

    heading("Sensitivity, pipeline failures and scope limits")
    table(["Failure", "Raw audit", "Reweighted estimate", "Without selected confounds"], [
        [NAMES[r["failure"]], f"{r['positive']}/{r['assessable']}", f"{100*r['weighted_frequency']:.1f}%",
         f"{s['positive']}/{s['assessable']}"]
        for r, s in zip(taxonomy, csv_rows(output / "taxonomy_without_mask_confounds.csv"))
    ], [2.6, 1.3, 1.8, 2.6])
    paragraph("Reweighting uses stratum population/sample weights and divides weighted positives by weighted assessable labels. It estimates the assessable saved-output population under the single reviewer's labels, not all 2,565 targets. It excludes unknown/NA evidence, and no design-adjusted confidence interval is claimed. The strong raw-versus-weighted difference for semantic drift reflects oversampling of large-expansion strata; it is not a count of fewer reviewed failures.")
    paragraph("The sensitivity column removes 36 cases whose confound tags contain mask, target_switch or low_resolution, leaving 84 reviewed outputs. It still includes geometry and occluder-residue cases, and is not a fully deconfounded sample. Texture and hidden-region inconsistency remain common after this exclusion. Unknown labels remain material: hidden-region inconsistency has 20/120 U labels; pattern has four U and 71 NA labels. Raw Wilson columns in the CSV are descriptive simple-binomial intervals, not design-corrected population intervals.")
    failures = csv_rows(output / "pipeline_failures.csv")
    table(["Pipeline reason", "Count", "% of all 2,565 targets"], [[r["reason"], r["count"], number(float(r["fraction_of_all_targets"])*100, 2)+"%"] for r in failures], [4.6, 1.2, 2.2])
    paragraph("These 1,614 pipeline failures have no completion to label for appearance. Separately, 16 audited cases carry an incomplete-geometry tag and six a final target-switch tag; these are incidental review flags, not a comprehensive geometry taxonomy. Heavy occlusion is an input condition, not an appearance failure type. The six requested appearance types are therefore kept distinct from shape errors, detection failures and input difficulty.")
    page()

    heading("Potential causes and follow-up tests")
    paragraph("The evidence supports testing stronger preservation of visible surface cues and identity, but it does not establish a particular mechanism as the cause. Three concrete follow-ups would separate the hypotheses:")
    table(["Hypothesis", "Controlled test", "Required outcome evidence"], [
        ["Surface evidence is weakly preserved", "Compare equally sized candidate sets with and without visible-boundary color/texture selection; hold detector, masks, prompt, seed schedule and compute fixed.", "Blind paired labels plus visible preservation, new-region plausibility and coverage. Include an equal-budget random-selection control."],
        ["Context encourages extra objects", "Vary surrounding context or object-only conditioning on the same targets; keep candidate budget fixed.", "Count confirmed new objects separately in raw RGB and returned masks; do not count old scene objects or residue as new."],
        ["High occlusion reduces usable evidence", "Build a category-balanced pseudo-occlusion set from complete objects with known hidden RGB; retain a natural-occlusion test set.", "Report hidden-region color/texture/pattern metrics by true occlusion and saved-output coverage; acknowledge synthetic-to-real shift."],
    ], [2, 3.4, 3.4])
    paragraph("Before interpreting population prevalence, add an independent human review of the same frozen sample, reconcile disagreements using the rubric and report inter-rater agreement. Larger category-stratified samples and repeated model seeds would reduce current sampling and run-variation limitations. These are proposed experiments; none was run as part of this deliverable.")
    heading("Evidence and reproducibility", 3)
    paragraph("The baseline and external annotation input hashes, sample seed, source hashes and final annotation hash are recorded in `analysis_manifest.json`. `baseline-verification.json` preserves the full/provenance verification result. `cocoa_mask_audit.csv` preserves the independent decoded-mask ratio and SAM-IoU checks. The COCOA archive source is [Wakeupbuddy/amodalAPI](https://github.com/Wakeupbuddy/amodalAPI); the exact download URL and archive hash are retained in `external/download_receipt.json`.")
    paragraph("Validation already completed in an isolated WSL CPU environment: 167 Python tests, all five Bash regressions and synthetic CPU smoke passed (`checks-cpu-env.log`). Tests cover ratio boundaries, negative outpainting offsets, missing evidence, deterministic sampling, unique annotation joins, stratified weights and failure-penalty denominators. No inference/runtime code was changed and no new GPU validation was needed for this offline analysis. Report validation is recorded separately in `report_validation.json`.")
    page()

    heading("Reproduce and navigate the deliverable")
    paragraph("Run from the repository root using an isolated CPU environment, separate from the pinned GPU environment. Reporting dependencies are `matplotlib`, `scipy`, `pycocotools`, `markdown`, `reportlab` and `pypdf`, in addition to the CPU package. Preserve the downloaded annotation JSONs and final review CSV. Run the following on the machine holding the recovered baseline; regenerate the analysis manifest there so its absolute paths are valid.")
    blocks.append(("code", "python scripts/analyze_appearance_failures.py \\\n+  --run-dir output/baseline-31548/output/pdmc-baseline-provenance-20261105 \\\n+  --dataset-root datasets/open_world_amodal \\\n+  --output-dir output/appearance-failures-20261107 \\\n+  --cocoa-annotations output/appearance-failures-20261107/external \\\n+  --review output/appearance-failures-20261107/review/annotations.csv\n\npython scripts/audit_appearance_evidence.py \\\n+  --analysis-dir output/appearance-failures-20261107\n\npython scripts/build_appearance_report.py \\\n+  --analysis-dir output/appearance-failures-20261107".replace("\n+", "\n")))
    paragraph("The analysis command re-derives diagnostics, deterministic panels and summaries while preserving the final annotations. Add `--reuse-derived` only when intentionally reusing unchanged local inputs. Visual labels require review; the scripts never infer the taxonomy from numerical thresholds. The final annotation table is the authoritative label source, superseding `annotations_initial.csv` and the initial working `visual_labels.json`.")
    table(["Artifact", "Contents"], [
        ["REPORT.md / REPORT.html / REPORT.pdf", "Complete report in editable, browser and portable formats"],
        ["review/index.html; annotations.csv", "Searchable offline viewer, 120 panels, labels, evidence and sampling weights"],
        ["taxonomy.csv / taxonomy.json", "Positive/assessable counts, severity, unknown/NA counts, raw intervals and weighted estimates"],
        ["occlusion_bins.csv", "True-occlusion subset: all/saved means, counts, confidence intervals"],
        ["expansion_proxy_bins.csv", "Output-dependent diagnostic, explicitly separate from true occlusion"],
        ["categories.csv; taxonomy_groups.csv", "Category coverage and sampled label breakdowns by source/category/proxy bin"],
        ["per_target_diagnostics.csv / .json", "All 2,565 targets, annotation joins, artifact hashes and aligned diagnostics"],
        ["figures/", "Exportable PNG/PDF/SVG plots; original review panels remain under review/"],
        ["delivery_manifest.json; report_validation.json", "Delivered-file hashes, source hashes and report consistency checks"],
    ], [3.8, 5])
    paragraph("The shareable ZIP excludes the CPU environment, external annotation archive and model/dataset artifacts. It includes the report, plots, review panels, tables and provenance receipts. Full regeneration also requires the preserved baseline, source images and external COCOA annotation files. This analysis leaves the accepted run, historical outputs and experiment registry unchanged.")
    return blocks


def render_markdown(blocks):
    pieces = []
    for block in blocks:
        kind = block[0]
        if kind == "heading":
            pieces.append("#" * block[1] + " " + block[2])
        elif kind == "paragraph":
            pieces.append(block[1])
        elif kind == "table":
            pieces.append(table_markdown(block[1], block[2]))
        elif kind == "figure":
            alt = block[2].split(". ", 1)[0]
            pieces.append(f"![{alt}]({block[1]})\n\n{block[2]}")
        elif kind == "code":
            pieces.append("```bash\n" + block[1] + "\n```")
        elif kind == "page":
            pieces.append("<!-- pagebreak -->")
    return "\n\n".join(pieces) + "\n"


CSS = """
:root { color-scheme: light; font-family: system-ui, sans-serif; color:#172d40; background:#eef2f5; }
body { max-width:1160px; margin:32px auto; background:white; padding:44px; line-height:1.55; }
h1,h2,h3 { color:#183e58; line-height:1.2; } h2 { margin-top:2em; }
a { color:#126188; } table { border-collapse:collapse; width:100%; margin:1.3em 0; font-size:.9rem; }
th,td { padding:10px; text-align:left; border-bottom:1px solid #d5dee5; vertical-align:top; }
th { background:#eaf1f5; } tr:nth-child(even) td { background:#f8fafb; }
img { max-width:100%; height:auto; } pre { background:#f1f4f7; padding:18px; overflow:auto; font-size:.84rem; }
code { overflow-wrap:anywhere; } nav { display:flex; gap:22px; margin-bottom:24px; flex-wrap:wrap; }
.pagebreak { margin:40px 0; border-top:1px solid #d5dee5; }
.controls { padding:16px; background:#eaf1f5; position:sticky; top:0; display:flex; gap:12px; flex-wrap:wrap; }
select,input { padding:8px; border:1px solid #96adbc; border-radius:4px; font:inherit; }
article { border-bottom:2px solid #d5dee5; padding:25px 0; } .tags { color:#5a361c; font-size:.9rem; }
@media(max-width:700px) { body { margin:0; padding:16px; } table { font-size:.75rem; } th,td { padding:5px; } }
@media print { body { margin:0; padding:0; background:white; } nav,.controls { display:none; } .pagebreak { break-before:page; border:0; margin:0; } }
"""


def html_document(title, body):
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title><style>{CSS}</style></head><body>{body}</body></html>'


def render_review(output, review, census=False):
    options = '<option value="">All failure types</option>' + ''.join(
        f'<option value="{k}">{escape(v)}</option>' for k, v in NAMES.items())
    categories = '<option value="">All categories</option>' + ''.join(
        f'<option>{escape(c)}</option>' for c in sorted({r['category'] for r in review}))
    scope = f"all {len(review):,} saved outputs" if census else f"all {len(review):,} sampled outputs"
    parts = ['<nav><a href="../REPORT.html">Report</a><a href="annotations.csv">Final annotations CSV</a><a href="revision_notes.md">Review revisions</a></nav>',
             f'<h1>Appearance audit: {scope}</h1>',
             '<p>Single assistant review. 0 = no observed defect; 1 = mild; 2 = moderate; 3 = severe; U = unknown; NA = not applicable. Only positive labels match a failure filter. Frequencies and sampling design are explained in the report. Click a panel to view full resolution.</p>',
             f'<div class="controls"><select id="failure" aria-label="Failure type">{options}</select><select id="category" aria-label="Category">{categories}</select><input id="query" aria-label="Search evidence or target ID" placeholder="Search ID or evidence"><span id="count">{len(review)} / {len(review)} cases</span></div>']
    for row in review:
        n = int(row['review_number'])
        source_check = output / f"review/source_checks/{n:04d}.jpg"
        source_link = (f'<p><a href="source_checks/{n:04d}.jpg">Full original source versus raw generated RGB</a></p>'
                       if source_check.exists() else '')
        positive = ' '.join(k for k in FAILURES if row[k] in ('1', '2', '3'))
        tags = ' | '.join(f'{NAMES[k]}: {row[k]}' for k in FAILURES)
        parts.append(f'<article data-failures="{positive}" data-category="{escape(row["category"], quote=True)}">'
                     f'<h2>{n:03d} - {escape(row["id"])}</h2>'
                     f'<p class="tags">{escape(tags)}</p>'
                     f'<a href="panels/{n:03d}.jpg"><img src="panels/{n:03d}.jpg" loading="lazy" alt="Original context, SAM visible target, saved completion, support masks and raw generated RGB for case {n}"></a>'
                     f'<p>{escape(row["evidence"])}</p>{source_link}'
                     f'<p>Confound: {escape(row["identity_or_mask_confound"] or "none recorded")}. Expansion-proxy stratum: {escape(row["proxy_bin"])}%. '
                     f'Inclusion probability: {float(row["inclusion_probability"]):.4f}. '
                     f'Co-occurrence scope: {escape(row.get("cooccurrence_output_scope") or "not labelled positive")}.</p></article>')
    parts.append('''<script>
const failure=document.querySelector('#failure'), category=document.querySelector('#category'), query=document.querySelector('#query');
function filter(){ let count=0; document.querySelectorAll('article').forEach(row=>{
const show=(!failure.value || row.dataset.failures.split(' ').includes(failure.value)) && (!category.value || row.dataset.category===category.value) && row.textContent.toLowerCase().includes(query.value.toLowerCase());
row.hidden=!show; if(show) count++; }); document.querySelector('#count').textContent=count+' / '+document.querySelectorAll('article').length+' cases'; }
[failure,category,query].forEach(el=>el.addEventListener('input',filter));
</script>''')
    (output / "review/index.html").write_text(html_document("Appearance audit", ''.join(parts)), encoding="utf-8")


def render_pdf(output, blocks, footer_text="Person 2 | Appearance failures | PD-MC accepted baseline | 7 November 2026"):
    import markdown
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, Spacer, PageBreak, Preformatted, KeepInFrame
    pagesize = landscape(A4)
    width = pagesize[0] - 88
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyReport", fontName="Helvetica", fontSize=9.8, leading=12.6, spaceAfter=7, textColor=colors.HexColor("#172d40")))
    styles.add(ParagraphStyle(name="CaptionReport", parent=styles["BodyReport"], fontSize=8.6, leading=10.6, textColor=colors.HexColor("#485b69"), spaceAfter=8))
    styles.add(ParagraphStyle(name="CellReport", parent=styles["BodyReport"], fontSize=8.2, leading=10.2, spaceAfter=0))
    styles.add(ParagraphStyle(name="CodeReport", fontName="Courier", fontSize=8, leading=10.5, spaceAfter=12))
    for name, size in (("Title", 25), ("Heading2", 18), ("Heading3", 12)):
        styles[name].fontSize = size
        styles[name].leading = size * 1.2
        styles[name].textColor = colors.HexColor("#183e58")
        styles[name].spaceAfter = 11
    styles["Title"].alignment = TA_LEFT

    def para(text, style="BodyReport"):
        html = markdown.markdown(str(text)).removeprefix("<p>").removesuffix("</p>")
        html = html.replace("<code>", '<font name="Courier">').replace("</code>", "</font>")
        return Paragraph(html, styles[style])

    story = []
    for block in blocks:
        kind = block[0]
        if kind == "heading":
            story.append(para(block[2], {1: "Title", 2: "Heading2", 3: "Heading3"}[block[1]]))
        elif kind == "paragraph":
            story.append(para(block[1]))
        elif kind == "page":
            story.append(PageBreak())
        elif kind == "code":
            story.append(Preformatted(block[1], styles["CodeReport"]))
        elif kind == "table":
            weights = block[3] or [1] * len(block[1])
            widths = [width * w / sum(weights) for w in weights]
            cells = [[para(f"**{c}**", "CellReport") for c in block[1]]]
            cells += [[para(c, "CellReport") for c in row] for row in block[2]]
            tab = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
            tab.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f0f5")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
                ("LINEBELOW", (0, 0), (-1, -1), .35, colors.HexColor("#d4dfe6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            tab.spaceAfter = 9
            story.append(tab)
        elif kind == "figure":
            image = Image(str(output / block[1]))
            height_limit = min(block[4], 365)
            factor = min(width * block[3] / image.imageWidth, height_limit / image.imageHeight)
            image.drawWidth, image.drawHeight = image.imageWidth * factor, image.imageHeight * factor
            image.hAlign = "CENTER"
            story.extend([image, Spacer(1, 5), para(block[2], "CaptionReport")])

    # Each logical section is a reviewable page; shrink modestly if needed,
    # preventing isolated captions or a trailing spacer from making a blank page.
    page_groups, current = [], []
    for item in story:
        if isinstance(item, PageBreak):
            page_groups.append(current)
            current = []
        else:
            current.append(item)
    if current:
        page_groups.append(current)
    framed = [KeepInFrame(width, pagesize[1] - 77, group, mode="shrink", hAlign="LEFT")
              for group in page_groups]
    story = []
    for i, frame in enumerate(framed):
        if i:
            story.append(PageBreak())
        story.append(frame)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#526d7d"))
        canvas.drawString(38, 20, footer_text)
        canvas.drawRightString(pagesize[0]-38, 20, str(doc.page))
        canvas.restoreState()

    document = SimpleDocTemplate(str(output / "REPORT.pdf"), pagesize=pagesize, rightMargin=38, leftMargin=38,
                                 topMargin=30, bottomMargin=35, title="Person 2 - Appearance Failures", author="Codex research audit")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return [float(getattr(frame, "_scale", 1)) for frame in framed]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.analysis_dir.resolve()
    manifest = json.loads((output / "analysis_manifest.json").read_text(encoding="utf-8"))
    if manifest["inputs"]["per_target.csv"] != EXPECTED_RESULTS_SHA256 or manifest["review_sha256"] != EXPECTED_REVIEW_SHA256:
        raise ValueError("This report narrative is frozen to the accepted baseline and final 120-case review")
    rows = json.loads((output / "per_target_diagnostics.json").read_text(encoding="utf-8"))
    sample = csv_rows(output / "review/sample.csv")
    review = read_review(output / "review/annotations.csv", [r["id"] for r in sample])
    if sha256(output / "review/annotations.csv") != manifest["review_sha256"]:
        raise ValueError("Final labels changed; rebuild analysis summaries first")
    taxonomy = json.loads((output / "taxonomy.json").read_text(encoding="utf-8"))
    for expected, actual in zip(taxonomy_counts(review), taxonomy):
        if any(actual[k] != v for k, v in expected.items()):
            raise ValueError("Taxonomy does not reproduce final labels")
    make_figures(output, taxonomy)
    blocks = report_blocks(output, manifest, rows, review, taxonomy)
    markdown_text = render_markdown(blocks)
    (output / "REPORT.md").write_text(markdown_text, encoding="utf-8")
    taxonomy_start = next(i for i, block in enumerate(blocks)
                          if block[0] == "heading" and block[2] == "Failure taxonomy and severity")
    taxonomy_end = next(i for i in range(taxonomy_start, len(blocks)) if blocks[i][0] == "page")
    (output / "TAXONOMY.md").write_text(render_markdown(blocks[taxonomy_start:taxonomy_end]), encoding="utf-8")
    import markdown
    html = markdown.markdown(markdown_text, extensions=["tables", "fenced_code"])
    html = html.replace("<!-- pagebreak -->", '<div class="pagebreak"></div>')
    navigation = '<nav><a href="REPORT.pdf">Download PDF</a><a href="review/index.html">Browse 120 reviewed cases</a><a href="taxonomy.csv">Taxonomy CSV</a><a href="occlusion_bins.csv">Occlusion-bin CSV</a></nav>'
    (output / "REPORT.html").write_text(html_document("Person 2 - Appearance Failures", navigation + html), encoding="utf-8")
    render_review(output, review)
    page_scales = render_pdf(output, blocks)
    rubric = "# Appearance annotation guide\n\n" + "\n\n".join(f"## {NAMES[k]}\n\n{DEFINITIONS[k]}" for k in FAILURES)
    rubric += "\n\n## Coding and attribution\n\nUse 0 for no observed defect, 1 mild, 2 moderate, 3 severe, U unassessable and NA inapplicable. Labels may overlap. Preserve evidence notes and identity/mask confounds. A gap or missing body part alone is a geometry failure, not wrong texture. A wrong final mask alone does not establish semantic drift. For co-occurrence, compare full original context with raw generated RGB and record whether the extra object reaches the returned mask. Hidden-region labels assess consistency with visible evidence; there is no true hidden RGB.\n\nThe final CSV contains one assistant's judgments, not independent human consensus. The retained initial CSV and revision notes document the same-reviewer check. The sample is stratified by generated expansion, not true occlusion. Do not replace U or NA with 0.\n"
    (output / "review/RUBRIC.md").write_text(rubric, encoding="utf-8")
    write_json(output / "report_build.json", {"built_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "python": sys.version, "source_sha256": sha256(Path(__file__)),
        "review_sha256": sha256(output / "review/annotations.csv"), "report_blocks": len(blocks),
        "pdf_page_scale_divisors": page_scales})
    print(json.dumps({"report": str(output / "REPORT.pdf"), "reviewed": len(review), "blocks": len(blocks)}))


if __name__ == "__main__":
    main()
