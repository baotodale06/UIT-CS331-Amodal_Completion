#!/usr/bin/env python3
"""Build the expanded-vocabulary census report from explicit frozen labels."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from amodal_benchmark.appearance import BIN_LABELS, FAILURES, read_review, taxonomy_counts, write_csv
from amodal_benchmark.dataset import sha256, write_json
from build_appearance_report import (NAMES, CAUSES, DEFINITIONS, csv_rows, html_document,
    make_figures, number, render_markdown, render_pdf, render_review, table_markdown)


def additional_tables(output, rows, review):
    by_id = {r["id"]: r for r in rows}
    grouped = []
    for field in ("gt_bin", "proxy_bin"):
        for value in (*BIN_LABELS, "unavailable"):
            group = [r for r in review if by_id[r["id"]][field] == value]
            grouped.extend({"group_field": field, "group": value, **r} for r in taxonomy_counts(group))
    write_csv(output / "taxonomy_by_occlusion.csv", grouped)
    clean = [r for r in review if not any(t in r["identity_or_mask_confound"] for t in
             ("mask", "target_switch", "low_resolution", "small_target", "occluder_residue"))]
    write_csv(output / "taxonomy_strict_sensitivity.csv", taxonomy_counts(clean))
    taxonomy = taxonomy_counts(review)
    bounds = [{"failure": r["failure"], "positive": r["positive"], "unknown": r["unassessable"],
               "applicable_or_unknown": len(review) - r["not_applicable"],
               "observed_positive_fraction": r["positive"] / (len(review) - r["not_applicable"]),
               "unknown_all_positive_fraction": (r["positive"] + r["unassessable"]) / (len(review) - r["not_applicable"])}
              for r in taxonomy]
    write_csv(output / "unknown_label_bounds.csv", bounds)
    write_csv(output / "cooccurrence_cases.csv", [r for r in review if r["cooccurring_object"] in ("1", "2", "3")])
    write_csv(output / "semantic_drift_cases.csv", [r for r in review if r["semantic_drift"] in ("1", "2", "3")])
    counts = Counter(max([int(r[k]) for k in FAILURES if r[k].isdigit()], default=-1) for r in review)
    summary = {"reviewed": len(review), "positive_any": sum(v for k, v in counts.items() if k > 0),
               "maximum_severity": dict(counts), "categories": len({r["category"] for r in review}),
               "source_counts": dict(Counter(r["source"] for r in review)),
               "confound_tags": dict(Counter(t for r in review for t in r["identity_or_mask_confound"].split(";") if t)),
               "strict_sensitivity_outputs": len(clean),
               "cooccurrence_scopes": dict(Counter(r["cooccurrence_output_scope"] for r in review if r["cooccurring_object"] in ("1", "2", "3"))),
               "method": "Single-assistant screen-level census, followed by targeted full-source checks; no independent human adjudication"}
    write_json(output / "census_summary.json", summary)
    return grouped, summary


def figures(output, grouped, review):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    for field, bins, name, title in (
        ("gt_bin", BIN_LABELS[1:7], "appearance_vs_occlusion", "Observed appearance failures vs annotated COCOA occlusion"),
        ("proxy_bin", BIN_LABELS, "appearance_vs_expansion_proxy", "Observed appearance failures vs model expansion (NOT true occlusion)")):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        for ax, key in zip(axes.flat, ("wrong_texture", "color_inconsistency", "pattern_inconsistency", "hidden_region_inconsistency")):
            values = [next(r for r in grouped if r["group_field"] == field and r["group"] == b and r["failure"] == key) for b in bins]
            y = [100 * r["frequency"] if r["frequency"] is not None else np.nan for r in values]
            ax.plot(range(len(bins)), y, marker="o", color="#306d88")
            for i, (v, r) in enumerate(zip(y, values)):
                if np.isfinite(v):
                    ax.annotate(f"{r['positive']}/{r['assessable']}", (i, v), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
            ax.set(ylim=(-3, 114), ylabel="Positive / assessable (%)", title=NAMES[key])
            ax.set_xticks(range(len(bins)), bins, rotation=20)
            ax.grid(alpha=.18)
        fig.suptitle(title + "\nSingle-assistant census; counts shown; unknown and NA excluded", fontsize=14)
        fig.supxlabel("Annotated occlusion (%)" if field == "gt_bin" else "Generated-area proxy (%) | Descriptive association, conditional on saved outputs")
        for ext in ("png", "svg", "pdf"):
            fig.savefig(output / f"figures/{name}.{ext}", dpi=180)
        plt.close(fig)
    cats = [c for c, n in Counter(r["category"] for r in review).most_common() if n >= 20]
    keys = ("wrong_texture", "color_inconsistency", "pattern_inconsistency", "hidden_region_inconsistency")
    stats = [{r["failure"]: r for r in taxonomy_counts([v for v in review if v["category"] == c])} for c in cats]
    values = np.array([[100 * s[k]["frequency"] if s[k]["frequency"] is not None else np.nan for k in keys] for s in stats])
    fig, ax = plt.subplots(figsize=(10, 11), layout="constrained")
    im = ax.imshow(np.ma.masked_invalid(values), cmap="YlOrBr", vmin=0, vmax=100, aspect="auto")
    for i, stat in enumerate(stats):
        for j, k in enumerate(keys):
            s = stat[k]
            ax.text(j, i, f"{s['positive']}/{s['assessable']}", ha="center", va="center", color="white" if values[i, j] > 65 else "black", fontsize=10)
    ax.set_yticks(range(len(cats)), cats)
    ax.set_xticks(range(len(keys)), [NAMES[k].replace(" inconsistency", "") for k in keys], rotation=12)
    ax.set_title("Category-wise observed failures\nGroups with at least 20 saved outputs; cells = positive / assessable")
    fig.colorbar(im, ax=ax, label="Frequency (%)")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(output / f"figures/category_appearance.{ext}", dpi=180)
    plt.close(fig)


def report_blocks(output, manifest, rows, review, taxonomy, summary):
    blocks = []
    def h(text, level=2): blocks.append(("heading", level, text))
    def p(text): blocks.append(("paragraph", text))
    def t(head, data, widths=None): blocks.append(("table", head, data, widths))
    def f(path, caption, width=1, height=330): blocks.append(("figure", path, caption, width, height))
    def page(): blocks.append(("page",))
    stats = manifest["baseline_statistics"]["Overall"]
    saved = [r for r in rows if r["status"] == "success"]
    gt = csv_rows(output / "occlusion_bins.csv")
    agreement = json.loads((output / "proxy_vs_ground_truth.json").read_text())
    mask = json.loads((output / "mask_diagnostics_summary.json").read_text())
    h("Person 2 - Appearance Failures", 1)
    p("**Expanded-vocabulary PD-MC | Complete saved-output census | 7 November 2026 (UTC)**")
    p(f"Analyzed all **2,565 target records** from `pdmc-open-world-sam-annotated` and visually reviewed **all 1,390 saved completions**. The remaining **1,175 targets (45.8%)** have no saved completion and are counted as pipeline failures, not assigned visual appearance labels. This report is about the annotation-expanded vocabulary run, not the earlier 120-case baseline audit.")
    p(f"**{summary['positive_any']}/1,390 outputs (46.9%) have at least one observed appearance defect.** Texture, color, repeated-pattern and hidden-region consistency failures dominate. Semantic identity changes are uncommon under a strict definition. Eleven confirmed secondary-object cases occur in raw generated RGB; all eleven extras are excluded from the final returned object mask.")
    t(["Evidence", "Coverage", "Scope"], [
        ["Pipeline and saved metrics", "2,565 targets; 1,390 saved", "Every target in the expanded-vocabulary run"],
        ["Visual census", "1,390/1,390 outputs; 284 recorded categories", "One assistant's screen-level judgments, with evidence per target"],
        ["True occlusion", "321 matched COCOA targets; 181 saved", "114 targets and 71 saved outputs fall in the requested 20-80% bins"],
        ["Expansion diagnostics", "1,390 outputs", "Output-dependent generated area; not a substitute for true occlusion"]], [1.2, 1.6, 2.5])
    p("Expanded vocabulary adds missing per-image annotation categories to Grounding DINO queries; matching uses semantic matcher v2 and references use SAM. Existing scores and immutable recovered outputs are reused. Historical execution/source/environment provenance is absent; artifact verification does not supply that missing provenance. No new inference or learned-metric computation was performed.")
    page()

    h("Failure taxonomy: observed frequency, severity and possible causes")
    p("Frequency = positive labels / assessable saved outputs for each failure. Labels overlap. Unknown (U) and inapplicable (NA) labels are excluded, and their counts are retained below. This is a census of this run's saved outputs, so no sampling weights or sampling-error claims are needed. Reviewer error and limited display resolution remain.")
    t(["Failure", "Frequency", "Mild / moderate / severe", "U / NA", "Possible cause (hypothesis)"], [
        [NAMES[r["failure"]], f"{r['positive']}/{r['assessable']} ({100*r['frequency']:.1f}%)",
         f"{r['mild']} / {r['moderate']} / {r['severe']}", f"{r['unassessable']} / {r['not_applicable']}", CAUSES[r["failure"]]] for r in taxonomy], [1.5, 1.1, 1.1, .6, 2.7])
    p("**Severity:** 0 = no observed defect; 1 = mild/local inconsistency, with overall surface identity preserved; 2 = clear consequential mismatch; 3 = extensive or strongly misleading corruption. U = insufficient or ambiguous evidence; NA = no assessable repeated pattern. A missing limb, hole, wrong final mask or severe shape distortion alone does not establish semantic drift.")
    p("Across outputs, maximum observed severity is mild for **347**, moderate for **243**, and severe for **62**. Another **733** have no positive recorded label, and **five** have no assessable label. These are not 738 certified correct outputs: some have unassessable hidden regions, incomplete geometry or other pipeline defects.")
    p("The eleven co-occurrence positives are moderate defects in raw RGB: nine same-category duplications and two context-object substitutions. The observed rate is 11/1,199 assessable raw outputs (0.9%), or 11/1,390 saved outputs (0.8%) as a conservative observed fraction. With 191 unknown cases, absence in the returned mask does not establish that regeneration never affects returned objects.")
    page()
    h("Severity distribution and operational definitions")
    f("figures/taxonomy_severity.png", "Each bar uses its own assessable denominator. Pattern frequency applies to the 601 outputs with assessable repeated structure, not all outputs.", .83, 245)
    t(["Failure", "What was counted"], [[NAMES[k], DEFINITIONS[k]] for k in FAILURES], [1.4, 4.8])
    p("Appearance labels describe visible incompatibilities in returned completions. Segmentation contamination and surviving occluders can contribute; the census does not causally attribute every defect to the generator. Hidden-region consistency means compatibility with visible object evidence, not correctness against unavailable hidden RGB. Shape inconsistency is recorded as a confound; heavy occlusion is a condition used for stratification, not an appearance failure type.")
    page()

    h("Performance versus true occlusion: the requested bins")
    f("figures/performance_vs_occlusion.png", "Required bins [20,30), [30,40), ..., [70,80)%. Output availability uses every matched target per bin. Metric curves separate failure-penalized means from saved-output means. Availability bars are 95% Wilson intervals; metric bars use 2,000 target bootstrap resamples.", .92, 365)
    p("The x-axis is annotated COCOA hidden area, not the model's predicted missing area. All four existing metrics retain the SAM-visible-reference protocol; they do not test whether the unobserved true texture or color was recovered. Target-level intervals do not include annotation-join uncertainty, within-image dependence, different model seeds or reviewer disagreement.")
    page()
    h("Occlusion-bin numbers and limits")
    t(["Occlusion %", "Targets", "Saved", "Saved %", "LPIPS all", "LPIPS saved", "SSIM saved", "CLIP saved"], [[
        r["bin"], r["targets"], r["saved"], number(100*float(r["saved_rate"]), 1), number(r["LPIPS_all_mean"]),
        number(r["LPIPS_saved_mean"]), number(r["SSIM_saved_mean"]), number(r["CLIP_saved_mean"], 2)] for r in gt[1:7]], [1.1,.6,.6,.7,.9,1,.95,.95])
    p("Output availability is 80.0%, 41.7%, 50.0%, 60.0%, 81.2% and 60.0% across the six bins. It is **not monotonic**. Saved-output LPIPS is also nonmonotonic; CLIP is lower in several higher-occlusion bins, but this small, category-confounded subset does not establish a causal degradation law. The 70-80% bin contains only five targets and three saved completions.")
    p("Ground-truth ratio: `r_occ = |H_gt| / |V_gt union H_gt|`. An original COCO image ID and trimmed case-insensitive annotation name must match exactly one COCOA region, with matching source dimensions. Of 845 COCOA targets, 321 join uniquely, 462 have no exact name and 62 are ambiguous. The other 1,720 targets have no ratio from this annotation source. Unique-name joins are conservative matches, not independent instance adjudication.")
    t(["Ratio availability", "Targets", "Saved"], [["Below 20%", gt[0]["targets"], gt[0]["saved"]],
        ["Requested [20,80)%", 114, 71], ["80-100%", gt[7]["targets"], gt[7]["saved"]],
        ["Unavailable", gt[8]["targets"], gt[8]["saved"]]], [2.2,1,1])
    p("Independent mask decoding reproduces all 321 supplied ratios and every reporting bin; maximum absolute discrepancy is 3.66e-10. No true-occlusion value is imputed for the 2,244 unmatched targets. The complete bin CSV includes feature similarity, all metric denominators and interval endpoints.")
    page()
    h("Visual failure frequencies within true-occlusion bins")
    f("figures/appearance_vs_occlusion.png", "Each point is positive / assessable among saved outputs with a matched ratio. Counts are printed because denominators are small and differ by failure. This panel covers 71 outputs in the requested bins.", .92, 365)
    p("The 50-60% bin has hidden-region inconsistency in 7/7 assessable saved outputs, but only nine outputs were saved and two hidden regions are unknown. The 60-70% bin has 5/7 and 70-80% has 1/2. These small conditional counts cannot establish that one occlusion level is intrinsically easier than another. Pipeline failures remain visible in the separate availability plot.")
    page()
    h("Full-output diagnostics: expansion is associated with more defects")
    f("figures/appearance_vs_expansion_proxy.png", "All saved outputs contribute to their generated-area stratum, subject to assessability. Expansion q = |M_pred minus V_SAM| / |M_pred| after integer crop/outpaint alignment. This is not true occlusion.", .92, 345)
    p("Texture failures rise from 29.9% in the below-20% expansion group to 82.6% in the 80-100% group; hidden-region inconsistency rises from 33.2% to 87.0%. Intermediate bins fluctuate. The association is descriptive: large expansion also includes outpainting, SAM undercoverage, background growth and target switches.")
    p(f"On 181 matched saved outputs, expansion versus true occlusion has Pearson r={agreement['pearson_r']:.3f}, Spearman rho={agreement['spearman_r']:.3f}, median absolute discrepancy {agreement['median_absolute_error_pp']:.2f} percentage points, and only {agreement['same_bin']}/181 share a bin. Median visible support retention is {100*mask['median_visible_retention']:.1f}%; 27 outputs retain under 50% and 13 under 10%. Median per-target visible-core Delta E76 is {mask['median_visible_delta_e76']:.2f}, a descriptive color diagnostic without an automatic failure threshold.")
    page()
    h("Category-wise analysis")
    groups = []
    for cat in ("cat", "bus", "couch", "laptop", "plate", "horse", "white plate", "keyboard", "bed", "zebra"):
        group = [r for r in review if r["category"] == cat]
        s = {r["failure"]: r for r in taxonomy_counts(group)}
        groups.append([cat, len(group), *[f"{s[k]['positive']}/{s[k]['assessable']}" for k in
                      ("wrong_texture", "color_inconsistency", "pattern_inconsistency", "hidden_region_inconsistency")]])
    t(["Category", "Reviewed", "Texture", "Color", "Pattern", "Hidden region"], groups, [1.3,.8,1,1,1,1.2])
    p("Keyboard patterns are especially fragile: 21/22 assessable outputs have irregular keys or labels. Bus graphics/text fail in 43/57; laptop repeated structures fail in 25/37. Plates expose material and color changes, often with food-remnant or mask confounds. Cat and horse examples show fur smoothing or palette changes; zebra examples expose stripe-scale discontinuities.")
    p("Categories retain the existing diagnostic field: annotation class/head where available, otherwise original label. Thus `plate` and `white plate` remain distinct. All 284 saved-output categories and all benchmark categories are tabulated; `figures/category_appearance.png` shows groups with at least 20 saved outputs. Category counts, occlusion mix, mask quality and assessability differ, so these are exploratory comparisons, not a controlled ranking.")
    p("Full category tables: `categories.csv` contains pipeline availability and existing score means; `taxonomy_groups.csv` contains six failure types by category, source and expansion bin. `taxonomy_by_occlusion.csv` adds true-occlusion strata. A 0/0 cell means no assessable pattern, not a 0% pattern failure rate.")
    page()
    h("Examples: texture, color and pattern")
    p("Panels show aligned original context, initial SAM visible support, returned completion, support change, and full generated RGB. Blue = visible support; orange = predicted support outside it; pink = lost visible support. Orange is not a ground-truth hidden mask. Examples illustrate labels and do not define frequencies.")
    for n, caption in [(1055, "Horse: brown coat gains red/purple patches and rough streaking; visible and new regions no longer share a consistent palette."),
                       (1200, "Cat: new belly becomes pink fold-like material instead of continuing tabby fur. Pink cloth in the source suggests context leakage, but this is not a proven cause."),
                       (1315, "Zebra: enlarged torso carries broad curved bands while the visible neck has fine stripes; stripe scale and continuity break.")]:
        f(f"review/panels/{n:03d}.jpg", f"Case {n}. {caption}", 1, 102)
    page()
    h("Examples: semantic identity and co-occurrence")
    f("review/panels/1321.jpg", "Case 1321, mouse: the completion develops a split shoe-like body and unrelated control layout. This was judged severe semantic drift, not merely a missing geometric part.", 1, 120)
    f("review/source_checks/0893.jpg", "Case 893, fork: full source has one fork; raw completion adds two others. They are outside the returned target mask. This is confirmed same-category duplication.", .78, 215)
    p("Other confirmed raw-RGB cases include two duplicated toilets (416, 518), another house (463), another elephant (465), a second cat (1125), extra plates (1121, 1230), another keyboard strip (1169), a new vase of flowers (628), and scissors replacing a knife (1163). All eleven were checked against full originals; full-source check images and explicit revisions are retained.")
    page()
    h("Attribution checks, sensitivity and uncertainty")
    strict = csv_rows(output / "taxonomy_strict_sensitivity.csv")
    loose = csv_rows(output / "taxonomy_without_mask_confounds.csv")
    t(["Failure", "Full census", "Remove selected mask/low-res confounds", "Also remove residue/small-target cases"], [
        [NAMES[r["failure"]], f"{r['positive']}/{r['assessable']}", f"{a['positive']}/{a['assessable']}",
         f"{b['positive']}/{b['assessable']}"] for r, a, b in zip(taxonomy, loose, strict)], [1.5,1,1.8,1.8])
    p(f"The strict sensitivity view retains {summary['strict_sensitivity_outputs']} outputs after excluding any mask, target-switch, low-resolution, small-target or occluder-residue tag. This is a selected subset, not a corrected causal estimate. Nine matched saved outputs have SAM-versus-COCOA-visible IoU below 0.5; identity/mask problems can alter both scores and perceived completion extent.")
    p("Full-source rechecks removed six provisional co-occurrence positives: basketballs, a cat on a dog, pliers, a dog near a taxi and two umbrellas were already present. One car-like bus extension became unknown because altered livery could explain it. Three additional uncertain candidates were checked; two became positives and one an existing-context negative. Four geometry-only semantic labels were removed, and one apparent display transformation became a mask confound. These are same-assistant checks, not independent raters.")
    p("Frequencies condition on assessability. `unknown_label_bounds.csv` reports observed-positive and all-unknown-positive bounds within applicable-or-unknown outputs; neither bound captures overlooked defects or rater error. Generic Wilson fields remain in the machine-readable taxonomy for software compatibility, but are not used as census uncertainty intervals. Independent human review is needed before treating these judgments as publication-quality consensus.")
    p("Likely causes are hypotheses: weak surface anchoring, context leakage, pattern priors and insufficient visible evidence. A fair next test would compare surface-consistency candidate selection at equal sampling budget, include pseudo-occlusion with known hidden RGB, and repeat with independent raters. No such intervention or improvement is established here.")
    page()
    h("Pipeline accounting and existing score denominators")
    t(["Metric", "All-target protocol", "Saved outputs", "Direction"], [[k,
        f"{number(stats[k], 4)} (n={stats[k+'_n']})",
        f"{number(sum(r[k] for r in saved if r[k] is not None)/sum(r[k] is not None for r in saved),4)} (n={sum(r[k] is not None for r in saved)})",
        "Lower" if k == "LPIPS" else "Higher"] for k in ("CLIP", "LPIPS", "Feature Similarity", "SSIM")], [1.2,1.8,1.8,.8])
    p("Failed targets receive LPIPS=1, feature similarity=0 and SSIM=0. CLIP omits failures. The 1,390 saved outputs include 1,178 non-cap termination states and 212 iteration-limited outputs; neither state certifies appearance or geometric completeness. There are no missing source images or reference failures in the recorded run.")
    t(["No-output reason", "Count", "% of 2,565 targets"], [[r["reason"], r["count"], number(100*float(r["fraction_of_all_targets"]),2)] for r in csv_rows(output / "pipeline_failures.csv")], [2.4,.8,1.2])
    p("The expanded-vocabulary run is annotation-guided. These results do not isolate the causal effect of vocabulary expansion, nor compare model quality against incompatible reference protocols. The earlier baseline report remains a separate artifact and should not be substituted for this census.")
    page()
    h("Reproduction and deliverables")
    p("Use a separate CPU environment with the repository installed plus numpy, Pillow, matplotlib, scipy, pycocotools, markdown, reportlab and pypdf. The full recovered expanded-vocabulary run, original benchmark images and external COCOA annotations are required for regeneration; the delivery ZIP includes reports, review evidence, numerical tables and analysis-source snapshots, not the models or dataset.")
    blocks.append(("code", "python scripts/expanded_appearance_review.py compile --census \\\n+  --analysis-dir output/appearance-expanded-20261107\n\npython scripts/analyze_appearance_failures.py \\\n+  --run-dir output/artifact-recovery-2026-11-05/downloads/pdmc-open-world-sam-annotated \\\n+  --dataset-root datasets/open_world_amodal \\\n+  --output-dir output/appearance-expanded-20261107 \\\n+  --cocoa-annotations output/appearance-expanded-20261107/external \\\n+  --review output/appearance-expanded-20261107/review/annotations.csv --per-bin 10000\n\npython scripts/audit_appearance_evidence.py --analysis-dir output/appearance-expanded-20261107\npython scripts/build_expanded_appearance_report.py --analysis-dir output/appearance-expanded-20261107\npython scripts/validate_expanded_appearance_report.py --analysis-dir output/appearance-expanded-20261107".replace("\n+", "\n")))
    t(["Artifact", "Content"], [["REPORT.pdf / REPORT.html / REPORT.md", "Complete findings, methods, plots and evidence"],
        ["TAXONOMY.md / taxonomy.csv", "Six failure types, frequencies, severity, unknown/NA counts"],
        ["review/index.html / annotations.csv", "Filterable offline census of all 1,390 outputs; per-case labels and evidence"],
        ["review/batches, additional_batches, adjudications.json", "Initial observations and explicit same-assistant revisions"],
        ["figures/; per_target_diagnostics.csv", "Exportable PNG/PDF/SVG plots and measurements for all 2,565 targets"],
        ["analysis_manifest.json; validation.json", "Input hashes, commands, coverage checks and artifact verification"]], [1.8,3])
    p("The intermediate 552-case sample and labels are archived for transparency; final reporting uses the full census and unit inclusion probability. The final `review/annotations.csv` is authoritative. This report does not modify historical experiment outputs, scientific configurations or the registry.")
    return blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.analysis_dir
    manifest = json.loads((output / "analysis_manifest.json").read_text())
    if manifest["inputs"]["per_target.csv"] != "b7466a86b480a705982d9c5706eade76900e91ced842dcbaad14d395dc7a8615":
        raise ValueError("Narrative is specific to the recovered expanded-vocabulary run")
    rows = json.loads((output / "per_target_diagnostics.json").read_text())
    saved_ids = [r["id"] for r in rows if r["status"] == "success"]
    review = read_review(output / "review/annotations.csv", saved_ids)
    review_digest = sha256(output / "review/annotations.csv")
    if len(review) != 1390 or review_digest != manifest["review_sha256"]:
        raise ValueError("Census identity or label manifest changed")
    if review_digest != "ebf4f6b036d8cf592fe115e2e4811da891d29e1fe56a37a3da0ac7090b718062":
        raise ValueError("Update the frozen narrative before reporting revised census labels")
    if any(float(r["inclusion_probability"]) != 1 for r in review):
        raise ValueError("Census must have unit inclusion probabilities")
    taxonomy = taxonomy_counts(review)
    grouped, summary = additional_tables(output, rows, review)
    make_figures(output, taxonomy, census=True)
    figures(output, grouped, review)
    blocks = report_blocks(output, manifest, rows, review, taxonomy, summary)
    md = render_markdown(blocks)
    (output / "REPORT.md").write_text(md, encoding="utf-8")
    import markdown
    body = markdown.markdown(md, extensions=["tables", "fenced_code"]).replace("<!-- pagebreak -->", '<div class="pagebreak"></div>')
    nav = '<nav><a href="REPORT.pdf">PDF report</a><a href="review/index.html">Browse all 1,390 outputs</a><a href="taxonomy.csv">Taxonomy CSV</a><a href="occlusion_bins.csv">Occlusion bins</a></nav>'
    (output / "REPORT.html").write_text(html_document("Expanded-vocabulary appearance failures", nav + body), encoding="utf-8")
    scales = render_pdf(output, blocks, footer_text="Person 2 | Expanded-vocabulary PD-MC | Census: 1,390 saved outputs | 7 November 2026 UTC")
    render_review(output, review, census=True)
    first = next(i for i, b in enumerate(blocks) if b[0] == "heading" and b[2].startswith("Failure taxonomy:"))
    last = next(i for i in range(first, len(blocks)) if blocks[i][0] == "page")
    (output / "TAXONOMY.md").write_text(render_markdown(blocks[first:last]), encoding="utf-8")
    rubric = "# Appearance census rubric\n\n" + table_markdown(["Failure", "Definition"], [[NAMES[k], DEFINITIONS[k]] for k in FAILURES])
    rubric += "\n\n0 = no observed defect; 1 = mild/local; 2 = clear consequential; 3 = extensive corruption; U = unknown; NA = no applicable repeated pattern. Labels overlap. U and NA never count as negative. Judge returned-object consistency; separately inspect raw RGB for extra objects and record their output scope. Compare full original context before confirming co-occurrence. A wrong initial/final mask or geometry distortion alone is not semantic drift. Hidden appearance has no RGB ground truth.\n\nAll 1,390 outputs were reviewed by one assistant at panel/contact-sheet resolution. Borderline cases can be missed. This is not independent human consensus. Severity is ordinal and causes are untested hypotheses.\n"
    (output / "review/RUBRIC.md").write_text(rubric, encoding="utf-8")
    revisions = json.loads((output / "review/adjudications.json").read_text())
    (output / "review/revision_notes.md").write_text("# Same-assistant source/identity rechecks\n\nInitial census labels and batch notes are preserved. The final CSV applies these explicit revisions. This is not an independent second rater.\n\n" + "\n\n".join(f"- Case {r['review_number']}: {r['evidence']}" for r in revisions) + "\n", encoding="utf-8")
    write_json(output / "report_build.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "review_sha256": sha256(output / "review/annotations.csv"), "pdf_page_scales": scales,
        "report_pages": len(scales), "reviewed": len(review), "source_sha256": sha256(Path(__file__))})
    print(json.dumps({"report": str(output / "REPORT.pdf"), "pages": len(scales), "reviewed": len(review), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
