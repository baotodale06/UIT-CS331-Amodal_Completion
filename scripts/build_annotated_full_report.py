#!/usr/bin/env python3
"""Build the completed annotated benchmark report from independently audited data."""
from datetime import datetime, timezone
import csv
import html
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from review_annotated_full import OUTPUT, STUDY, DESIGN, RECOVERY, read, require

ORDER = ('LPIPS', 'CLIP', 'Feature Similarity', 'SSIM')
LABELS = {'LPIPS':'LPIPS (lower is better)', 'CLIP':'CLIP (shared successes)',
          'Feature Similarity':'VGG feature similarity', 'SSIM':'SSIM'}


def interval(values):
    return f'[{values[0]:+.3f}%, {values[1]:+.3f}%]'


def metric_table(metrics):
    lines = ['| Metric | Control | Surface-PMC | Relative gain | Nominal 95% interval | n |',
             '| --- | ---: | ---: | ---: | --- | ---: |']
    for metric in ORDER:
        row = metrics[metric]
        lines.append(f'| {LABELS[metric]} | {row["baseline"]:.6f} | {row["method"]:.6f} | '
                     f'{row["relative_gain_percent"]:+.3f}% | {interval(row["relative_ci_percent"])} | {row["n"]} |')
    return lines


def figures(audit, cohort):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    scopes = [('Full benchmark (2,565 targets)', audit['metrics'], -.22, '#176B87'),
              ('Unseen subset (2,017 targets)', audit['unseen_metrics'], 0, '#BD5C18'),
              ('Primary cohort (204 targets)', cohort['metrics'], .22, '#6F747A')]
    for label, metrics, offset, color in scopes:
        for i, metric in enumerate(ORDER):
            row = metrics[metric]
            low, high = row['relative_ci_percent']
            ax.plot([low, high], [i+offset]*2, color=color, linewidth=1.6)
            ax.plot(row['relative_gain_percent'], i+offset, 'o', color=color,
                    label=label if i == 0 else None, markersize=5)
    ax.axvline(0, color='#555', linewidth=1)
    ax.axvline(3, color='#888', linestyle=':', linewidth=1.5, label='Earlier 3% practical target')
    ax.set(yticks=range(4), yticklabels=['LPIPS', 'CLIP', 'VGG similarity', 'SSIM'],
           xlabel='Relative gain (%) with nominal 95% image-cluster intervals',
           title='Three overlapping analysis scopes; primary cohort stays inconclusive')
    ax.invert_yaxis()
    ax.legend(loc='upper center', bbox_to_anchor=(.5,-.17), ncols=2, fontsize=9, frameon=False)
    fig.tight_layout()
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(OUTPUT/('scope-comparison.'+extension), dpi=180, bbox_inches='tight')
    plt.close(fig)


def pdf(markdown):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
    for name, filename in (('Arial', 'arial.ttf'), ('Arial-Bold', 'arialbd.ttf')):
        pdfmetrics.registerFont(TTFont(name, 'C:/Windows/Fonts/'+filename))
    pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='Arial-Bold')
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = 'Arial'
    styles['Normal'].fontSize, styles['Normal'].leading = 9, 12.5
    styles['Heading1'].fontName = styles['Heading2'].fontName = 'Arial-Bold'
    styles['Heading1'].fontSize = 18
    styles['Heading2'].fontSize = 12
    styles.add(ParagraphStyle('Cell', fontName='Arial', fontSize=7.4, leading=10, alignment=TA_LEFT))
    def rich(text):
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = html.escape(text)
        return re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text).replace('`', '')
    story, lines, index = [], markdown.splitlines(), 0
    while index < len(lines):
        line = lines[index]
        if not line:
            index += 1
            continue
        if line.startswith('|'):
            data = []
            while index < len(lines) and lines[index].startswith('|'):
                cells = [cell.strip() for cell in lines[index].strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+', cell) for cell in cells):
                    data.append([Paragraph(rich(cell), styles['Cell']) for cell in cells])
                index += 1
            count = len(data[0])
            widths = {6:[112,66,66,66,118,42], 7:[74,53,53,53,53,53,131],
                      5:[170,80,80,70,70], 4:[190,94,94,92], 3:[270,100,100]}.get(count)
            if data[0][0].getPlainText() == 'Source':
                widths = [65,48,70,70,70,147]
            table = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8F0F3')),
                ('VALIGN',(0,0),(-1,-1),'TOP'), ('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#77919F')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F6F8FA')]),
                ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            story.extend([table, Spacer(1,10)])
            continue
        if line.startswith('!['):
            path = OUTPUT/re.search(r'\(([^)]+)\)',line).group(1)
            width, height = ImageReader(str(path)).getSize()
            story.append(Image(str(path), width=470, height=470*height/width))
        elif line.startswith('# '):
            story.append(Paragraph(rich(line[2:]), styles['Heading1']))
        elif line.startswith('## '):
            story.append(Paragraph(rich(line[3:]), styles['Heading2']))
        else:
            story.extend([Paragraph(rich(line), styles['Normal']), Spacer(1,7)])
        index += 1
    def footer(canvas, document):
        canvas.setFont('Arial',8)
        canvas.setFillColor(colors.HexColor('#66717A'))
        canvas.drawString(52,28,'Annotated Surface-PMC | Completed benchmark review | 14 November 2026')
        canvas.drawRightString(543,28,str(document.page))
    SimpleDocTemplate(str(OUTPUT/'REPORT.pdf'), pagesize=(595,842), leftMargin=52,
        rightMargin=53, topMargin=42, bottomMargin=48,
        title='Annotated Surface-PMC: completed full benchmark', author='Amodal completion research team').build(
            story, onFirstPage=footer, onLaterPages=footer)


def main():
    audit, pairs = read(OUTPUT/'audit.json'), read(OUTPUT/'paired-records.json')
    require(audit['passed'], 'Review audit must pass first')
    require(audit['source_sha256'] == sha256(ROOT/'scripts/review_annotated_full.py'), 'Audit source changed')
    comparison = read(STUDY/'full-comparison/comparison.json')
    cohort = read(STUDY/'cohort-comparison/comparison.json')
    evidence = OUTPUT/'evidence'
    evidence.mkdir(exist_ok=True)
    copies = {STUDY/'full-comparison'/name: evidence/name for name in
        ('comparison.json','independent-audit.json','baseline-verification.json','surface-verification.json')}
    copies.update({ROOT/'docs/SURFACE_ANNOTATED_METHODS_20261112.md': evidence/'METHODS.md',
                   STUDY/'report/REPORT.pdf': evidence/'PRIMARY_COHORT_REPORT.pdf',
                   DESIGN/'method-freeze.json': evidence/'method-freeze.json',
                   DESIGN/'study-freeze.json': evidence/'study-freeze.json',
                   DESIGN/'unseen-inventory.json': evidence/'unseen-inventory.json',
                   RECOVERY/'remote-receipt.json': evidence/'export-receipt.json',
                   RECOVERY/'remote-manifest.json': evidence/'export-manifest.json',
                   RECOVERY/'recovery-receipt.json': evidence/'recovery-receipt.json'})
    for source, target in copies.items():
        if target.exists():
            require(sha256(source) == sha256(target), 'Preserve existing differing evidence')
        else:
            shutil.copyfile(source,target)
    rows = []
    for pair in pairs:
        row = {key:value for key,value in pair.items() if key not in ('baseline','surface')}
        for metric in ORDER:
            for side in ('baseline','surface'):
                row[side+'_'+metric] = pair[side][metric]
        row['absolute_lpips_gain'] = pair['baseline']['LPIPS']-pair['surface']['LPIPS']
        rows.append(row)
    with (OUTPUT/'paired-records.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figures(audit,cohort)
    lpips = audit['metrics']['LPIPS']
    c = audit['coverage']
    lines = ['# Annotated Surface-PMC: completed full benchmark', '',
        '14 November 2026. This completes the frozen 12 November continuation. Both arms use annotated vocabulary, matched fresh SAM references, and the existing pretrained generator. No new training, fitting, retuning, or inference was launched for this review.', '',
        f'**The full operational benchmark improves LPIPS by {lpips["relative_gain_percent"]:.3f}% '
        f'{interval(lpips["relative_ci_percent"])}. The earlier 3% practical target is not met.** '
        'The independent 204-target primary cohort remains inconclusive; the larger full benchmark does not replace that primary test.', '',
        'The full comparison, provenance verification, registration and independent cluster audit completed on 13 November. Local checksum, provenance, registry, selector and statistical replay now pass. This is evidence of a modest gain under the specified visible-reference protocol, not proof of correct unseen surfaces.', '',
        '## Full operational comparison', '',
        f'{audit["targets"]:,} targets from {audit["images"]:,} images. Positive relative gain is favorable. LPIPS, VGG similarity and SSIM include all failures with the original penalties; CLIP uses only paired shared successes.', '',
        *metric_table(audit['metrics']), '',
        f'Absolute LPIPS reduction: {lpips["absolute_gain"]:.8f}, nominal 95% interval '
        f'[{lpips["absolute_ci"][0]:+.8f}, {lpips["absolute_ci"][1]:+.8f}]. '
        'Full results include images previously used in controlled development or training; this is an operational benchmark comparison.', '',
        '## Unseen-image sensitivity and the original primary cohort', '',
        f'The frozen unseen inventory contains {audit["unseen_targets"]:,} targets from {audit["unseen_images"]:,} images. '
        'It excludes prior images by identity and source bytes. Only the original primary cohort received the exhaustive near-copy screen. These sensitivity intervals are nominal and exploratory.', '',
        *metric_table(audit['unseen_metrics']), '',
        'The original 192-image / 204-target primary cohort has LPIPS +0.250% [-0.269%, +0.810%], with 87 saved outputs in each arm. '
        'It remains inconclusive, including after the full results are available. Full, unseen and primary analyses overlap; they are not three independent replications.', '',
        '![Relative gains across the three scopes](scope-comparison.png)', '',
        '## Coverage, failures and source-level results', '',
        f'The control saves 1,394 outputs and retains 1,171 failures; Surface-PMC saves 1,399 and retains 1,166 failures. '
        f'There are {c["shared_success"]:,} shared successes, {c["gained"]} gained outputs, {c["lost"]} lost outputs and '
        f'{c["both_failed"]:,} failures in both arms. The net coverage change is +5 targets (+{500/2565:.3f} percentage points). '
        f'Iteration-capped saved outputs change from {audit["capped_successes"]["baseline"]} to '
        f'{audit["capped_successes"]["surface"]}. A saved output or non-cap stop does not establish complete or correct hidden geometry.', '',
        '| Source | Targets | Control saved | Surface saved | LPIPS gain | 95% interval |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for source, values in audit['per_source'].items():
        first,second = (comparison[side+'_statistics'][source] for side in ('baseline','surface'))
        value = values['LPIPS']
        lines.append(f'| {source} | {first["targets"]} | {first["successes"]} | {second["successes"]} | '
                     f'{value["relative_gain_percent"]:+.3f}% | {interval(value["relative_ci_percent"])} |')
    lines += ['', 'Source intervals are post-hoc descriptive and unadjusted for multiple comparisons. They do not establish category-wide or causal mechanisms.', '',
        '| Failure reason | Control | Surface-PMC |', '| --- | ---: | ---: |']
    for reason in sorted(set(audit['failure_reasons']['baseline']) | set(audit['failure_reasons']['surface'])):
        lines.append(f'| {reason} | {audit["failure_reasons"]["baseline"].get(reason,0)} | {audit["failure_reasons"]["surface"].get(reason,0)} |')
    lines += ['', 'Of the 1,166 Surface-PMC failures, 1,156 stop before candidate sampling. The current downstream selector does not run on these cases. No-occlusion decisions, ambiguous instances and unmatched labels dominate this upstream bottleneck.', '',
        'The diagnostic decomposition below partitions the full absolute LPIPS difference using the full 2,565-target denominator. These contributions add to the full difference; they are descriptive, not isolated causal effects.', '',
        '| Transition group | Targets | Absolute LPIPS contribution |', '| --- | ---: | ---: |']
    for group in ('shared_success','gained','lost','both_failed'):
        lines.append(f'| {group.replace("_"," ")} | {c[group]} | {audit["lpips_gain_contributions"][group]:+.8f} |')
    shared = audit['shared_success_metrics']['LPIPS']
    lines += ['', f'Conditional LPIPS on shared successes improves {shared["relative_gain_percent"]:.3f}% '
        f'{interval(shared["relative_ci_percent"])}. Its denominator differs from the all-target endpoint. '
        'The overall unpaired CLIP means are 28.159741 (control n=1,394) and 28.533665 (Surface-PMC n=1,399); '
        'the paired CLIP estimate in the results table uses 1,387 shared successes.', '',
        '## Method, estimand and uncertainty', '',
        'The intervention bundles K=8 candidate generation, the unchanged 22-feature ridge selector, observation anchoring, affine correction and pool consensus inside progressive completion. The control uses K=1. Annotated scene queries are present in both arms; candidate masks retain target-only segmentation. The selector parameters and corrections match the historical frozen method.', '',
        'For LPIPS, Δ = (1/N) Σᵢ(Lcontrol,i − Lsurface,i), and relative gain = 100 × Δ / mean(Lcontrol). '
        'Failed completions have LPIPS=1, VGG similarity=0 and SSIM=0. CLIP has no invented failure score. '
        'Higher-is-better metrics reverse the difference direction. The bootstrap resamples source images within each dataset source, retains all targets belonging to an image, and uses 5,000 draws with seed 20261110. Intervals are percentile 2.5%–97.5%; secondary/source analyses have no multiplicity adjustment.', '',
        'Fresh control references were copied exactly before Surface-PMC execution. Paired input/reference hashes, seeds, model/checkpoint configuration, metric operations and installed packages match. The prior one-pixel historical-reference drift and its amendment remain documented. '
        'The comparison measures the complete intervention plus extra sampling; it cannot isolate selection from correction or increased compute. See [the preserved mathematical methods](evidence/METHODS.md).', '',
        '## Compute and completed jobs', '',
        f'{audit["decisions_replayed"]:,} saved candidate pools contain {audit["recorded_candidates"]:,} candidate images, with '
        f'{audit["fallback_decisions"]} all-invalid fallbacks. These counts describe recorded pools, not any discarded work before timeout.', '',
        '| Arm | Jobs including cohort | Allocation wall hours | Reserved GPU-hours |',
        '| --- | --- | ---: | ---: |']
    for side in ('baseline','surface'):
        value = audit['compute'][side]
        lines.append(f'| {side} | {", ".join(value["jobs"])} | {value["allocation_seconds"]/3600:.3f} | {value["reserved_gpu_hours"]:.3f} |')
    lines += ['', f'The allocation-time ratio is {audit["compute"]["allocation_ratio"]:.3f}×. '
        'Both arms used four Quadro RTX 6000 GPUs and four workers. Cohort work is counted because its results were reused in the full run. '
        'Reserved GPU-hours include setup, scoring, verification and idle worker time; they are not measured utilization or pure model latency. '
        'Failed historical-reference attempts, smoke, the cohort pool diagnostic, CPU audits and queue time are excluded from this two-arm comparison.', '',
        'Surface job 39490 reached its 24-hour limit. The pre-submitted unchanged continuation 39491 completed. '
        'Finalizer 39492 verified both complete runs and registered them; independent full/unseen audit 39497 and metadata export 39495 completed. '
        'No job remains to resume in this study. Accounting timestamps in the raw receipt use cluster local time; elapsed seconds are used for cost.', '',
        '## Evidence checks and limitations', '',
        f'All {audit["recovered_metadata_members"]:,} metadata members match independent remote SHA-256 values. '
        f'Local execution provenance passes for both arms, all 408 original cohort result hashes are preserved, and '
        f'{audit["historical_result_hash_checks"]:,} historical snapshot-to-result hashes agree. '
        f'All {audit["decisions_replayed"]:,} stored-feature selector choices and losses replay, including seeds and decision hashes. '
        'Full and unseen means, denominators, transitions and bootstrap intervals reproduce the separate cluster audit within 1e-10. '
        'Both registry entries match execution, protocol, summary and metric identities.', '',
        'The full native PNGs remain on the cluster, where full artifact and provenance verification passed. This local review does not rerun the neural metrics, feature extraction or full RGB-correction replay. '
        'The primary cohort already has separate native recovery and exact local/cluster correction replay. These scopes must not be conflated.', '',
        'Visible-reference appearance scores and CLIP do not establish natural hidden RGB accuracy, amodal shape correctness or human preference. '
        'Changed masks, crop geometry and progressive trajectories can affect appearance scores. The cohort-only fixed-pool diagnostic does not identify the selector contribution over the full progressive run.', '',
        '## Research decision and next experiment', '',
        'Record Surface-PMC as a modest operational improvement under this protocol, with the original primary transfer test still inconclusive and the earlier 3% practical target unmet. '
        'The approximately 4.73× allocation cost does not justify automatically replacing the annotated PD-MC operational baseline. The lower iteration-cap count (209 to 199) is descriptive and does not establish hidden-object correctness.', '',
        'The next mechanism experiment should use a new frozen design with equal-budget progressive random selection and a correction-only ablation, plus independent geometry or blinded human assessment. '
        'Existing controlled, cohort and full outcomes must be treated as inspected data for any future tuning. No such experiment has been launched by this continuation.', '',
        '## Files and reproduction', '',
        '[Local audit](audit.json), [all 2,565 paired targets](paired-records.csv), [selector replay](decision-replay.json), '
        '[original full comparison](evidence/comparison.json), [cluster audit](evidence/independent-audit.json), '
        '[primary cohort PDF](evidence/PRIMARY_COHORT_REPORT.pdf), and [job/registry recovery receipt](remote-evidence/receipt.json) are retained beside this report.', '',
        'From the original workspace, reproduce the metadata review with `python scripts/review_annotated_full.py` and rebuild this report with '
        '`python scripts/build_annotated_full_report.py`. These commands preserve the original full comparison and remote audit. '
        'The portable package contains the independently hashed full metadata archive and receipts; native full PNGs are outside that package.', '']
    markdown = '\n'.join(lines)
    (OUTPUT/'REPORT.md').write_text(markdown,encoding='utf-8')
    pdf(markdown)
    write_json(OUTPUT/'report-build.json', dict(at=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha256(Path(__file__)), audit_sha256=sha256(OUTPUT/'audit.json'),
        report_sha256=sha256(OUTPUT/'REPORT.md'), pdf_sha256=sha256(OUTPUT/'REPORT.pdf'),
        csv_rows=len(rows), evidence={target.relative_to(OUTPUT).as_posix():sha256(target) for target in copies.values()}))
    print(json.dumps(dict(report=str(OUTPUT/'REPORT.md'), pdf=str(OUTPUT/'REPORT.pdf'), paired_rows=len(rows))))


if __name__ == '__main__':
    main()
