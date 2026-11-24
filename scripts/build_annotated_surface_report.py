#!/usr/bin/env python3
"""Build a scientific report, figures and complete gallery from audited results."""
from collections import Counter
from datetime import datetime, timezone
import csv
import html
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json
from amodal_benchmark.references import reference_paths
from amodal_benchmark.protocol import METRICS

LABELS = {'CLIP': 'CLIP', 'LPIPS': 'LPIPS', 'Feature Similarity': 'VGG similarity', 'SSIM': 'SSIM'}


def number(value, precision=6):
    return 'unavailable' if value is None else f'{value:.{precision}f}'


def gain_text(value):
    return 'unavailable' if value is None else f'{value:+.3f}%'


def interval_text(values):
    return 'unavailable' if any(v is None for v in values) else f'[{values[0]:+.3f}, {values[1]:+.3f}]%'


def thumbnail(source, destination, size=(340, 280)):
    with Image.open(source) as image:
        image = image.convert('RGBA')
        image.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', size, (247, 248, 250))
        canvas.paste(image, ((size[0]-image.width)//2, (size[1]-image.height)//2), image)
        canvas.save(destination, quality=92)


def figures(report, audit, pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    items = [(LABELS[m], audit['metrics'][m]) for m in METRICS if audit['metrics'][m] is not None]
    fig, ax = plt.subplots(figsize=(8.4, 3.3))
    for i, (label, row) in enumerate(items):
        low, high = row['relative_ci_percent']
        if low is not None:
            ax.hlines(i, low, high, color='#176b87', linewidth=2)
        if row['relative_gain_percent'] is not None:
            ax.plot(row['relative_gain_percent'], i, 'o', color='#176b87')
    ax.axvline(0, color='#999999', linestyle='--', linewidth=1)
    ax.set(yticks=range(len(items)), yticklabels=[name for name, _ in items],
           xlabel='Relative gain (%) with nominal 95% image-cluster interval',
           title='Annotated vocabulary: frozen 204-target comparison')
    ax.invert_yaxis()
    fig.tight_layout()
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(report / ('paired-gains.' + extension), dpi=180)
    plt.close(fig)
    rows = [('Overall', audit['metrics']['LPIPS'])] + [(source, values['LPIPS']) for source, values in audit['per_source'].items()]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    for i, (source, row) in enumerate(rows):
        if row is None:
            continue
        low, high = row['absolute_ci']
        ax.hlines(i, low, high, color='#176b87', linewidth=2)
        ax.plot(row['absolute_gain'], i, 'o', color='#176b87')
    ax.axvline(0, color='#999999', linestyle='--', linewidth=1)
    ax.set(yticks=range(len(rows)), yticklabels=[name for name, _ in rows],
           xlabel='Absolute LPIPS gain: control minus Surface-PMC',
           title='All targets retained; source analyses are exploratory')
    ax.invert_yaxis()
    fig.tight_layout()
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(report / ('source-lpips.' + extension), dpi=180)
    plt.close(fig)


def method_figure(report):
    """Render the shared inputs and the bundled progressive intervention."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.set(xlim=(0, 10), ylim=(0, 6))
    ax.axis('off')
    def box(x, y, width, height, text, color='#e9f2f6'):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
            boxstyle='round,pad=0.07', linewidth=.8, edgecolor='#6c8390', facecolor=color))
        ax.text(x+width/2, y+height/2, text, ha='center', va='center', fontsize=10)
    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
            mutation_scale=11, color='#536e7d', linewidth=1.2))
    box(1.3, 4.8, 7.4, .85, 'Shared observed RGB, annotated scene queries, matcher v2\nExact paired SAM references; same model weights and runtime')
    label_background = dict(facecolor='white', edgecolor='none', pad=3)
    ax.text(2.2, 4.35, 'Annotated PD-MC control', ha='center', weight='bold', fontsize=11,
            bbox=label_background, zorder=5)
    ax.text(7.2, 4.35, 'Transferred Surface-PMC', ha='center', weight='bold', fontsize=11,
            bbox=label_background, zorder=5)
    box(.45, 3.15, 3.5, .85, 'One diffusion candidate\nper progressive pass')
    box(5.15, 3.15, 4.1, .85, 'Eight diffusion candidates per pass\nTarget-only candidate segmentation')
    box(5.15, 1.85, 4.1, .85, 'Frozen 22-feature ranker\nAffine + pool-consensus RGB correction')
    box(.45, .55, 3.5, .85, 'Upstream occlusion check\nContinue or save / fail')
    box(5.15, .55, 4.1, .85, 'Resegment and check occlusion\nContinue or save / fail')
    for first, second in (((2.2,4.8),(2.2,4.02)),((7.2,4.8),(7.2,4.02)),
                           ((2.2,3.15),(2.2,1.43)),((7.2,3.15),(7.2,2.72)),
                           ((7.2,1.85),(7.2,1.43))):
        arrow(first, second)
    ax.text(2.2, 2.25, 'Original stopping rules\nIteration indices 0–5', ha='center', va='center', fontsize=9,
            bbox=label_background, zorder=5)
    ax.text(5, .03, 'Paired evaluation retains every target and failure. K=1 versus K=8 measures the bundled method plus sampling.',
            ha='center', va='bottom', fontsize=8.2)
    fig.tight_layout(pad=.3)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(report/('method-diagram.'+extension), dpi=180)
    plt.close(fig)


def equation_figures(report):
    """Export publication-portable definitions without depending on test outcomes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = {
        'method-equations': [
            ('Frozen feature scaling and selection',
             r'$z_{kj}=(\mathrm{impute}(f_{kj},m_j)-\mu_j)/\sigma_j,\quad r_k=\sum_{j=1}^{22}w_jz_{kj}$',
             r'$k^*=\mathrm{argmin}_{k\in\mathcal{V}}\ r_k$',
             'Choose candidate zero if the valid set is empty. No evaluation reference enters this rule.'),
            ('Historical within-pool ridge fit; transferred without refitting',
             r'$\tilde{z}_{ik}=z_{ik}-\overline{z}_i,\quad \tilde{y}_{ik}=y_{ik}-\overline{y}_i$',
             r'$\hat{w}=\mathrm{argmin}_w\ \Vert\tilde{y}-\tilde{Z}w\Vert_2^2+100\Vert w\Vert_2^2$',
             'The prior artificial-occlusion development targets combine perceptual errors and CLIP.'),
            ('Bounded observation-anchored affine correction, per channel',
             r'$a=\mathrm{clip}\left(\frac{\mathrm{mean}[(x-m_x)(y-m_y)]+400}{\mathrm{mean}[(x-m_x)^2]+400},0.8,1.25\right)$',
             r'$\delta_k(p)=\mathrm{clip}((C_k(p)-m_x)(a-1)+m_y-m_x,-32,32)$',
             'Apply outside measured visible support, then restore observed pixels; support guards apply.'),
            ('Masked low-frequency pool-consensus correction',
             r'$u_k=\frac{G_4*[S_k(T-C_k^\prime)]}{\max(G_4*S_k,10^{-6})}$',
             r'$C_k^{\prime\prime}=C_k^\prime+0.5\,\mathrm{clip}(u_k,-32,32)\quad\mathrm{on}\ S_k\setminus V$',
             'S requires at least four of eight masks; T is the masked median. Other pixels are retained.')],
        'analysis-equations': [
            ('Paired gain, with positive values always favorable',
             r'$d_i=h(B_i-A_i),\quad\hat{\Delta}=\frac{1}{N}\sum_{i=1}^{N}d_i$',
             r'$\hat{g}=100\hat{\Delta}/|\overline{A}|$',
             'h = -1 for LPIPS, +1 for similarities. A is control; B is Surface-PMC.'),
            ('Source-stratified image-cluster bootstrap',
             r'$\hat{\Delta}^{(b)}=\frac{\sum_s\sum_j D_{s,I_{sj}^{(b)}}}{\sum_s\sum_j n_{s,I_{sj}^{(b)}}}$',
             r'$\mathrm{CI}_{95\%}=[q_{0.025}(\hat{\Delta}^{(b)}),\ q_{0.975}(\hat{\Delta}^{(b)})]$',
             'D is the image sum of target gains, n its target count. Resample images within each source.'),
            ('Exact failure-inclusive LPIPS decomposition',
             r'$\hat{\Delta}=\sum_{g\in\{\mathrm{shared,gained,lost,both\ failed}\}}\frac{1}{204}\sum_{i\in g}(A_i-B_i)$',
             r'$L_i^{\mathrm{failure}}=1$',
             'Every contribution uses the same 204-target denominator. CLIP instead uses shared successes.'),
            ('Final saved-pool selection and headroom identities',
             r'$\overline{e}-e_{k^*}=(\overline{e}-\overline{e}_{valid})+(\overline{e}_{valid}-e_{k^*})$',
             r'$\overline{e}-e_{k^*}=(\overline{e}-\min_k e_k)-(e_{k^*}-\min_k e_k)$',
             'These lower-is-better identities condition on the final mask and earlier corrected trajectory.')]
    }
    for name, rows in groups.items():
        fig, ax = plt.subplots(figsize=(10.2, 9.4))
        ax.set(xlim=(0,1), ylim=(0,1))
        ax.axis('off')
        for i, (title, first, second, caption) in enumerate(rows):
            top = .97-.25*i
            ax.text(.025,top,title,va='top',fontsize=12,weight='bold',color='#21485c')
            ax.text(.04,top-.055,first,va='top',fontsize=13)
            ax.text(.04,top-.132,second,va='top',fontsize=13)
            ax.text(.025,top-.199,caption,va='top',fontsize=8.7)
        fig.tight_layout(pad=.4)
        for extension in ('png','pdf','svg'):
            fig.savefig(report/(name+'.'+extension),dpi=180)
        plt.close(fig)


def gallery(report, study, pairs, official):
    images = report / 'images'
    images.mkdir(exist_ok=True)
    items, manifest = [], {}
    for index, row in enumerate(pairs):
        identity, source_row = row['id'], official[row['id']]
        key = f'case-{index+1:03d}'
        panels = [('Source image', ROOT / 'datasets/open_world_amodal' / source_row['image'])]
        reference = reference_paths(study / 'baseline-cohort/visible_references', source_row)['crop']
        panels.append(('Visible SAM reference', reference))
        panels += [(title, study / (side+'-cohort') / 'targets' / identity / 'completion.png'
                    if row[side+'_status'] == 'success' else None)
                   for side, title in (('baseline', 'Annotated PD-MC'), ('surface', 'Surface-PMC'))]
        tiles = []
        for position, (title, source) in enumerate(panels):
            if source is not None:
                name = key + '-' + str(position) + '.jpg'
                thumbnail(source, images / name)
                manifest['images/'+name] = dict(source=source.relative_to(ROOT).as_posix(), source_sha256=sha256(source),
                                              thumbnail_sha256=sha256(images/name))
                content = f'<img loading="lazy" src="images/{name}" alt="{html.escape(title)}">'
            else:
                side = 'baseline' if position == 2 else 'surface'
                content = '<div class="missing">' + html.escape(row[side+'_status'] + ': ' + str(row[side+'_reason'])) + '</div>'
            tiles.append(f'<figure>{content}<figcaption>{title}</figcaption></figure>')
        delta = row['baseline']['LPIPS'] - row['surface']['LPIPS']
        first, second = row['baseline_status'] == 'success', row['surface_status'] == 'success'
        transition = 'shared' if first and second else 'gained' if second else 'lost' if first else 'both_failed'
        direction = 'gain' if delta > 1e-12 else 'loss' if delta < -1e-12 else 'tie'
        metric_rows = ''.join('<tr><th>'+LABELS[m]+'</th><td>'+number(row['baseline'][m], 4)+'</td><td>'+
                              number(row['surface'][m], 4)+'</td></tr>' for m in METRICS)
        items.append(dict(id=identity, key=key, source=row['source'], direction=direction, transition=transition,
                          delta=delta, html=f'<article data-direction="{direction}" data-transition="{transition}" '
                          f'data-search="{html.escape(identity.lower()+" "+row["label"].lower(), quote=True)}">'
                          f'<h2>{key}: {html.escape(identity)}</h2><p>LPIPS gain {delta:+.6f}; {transition}; '
                          f'{row["surface_passes"]} Surface-PMC passes.</p><div class="panels">'+''.join(tiles)+
                          '</div><table><tr><th>Metric</th><th>Control</th><th>Surface</th></tr>'+metric_rows+'</table></article>'))
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Annotated Surface-PMC: all 204 cases</title>
<style>body{font:16px/1.5 system-ui,sans-serif;color:#172534;background:#f4f6f8;margin:0}header,main{max-width:1480px;margin:auto;padding:24px}header{position:sticky;top:0;background:#fff;z-index:2;border-bottom:1px solid #ccd7df}h1{font-size:26px;margin:0}h2{font-size:18px;overflow-wrap:anywhere}article{background:#fff;padding:20px;margin:18px 0;border:1px solid #d4dee5;border-radius:10px}.panels{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}figure{margin:0}img{width:100%;height:auto;display:block}figcaption{text-align:center}.missing{height:280px;display:flex;align-items:center;justify-content:center;background:#edf0f3;padding:10px;box-sizing:border-box;overflow-wrap:anywhere}table{border-collapse:collapse;margin-top:12px;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:3px 22px 3px 0}select,input{font:inherit;padding:7px;margin:10px 8px 0 0}a{color:#176b87}[hidden]{display:none!important}@media(max-width:750px){.panels{grid-template-columns:repeat(2,minmax(0,1fr))}header{position:static}.missing{height:220px}}</style>
<header><h1>Surface-PMC on annotated vocabulary</h1><p>All 204 frozen targets. Displayed crops use each output's own geometry; thumbnails are not pixel-aligned.
Visible references do not reveal the true hidden surface. A saved output is not a certificate of visual completeness.</p>
<a href="REPORT.pdf">Research report (PDF)</a> · <a href="REPORT.md">Report and evidence links</a> · <a href="paired_records.csv">All paired values (CSV)</a><br>
<input id="search" placeholder="Search image, source or label" aria-label="Search cases">
<select id="direction" aria-label="LPIPS change"><option value="">All LPIPS changes</option><option value="gain">LPIPS gains</option><option value="loss">LPIPS losses</option><option value="tie">LPIPS ties</option></select>
<select id="transition" aria-label="Coverage"><option value="">All coverage states</option><option value="shared">Shared successes</option><option value="gained">Recovered successes</option><option value="lost">Lost successes</option><option value="both_failed">Both failed</option></select>
<span id="count"></span></header><main>''' + ''.join(item['html'] for item in items) + '''</main><script>
const articles=[...document.querySelectorAll('article')], search=document.getElementById('search'), direction=document.getElementById('direction'), transition=document.getElementById('transition');
function filter(){let n=0;for(const a of articles){a.hidden=!(a.dataset.search.includes(search.value.toLowerCase())&&(!direction.value||a.dataset.direction===direction.value)&&(!transition.value||a.dataset.transition===transition.value));if(!a.hidden)n++;}document.getElementById('count').textContent=n+' / '+articles.length+' cases';}
for(const e of [search,direction,transition])e.addEventListener('input',filter);filter();</script></html>'''
    (report / 'gallery.html').write_text(page, encoding='utf-8')
    write_json(report / 'gallery-manifest.json', dict(cases=len(items), panels=manifest,
        direction_counts=dict(Counter(row['direction'] for row in items)),
        transition_counts=dict(Counter(row['transition'] for row in items))))
    return items


def qualitative_examples(report, items):
    """Show deterministic metric extremes, explicitly separate from prevalence."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    shared = [row for row in items if row['transition'] == 'shared']
    positive = sorted((row for row in shared if row['delta'] > 1e-12),
                      key=lambda row: (-row['delta'], row['id']))[:2]
    negative = sorted((row for row in shared if row['delta'] < -1e-12),
                      key=lambda row: (row['delta'], row['id']))[:2]
    selected = positive + negative
    fig, axes = plt.subplots(max(1, len(selected)), 4,
                             figsize=(12, 2.85 * max(1, len(selected))), squeeze=False)
    for axis in axes.flat:
        axis.axis('off')
    titles = ('Observed source', 'Visible SAM reference', 'Annotated PD-MC', 'Surface-PMC')
    manifest = json.loads((report / 'gallery-manifest.json').read_text())['panels']
    evidence = []
    for index, row in enumerate(selected):
        panels = []
        for position, title in enumerate(titles):
            relative = 'images/' + row['key'] + '-' + str(position) + '.jpg'
            with Image.open(report / relative) as image:
                axes[index, position].imshow(image)
            axes[index, position].set_title(title, fontsize=9)
            panels.append(dict(path=relative, sha256=sha256(report / relative)))
        axes[index, 0].text(0, -.10, row['id'] + f' | LPIPS gain {row["delta"]:+.6f}',
                           transform=axes[index, 0].transAxes, fontsize=8, va='top')
        geometry = {}
        for side, position in (('baseline', 2), ('surface', 3)):
            native = ROOT / manifest[panels[position]['path']]['source']
            with Image.open(native) as native_image:
                alpha = np.array(native_image.convert('RGBA'))[..., 3]
                geometry[side] = dict(width=native_image.width, height=native_image.height,
                    nonzero_alpha_pixels=int(np.count_nonzero(alpha)),
                    native_sha256=sha256(native))
            record = json.loads((native.parent / 'result.json').read_text())
            geometry[side]['offset_rc'] = record['geometry']['offset_rc']
        evidence.append(dict(id=row['id'], delta=row['delta'], panels=panels, native_geometry=geometry))
    if not selected:
        axes[0, 0].text(0, .5, 'No nonzero shared-success LPIPS differences.')
    fig.subplots_adjust(left=.02, right=.99, top=.97, bottom=.07, hspace=.40, wspace=.05)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(report / ('qualitative-examples.' + extension), dpi=180)
    plt.close(fig)
    write_json(report / 'qualitative-examples.json', dict(
        selection='Largest two positive and two negative LPIPS differences among shared successes; ID breaks ties.',
        limitation='Metric-selected extremes, not a representative or blinded human evaluation. Crops differ between methods.',
        cases=evidence))
    return evidence


def pdf_report(report, sections, metric_rows, source_rows, pool_rows, coefficients):
    import matplotlib
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as PDFImage
    fonts = Path(matplotlib.get_data_path()) / 'fonts/ttf'
    pdfmetrics.registerFont(TTFont('DejaVu', str(fonts/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('DejaVuBold', str(fonts/'DejaVuSans-Bold.ttf')))
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = 'DejaVu'
    styles['BodyText'].fontSize = 9
    styles['BodyText'].leading = 13
    styles['Heading1'].fontName = styles['Heading2'].fontName = 'DejaVuBold'
    small = ParagraphStyle('smallcell', parent=styles['BodyText'], fontSize=7.4, leading=10)
    flow = []
    def paragraph(text, style='BodyText'):
        flow.append(Paragraph(html.escape(text), styles[style]))
        flow.append(Spacer(1, 6))
    def table(headers, rows, widths=None):
        wrapped = [[Paragraph(html.escape(str(v)), small) for v in row] for row in [headers, *rows]]
        value = Table(wrapped, colWidths=widths, repeatRows=1, hAlign='LEFT')
        value.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eef4')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),6),
            ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#93aabd'))]))
        flow.extend([value, Spacer(1, 10)])
    paragraph('Surface-PMC on annotated vocabulary', 'Title')
    paragraph(sections['conclusion'], 'Heading2')
    paragraph(sections['scope'])
    table(['Metric', 'Control', 'Surface', 'Relative gain', '95% interval', 'n'], metric_rows, [78,63,63,75,128,30])
    flow.append(PDFImage(str(report/'paired-gains.png'), width=480, height=189))
    paragraph(sections['statistics'])
    paragraph(sections['reference'])
    flow.append(PageBreak())
    paragraph('Method and frozen transfer design', 'Heading1')
    flow.append(PDFImage(str(report/'method-diagram.png'), width=480, height=278.4))
    for name in ('question', 'method', 'rank', 'correction', 'selection'):
        paragraph(sections[name])
    flow.append(PageBreak())
    paragraph('Exact 22-feature ranker', 'Heading2')
    paragraph('The coefficients below apply after frozen median imputation and standardization. Full precision center, scale, medians and weights are in ranker-parameters.csv. No coefficients were fitted on this cohort.')
    table(['Feature', 'Coefficient', 'Feature', 'Coefficient'],
          [[coefficients[i][0], f'{coefficients[i][1]:+.6f}', coefficients[i+11][0], f'{coefficients[i+11][1]:+.6f}'] for i in range(11)],
          [160,78,160,78])
    flow.append(PageBreak())
    paragraph('Failure accounting and source sensitivity', 'Heading1')
    paragraph(sections['coverage'])
    table(['Source', 'Targets', 'Control saved', 'Surface saved', 'LPIPS gain', '95% interval'], source_rows,
          [68,48,72,72,78,99])
    flow.append(PDFImage(str(report/'source-lpips.png'), width=480, height=206))
    paragraph(sections['decomposition'])
    paragraph(sections['cost'])
    flow.append(PageBreak())
    paragraph('Fixed final-pool mechanism diagnostic', 'Heading1')
    paragraph(sections['pool'])
    if pool_rows:
        oracle_rows = [row for row in pool_rows if row[0] == 'Metric-specific oracle vs control']
        policy_rows = [row for row in pool_rows if row[0] != 'Metric-specific oracle vs control']
        table(['Contrast', 'Metric', 'Relative gain', '95% interval', 'n'], policy_rows, [165,75,85,110,42])
        flow.append(PageBreak())
        paragraph('Reference-informed raw-pool headroom', 'Heading1')
        paragraph('Each row chooses its own best raw candidate using the evaluation metric. These different oracle choices are individual bounds, not a jointly attainable or deployable method.')
        table(['Contrast', 'Metric', 'Relative gain', '95% interval', 'n'], oracle_rows, [165,75,85,110,42])
    if sections.get('pool_findings'):
        paragraph(sections['pool_findings'])
    paragraph(sections['pool_limits'])
    flow.append(PageBreak())
    paragraph('Evidence, reproducibility and next inference', 'Heading1')
    for name in ('audit', 'examples', 'full', 'decision', 'limits'):
        paragraph(sections[name])
    paragraph('Companion files: gallery.html (all 204 targets), paired_records.csv, paired_records.json, audit.json, pool-audit.json, gallery-manifest.json, ranker-parameters.csv, and the evidence directory. All figures are also exported as PDF and SVG.')
    flow.append(PageBreak())
    paragraph('Illustrative shared-success gains and losses', 'Heading1')
    paragraph('The largest two positive and two negative LPIPS differences are selected mechanically, with target ID breaking ties. These extremes do not estimate prevalence or human preference. All 204 cases, including failures, remain in gallery.html. Each column retains its own crop geometry; the panels are not pixel-aligned.')
    with Image.open(report/'qualitative-examples.png') as example_image:
        example_height = 500 * example_image.height / example_image.width
    flow.append(PDFImage(str(report/'qualitative-examples.png'), width=500, height=example_height))
    paragraph(sections['example_findings'])
    for filename, title in (('method-equations', 'Mathematical method definitions'),
                            ('analysis-equations', 'Statistical estimand and diagnostic identities')):
        flow.append(PageBreak())
        paragraph(title, 'Heading1')
        flow.append(PDFImage(str(report/(filename+'.png')),width=500,height=460.8))
        paragraph('Full assumptions, numerical guards, historical training targets and interpretation limits are in METHODS.md. These equations describe the frozen implementation; they do not imply hidden-surface accuracy.')
    def footer(canvas, doc):
        canvas.setFont('DejaVu', 7)
        canvas.drawString(42, 25, 'Annotated vocabulary · frozen transfer study · 12 November 2026')
        canvas.drawRightString(A4[0]-42, 25, str(doc.page))
    SimpleDocTemplate(str(report/'REPORT.pdf'), pagesize=A4, rightMargin=42, leftMargin=42,
                      topMargin=40, bottomMargin=42, title='Surface-PMC on annotated vocabulary').build(flow, onFirstPage=footer, onLaterPages=footer)


def main():
    design = ROOT / 'output/surface-annotated-20261112-design'
    active = json.loads((design/'active-study.json').read_text())
    study = ROOT / active['study']
    report = study / 'report'
    audit = json.loads((report/'audit.json').read_text())
    pairs = json.loads((report/'paired_records.json').read_text())
    comparison = json.loads((study/'cohort-comparison/comparison.json').read_text())
    freeze = json.loads((design/'method-freeze.json').read_text())
    if not audit['passed'] or audit['targets'] != 204 or len(pairs) != 204:
        raise ValueError('Require a successful complete cohort audit')
    pool = json.loads((report/'pool-audit.json').read_text()) if (report/'pool-audit.json').is_file() else None
    if pool is not None and not pool['passed']:
        raise ValueError('Pool audit has not passed')
    methods = ROOT/'docs/SURFACE_ANNOTATED_METHODS_20261112.md'
    shutil.copy2(methods, report/'METHODS.md')
    evidence = report/'evidence'
    evidence.mkdir(exist_ok=True)
    for name in ('cohort.json','method-freeze.json','study-freeze.json','fresh-reference-amendment.json',
                 'pool-freshrefs-freeze.json','freshrefs-submission.json','reference-drift-diagnostic.json'):
        shutil.copy2(design/name, evidence/name)
    for name in ('SURFACE_ANNOTATED_20261112.md','SURFACE_ANNOTATED_POOL_DIAGNOSTIC_20261112.md',
                 'SURFACE_ANNOTATED_REFERENCE_AMENDMENT_20261112.md','SURFACE_ANNOTATED_CONTINUATION_20261112.md'):
        shutil.copy2(ROOT/'docs'/name, evidence/name)
    shutil.copy2(study/'cohort-comparison/comparison.json', evidence/'comparison.json')
    status_paths = sorted((design/'status').glob('*.json'))
    status = json.loads(status_paths[-1].read_text()) if status_paths else None
    if status:
        write_json(evidence/'job-status.json', status)
    submission = json.loads((design/'freshrefs-submission.json').read_text())
    costs = {}
    for job in submission['jobs']:
        if job['stage'] in ('baseline-cohort', 'surface-cohort') and status:
            state = status['jobs'][job['job_id']]
            elapsed = state['elapsed']
            days, clock = elapsed.split('-') if '-' in elapsed else ('0', elapsed)
            hours, minutes, seconds = map(int, clock.split(':'))
            costs[job['stage']] = dict(job_id=job['job_id'], state=state['state'], elapsed=elapsed,
                reserved_gpu_hours=4*(int(days)*24+hours+minutes/60+seconds/3600))
    write_json(report/'cohort-compute.json', costs)
    official = {row['id']: row for row in read_official(ROOT/'datasets/open_world_amodal')[0]}
    with (report/'paired_records.csv').open('w', encoding='utf-8', newline='') as stream:
        columns = ['id','source','image_identity','baseline_status','surface_status','baseline_reason','surface_reason',
                   'baseline_iterations','surface_iterations','surface_passes','lpips_gain']
        columns += [side+'_'+metric for side in ('baseline','surface') for metric in METRICS]
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        for row in pairs:
            values = {key:row[key] for key in columns if key in row}
            values['lpips_gain'] = row['baseline']['LPIPS']-row['surface']['LPIPS']
            values.update({side+'_'+metric:row[side][metric] for side in ('baseline','surface') for metric in METRICS})
            writer.writerow(values)
    ranker = freeze['ranker']
    with (report/'ranker-parameters.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['feature','coefficient','median_imputation','center','scale'])
        writer.writerows(zip(ranker['feature_names'],ranker['coefficients'],ranker['medians'],ranker['center'],ranker['scale']))
    figures(report, audit, pairs)
    method_figure(report)
    equation_figures(report)
    items = gallery(report, study, pairs, official)
    examples = qualitative_examples(report, items)
    primary = audit['metrics']['LPIPS']
    low, high = primary['absolute_ci']
    verdict = ('The primary LPIPS endpoint improves' if low > 0 else
               'The primary LPIPS endpoint worsens' if high < 0 else 'The primary LPIPS result is inconclusive')
    baseline = audit['baseline_statistics']['Overall']
    surface = audit['surface_statistics']['Overall']
    sections = dict(
        conclusion=verdict + f': relative gain {gain_text(primary["relative_gain_percent"])}, nominal 95% interval {interval_text(primary["relative_ci_percent"])}.',
        scope='Completed frozen transfer experiment: 204 targets from 192 source images, with every failure retained. Both arms use annotated detector vocabulary, semantic matcher v2, identical newly frozen SAM references and the same runtime. This cohort is separate from the longer 2,565-target continuation.',
        question='Research question: does the existing corrected, eight-candidate progressive Surface-PMC method improve the team’s annotated-vocabulary baseline? The generator and feature networks remain pretrained. The 22-feature ridge selector transfers unchanged from the final 10 November method. No new training or tuning was performed.',
        method='At each progressive pass, eight candidates use seed slots 11, 22, 33, 44, 55, 66, 77, 88. Their actual seeds are SHA256(surface-progressive-v1:target_seed:iteration:candidate_seed), interpreted from the first eight little-endian bytes modulo 2^63−1. The original observed target pixels anchor surface features; generated pixels never become new observations. Initial detection and progressive stopping use annotated scene vocabulary; candidate masks use the original target-only query.',
        rank='Selection equation: z[k,j] = (impute(f[k,j], median[j]) − center[j]) / scale[j]; score[k] = Σ_j coefficient[j] z[k,j]; k* = argmin over valid candidates score[k]. If all eight candidates are invalid, the frozen fallback is candidate zero. The selector uses boundary/color/texture, preservation, expansion, consensus, mask disagreement, observed perceptual/deep features and category-text cosine. CLIP is therefore an explicit selection feature, not an independent selector-blind endpoint.',
        correction='After selection scoring, each candidate receives the frozen observation-anchored affine correction (strength 1, bounded RGB shift 32, fitted channel slopes clipped to [0.8,1.25]) and low-frequency pool-consensus correction (strength 0.5, Gaussian sigma 4, minimum mask agreement 0.5). Original visible pixels are restored before native JPEG serialization; final JPEG pixels may differ. Candidate generation, selection, correction and altered progressive trajectories form one combined intervention.',
        selection='The cohort was selected by a pre-generation source-stratified image hash rule, retaining all targets of selected images: VG 107 images/109 targets, COCOA 58/67, free_images 14/14, LAION 13/14. All 477 previous controlled-study images were excluded by identity/source bytes; decoded/thumbnail checks rejected one near-copy before generation. The final cohort has no flagged prior-image overlaps. This screen cannot exclude every transformed duplicate.',
        statistics='For LPIPS, d_i = control_i − surface_i; for similarity metrics, d_i = surface_i − control_i. Absolute gain = mean(d_i); relative gain = 100 mean(d_i) / |mean(control_i)|. Nominal 95% percentile intervals use 5,000 source-stratified image-cluster resamples, keeping every target from each sampled image. LPIPS is the sole primary endpoint; secondary and source intervals are exploratory and not multiplicity-adjusted. LPIPS failures score 1; VGG/SSIM failures score 0. CLIP is paired only on shared successes, with no invented failure score.',
        reference='Protocol amendment: the original historical-reference attempts failed an exact SAM check. An isolated probe reproduced one changed mask pixel, which changes five eroded alpha pixels. The fresh control generated new references, copied with hashes before Surface-PMC inference; its exact-mask and paired-reference checks remain mandatory. Historical attempts and original freezes are preserved. The reason for the numerical drift is not isolated.',
        coverage=f'Annotated control saved {baseline["successes"]}/204 outputs, with {baseline["failures"]} failures; Surface-PMC saved {surface["successes"]}/204, with {surface["failures"]} failures. Shared successes: {comparison["shared_successes"]}; recovered: {comparison["gained_successes"]}; lost: {comparison["lost_successes"]}. A successful record means a saved completion, not verified hidden geometry or visual completeness.',
        decomposition='Exact decomposition of the all-target absolute LPIPS gain: ' + '; '.join(f'{name} {value:+.8f}' for name,value in audit['lpips_gain_decomposition'].items()) + '. Each contribution uses the same 204-target denominator. Shared-success analysis is a diagnostic and does not replace the failure-inclusive primary.',
        cost=f'Surface-PMC logged {audit["decisions_replayed"]} progressive candidate pools and {audit["generated_candidates"]} candidate images, including {audit["fallback_decisions"]} all-invalid fallback decisions. Both runs requested four Quadro RTX 6000 GPUs and four workers. The control uses K=1 and the method K=8: the comparison measures method plus extra sampling, not the selector’s isolated effect at equal compute. Allocation wall times and job states are preserved in evidence/job-status.json.',
        pool='The secondary diagnostic scores all eight saved raw candidates from each successful final Surface-PMC pass, retaining the corrected final mask, native crop/JPEG serialization, earlier trajectory and failures. It compares the frozen raw choice against uniform expectation over all candidates, uniform over valid candidates, first-seed candidate and category-text selection. Corrected versus selected raw isolates the final RGB correction under these fixed conditions.' if pool else 'The pre-specified final-pool diagnostic is pending; no diagnostic result is inferred from the progressive result.',
        pool_limits='This diagnostic does not rerun progressive random trajectories. Fixed corrected masks may favor the selected trajectory. Reference-informed oracle choices differ by metric and are only individual upper bounds; they are not deployable selectors or a jointly attainable score vector. No tuning uses these results.',
        audit=f'Independent local audit reproduced all {audit["targets"]} target records, reference/input identities, failure counts and paired intervals; replayed {audit["decisions_replayed"]} selector decisions from logged observable features and exact corrected RGB at saved candidate resolution; and checked {audit["pool_image_hashes_checked"]} pool-artifact hashes. Local feature extraction and neural metrics were not rerun. ' + (f'The diagnostic additionally replayed {pool["candidate_derivations"]} native candidate derivations; maximum original-score replay drift was {max(pool["cache_replay_drift"].values()):.3g}.' if pool else 'Remote neural-metric replay is part of the pending diagnostic.'),
        examples='The companion gallery contains every frozen target, including all failed targets. Filters expose metric gains, losses and coverage states; no recovered or lost successes were observed in this cohort. The displayed image crops retain each method’s own geometry. The largest metric gains and losses are convenience views, not a blinded human evaluation or a representative qualitative sample.',
        full='The full 2,565-target annotated control and Surface-PMC continuation are separate operational benchmark runs. They include previously used training/development images and are not wholly unseen generalization. The frozen unseen-image inventory supports a later sensitivity analysis. Only complete runs passing provenance verification can be registered. Valid longer inference remains running beyond the interactive work window.',
        decision=('The completed cohort supports an improvement in failure-inclusive visible-reference LPIPS for this transferred method; full-benchmark generality and equal-budget progressive selection remain open.' if low > 0 else
                  'The completed cohort does not establish a primary improvement. Preserve this result and await the full operational comparison; do not retune against the frozen cohort or replace its denominator.'),
        limits='Natural hidden RGB is unavailable. SAM-visible-reference appearance metrics measure a different quantity from true hidden-surface reconstruction. More candidates, RGB correction, mask changes and progressive path changes are bundled. A point gain or nonzero confidence interval does not prove amodal shape accuracy, causal selector benefit at equal compute, or human preference. There is no predeclared 3% pass threshold for this annotated study.')
    sections['cost'] += ' Cohort allocation totals: ' + '; '.join(
        f'{name} job {row["job_id"]}, {row["elapsed"]} wall time, {row["reserved_gpu_hours"]:.3f} reserved GPU-hours'
        for name,row in costs.items()) + '. Reserved GPU-hours include idle/scoring/verification time and are not measured GPU utilization. Failed historical attempts and separate smoke/diagnostic allocations are additional costs.'
    sections['question'] += ' The historical selector fit used 183 development images with artificial occlusion. The complete derivation and exact metric operations are in METHODS.md.'
    sections['limits'] += ' Observed LPIPS features share a metric family with the primary endpoint; historical fitting also used perceptual and CLIP objectives. These scores are not wholly independent perceptual assessors of the selector. The bootstrap conditions on the frozen model and candidate seeds; it does not include retraining or seed-to-seed uncertainty.'
    sections['coverage'] += f' Iteration-limited saved outputs: control {audit["capped_successes"]["baseline"]}, Surface-PMC {audit["capped_successes"]["surface"]}. The unchanged inclusive maximum of 5 permits six progressive passes.'
    sections['coverage'] += (' Targets that never reached surface sampling: '
        + '; '.join(f'{name}: {count}' for name, count in audit['no_surface_sampling_reasons'].items())
        + '. These upstream gates account for all 117 shared failures in this cohort. '
        'Changing candidate selection inside a pass cannot affect a target that never enters that pass.')
    sections['example_findings'] = ('Post-hoc inspection of these four metric-selected examples: '
        + '; '.join(f'{row["id"]}: control {row["native_geometry"]["baseline"]["nonzero_alpha_pixels"]:,} '
                    f'versus Surface-PMC {row["native_geometry"]["surface"]["nonzero_alpha_pixels"]:,} foreground pixels'
                    for row in examples)
        + '. The native widths, heights, offsets and image hashes are retained in qualitative-examples.json. '
        'The teddy and saxophone gains accompany smaller masks; the phonebooth and table losses accompany larger masks. '
        'The table control retains only a small fragment, while Surface-PMC retains more of the table yet scores worse. '
        'These observations illustrate why visible-reference scores cannot certify hidden-shape accuracy. '
        'They are assistant visual observations, not blinded human ratings, and establish no cohort-wide prevalence or causal effect.')
    shared_lpips = audit['shared_success_metrics']['LPIPS']
    if shared_lpips is not None:
        sections['coverage'] += (' On the shared-success diagnostic, LPIPS relative gain is '
            + gain_text(shared_lpips['relative_gain_percent']) + ' with interval '
            + interval_text(shared_lpips['relative_ci_percent'])
            + f' (n={shared_lpips["n"]}); this conditional result does not replace the 204-target primary.')
    if pool:
        selector = pool['contrasts']['frozen_raw_vs_uniform_raw']
        correction = pool['contrasts']['corrected_vs_frozen_raw']
        oracle = pool['contrasts']['oracle_LPIPS_vs_baseline']['LPIPS']
        sections['pool_findings'] = (
            'At fixed final masks and earlier trajectories, frozen raw selection versus uniform raw choice gives LPIPS '
            + gain_text(selector['LPIPS']['relative_gain_percent']) + ' ('
            + interval_text(selector['LPIPS']['relative_ci_percent']) + ') and CLIP '
            + gain_text(selector['CLIP']['relative_gain_percent']) + ' ('
            + interval_text(selector['CLIP']['relative_ci_percent']) + '). '
            'Final RGB correction versus the selected raw image gives LPIPS '
            + gain_text(correction['LPIPS']['relative_gain_percent']) + ' ('
            + interval_text(correction['LPIPS']['relative_ci_percent']) + ') and CLIP '
            + gain_text(correction['CLIP']['relative_gain_percent']) + ' ('
            + interval_text(correction['CLIP']['relative_ci_percent']) + '). '
            'The per-metric raw LPIPS oracle versus the annotated control gives '
            + gain_text(oracle['relative_gain_percent']) + ' ('
            + interval_text(oracle['relative_ci_percent']) + '). '
            'All intervals here are nominal exploratory 95% intervals. The raw-pool oracle bounds only these '
            'saved raw candidates at fixed masks/trajectories; corrected RGB is outside that candidate set. '
            'These observations do not isolate the effect of selection at earlier progressive steps.')
    repeated = audit['repeated_smoke']
    sections['audit'] += f' Repeated eight-target GPU smoke: {repeated["exact_native_artifact_matches"]} native RGB/mask artifact hashes match the cohort, with {len(repeated["differences"])} targets showing status or native-artifact differences.'
    reference_audit = design/'freshrefs-reference-audit.json'
    if reference_audit.is_file():
        changed = json.loads(reference_audit.read_text())
        shutil.copy2(reference_audit,evidence/reference_audit.name)
        sections['reference'] += f' Completed cohort audit found {changed["changed_masks"]} historical/current mask changes, totaling {changed["total_changed_mask_pixels"]} changed pixels, with {changed["changed_geometry"]} changed crop geometries.'
    metric_rows = [[LABELS[m],number(v['baseline']),number(v['method']),gain_text(v['relative_gain_percent']),interval_text(v['relative_ci_percent']),v['n']]
                   for m,v in audit['metrics'].items() if v is not None]
    source_rows = []
    for source in audit['per_source']:
        first, second = audit['baseline_statistics'][source], audit['surface_statistics'][source]
        v = audit['per_source'][source]['LPIPS']
        source_rows.append([source,first['targets'],first['successes'],second['successes'],gain_text(v['relative_gain_percent']),interval_text(v['relative_ci_percent'])])
    pool_rows = []
    if pool:
        for name in ('frozen_raw_vs_uniform_raw','frozen_raw_vs_uniform_valid','frozen_raw_vs_first_raw','frozen_raw_vs_semantic_only','corrected_vs_frozen_raw'):
            for metric in METRICS:
                row = pool['contrasts'][name][metric]
                if row:
                    pool_rows.append([name.replace('_vs_', ' vs ').replace('_',' '),LABELS[metric],gain_text(row['relative_gain_percent']),interval_text(row['relative_ci_percent']),row['n']])
        for metric in METRICS:
            row = pool['contrasts']['oracle_' + metric + '_vs_baseline'][metric]
            if row:
                pool_rows.append(['Metric-specific oracle vs control', LABELS[metric],
                    gain_text(row['relative_gain_percent']), interval_text(row['relative_ci_percent']), row['n']])
    def md_table(headers, rows):
        return ['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |'] + ['| '+' | '.join(map(str,row))+' |' for row in rows]
    lines = ['# Surface-PMC on annotated vocabulary','',sections['conclusion'],'',sections['scope'],'',
             '[PDF](REPORT.pdf) · [All 204 cases](gallery.html) · [Paired CSV](paired_records.csv) · [Audit](audit.json)','',
             '## Frozen primary and secondary results','']
    lines += md_table(['Metric','Control','Surface','Relative gain','Nominal 95% interval','Paired n'],metric_rows)
    lines += ['',sections['statistics'],'','![Paired gains](paired-gains.png)','',
              '## Research question and method','',sections['question'],'','![Paired method diagram](method-diagram.png)','',sections['method'],'',sections['rank'],'',sections['correction'],'',
              r'$$z_{kj} = (\operatorname{impute}(f_{kj},m_j)-\mu_j)/\sigma_j,\qquad k^* = \arg\min_{k\in\mathcal V}\sum_{j=1}^{22} w_j z_{kj}.$$','',
              '[Full-precision frozen parameters](ranker-parameters.csv) · [Mathematical methods and training derivation](METHODS.md) · [Method equations (PDF)](method-equations.pdf) · [Statistical equations (PDF)](analysis-equations.pdf).','',sections['selection'],'',
              '## Reference-runtime amendment','',sections['reference'],'',
              '[Exact amendment and failed-attempt lineage](evidence/SURFACE_ANNOTATED_REFERENCE_AMENDMENT_20261112.md)','',
              '## Coverage and source sensitivity','',sections['coverage'],'']
    lines += md_table(['Source','Targets','Control saved','Surface saved','LPIPS gain','95% interval'],source_rows)
    lines += ['',sections['decomposition'],'','![Source sensitivity](source-lpips.png)','',sections['cost'],'',
              '## Fixed final-pool diagnostic','',sections['pool'],'']
    if pool_rows:
        lines += md_table(['Contrast','Metric','Relative gain','95% interval','n'],pool_rows)
        lines += ['','[Complete independent pool audit, including individual oracle bounds](pool-audit.json)','']
    lines += [sections.get('pool_findings',''),'',sections['pool_limits'],'','## Evidence and interpretation','',sections['audit'],'',sections['examples'],'',
              '![Largest two shared-success LPIPS gains and losses](qualitative-examples.png)','',
              '[Exact example-selection rule and image hashes](qualitative-examples.json). These metric-selected extremes are not representative prevalence estimates.','',sections['example_findings'],'',sections['limits'],'',
              '## Full-benchmark continuation','',sections['full'],'',
              '[Exact continuation jobs and recovery instructions](evidence/SURFACE_ANNOTATED_CONTINUATION_20261112.md).','',sections['decision'],'']
    (report/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    pdf_report(report, sections, metric_rows, source_rows, pool_rows, list(zip(ranker['feature_names'],ranker['coefficients'])))
    write_json(report/'report-content.json',dict(at=datetime.now(timezone.utc).isoformat(),sections=sections,
        metric_rows=metric_rows,source_rows=source_rows,pool_rows=pool_rows,
        audit_sha256=sha256(report/'audit.json'),pool_audit_sha256=sha256(report/'pool-audit.json') if pool else None,
        methods_sha256=sha256(report/'METHODS.md'),
        builder_sha256=sha256(Path(__file__))))
    print(json.dumps(dict(report=str(report),cohort_targets=204,pool_diagnostic_included=pool is not None,verdict=verdict)))


if __name__ == '__main__':
    main()
