#!/usr/bin/env python3
"""Build measured Surface-PMC tables, figures, complete case gallery and PDF."""
import argparse
from collections import Counter
import csv
from html import escape
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.surface import METHODS
from amodal_benchmark.surface_evaluation import summarize_cases, stratified_bootstrap

NAMES = {'fixed_seed_11': 'Fixed seed 11', 'uniform_random': 'Uniform selection (expected)',
    'uniform_valid_surface': 'Uniform among valid candidates',
    'visible_preservation': 'Visible-pixel preservation',
    'boundary': 'Boundary', 'color': 'Color', 'texture': 'Texture', 'boundary_color': 'Boundary + color',
    'boundary_texture': 'Boundary + texture', 'surface_full': 'Full surface', 'local_surface': 'Local surface',
    'development_selected': 'Development-selected rule', 'oracle_hidden_mae': 'Hidden-MAE oracle (unavailable)'}
ORDER = ['fixed_seed_11', 'uniform_random', 'uniform_valid_surface', 'visible_preservation', *METHODS, 'development_selected', 'oracle_hidden_mae']


def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def attach_perceptual(experiment, rows):
    for row in rows:
        path = experiment / 'perceptual' / (row['case_id'] + '.json')
        if path.exists():
            perceptual = json.loads(path.read_text())
            by_seed = {v['seed']: v['masked_spatial_lpips'] for v in perceptual['values']}
            for metric in row['metrics']:
                metric['masked_spatial_lpips'] = by_seed[metric['seed']]


def perceptual_summary(rows):
    complete = [row for row in rows if all(m.get('masked_spatial_lpips') is not None for m in row['metrics'])]
    if not complete:
        return dict(n=0, methods={})
    result = {}
    sources = [row['source'] for row in complete]
    for method in ORDER:
        values, gains = [], []
        for row in complete:
            pool = np.array([m['masked_spatial_lpips'] for m in row['metrics']])
            if method == 'uniform_random':
                value = float(pool.mean())
            elif method == 'uniform_valid_surface':
                valid_indices = [i for i, loss in enumerate(row['selected']['surface_full']['losses']) if loss is not None] or [0]
                value = float(pool[valid_indices].mean())
            else:
                value = float(pool[row['selected'][method]['index']])
            values.append(value)
            gains.append(float(pool.mean() - value))
        result[method] = dict(lpips=stratified_bootstrap(values, sources),
                              gain_vs_uniform=stratified_bootstrap(gains, sources))
    return dict(n=len(complete), excluded_incomplete_pools=len(rows) - len(complete), methods=result)


def panel(row, experiment, destination):
    input_dir = experiment / 'inputs' / row['case_id']
    truth_dir = experiment / 'evaluation_only' / row['case_id']
    folder = experiment / 'full/candidates' / row['case_id']
    selected = row['selected']['surface_full']['index']
    local = row['selected']['local_surface']['index']
    oracle = row['selected']['oracle_hidden_mae']['index']
    expected = np.mean([m['hidden_core_mae'] for m in row['metrics']])
    gain = expected - row['metrics'][selected]['hidden_core_mae']
    width, cell, top = 1536, 256, 60
    result = Image.new('RGB', (width, 418), '#fafafa')
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
        small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    except OSError:
        font = small = ImageFont.load_default()
    draw.text((10, 8), f"{row['cohort']} / {row['source']} / {row['label']} / {row['case_id']}", fill='#17212b', font=font)
    draw.text((10, 32), f"Uniform expected MAE {expected:.4f}; full-surface gain {gain:+.4f}. "
        'Truth is evaluation only. All four seeds are shown. * = fixed-seed fallback.', fill='#17212b', font=small)
    for index, (label, path) in enumerate([('Observed input', input_dir / 'observed.png'),
            ('Held-out source truth', truth_dir / 'truth.png')]):
        result.paste(Image.open(path).convert('RGB').resize((cell, cell)), (index * cell, top))
        draw.text((index * cell + 7, top + cell + 8), label, fill='black', font=font)
    for index, metric in enumerate(row['metrics']):
        seed = metric['seed']
        path = folder / f'seed-{seed}.png'
        left = (index + 2) * cell
        if path.exists():
            result.paste(Image.open(path).convert('RGB').resize((cell, cell)), (left, top))
        tags = []
        if index == selected: tags.append('FULL*' if row['selected']['surface_full']['fallback'] else 'FULL')
        if index == local: tags.append('LOCAL*' if row['selected']['local_surface']['fallback'] else 'LOCAL')
        if index == oracle: tags.append('ORACLE')
        draw.text((left + 7, top + cell + 7), f"Seed {seed}: {' / '.join(tags)}", fill='#164a66', font=small)
        score = row['scores'][index]
        draw.text((left + 7, top + cell + 28), f"Hidden MAE {metric['hidden_core_mae']:.4f}", fill='black', font=small)
        terms = '/'.join(f"{score[k]:.3f}" if score.get(k) is not None else 'NA' for k in ('boundary', 'color', 'texture'))
        draw.text((left + 7, top + cell + 48), f"B/C/T: {terms}", fill='black', font=small)
        draw.text((left + 7, top + cell + 68), f"Valid {score.get('valid', False)}; hidden mask recall "
                  f"{metric['hidden_mask_recall']:.2f}", fill='black', font=small)
    result.save(destination, quality=90)
    return float(gain)


def mask_panel(row, experiment, destination):
    """Make candidate-dependent support and fixed evaluation coverage visible."""
    case = row['case_id']
    read_mask = lambda path: np.array(Image.open(path).convert('L')) > 0
    visible = read_mask(experiment / 'inputs' / case / 'visible.png')
    core = read_mask(experiment / 'evaluation_only' / case / 'hidden_core.png')
    result = Image.new('RGB', (1536, 340), '#fafafa')
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 8), 'MASK DIAGNOSTIC: green = visible support; orange = held-out core; '
              'cyan = core recovered by predicted mask; gray = other predicted pixels.', fill='black', font=font)
    result.paste(Image.open(experiment / 'inputs' / case / 'observed.png').convert('RGB').resize((256, 256)), (0, 40))
    support = np.zeros((*visible.shape, 3), np.uint8)
    support[visible], support[core] = (42, 155, 87), (235, 145, 35)
    result.paste(Image.fromarray(support).resize((256, 256), Image.Resampling.NEAREST), (256, 40))
    draw.text((8, 303), 'Observed input', fill='black', font=font)
    draw.text((264, 303), 'Fixed evaluation support', fill='black', font=font)
    for index, metric in enumerate(row['metrics']):
        path = experiment / 'full/candidates' / case / f"seed-{metric['seed']}-mask.png"
        predicted = read_mask(path) if path.exists() else np.zeros(visible.shape, bool)
        colored = np.zeros_like(support)
        colored[predicted] = (125, 125, 125)
        colored[visible] = (42, 155, 87)
        colored[core] = (235, 145, 35)
        colored[core & predicted] = (40, 185, 210)
        left = (index + 2) * 256
        result.paste(Image.fromarray(colored).resize((256, 256), Image.Resampling.NEAREST), (left, 40))
        draw.text((left + 8, 303), f"Seed {metric['seed']}: hidden recall {metric['hidden_mask_recall']:.2f}", fill='black', font=font)
    result.save(destination, quality=90)


def plots(summary, cohorts, rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    selected = ['fixed_seed_11', 'uniform_random', 'visible_preservation', 'boundary', 'color', 'texture', 'surface_full',
                'local_surface', 'development_selected', 'oracle_hidden_mae']
    fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
    for i, method in enumerate(selected):
        estimate = summary[method]['hidden_core_mae']
        color = '#007a80' if method == 'surface_full' else '#d28535' if method == 'local_surface' else '#6f7b86'
        ax.errorbar(estimate['mean'], i, xerr=[[estimate['mean'] - estimate['ci_low']],
            [estimate['ci_high'] - estimate['mean']]], fmt='o', color=color, capsize=4)
    ax.set_yticks(range(len(selected)), [NAMES[m] for m in selected])
    ax.invert_yaxis()
    ax.set_xlabel('Held-out target RGB MAE / 255 (lower is better)')
    ax.set_title(f'Pooled image-disjoint test cases: n={len(rows)}; 95% source-stratified bootstrap intervals')
    fig.savefig(output / 'hidden_mae.png', dpi=180)
    fig.savefig(output / 'hidden_mae.pdf')
    plt.close(fig)


    selected = ['boundary', 'color', 'texture', 'boundary_color', 'boundary_texture', 'surface_full', 'local_surface', 'development_selected']
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    colors = ['#6f7b86', '#007a80', '#a45c27']
    for j, (cohort, estimates) in enumerate(cohorts.items()):
        for i, method in enumerate(selected):
            estimate = estimates[method]['gain_vs_uniform']
            ax.errorbar(estimate['mean'], i + (j - 1) * .22,
                xerr=[[estimate['mean'] - estimate['ci_low']], [estimate['ci_high'] - estimate['mean']]],
                fmt='o', markersize=4, capsize=3, color=colors[j], label=cohort if i == 0 else None)
    ax.axvline(0, color='black', linewidth=.8)
    ax.set_yticks(range(len(selected)), [NAMES[m] for m in selected])
    ax.invert_yaxis()
    ax.set_xlabel('Paired MAE gain over expected uniform selection (positive is better)')
    ax.legend(loc='best')
    fig.savefig(output / 'paired_gains.png', dpi=180)
    fig.savefig(output / 'paired_gains.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for ax, method in zip(axes, ('surface_full', 'local_surface')):
        xs, ys = [], []
        for row in rows:
            losses = row['selected'][method]['losses']
            valid = [(loss, metric['hidden_core_mae']) for loss, metric in zip(losses, row['metrics']) if loss is not None]
            if len(valid) >= 2:
                values = np.array(valid)
                if values[:, 0].std() > 1e-8:
                    xs.extend(((values[:, 0] - values[:, 0].mean()) / values[:, 0].std()).tolist())
                    ys.extend((values[:, 1] - values[:, 1].mean()).tolist())
        ax.scatter(xs, ys, s=12, alpha=.35, color='#007a80')
        ax.axhline(0, color='gray', lw=.6)
        ax.axvline(0, color='gray', lw=.6)
        correlation = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else float('nan')
        ax.set_title(f'{NAMES[method]}: within-pool association r={correlation:.3f}')
        ax.set_xlabel('Compatibility loss standardized within pool')
        ax.set_ylabel('Hidden MAE minus pool mean')
    fig.savefig(output / 'score_error_association.png', dpi=180)
    fig.savefig(output / 'score_error_association.pdf')
    plt.close(fig)


def explanation_plot(row, scales, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3.6), layout='constrained')
    bottoms = np.zeros(len(row['metrics']))
    colors = {'boundary': '#447a96', 'color': '#d19b40', 'texture': '#6b9c78'}
    valid = [value is not None for value in row['selected']['surface_full']['losses']]
    for key, weight in METHODS['surface_full'].items():
        values = np.array([score[key] / scales[key] * weight if ok else 0.
                           for score, ok in zip(row['scores'], valid)])
        ax.bar(range(len(values)), values, bottom=bottoms, label=key, color=colors[key])
        bottoms += values
    for i, metric in enumerate(row['metrics']):
        text = f"MAE {metric['hidden_core_mae']:.3f}" if valid[i] else 'INVALID / fallback only'
        if i == row['selected']['surface_full']['index']:
            text += '\nSELECTED'
        ax.text(i, bottoms[i] + max(.03, bottoms.max() * .025), text, ha='center', fontsize=9)
    ax.set_xticks(range(len(row['metrics'])), ['Seed ' + str(m['seed']) for m in row['metrics']])
    ax.set_ylim(0, max(.2, bottoms.max() * 1.35))
    ax.set_ylabel('Weighted, development-normalized loss')
    ax.set_title(row['label'] + ': why full surface chose this candidate (lower is better)')
    ax.legend(loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_pdf(report, output, example_rows):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage, PageBreak, Table, TableStyle
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='SmallText', parent=styles['BodyText'], fontSize=8.5, leading=12))
    doc = SimpleDocTemplate(str(output / 'REPORT.pdf'), pagesize=(8.27 * inch, 11.69 * inch),
                            leftMargin=42, rightMargin=42, topMargin=38, bottomMargin=38,
                            title='Surface consistency: controlled experiments and limits')
    flow = []
    def para(text, style='BodyText'):
        flow.append(Paragraph(escape(text).replace('\n', '<br/>'), styles[style]))
        flow.append(Spacer(1, 8))
    def figure(name, width=510):
        path = output / 'figures' / name
        with Image.open(path) as img: height = width * img.height / img.width
        flow.append(PDFImage(str(path), width=width, height=height))
        flow.append(Spacer(1, 8))
    para('Surface consistency: what the experiments establish', 'Title')
    para(report['conclusion'], 'Heading2')
    para(report['overview'])
    para(report['primary_result'])
    for text in report['cohort_results']: para(text, 'SmallText')
    figure('hidden_mae.png')
    para(report['interpretation'], 'SmallText')
    flow.append(PageBreak())
    para('Theory, assumptions and counterexamples', 'Heading1')
    for text in report['theory']: para(text)
    counterexample = output.parent / 'counterexamples/counterexamples.png'
    if counterexample.exists():
        flow.append(PDFImage(str(counterexample), width=425, height=425 * 1175 / 1040))
    flow.append(PageBreak())
    para('Experimental contract', 'Heading1')
    for text in report['design']: para(text)
    para(report['cost'], 'SmallText')
    flow.append(PageBreak())
    para('Selection rules on the same candidate pools', 'Heading1')
    para(report['local_result'], 'SmallText')
    figure('paired_gains.png')
    para('Gain is uniform-selection expected MAE minus selected MAE. The controls and selectors use exactly the same four candidates. Intervals crossing zero are inconclusive; ablations are exploratory.', 'SmallText')
    table_rows = [['Rule', 'MAE', 'Gain vs uniform', 'Fallbacks']]
    for row in report['table']:
        table_rows.append([NAMES[row['method']], f"{row['mae']:.4f}",
                           f"{row['gain']:+.4f} [{row['low']:+.4f}, {row['high']:+.4f}]", str(row['fallbacks'])])
    table = Table(table_rows, colWidths=[178, 48, 195, 68])
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce9ec')),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8),
        ('BOTTOMPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),4),
        ('LINEBELOW',(0,0),(-1,0),.6,colors.grey)]))
    flow.append(table)
    flow.append(PageBreak())
    para('Why selection succeeds or fails', 'Heading1')
    figure('score_error_association.png')
    for text in report['diagnostics']: para(text)
    flow.append(PageBreak())
    para('Testing the appearance assumption', 'Heading1')
    for text in report['posthoc_explanation']: para(text)
    para(report['perceptual_text'])
    for text in report['perceptual_cohorts']: para(text, 'SmallText')
    para(report['historical_text'])
    flow.append(PageBreak())
    para('Which pixels the score and evaluation use', 'Heading1')
    para('The surface score uses each candidate mask. Reconstruction error uses the same held-out target core for every candidate, including pixels that its predicted mask misses. Orange in the candidate panels marks missed held-out support; cyan marks recovered support. Gray pixels are other predicted support and are not automatically errors: complete-object amodal ground truth is unavailable.')
    for row in example_rows[:3]:
        para(row['title'], 'Heading2')
        path = output / 'gallery' / (row['case']['case_id'] + '-masks.jpg')
        flow.append(PDFImage(str(path), width=510, height=510 * 340 / 1536))
        flow.append(Spacer(1, 12))
    for row in example_rows:
        flow.append(PageBreak())
        para(row['title'], 'Heading1')
        para(row['text'])
        path = output / 'gallery' / (row['case']['case_id'] + '.jpg')
        flow.append(PDFImage(str(path), width=510, height=510 * 418 / 1536))
        # Enlarged comparison retains observed evidence, truth and selected candidate.
        case = row['case']
        experiment = Path(case['experiment_path'])
        chosen = case['metrics'][case['selected']['surface_full']['index']]['seed']
        triptych = []
        for label, path in [('Observed', experiment / 'inputs' / case['case_id'] / 'observed.png'),
                            ('Held-out truth', experiment / 'evaluation_only' / case['case_id'] / 'truth.png'),
                            ('Full-surface selection', experiment / 'full/candidates' / case['case_id'] / f'seed-{chosen}.png')]:
            triptych.append([Paragraph(escape(label), styles['SmallText']), Spacer(1, 5),
                             PDFImage(str(path), width=164, height=164)])
        flow.append(Table([triptych], colWidths=[170, 170, 170]))
        figure('explanation-' + case['case_id'] + '.png')
    flow.append(PageBreak())
    para('Limits, reproducibility and next decision', 'Heading1')
    for text in report['limitations']: para(text)
    for text in report['reproducibility']: para(text, 'SmallText')
    para('Reproduce with scripts/prepare_surface_experiment.py, scripts/run_surface_slurm.sh, scripts/analyze_surface_experiment.py, scripts/measure_surface_perceptual.py, scripts/validate_surface_experiment.py and scripts/build_surface_report.py. Exact commands, hashes, job IDs, per-candidate scores, selected seeds, metrics and source snapshots accompany this report.')
    def footer(canvas, document):
        canvas.setFont('Helvetica', 8)
        canvas.drawString(42, 23, 'Surface-PMC research | 8 November 2026 | Controlled held-out-pixel experiment')
        canvas.drawRightString(553, 23, str(document.page))
    doc.build(flow, onFirstPage=footer, onLaterPages=footer)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--replication', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for name in ('figures', 'tables', 'gallery'):
        (output / name).mkdir(exist_ok=True)
    freeze = json.loads((args.pilot / 'analysis/development_freeze.json').read_text())
    rows, cohort_rows = [], {}
    for name, experiment in [('Pilot test', args.pilot), ('Replication', args.replication)]:
        group = json.loads((experiment / 'analysis/test_cases.json').read_text())
        attach_perceptual(experiment, group)
        for row in group:
            row.update(cohort=name, experiment_path=str(experiment))
        rows.extend(group)
        cohort_rows[name] = group
    if len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate cases in pooled report')
    summary = summarize_cases(rows, 'test')
    cohorts = {name: summarize_cases(group, 'test') for name, group in cohort_rows.items()}
    cohorts['Pooled'] = summary
    perceptual = perceptual_summary(rows)
    perceptual_cohorts = {name: perceptual_summary(group) for name, group in cohort_rows.items()}
    diagnostic_path = args.pilot / 'diagnostics/summary.json'
    diagnostic = json.loads(diagnostic_path.read_text())['pooled'] if diagnostic_path.exists() else {}
    write_json(output / 'pooled_test_summary.json', dict(n=len(rows), methods=summary, cohorts=cohorts,
        perceptual=perceptual, perceptual_cohorts=perceptual_cohorts,
        development_method=freeze['development_method'], freeze_sha256=sha256(args.pilot / 'analysis/development_freeze.json')))
    candidate_rows, selected_rows = [], []
    for row in rows:
        for i, metric in enumerate(row['metrics']):
            score = row['scores'][i]
            candidate_rows.append(dict(case_id=row['case_id'], cohort=row['cohort'], source=row['source'], label=row['label'],
                **metric, valid=score['valid'], invalid_reasons=';'.join(score.get('invalid_reasons', [])),
                full_surface_loss=row['selected']['surface_full']['losses'][i],
                local_surface_loss=row['selected']['local_surface']['losses'][i],
                **{f'{key}_weighted_contribution': (score[key] / freeze['scales'][key] * weight
                    if score.get(key) is not None and score['valid'] else None)
                   for key, weight in METHODS['surface_full'].items()},
                **{k: score.get(k) for k in ('boundary', 'color', 'texture', 'local_boundary', 'local_texture', 'hidden_pixels', 'visible_recall')}))
        for method, selection in row['selected'].items():
            metric = row['metrics'][selection['index']]
            selected_rows.append(dict(case_id=row['case_id'], cohort=row['cohort'], source=row['source'], method=method,
                seed=metric['seed'], fallback=selection['fallback'], hidden_core_mae=metric['hidden_core_mae'],
                gain_vs_uniform=np.mean([m['hidden_core_mae'] for m in row['metrics']]) - metric['hidden_core_mae'],
                masked_spatial_lpips=metric.get('masked_spatial_lpips'), hidden_mask_recall=metric['hidden_mask_recall']))
    csv_write(output / 'tables/candidates.csv', candidate_rows)
    csv_write(output / 'tables/selections.csv', selected_rows)
    table = [dict(method=method, mae=summary[method]['hidden_core_mae']['mean'],
        gain=summary[method]['gain_vs_uniform']['mean'], low=summary[method]['gain_vs_uniform']['ci_low'],
        high=summary[method]['gain_vs_uniform']['ci_high'], fallbacks=summary[method]['fallbacks'],
        wins=summary[method]['wins'], losses=summary[method]['losses'], ties=summary[method]['ties']) for method in ORDER]
    csv_write(output / 'tables/pooled_methods.csv', table)
    csv_write(output / 'tables/cohort_methods.csv', [dict(cohort=cohort, method=method,
        mae=values['hidden_core_mae']['mean'], **{f'gain_{key}': value for key, value in values['gain_vs_uniform'].items()})
        for cohort, methods in cohorts.items() for method, values in methods.items()])
    csv_write(output / 'tables/source_methods.csv', [dict(method=method, source=source, **values)
        for method, method_values in summary.items() for source, values in method_values['by_source'].items()])
    if perceptual['n']:
        csv_write(output / 'tables/perceptual_methods.csv', [dict(method=method,
            **{f'lpips_{key}': value for key, value in values['lpips'].items()},
            **{f'gain_{key}': value for key, value in values['gain_vs_uniform'].items()})
            for method, values in perceptual['methods'].items()])
        csv_write(output / 'tables/perceptual_cohort_methods.csv', [dict(cohort=name, method=method,
            **{f'lpips_{key}': value for key, value in values['lpips'].items()},
            **{f'gain_{key}': value for key, value in values['gain_vs_uniform'].items()})
            for name, group in perceptual_cohorts.items() for method, values in group['methods'].items()])
    plots(summary, cohorts, rows, output / 'figures')
    gains = {}
    gallery_items = []
    for row in rows:
        gain = panel(row, Path(row['experiment_path']), output / 'gallery' / (row['case_id'] + '.jpg'))
        mask_panel(row, Path(row['experiment_path']), output / 'gallery' / (row['case_id'] + '-masks.jpg'))
        gains[row['case_id']] = gain
        gallery_items.append(dict(case_id=row['case_id'], label=row['label'], cohort=row['cohort'], source=row['source'],
                                  gain=gain, image=row['case_id'] + '.jpg', masks=row['case_id'] + '-masks.jpg',
                                  fallback=row['selected']['surface_full']['fallback'],
                                  eligible_candidates=sum(v is not None for v in row['selected']['surface_full']['losses'])))
    write_json(output / 'gallery/cases.json', gallery_items)
    payload = json.dumps(gallery_items).replace('</', '<\\/')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Surface consistency: all held-out cases</title>
<style>body{font:16px system-ui;margin:24px;background:#f5f7f8;color:#17212b}header{position:sticky;top:0;background:#f5f7f8;padding:12px 0}input,select{padding:8px;margin-right:10px}article{background:white;margin:20px 0;padding:12px}img{width:100%;height:auto}a{color:#075d76}</style>
<header><h1>All held-out cases</h1><p>Positive gain means lower hidden RGB error than expected uniform selection. Truth and oracle are evaluation only.</p>
<input id="q" placeholder="Search label or case ID"><select id="filter"><option value="all">All cases</option><option value="gain">Full-surface gains</option><option value="loss">Full-surface losses</option><option value="fallback">All-invalid fallback</option><option value="pilot">Pilot test</option><option value="replication">Replication</option></select><span id="count"></span></header><main id="cases"></main>
<script>const data=__DATA__;function render(){const q=document.querySelector('#q').value.toLowerCase(),f=document.querySelector('#filter').value;const rows=data.filter(r=>(r.case_id+' '+r.label+' '+r.source).toLowerCase().includes(q)&&(f==='all'||f==='gain'&&r.gain>1e-10||f==='loss'&&r.gain< -1e-10||f==='fallback'&&r.fallback||f==='pilot'&&r.cohort==='Pilot test'||f==='replication'&&r.cohort==='Replication'));document.querySelector('#count').textContent=rows.length+' / '+data.length;const main=document.querySelector('#cases');main.replaceChildren();for(const r of rows){const a=document.createElement('article'),h=document.createElement('p'),im=document.createElement('img');h.textContent=r.cohort+' | '+r.label+' | gain '+r.gain.toFixed(4)+' | eligible '+r.eligible_candidates+'/4'+(r.fallback?' | FIXED-SEED FALLBACK':'');im.src=r.image;im.loading='lazy';im.alt=r.case_id+' all four candidates';const detail=document.createElement('details'),label=document.createElement('summary'),mask=document.createElement('img');label.textContent='Inspect score and evaluation supports';mask.src=r.masks;mask.loading='lazy';mask.alt=r.case_id+' mask coverage';detail.append(label,mask);a.append(h,im,detail);main.append(a)}}document.querySelector('#q').oninput=render;document.querySelector('#filter').onchange=render;render();</script></html>'''.replace('__DATA__', payload)
    (output / 'gallery/index.html').write_text(html, encoding='utf-8')
    primary = summary['surface_full']['gain_vs_uniform']
    if primary['ci_low'] > 0:
        conclusion = 'The full surface rule improves held-out pixel reconstruction in this controlled experiment.'
    elif primary['ci_high'] < 0:
        conclusion = 'The full surface rule worsens held-out pixel reconstruction in this controlled experiment.'
    else:
        conclusion = 'The experiment does not establish a reliable full-surface improvement over equal-budget random selection.'
    uniform = summary['uniform_random']['hidden_core_mae']['mean']
    selected_mae = summary['surface_full']['hidden_core_mae']['mean']
    headroom = summary['oracle_hidden_mae']['gain_vs_uniform']['mean']
    historical_path = args.pilot / 'historical-census/summary.json'
    historical = json.loads(historical_path.read_text()) if historical_path.exists() else {}
    association = next((r for r in historical.get('associations', []) if r['label'] == 'hidden_region_inconsistency' and r['subset'] == 'all'), None)
    historical_text = (f"The exploratory historical audit scores {historical.get('valid', 0)}/{historical.get('total', 0)} outputs. "
        + (f"For pre-existing assistant hidden-inconsistency labels, full-surface loss has descriptive AUC {association['auc_higher_loss_predicts_failure']:.3f} on {association['n']} assessable scored cases. " if association else '')
        + 'These are one-candidate historical outputs with assistant labels, not an independent human evaluation or a reranking test.')
    if perceptual['n']:
        lp = perceptual['methods']['surface_full']['gain_vs_uniform']
        local_lp = perceptual['methods']['local_surface']['gain_vs_uniform']
        perceptual_text = (f"Masked spatial LPIPS provides a separate perceptual check on {perceptual['n']} complete pools: "
            f"full-surface gain {lp['mean']:+.4f}, 95% CI [{lp['ci_low']:+.4f}, {lp['ci_high']:+.4f}]; "
            f"local-surface gain {local_lp['mean']:+.4f} [{local_lp['ci_low']:+.4f}, {local_lp['ci_high']:+.4f}]. "
            'Positive is better. This is not a human preference score. Both images are neutral outside the held-out core; '
            'spatial AlexNet v0.1 LPIPS is averaged on that core. Mask edges affect receptive fields.')
    else:
        perceptual_text = 'Perceptual metrics are unavailable; no perceptual-improvement claim is made.'
    perceptual_cohort_text = []
    for name, group in perceptual_cohorts.items():
        if group['n']:
            parts = []
            for method in ('surface_full', 'local_surface'):
                value = group['methods'][method]['gain_vs_uniform']
                parts.append(f"{NAMES[method]} {value['mean']:+.4f} [{value['ci_low']:+.4f}, {value['ci_high']:+.4f}]")
            perceptual_cohort_text.append(f"{name} LPIPS gains, n={group['n']}: " + '; '.join(parts) + '.')
    all_records = [json.loads(path.read_text()) for experiment in (args.pilot, args.replication)
                   for path in (experiment / 'full/candidates').rglob('seed-*.json')]
    times = np.array([r['total_seconds'] for r in all_records])
    scoring = np.array([r.get('scoring_seconds', 0) for r in all_records])
    peaks = [r['peak_allocated_bytes'] / 2**30 for r in all_records]
    status = Counter(r['status'] for r in all_records)
    eligible_candidates = sum(value is not None for row in rows for value in row['selected']['surface_full']['losses'])
    fallback_cases = ', '.join(row['case_id'] + ' (' + row['label'] + ')' for row in rows if row['selected']['surface_full']['fallback']) or 'none'
    cohort_results = []
    for name in ('Pilot test', 'Replication'):
        value = cohorts[name]['surface_full']
        gain = value['gain_vs_uniform']
        cohort_results.append(f"{name}, n={gain['n']}: full-surface MAE {value['hidden_core_mae']['mean']:.4f}; "
            f"gain over all-candidate uniform selection {gain['mean']:+.5f}, 95% CI [{gain['ci_low']:+.5f}, {gain['ci_high']:+.5f}].")
    local = summary['local_surface']['gain_vs_uniform']
    local_result = (f"The fixed local-seam variant has pooled gain {local['mean']:+.4f} "
        f"[{local['ci_low']:+.4f}, {local['ci_high']:+.4f}]. It is an exploratory ablation. "
        f"The development-selected rule was {NAMES[freeze['development_method']]}, so the local rule was not selected by development performance. "
        'Intervals are nominal and do not correct for comparisons across the ablation menu.')
    for name in ('Pilot test', 'Replication'):
        value = cohorts[name]['local_surface']['gain_vs_uniform']
        local_result += (f" {name} local-surface gain: {value['mean']:+.4f} "
                         f"[{value['ci_low']:+.4f}, {value['ci_high']:+.4f}].")
    reproducibility = []
    for name, experiment in [('Pilot', args.pilot), ('Replication', args.replication)]:
        validation_path = experiment / 'validation.json'
        if validation_path.exists():
            checked = json.loads(validation_path.read_text())
            reproducibility.append(f"{name} pinned-runtime replay: {checked['candidate_metric_replays']} candidate metrics, "
                f"{checked['all_policy_replays']} policy decisions and {checked['artifact_checks']} artifact checks; "
                f"maximum raw-score difference {checked['max_score_replay_difference']:.1g}. "
                'Preparation was replayed from original source pixels and SAM masks; transport hashes were independently checked.')
        portability_path = experiment / 'portability-windows-numpy254/summary.json'
        if portability_path.exists():
            audit = json.loads(portability_path.read_text())
            changes = {key: value for key, value in audit['decision_changes'].items() if value}
            reproducibility.append(f"{name} numerical portability audit under NumPy {audit['runtime']['numpy']}, "
                f"Pillow {audit['runtime']['pillow']}: changed decisions {changes or 'none'}. "
                f"Largest raw-term difference {audit['max_term_difference']['absolute_difference']:.5f}. "
                'Frozen decisions and original reported scores remain unchanged.')
    reproducibility.append('Exact scorer replay used Python 3.10, NumPy 1.23.5 and Pillow 9.5.0. '
        'Do not loosen the replay tolerance or replace the frozen results after upgrading numerical packages. '
        'The portability audit changes one boundary-plus-color pilot decision; full and local decisions are unchanged in both cohorts. '
        'A CPU replay verifies saved generation artifacts; it does not guarantee regeneration on arbitrary hardware.')
    explanations = []
    if diagnostic:
        decomp = diagnostic['decomposition']
        explanations.append(f"Post-hoc exact decomposition: pooled total gain {decomp['total_gain']['mean']:+.4f} "
            f"= filter contribution {decomp['filter_gain']['mean']:+.4f} + appearance-ranking contribution {decomp['ranking_gain']['mean']:+.4f}. "
            f"Remaining regret relative to the best available candidate is {decomp['total_regret']['mean']:.4f}, "
            f"comprising ranking regret {decomp['ranking_regret']['mean']:.4f} and guard-exclusion regret {decomp['exclusion_regret']['mean']:.4f}.")
        for method in ('surface_full', 'local_surface'):
            value = diagnostic['pair_concordance'][method]
            if value:
                explanations.append(f"{NAMES[method]} orders pairs of valid candidates correctly with mean per-image concordance "
                    f"{value['mean']:.3f} [{value['ci_low']:.3f}, {value['ci_high']:.3f}], n={value['n']}. "
                    'Score ties receive half credit; error ties are omitted. Chance ordering is 0.5. This explanatory statistic never changes selection.')
        truth_check = diagnostic.get('fixed_support_truth')
        if truth_check:
            explanations.append(f"On identical evaluation-only support for the source truth and selected RGB, "
                f"the selected reconstruction scores more compatible than the actual source in "
                f"{truth_check['erroneous_selected_rgb_scores_better_than_truth']}/{truth_check['n']} cases. "
                'This shows that compatibility can prefer a reconstruction with pixel error even after removing the changing-mask confound. It does not identify the visual cause without inspection.')
        contrast = diagnostic['exploratory_paired_contrasts']['local_gain_over_visible_preservation']
        explanations.append(f"The exploratory paired local-seam gain over visible-pixel preservation is "
            f"{contrast['mean']:+.4f} [{contrast['ci_low']:+.4f}, {contrast['ci_high']:+.4f}]. "
            'A benefit over random selection alone does not establish superiority to this simpler observable control.')
        contrast = diagnostic['exploratory_paired_contrasts']['local_gain_over_full']
        explanations.append(f"Directly comparing local and full rules gives paired MAE gain "
            f"{contrast['mean']:+.4f} [{contrast['ci_low']:+.4f}, {contrast['ci_high']:+.4f}]. "
            'The current data do not establish local superiority to full surface. A positive interval for one method '
            'and an interval crossing zero for another are not themselves a significant difference between methods.')
    report = dict(conclusion=conclusion,
        overview=f"Frozen pretrained PMC, four candidates per image, 24 development images and {len(rows)} held-out image identities: 72 pilot test plus 87 prespecified replication. All selectors share the same seeds and generated images. No diffusion model was trained.",
        cohort_results=cohort_results, local_result=local_result, posthoc_explanation=explanations,
        reproducibility=reproducibility, perceptual_cohorts=perceptual_cohort_text,
        primary_result=f"Pooled hidden RGB MAE: uniform selection {uniform:.4f}; full surface {selected_mae:.4f}. Paired gain {primary['mean']:+.4f} (95% CI {primary['ci_low']:+.4f} to {primary['ci_high']:+.4f}), equivalent to {primary['mean'] * 255:+.2f} intensity levels or {100 * primary['mean'] / uniform:+.1f}% relative error reduction.",
        interpretation='The primary result concerns recovery of artificially hidden, originally observed SAM-supported pixels. It does not prove correct reconstruction of naturally hidden surfaces, human preference, improved amodal shape or end-to-end progressive completion.',
        theory=[
            'Surface consistency is a compatibility prior: visible color, texture and seam continuity may help choose between candidates when they share one material or pattern. It is not hidden ground truth.',
            'Identifiability limit: identical visible evidence can correspond to different hidden materials or parts. Any observable-only selector can therefore fail on a legitimate hidden transition.',
            'Constructed examples show that global means/histograms cannot resolve stripe phase; a local seam term can. Both variants prefer an incorrect uniform fill over a true unseen material change. These examples test mechanisms, not natural-image success rates.'],
        design=[
            'The pilot contains 96 image-disjoint cases, 24 per source. Six per source are development and 18 per source are test. Replication excludes all pilot identities and image bytes, with 24 VG, 24 COCOA, 24 free-images and 15 LAION cases. LAION eligibility was exhausted; the amendment was recorded before test errors were inspected.',
            'Cached visible SAM support defines a square context crop at 512 pixels. A hash-selected half-plane hides approximately 35% of visible target pixels. RGB is removed after resizing. Hidden truth and support are kept in evaluation-only files; generation and scoring consume only observations.',
            'The pinned Mixed Context sampler uses step 35 of 50 and explicit seeds 11,22,33,44. Grounding DINO plus SAM predict each candidate target mask. This is one shared completion step with known artificial occlusion; ordering, the progressive loop and released occluder dilation are outside this experiment.',
            'Full surface combines original-visible anchored Lab boundary distance, Lab histogram Hellinger distance and masked gradient texture distance with weights .40/.25/.35. Local surface combines paired seam color and local texture. Positive development medians normalize terms; all settings are frozen before test analysis.',
            f"The separately reported development-selected rule is {NAMES[freeze['development_method']]}. It was chosen on 24 development cases from the fixed menu including the fixed-seed control, then held unchanged for both test cohorts.",
            'Primary MAE is evaluated on all held-out target-core pixels, independent of predicted mask size. Missing generations receive MAE 1. LPIPS masks both images outside that same core. Source-stratified paired bootstrap intervals use 5,000 resamples; ablations are exploratory.'],
        cost=f"Generated records: {dict(status)}; {len(all_records)} candidates. Median candidate time including generation, segmentation and scoring {np.median(times):.2f}s; median scorer time {np.median(scoring):.3f}s; summed candidate-processing time {times.sum()/3600:.2f} GPU-worker hours, excluding initial model loading. Peak allocated VRAM {max(peaks):.2f} GiB. Four candidates require four sampling calls; a lower error than K=1 alone cannot isolate the scorer.",
        diagnostics=[
            f"Full surface has defined scores for {eligible_candidates}/{sum(len(row['metrics']) for row in rows)} held-out candidates. All-invalid pools use the declared fixed-first fallback: {fallback_cases}. Missing terms and rejected masks remain in the tables and their RGB errors remain in evaluation.",
            f"Available oracle gain is {headroom:.4f}; full-surface oracle regret is {summary['surface_full']['oracle_regret']:.4f}. Selection gain equals available gain minus regret. The oracle uses hidden truth and is unavailable during inference.",
            f"Full surface improves on the pool mean in {summary['surface_full']['wins']}/{len(rows)} cases, ties in {summary['surface_full']['ties']}, and loses in {summary['surface_full']['losses']}. It falls back to seed 11 in {summary['surface_full']['fallbacks']} pools. These counts include every test case.",
            f"Mean held-out mask recall is {summary['surface_full']['hidden_mask_recall']:.3f} for full surface versus {summary['uniform_random']['hidden_mask_recall']:.3f} for uniform selection. Visible RGB MAE is {summary['surface_full']['visible_mae']:.4f} versus {summary['uniform_random']['visible_mae']:.4f}. These diagnostics do not certify full amodal geometry.",
            f"The simple visible-pixel-preservation control has hidden MAE {summary['visible_preservation']['hidden_core_mae']['mean']:.4f}. It selects using known visible RGB only, under the same validity guards, and was declared before test errors. It is an additional diagnostic control, outside the frozen development selection menu.",
            f"Uniform selection restricted to exactly the candidates valid for full surface has MAE {summary['uniform_valid_surface']['hidden_core_mae']['mean']:.4f}. Full-surface gain beyond these guards alone is {summary['surface_full']['gain_vs_valid_uniform']['mean']:+.4f} [{summary['surface_full']['gain_vs_valid_uniform']['ci_low']:+.4f}, {summary['surface_full']['gain_vs_valid_uniform']['ci_high']:+.4f}]. This separates the mask-validity filter from the appearance-ranking contribution.",
            'The scatter removes between-image difficulty by centering errors and standardizing scores within candidate pools. Its correlation is descriptive. A low optimized compatibility loss alone is not evidence of better hidden reconstruction.'],
        perceptual_text=perceptual_text, historical_text=historical_text, table=table,
        limitations=[
            'The experiment uses pseudo-occlusion of observed pixels. Original SAM masks can contain errors; the original target need not be fully visible. Crop construction uses pre-occlusion support. Real hidden appearance remains unobserved.',
            'Pixel error penalizes plausible alternative geometry and pose. LPIPS is a supporting metric with an explicit masked protocol; no independent human preference ratings were collected. The historical census uses prior assistant judgments.',
            'Candidate masks vary. Guards reject missing/tiny hidden regions and severe loss of visible support, but do not prevent every mask-based shortcut or semantic error. Error is therefore measured on fixed held-out support.',
            'These results apply to a fixed four-seed menu and one sampling step. They do not establish a new full benchmark score, full progressive-loop behavior, a trained method or superiority to current research methods.',
            'Preserve the frozen test sets. Any revised scorer should be tuned on development data and evaluated on a new independent cohort. The next useful question is whether local evidence and an explicit material-change uncertainty model improve generalization without selecting smaller or smoother objects.'])
    ordered = sorted(rows, key=lambda row: gains[row['case_id']])
    notes_path = args.pilot / 'visual_notes.json'
    visual_notes = json.loads(notes_path.read_text()) if notes_path.exists() else {}
    example_specs = [('Largest full-surface loss (selected by measured error)', ordered[0]),
                     ('Largest full-surface gain (selected by measured error)', ordered[-1]),
                     ('Middle-ranked full-surface outcome', ordered[len(ordered)//2])]
    for extra in visual_notes.get('extra_examples', []):
        row = next(row for row in rows if row['case_id'] == extra['case_id'])
        if all(row['case_id'] != existing['case_id'] for _, existing in example_specs):
            example_specs.append((extra['title'], row))
    examples = []
    for title, row in example_specs:
        chosen = row['selected']['surface_full']['index']
        best = row['selected']['oracle_hidden_mae']['index']
        caption = f"{row['label']} ({row['cohort']}). Full surface selects seed {row['metrics'][chosen]['seed']}, while the unavailable hidden-MAE oracle selects seed {row['metrics'][best]['seed']}. Gain versus the pool mean is {gains[row['case_id']]:+.4f}. The gallery exposes every case, including all losses. "
        caption += visual_notes.get('cases', {}).get(row['case_id'], '')
        examples.append(dict(title=title, case=row, text=caption))
        explanation_plot(row, freeze['scales'], output / 'figures' / ('explanation-' + row['case_id'] + '.png'))
    write_json(output / 'report_data.json', report)
    write_json(output / 'illustrative_cases.json', examples)
    lines = ['# Surface consistency: controlled results', '', '**' + conclusion + '**', '', report['overview'], '',
             report['primary_result'], '', *cohort_results, '', report['interpretation'], '', '## Results on shared candidate pools', '',
             '| Rule | Hidden MAE | Gain vs uniform [95% CI] | Fallbacks |', '| --- | ---: | ---: | ---: |']
    for row in table:
        lines.append(f"| {NAMES[row['method']]} | {row['mae']:.4f} | {row['gain']:+.4f} [{row['low']:+.4f}, {row['high']:+.4f}] | {row['fallbacks']} |")
    lines.extend(['', local_result, '', '![Paired gains](figures/paired_gains.png)', '', '## What the theory means', ''])
    lines.extend(report['theory'])
    lines.extend(['', '## Design and leakage controls', ''])
    lines.extend(text + '\n' for text in report['design'])
    lines.extend(['', '## Explanation and independent diagnostics', '', '![Within-pool association](figures/score_error_association.png)', ''])
    lines.extend(text + '\n' for text in report['diagnostics'])
    lines.extend(text + '\n' for text in report['posthoc_explanation'])
    lines.extend(['', perceptual_text, '', *perceptual_cohort_text, '', historical_text, '', '## Compute and failures', '', report['cost'], '',
                  '## Examples and complete evidence', '', '[Open all held-out cases](gallery/index.html). The gallery exposes every candidate, support masks and the deterministic choices. The first three examples are the largest loss, largest gain and middle-ranked result. Additional examples and visual descriptions are explicitly post-hoc assistant observations, not independent human ratings.'])
    for example in examples:
        lines.extend(['', '### ' + example['title'], '', example['text'], '',
                      f"![{example['case']['label']}](gallery/{example['case']['case_id']}.jpg)", '',
                      f"![Weighted score explanation](figures/explanation-{example['case']['case_id']}.png)"])
    lines.extend(['', '## Limits and next decision', ''])
    lines.extend('- ' + text for text in report['limitations'])
    lines.extend(['', '## Reproducibility checks', ''])
    lines.extend(text + '\n' for text in reproducibility)
    lines.extend(['', 'Tables: [candidate scores and metrics](tables/candidates.csv), [selection decisions](tables/selections.csv), [method summary](tables/pooled_methods.csv).', '',
                  'Additional tables: [separate cohorts](tables/cohort_methods.csv), [sources](tables/source_methods.csv), [perceptual diagnostics](tables/perceptual_methods.csv).', '',
                  'The [theory derivations](../../../docs/SURFACE_CONSISTENCY_THEORY.md) distinguish an observable compatibility prior, a ranking condition and an identifiability limit. Post-hoc explanation code does not alter the frozen experiment.', '',
                  'Reproduction and job records are in the parent experiment folder. Source, input and output checksums are retained; the original benchmark and historical outputs remain separate.'])
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    make_pdf(report, output, examples)
    write_json(output / 'build.json', dict(report_sha256=sha256(output / 'REPORT.md'), pdf_sha256=sha256(output / 'REPORT.pdf'),
        builder_sha256=sha256(Path(__file__)), test_cases=len(rows), candidate_rows=len(candidate_rows),
        selection_rows=len(selected_rows), gallery_panels=len(gallery_items), development_method=freeze['development_method']))
    print(report['conclusion'])
    print(report['primary_result'])


if __name__ == '__main__':
    main()
