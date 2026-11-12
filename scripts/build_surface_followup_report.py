#!/usr/bin/env python3
"""Package a frozen fresh-test result, all cases, controls, and failed full-run attempt."""
import argparse
from html import escape
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json

PRIMARY = ('hidden_core_mae', 'object_lpips', 'dreamsim', 'clip_text_cosine')
LABELS = dict(hidden_core_mae='Hidden RGB MAE', object_lpips='Object LPIPS',
    dreamsim='DreamSim', clip_text_cosine='CLIP text cosine',
    hidden_spatial_lpips='Hidden spatial LPIPS', object_ssim='Object SSIM',
    object_feature_similarity='VGG feature similarity')


def font(size):
    for name in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'C:/Windows/Fonts/arial.ttf'):
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def object_view(image, mask):
    rgb = np.array(Image.open(image).convert('RGB'))
    support = np.array(Image.open(mask).convert('L')) > 0
    rgb[~support] = 0
    return Image.fromarray(rgb)


def build_panels(args, cases, freeze):
    inventory = []
    for case in cases:
        case_id = case['case_id']
        row = json.loads((args.metrics_dir / (case_id + '.json')).read_text())
        if row['freeze_sha256'] != sha256(args.freeze):
            raise ValueError('Gallery metric policy differs from the frozen method')
        index = row['selection']['trained_surface']
        seed = freeze['candidate_seeds'][index]
        first = args.run_dir / 'candidates' / case_id
        chosen = first if (first / f'seed-{seed}.png').exists() else args.extra_run_dir / 'candidates' / case_id
        selected = args.metrics_dir / 'selected' / case_id
        inputs, truth = args.experiment_dir / 'inputs' / case_id, args.experiment_dir / 'evaluation_only' / case_id
        for name, expected in row['selected_artifacts'].items():
            if sha256(selected / name) != expected:
                raise ValueError('Selected scientific artifact changed')
        reference = object_view(truth / 'truth.png', truth / 'full.png')
        hidden = Image.open(truth / 'hidden_core.png').convert('L')
        outline = np.array(hidden.filter(ImageFilter.MaxFilter(3))) > np.array(hidden.filter(ImageFilter.MinFilter(3)))
        reference_rgb = np.array(reference)
        reference_rgb[outline] = (30, 210, 245)
        frames = [Image.open(inputs / 'observed.png').convert('RGB'),
            object_view(first / 'seed-11.png', first / 'seed-11-mask.png'),
            object_view(chosen / f'seed-{seed}.png', chosen / f'seed-{seed}-mask.png'),
            object_view(selected / 'surface.png', selected / 'mask.png'), Image.fromarray(reference_rgb)]
        titles = ['Occluded observation', 'Fixed seed 11', f'Selected seed {seed}, raw',
                  'Frozen surface method', 'Withheld RGB (cyan outline)']
        panel = Image.new('RGB', (1280, 292), '#f6f7f9')
        draw = ImageDraw.Draw(panel)
        for col, (title, frame) in enumerate(zip(titles, frames)):
            draw.text((col * 256 + 7, 9), title, fill='#14233b', font=font(14))
            panel.paste(frame.resize((256, 256), Image.Resampling.LANCZOS), (col * 256, 36))
        destination = args.output_dir / 'panels' / (case_id + '.webp')
        destination.parent.mkdir(parents=True, exist_ok=True)
        panel.save(destination, quality=92, method=6)
        values = {}
        for key in (*PRIMARY, 'hidden_spatial_lpips'):
            raw = [item[key] for item in row['raw']]
            baseline = float(np.mean(raw)) if all(item is not None for item in raw) else None
            method = row['refined'][index][key]
            values[key] = dict(uniform=baseline, seed_11=row['raw'][0][key], method=method,
                absolute_gain=(method - baseline) * (1 if key == 'clip_text_cosine' else -1)
                    if method is not None and baseline is not None else None)
        inventory.append(dict(case_id=case_id, source=case['source'], label=case['label'],
            seed=seed, panel='panels/' + destination.name, panel_sha256=sha256(destination),
            metrics=values, valid_candidates=len(row['valid_indices']),
            fallback=not row['valid_indices'], generation_failures=row['generation_failures'],
            selected_scientific_sha256=row['selected_artifacts']))
    write_json(args.output_dir / 'gallery_cases.json', inventory)
    return inventory


def gallery(output, cases):
    cards = []
    for case in cases:
        mae = case['metrics']['hidden_core_mae']
        gain = mae['absolute_gain']
        cards.append(f'<article data-source="{escape(case["source"])}" data-gain="{gain:.8f}">'
            f'<h2>{escape(case["case_id"])} — {escape(case["label"])}</h2>'
            f'<p>Hidden MAE: expected uniform {mae["uniform"]:.4f}; surface {mae["method"]:.4f}; '
            f'absolute gain {gain:+.4f}. Valid candidates: {case["valid_candidates"]}. '
            f'Fallback: {str(case["fallback"]).lower()}.</p>'
            f'<a href="../{case["panel"]}"><img loading="lazy" src="../{case["panel"]}" '
            f'alt="Observation, seed 11, selected raw candidate, final surface, and withheld source RGB for {escape(case["label"])}"></a></article>')
    source_options = ''.join(f'<option>{escape(source)}</option>' for source in sorted({case['source'] for case in cases}))
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Surface-PMC: every fresh test case</title><style>
body{font:16px system-ui;background:#eef2f7;color:#15253c;margin:24px auto;max-width:1320px;padding:0 16px}
header{position:sticky;top:0;background:#eef2f7;padding:12px 0;border-bottom:1px solid #bcc7d6;z-index:2}
h1{font-size:26px}h2{font-size:18px}article{background:white;margin:22px 0;padding:16px;border-radius:8px}
img{display:block;width:100%;height:auto}select,input{padding:8px;margin:3px;border:1px solid #a8b6c7;border-radius:4px}
article[hidden]{display:none}p{line-height:1.5}a{color:#174b8f}</style>
<header><h1>Surface-PMC: every fresh test case</h1><p>Artificially hidden originally visible pixels. No true amodal RGB ground truth is claimed.
The fixed-seed image is an example control; the primary uniform control is an expectation over the entire common pool.</p>
<input id="search" placeholder="Search case or label" aria-label="Search case or label">
<select id="source" aria-label="Source"><option value="">All sources</option>''' + source_options + '''</select>
<select id="outcome" aria-label="MAE outcome"><option value="">All outcomes</option><option value="gain">MAE gains</option><option value="loss">MAE losses</option></select>
<span id="count"></span> · <a href="../REPORT.md">Report</a> · <a href="../gallery_cases.json">Case data</a></header>
''' + '\n'.join(cards) + '''<script>
const cards=[...document.querySelectorAll('article')];
function filter(){const query=document.querySelector('#search').value.toLowerCase(),source=document.querySelector('#source').value,outcome=document.querySelector('#outcome').value;
let count=0;for(const card of cards){const gain=Number(card.dataset.gain);card.hidden=!!((query&&!card.textContent.toLowerCase().includes(query))||(source&&card.dataset.source!==source)||(outcome==='gain'&&gain<=0)||(outcome==='loss'&&gain>=0));if(!card.hidden)count++;}document.querySelector('#count').textContent=count+' / '+cards.length+' cases';}
for(const element of document.querySelectorAll('input,select'))element.addEventListener('input',filter);filter();
</script></html>'''
    folder = output / 'gallery'
    folder.mkdir(exist_ok=True)
    (folder / 'index.html').write_text(html, encoding='utf-8')


def plot_result(output, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys = [*PRIMARY, 'hidden_spatial_lpips']
    values = [summary['metrics'][key]['trained_surface_refined']['uniform_raw'] for key in keys]
    points = np.array([value['relative_gain_percent'] for value in values])
    intervals = np.array([value['relative_ci_percent'] for value in values])
    fig, ax = plt.subplots(figsize=(8.7, 4.2), layout='constrained')
    ax.axvspan(3, 5, color='#d5eee4', label='Requested 3–5% gain')
    ax.axvline(0, color='#586779', lw=1)
    for i, (point, (low, high)) in enumerate(zip(points, intervals)):
        ax.plot([low, high], [i, i], color='#173e70', lw=2)
        ax.plot(point, i, 'o', color='#173e70')
        ax.annotate(f'{point:+.2f}%', (point, i), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9)
    ax.set_yticks(range(len(keys)), [LABELS[key] for key in keys])
    ax.invert_yaxis()
    ax.set_xlabel('Relative improvement over uniform selection from the same eight candidates (%)')
    ax.set_title(f'Frozen fresh test: {summary["images"]} images; paired 95% bootstrap intervals', loc='left', fontsize=12)
    ax.grid(axis='x', alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='best', frameon=False, fontsize=9)
    for suffix in ('png','pdf','svg'):
        fig.savefig(output / ('fresh-metric-gains.' + suffix), dpi=180)
    plt.close(fig)


def make_report(args, summary, freeze, cases):
    replay = json.loads((args.experiment_dir / 'analysis-eight/replay-verification.json').read_text())
    values = {key:summary['metrics'][key]['trained_surface_refined']['uniform_raw'] for key in PRIMARY}
    meets = all(value['relative_gain_percent'] >= 3 for value in values.values())
    ranker_kind = freeze['ranker'].get('kind', 'ridge')
    ranker_description = (f"A small gradient-boosted ranker ({len(freeze['ranker']['trees'])} trees, "
                          f"depth {freeze['ranker']['depth']}, minimum leaf {freeze['ranker']['minimum_leaf']})"
                          if ranker_kind == 'gradient_boosting' else 'A small ridge ranker')
    lines = ['# Surface-PMC follow-up: frozen fresh-test results', '',
        ('The frozen controlled method meets the 3% relative point-estimate target on all four primary endpoints.' if meets else
         'The frozen controlled method does not meet the 3% relative point-estimate target on every primary endpoint.'), '',
        f'{summary["images"]} image identities, disjoint by identity and source-image SHA-256 from all 183 development images. '
        f'Source counts: {summary["sources"]}. Candidate budget: {len(freeze["candidate_seeds"])} shared seeds per image.', '',
        '**Scope:** one-step reconstruction of artificially hidden, previously observed object pixels. '
        'The source SAM support is a visible-object proxy. These measurements are not true hidden amodal ground truth '
        'and do not establish an improved full progressive benchmark.', '',
        '## Primary results', '',
        '| Metric | Uniform baseline | Frozen surface | Relative improvement | 95% interval | n |',
        '| --- | ---: | ---: | ---: | --- | ---: |']
    for key in (*PRIMARY, 'hidden_spatial_lpips'):
        value = summary['metrics'][key]['trained_surface_refined']['uniform_raw']
        low, high = value['relative_ci_percent']
        lines.append(f'| {LABELS[key]} | {value["baseline"]:.6f} | {value["method"]:.6f} | '
                     f'{value["relative_gain_percent"]:+.2f}% | [{low:+.2f}%, {high:+.2f}%] | {value["n"]} |')
    lines += ['', 'Positive improvements follow each metric’s direction. MAE, LPIPS and DreamSim are lower-is-better; '
              'CLIP cosine is higher-is-better. Intervals use 5,000 paired source-stratified image bootstrap resamples. '
              'They are nominal intervals, without a multiple-comparison correction. Hidden spatial LPIPS is supplementary.', '',
              '![Fresh-test metric improvements](fresh-metric-gains.png)', '',
              '## Pre-specified selection-only component', '',
              'The already measured `trained_surface_raw` component uses the identical frozen candidate choices '
              'without color or consensus correction. Its four relative point gains all exceed 3%. This is a '
              'pre-specified secondary component, not a replacement for the frozen primary method. Favoring it '
              'after these results would require independent confirmation before making a new primary claim. '
              'The MAE and CLIP intervals extend below the 3% threshold.', '',
              '| Metric | Uniform baseline | Selection only | Relative improvement | 95% interval |',
              '| --- | ---: | ---: | ---: | --- |']
    for key in PRIMARY:
        value = summary['metrics'][key]['trained_surface_raw']['uniform_raw']
        low, high = value['relative_ci_percent']
        lines.append(f'| {LABELS[key]} | {value["baseline"]:.6f} | {value["method"]:.6f} | '
                     f'{value["relative_gain_percent"]:+.2f}% | [{low:+.2f}%, {high:+.2f}%] |')
    lines += ['', 'Corrections improve reconstruction errors but reduce category-text CLIP. The complete gallery '
              'already shows the selected raw output beside the corrected output for every image.', '',
              '## Equal-budget and component controls', '',
              'Every control uses the same generated pool. Refined controls receive exactly the same color calibration '
              'and consensus correction as the proposed selector. This separates selection from postprocessing.', '',
              'Each row reports the corrected frozen method\'s relative improvement over the named control, '
              'not that control\'s improvement over uniform sampling.', '',
              '| Comparison control | MAE gain | Object LPIPS gain | DreamSim gain | CLIP gain |',
              '| --- | ---: | ---: | ---: | ---: |']
    for control in ('seed_11_raw','uniform_valid_raw','local_surface_raw','semantic_only_raw',
                    'uniform_refined','uniform_valid_refined','consensus_only_refined',
                    'local_surface_refined','semantic_only_refined','visible_deep_only_refined'):
        if all(control in summary['metrics'][key]['trained_surface_refined'] for key in PRIMARY):
            gains = [summary['metrics'][key]['trained_surface_refined'][control]['relative_gain_percent'] for key in PRIMARY]
            lines.append('| ' + control + ' | ' + ' | '.join(f'{gain:+.2f}%' for gain in gains) + ' |')
    lines += ['', 'The complete paired intervals, source breakdowns, denominators and raw/refined policies are retained '
              'in [summary.json](../analysis-eight/summary.json) and [comparisons.csv](../analysis-eight/comparisons.csv). '
              'Additional sampling cost is not attributed to surface scoring. CLIP is explicitly used as an observable '
              'ranking feature, so the semantic-only control is essential.', '',
              '## Frozen method and validation', '',
              'The pretrained generator and feature networks remain frozen. ' + ranker_description + ' is fitted on the old '
              'development images using within-image centered targets and leave-one-source-out model selection. '
              'Development utility balances MAE, object LPIPS, DreamSim and CLIP, weighted to the predeclared fresh-cohort '
              'source counts. Observable features measure local surface continuity, visible preservation, sample agreement '
              'and pretrained appearance/semantic compatibility. Hidden RGB never enters inference.', '',
              'The selected output receives bounded observation-residual calibration and the frozen consensus correction. '
              'The correction transports smooth RGB residuals only where generated masks agree; it preserves selected '
              'mask geometry. The fixed hidden MAE support includes pixels missed by the predicted mask.', '',
              '```json', json.dumps(dict(ranker_kind=ranker_kind, refinement=freeze['refinement'], ensemble=freeze.get('ensemble'),
                  selection_rule=freeze['ranker'].get('selection_rule'),
                  regularization=freeze['ranker']['regularization']), indent=2), '```', '',
              f'Generation failures: {summary["generation_failures"]}; candidate errors: {summary["candidate_errors"]}. '
              f'All-invalid pools using the seed-11 fallback: {sum(case["fallback"] for case in cases)}.', '',
              'Observation-invariance tests randomize all unobserved RGB without changing inference features or decisions. '
              'The deployable pretrained feature provider matched the development provider exactly on the GPU audit. '
              'Original baseline outputs, historical studies, held-out decisions and failure accounting are preserved.', '',
              f'Independent replay passed on {replay["images"]} images and {replay["candidates"]} candidates, '
              f'including {replay["mae_checks"]} hidden-MAE checks and exact selected RGB/masks. '
              'The strict primitive-score replay uses the original generation CPU and pinned environment. '
              'Other CPU dispatch paths can move float32 color values across histogram boundaries; '
              'failed cross-runtime attempts remain preserved and the tolerance was not relaxed. '
              'Neural metrics were measured on the GPU; CPU replay verifies their saved inputs, not their networks.', '',
              'The original five-hour research window ended before final fresh evaluation. '
              'These results must not be described as completed within that window.', '',
              '## Every test case', '',
              '[Open the complete searchable gallery](gallery/index.html). Every image is included, with gains, losses '
              'and fallbacks. The reference panel outlines the hidden evaluation core. Gallery images are display derivatives; '
              'native scientific artifact hashes are in [gallery_cases.json](gallery_cases.json). No human preference claim is made.', '',
              '## Full-benchmark calibration attempt', '']
    if args.full_comparison:
        full = json.loads(args.full_comparison.read_text())
        lines += ['A separately frozen calibration-only derivation was evaluated on all 2,565 real-occlusion benchmark targets. '
                  'It preserved 951 saved outputs and 1,614 failures. It missed the requested target. '
                  'This approach is retained; it is not presented as successful full-benchmark improvement. '
                  'That benchmark has no hidden RGB ground truth, so no hidden MAE claim is possible.', '',
                  '| Full benchmark metric | Paired baseline | Calibration | Relative gain | n |',
                  '| --- | ---: | ---: | ---: | ---: |',
                  *[f'| {key} | {value["baseline"]:.6f} | {value["method"]:.6f} | '
                    f'{value["relative_gain_percent"]:+.3f}% | {value["n"]} |' for key,value in full['metrics'].items()], '',
                  f'Full-comparison evidence SHA-256: `{sha256(args.full_comparison)}`.', '']
    lines += ['The progressive runtime adapter is a separate integration. Its completed or pending GPU results must be '
              'reported independently; the controlled result above is not a substitute for a full progressive evaluation.', '',
              '## Reproducibility', '', f'Method freeze SHA-256: `{sha256(args.freeze)}`.',
              f'Fresh summary SHA-256: `{sha256(args.summary)}`.',
              f'Report builder SHA-256: `{sha256(Path(__file__))}`.', '',
              'See the final method JSON, generation contracts, per-case decisions, model-search records, runtime snapshots '
              'and archive checksum receipts for exact seeds, weights, source versions and environments.']
    (args.output_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(args.output_dir / 'result-status.json', dict(meets_all_primary_point_targets=meets,
        selection_only_component_meets_all_point_targets=all(
            summary['metrics'][key]['trained_surface_raw']['uniform_raw']['relative_gain_percent'] >= 3
            for key in PRIMARY),
        selection_only_is_secondary=True,
        each_primary_interval_excludes_no_improvement=all(value['relative_ci_percent'][0] > 0 for value in values.values()),
        true_amodal_ground_truth=False, full_progressive_improvement_established=False,
        freeze_sha256=sha256(args.freeze), summary_sha256=sha256(args.summary), images=len(cases)))
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage, Preformatted
    styles = getSampleStyleSheet()
    document = []
    code, block = False, []
    for line in lines:
        if line.startswith('```'):
            if code:
                document += [Preformatted('\n'.join(block), styles['Code']), Spacer(1, 8)]
                block = []
            code = not code
            continue
        if code:
            block.append(line)
        elif line.startswith('!['):
            document.append(PDFImage(str(args.output_dir / 'fresh-metric-gains.png'), width=6.6*inch, height=3.2*inch))
        elif line.startswith('|'):
            if not line.startswith('| ---'):
                document.append(Paragraph(escape(line.strip('|').replace('|', ' · ')), styles['BodyText']))
        elif line:
            style = styles['Title'] if line.startswith('# ') else styles['Heading2'] if line.startswith('## ') else styles['BodyText']
            document += [Paragraph(escape(line.lstrip('# ')), style), Spacer(1, 7)]
    SimpleDocTemplate(str(args.output_dir / 'REPORT.pdf'), rightMargin=38, leftMargin=38,
                      topMargin=38, bottomMargin=38).build(document)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('experiment-dir','run-dir','extra-run-dir','metrics-dir','freeze','summary','output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--full-comparison', type=Path)
    p.add_argument('--panels-only', action='store_true')
    p.add_argument('--reuse-panels', action='store_true')
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(a.freeze.read_text())
    if freeze['status'] != 'frozen_before_fresh_test':
        raise ValueError('The fresh report requires a final pre-test freeze')
    cases = json.loads((a.experiment_dir / 'cases.json').read_text())
    if set(case['image_identity'] for case in cases) & set(freeze['development']['image_identities']):
        raise ValueError('Fresh gallery overlaps development')
    inventory = json.loads((a.output_dir / 'gallery_cases.json').read_text()) if a.reuse_panels else build_panels(a, cases, freeze)
    if {case['case_id'] for case in inventory} != {case['case_id'] for case in cases}:
        raise ValueError('Gallery inventory is incomplete')
    for case in inventory:
        if sha256(a.output_dir / case['panel']) != case['panel_sha256']:
            raise ValueError('Gallery panel checksum mismatch')
    gallery(a.output_dir, inventory)
    if a.panels_only:
        return
    summary = json.loads(a.summary.read_text())
    if summary['freeze_sha256'] != sha256(a.freeze) or summary['images'] != len(cases):
        raise ValueError('Fresh summary and gallery identities differ')
    replay = json.loads((a.experiment_dir / 'analysis-eight/replay-verification.json').read_text())
    if (replay.get('passed') is not True or replay['freeze_sha256'] != sha256(a.freeze)
            or replay['images'] != len(cases)):
        raise ValueError('Complete strict replay is required before the final report')
    plot_result(a.output_dir, summary)
    make_report(a, summary, freeze, inventory)
    print(a.output_dir / 'REPORT.md')


if __name__ == '__main__':
    main()
