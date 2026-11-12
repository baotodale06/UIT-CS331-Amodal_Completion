#!/usr/bin/env python3
"""Audit recovered confirmation evidence and publish its complete controlled result."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.surface_policy import verify_frozen_surface_sources
from amodal_benchmark.surface_refinement import predict_rank
from finalize_surface_confirmation import check_cohort

PRIMARY = ('hidden_core_mae', 'object_lpips', 'dreamsim', 'clip_text_cosine')
EXTRA = ('hidden_spatial_lpips', 'object_feature_similarity', 'object_ssim')
LABELS = dict(hidden_core_mae='Hidden RGB MAE', object_lpips='Object LPIPS',
    dreamsim='DreamSim', clip_text_cosine='CLIP cosine',
    hidden_spatial_lpips='Hidden spatial LPIPS', object_feature_similarity='VGG similarity',
    object_ssim='Object SSIM')
HIGHER = {'clip_text_cosine', 'clip_text_logit', 'object_feature_similarity', 'object_ssim'}


def rgb(path):
    return np.array(Image.open(path).convert('RGB'))


def mask(path):
    return np.array(Image.open(path).convert('L')) > 0


def independent_summary(control, method, sources, higher):
    control, method = np.asarray(control), np.asarray(method)
    sign = 1 if higher else -1
    rng = np.random.default_rng(20261110)
    first, second = np.zeros(5000), np.zeros(5000)
    sources = np.asarray(sources)
    for source in sorted(set(sources)):
        indices = np.flatnonzero(sources == source)
        sampled = indices[rng.integers(len(indices), size=(5000, len(indices)))]
        first += control[sampled].sum(axis=1)
        second += method[sampled].sum(axis=1)
    first /= len(control)
    second /= len(control)
    if np.any(first <= 0):
        raise ValueError('Nonpositive bootstrap baseline requires explicit analysis')
    return dict(n=len(control), baseline=float(control.mean()), method=float(method.mean()),
        relative_gain_percent=float(100 * sign * (method-control).mean() / control.mean()),
        relative_ci_percent=np.percentile(100 * sign * (second-first) / first, [2.5, 97.5]).tolist())


def display_font(size):
    for path in ('C:/Windows/Fonts/arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_panel(case, row, original, recovered, output):
    name = case['case_id']
    selected = recovered / 'selection-only' / name
    baseline = recovered / 'full/candidates' / name
    truth = original / 'evaluation_only' / name
    def isolated(image, support):
        value = rgb(image)
        value[~mask(support)] = 0
        return Image.fromarray(value)
    frames = [Image.open(original/'inputs'/name/'observed.png').convert('RGB'),
        isolated(baseline/'seed-11.png', baseline/'seed-11-mask.png'),
        isolated(selected/'selected.png', selected/'mask.png'),
        isolated(truth/'truth.png', truth/'full.png')]
    panel = Image.new('RGB', (1024, 288), '#f5f6f8')
    draw = ImageDraw.Draw(panel)
    for index, (title, frame) in enumerate(zip(
            ['Observed input', 'Fixed seed 11 control', 'Selected unmodified candidate', 'Withheld complete-object RGB'], frames)):
        draw.text((index*256+7, 8), title, font=display_font(13), fill='#13243a')
        panel.paste(frame.resize((256, 256), Image.Resampling.LANCZOS), (index*256, 32))
    path = output / 'panels' / (name + '.webp')
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path, quality=92, method=6)
    values = [item['hidden_core_mae'] for item in row['raw']]
    chosen = row['raw'][row['selection']['trained_surface']]['hidden_core_mae']
    return dict(case_id=name, label=case['label'], source=case['source'],
        panel='panels/'+path.name, panel_sha256=sha256(path),
        uniform_mae=float(np.mean(values)), seed_11_mae=values[0], selected_mae=chosen,
        gain=float(np.mean(values)-chosen), selected_seed=row['candidate_seeds'][row['selection']['trained_surface']])


def publish(output, audit, panels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values = audit['uniform_comparison']
    ordered = PRIMARY + EXTRA
    fig, ax = plt.subplots(figsize=(9.1, 5.0), layout='constrained')
    for index, metric in enumerate(ordered):
        row = values[metric]
        mean = row['relative_gain_percent']
        low, high = row['relative_ci_percent']
        color = '#176894' if metric in PRIMARY else '#787878'
        ax.errorbar(mean, index, xerr=[[mean-low], [high-mean]], fmt='o', color=color, capsize=4)
    ax.axvline(0, color='#999999', linewidth=.8)
    ax.axvline(3, color='#a24820', linestyle='--', label='3% improvement threshold')
    ax.axvspan(3, 5, color='#d4e3d3', alpha=.35)
    ax.set_yticks(range(len(ordered)), [LABELS[key]+(' *' if key in PRIMARY else '') for key in ordered])
    ax.invert_yaxis()
    ax.set_xlabel('Relative improvement (%) vs. uniform selection from the same eight candidates')
    ax.set_title('Independent 128-image Surface-PMC confirmation\nNominal 95% paired, source-stratified bootstrap intervals')
    ax.legend(loc='lower right', frameon=False)
    ax.grid(axis='x', alpha=.15)
    fig.savefig(output/'improvement-intervals.png', dpi=180)
    fig.savefig(output/'improvement-intervals.pdf')
    plt.close(fig)
    lines = ['# Independently confirmed Surface-PMC selection', '',
        '**The four predeclared controlled endpoints exceed 3% improvement, including every nominal 95% lower bound.**', '',
        'The frozen selector chooses one unmodified output from eight PMC candidates. '
        'The 128 test images (64 VG, 64 COCOA) were excluded from all 349 earlier images '
        'before generation. There are zero generation failures among 1,024 candidates.', '',
        '| Metric | Uniform baseline | Surface selection | Relative improvement | Nominal 95% interval |',
        '| --- | ---: | ---: | ---: | --- |']
    for key in PRIMARY:
        row = values[key]; low, high = row['relative_ci_percent']
        lines.append(f'| {LABELS[key]} | {row["baseline"]:.6f} | {row["method"]:.6f} | '
                     f'{row["relative_gain_percent"]:+.3f}% | [{low:+.3f}%, {high:+.3f}%] |')
    lines += ['', 'Each metric uses all 128 images. Lower is better for MAE, LPIPS, and DreamSim; '
        'higher is better for CLIP. The intervals use 5,000 paired resamples within source. '
        'They are individual nominal intervals, not a simultaneous confidence band.', '',
        '![Improvement intervals](improvement-intervals.png)', '',
        '## Additional recorded metrics', '',
        '| Metric | Relative improvement | Nominal 95% interval |',
        '| --- | ---: | --- |']
    for key in EXTRA:
        row = values[key]; low, high = row['relative_ci_percent']
        lines.append(f'| {LABELS[key]} | {row["relative_gain_percent"]:+.3f}% | [{low:+.3f}%, {high:+.3f}%] |')
    lines += ['', '**SSIM improves by 1.67%, below 3%.** The result does not meet a claim '
        'that every recorded diagnostic improves by at least 3%.', '',
        '## Relevant controls', '',
        'These are gains of surface selection over each named control; all use the same saved candidate pools.', '',
        '| Control | MAE gain | LPIPS gain | DreamSim gain | CLIP gain |',
        '| --- | ---: | ---: | ---: | ---: |']
    for control, label in [('seed_11_raw','Fixed seed 11'), ('uniform_valid_raw','Uniform over valid candidates'),
                           ('semantic_only_raw','CLIP-only selection'), ('local_surface_raw','Local surface rule')]:
        points = audit['control_comparisons'][control]
        lines.append('| '+label+' | '+' | '.join(f'{points[k]["relative_gain_percent"]:+.3f}%' for k in PRIMARY)+' |')
    lines += ['', 'The selector improves reconstruction over CLIP-only selection but gives up 0.632% CLIP. '
        'CLIP is an explicit selector input. The seed-11 comparison uses more sampling compute '
        'for the surface method; it is not an equal-compute generator comparison.', '',
        '## Evidence and limits', '',
        f'Local verification checks {audit["archive_member_checks"]:,} archive members, '
        'all 1,024 candidate-record identities, all 128 rank decisions, unchanged selected RGB/masks, '
        'input/evaluation hashes, selection-before-measurement timing, and all metric means and intervals. '
        'Hidden MAE was independently recomputed from the selected PNGs and withheld RGB for every image. '
        'The original generation-node replay covers every candidate and passed. '
        'Pretrained neural metrics were not rerun during this local audit.', '',
        'This is a one-step artificial-occlusion experiment on VG and COCOA, using the paper\'s '
        'CLIP/DreamSim/LPIPS metric families plus hidden RGB MAE. It is not the original paper\'s '
        'COCO/Open Images benchmark, a human preference study, or proof of full progressive improvement. '
        'The earlier 166-image corrected primary remains a missed-CLIP result; this new cohort independently '
        'confirms its previously secondary selection-only component. No parameters were retuned on this cohort. '
        'The original five-hour deadline was missed.', '',
        'The separate full progressive run timed out after 2,356/2,565 targets and is being resumed '
        'unchanged as job39387; finalizer37814 waits for it. Its corrected policy differs from this raw selector.', '',
        '[Audit receipt](audit.json) · [All 128 cases, including losses](gallery.html) · '
        '[Plot PDF](improvement-intervals.pdf) · [Report PDF](REPORT.pdf)', '']
    report = '\n'.join(lines)
    (output/'REPORT.md').write_text(report, encoding='utf-8')
    cards = []
    for case in panels:
        cards.append(f'<article data-source="{escape(case["source"])}" data-gain="{case["gain"]}"><h2>'
            f'{escape(case["case_id"])} — {escape(case["label"])}</h2><p>Expected uniform MAE '
            f'{case["uniform_mae"]:.4f}; fixed seed11 {case["seed_11_mae"]:.4f}; selected '
            f'{case["selected_mae"]:.4f}; gain vs uniform {case["gain"]:+.4f}; seed {case["selected_seed"]}.</p>'
            f'<img loading="lazy" src="{escape(case["panel"])}" alt="Observed, fixed-seed control, '
            'surface-selected output and withheld complete-object RGB"></article>')
    gallery = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>128-image Surface-PMC confirmation</title><style>body{font:16px system-ui;background:#f5f6f8;color:#182b40;margin:24px auto;max-width:1100px;padding:0 16px}article{background:white;padding:16px;margin:18px 0;border:1px solid #dbe0e5;border-radius:8px}img{width:100%;height:auto}h2{font-size:18px}input,select{font:inherit;padding:8px;margin:4px}p{line-height:1.5}</style>
<h1>Independent Surface-PMC confirmation: all 128 cases</h1><p>128 frozen images; 86 MAE improvements and 42 losses versus expected uniform selection. The displayed fixed-seed image is a separate control; uniform expectation has no single representative image. This gallery is not a new human evaluation.</p>
<input id="search" placeholder="Search case or category"><select id="source"><option value="">All sources</option><option>VG</option><option>COCOA</option></select><select id="outcome"><option value="">All outcomes</option><option value="gain">MAE improvements</option><option value="loss">MAE losses</option></select><p id="count"></p>'''+''.join(cards)+'''
<script>const q=document.querySelector('#search'),s=document.querySelector('#source'),o=document.querySelector('#outcome');function filter(){let n=0;document.querySelectorAll('article').forEach(a=>{let ok=a.textContent.toLowerCase().includes(q.value.toLowerCase())&&(!s.value||a.dataset.source===s.value)&&(!o.value||(o.value==='gain'?Number(a.dataset.gain)>0:Number(a.dataset.gain)<0));a.hidden=!ok;if(ok)n++});document.querySelector('#count').textContent=n+' cases shown'}[q,s,o].forEach(x=>x.addEventListener('input',filter));filter()</script>'''
    (output/'gallery.html').write_text(gallery, encoding='utf-8')
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as PDFImage
    styles = getSampleStyleSheet()
    flow = [Paragraph('Independent Surface-PMC confirmation', styles['Title']),
        Paragraph('128 unseen VG/COCOA images. All four predeclared controlled endpoints exceed 3%, including their nominal 95% lower bounds.', styles['BodyText']), Spacer(1,12)]
    table = [['Metric','Baseline','Selected','Gain','95% interval']]
    for key in PRIMARY:
        row=values[key]; low,high=row['relative_ci_percent']
        table.append([LABELS[key],f'{row["baseline"]:.6f}',f'{row["method"]:.6f}',f'{row["relative_gain_percent"]:.2f}%',f'[{low:.2f}, {high:.2f}]%'])
    rendered=Table(table,hAlign='LEFT',colWidths=[112,76,76,59,111])
    rendered.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6edf3')),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#aab5bd')),('FONTSIZE',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    flow.extend([rendered,Spacer(1,12),PDFImage(str(output/'improvement-intervals.png'),width=445,height=245)])
    for paragraph in [
        'Baseline: expected uniform selection from the same eight generated candidates. Selected RGB and masks are unchanged generator outputs. Zero failures among 1,024 candidates; 86 image-level MAE gains and 42 losses.',
        'Additional recorded metrics: VGG similarity +3.73%; SSIM +1.67% (below 3%). Compared with CLIP-only selection, MAE improves 4.38%, LPIPS 6.72%, and DreamSim 11.49%, while CLIP decreases 0.63%. CLIP is also a selector input.',
        'Verified: archive hashes, candidate identities, 128 rank decisions, exact selected outputs, timing, independent selected-image hidden MAE, and metric means/intervals. Native all-candidate replay passed. This local audit did not rerun pretrained metric networks.',
        'Scope: one-step artificial occlusion, not the original paper benchmark or full progressive improvement. The full run is separately resuming after a time limit. The original five-hour deadline was missed. Complete records and every case are in REPORT.md, audit.json and gallery.html.']:
        flow.extend([Spacer(1,9),Paragraph(paragraph,styles['BodyText'])])
    SimpleDocTemplate(str(output/'REPORT.pdf'),pagesize=A4,rightMargin=40,leftMargin=40,topMargin=38,bottomMargin=38).build(flow)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recovery-dir',type=Path,default=ROOT/'output/surface-job-recovery-20261110')
    p.add_argument('--experiment-dir',type=Path,default=ROOT/'output/surface-confirmation128-20261110')
    p.add_argument('--output-dir',type=Path,default=ROOT/'output/surface-confirmation128-20261110/report')
    args=p.parse_args(); recovered=args.recovery_dir/'confirmation/output/surface-confirmation128-20261110'
    frozen_path=args.recovery_dir/'confirmation/output/surface-confirmation128-20261110-design/selection-only-freeze.json'
    frozen=json.loads(frozen_path.read_text()); verify_frozen_surface_sources(frozen)
    cases=json.loads((recovered/'cases.json').read_text()); check_cohort(cases,frozen)
    expected_digest='711f49db26e19d0655c64b59682d9ae50c0841d38071096ad9dc7f83d733d4c4'
    assert sha256(frozen_path)==expected_digest
    parent=json.loads((ROOT/'output/surface-followup-20261110-design/final-method-freeze.json').read_text())
    assert frozen['ranker']==parent['ranker'] and frozen['candidate_seeds']==parent['candidate_seeds']
    for relative,expected in frozen['confirmation']['input_sha256'].items():
        assert sha256(ROOT/relative)==expected,relative
    archive_checks=0
    for name,archive_name in [('confirmation-receipt.json','confirmation-evidence.tar.gz'),('confirmation-support-receipt.json','confirmation-support-evidence.tar.gz')]:
        receipt=json.loads((args.recovery_dir/name).read_text())
        assert sha256(args.recovery_dir/archive_name)==receipt.get('sha256',receipt.get('archive_sha256'))
        for relative,expected in receipt['member_sha256'].items():
            assert sha256(args.recovery_dir/'confirmation'/relative)==expected,relative
            archive_checks+=1
    contract=json.loads((recovered/'inference/contract.json').read_text())
    assert contract['freeze_sha256']==expected_digest and contract['evaluation_data_access'] is False
    for index in range(4):
        runtime=json.loads((recovered/f'metrics/runtime-worker-{index}.json').read_text())
        assert datetime.fromisoformat(contract['finished_at']) <= datetime.fromisoformat(runtime['started_at'])
    replay=json.loads((recovered/'analysis/replay-verification.json').read_text())
    assert replay['passed'] and replay['images']==128 and replay['freeze_sha256']==expected_digest
    summary=json.loads((recovered/'analysis/summary.json').read_text())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]; panels=[]; mae_errors=[]; rank_errors=[]
    for case in cases:
        name=case['case_id']; row=json.loads((recovered/'metrics'/(name+'.json')).read_text())
        decision=json.loads((recovered/'inference/decisions'/(name+'.json')).read_text())
        assert row['raw']==row['refined']==row['calibrated']
        assert row['candidate_seeds']==frozen['candidate_seeds'] and row['freeze_sha256']==expected_digest
        index,losses=predict_rank(decision['observable_scores'],frozen['ranker'])
        assert index==decision['index']==row['selection']['trained_surface']
        calculated_losses=np.asarray(losses,dtype=float)
        recorded_losses=np.asarray(decision['losses'],dtype=float)
        finite=np.isfinite(calculated_losses)
        assert np.array_equal(finite,np.isfinite(recorded_losses))
        error=float(np.max(np.abs(calculated_losses[finite]-recorded_losses[finite]))) if finite.any() else 0.
        assert error<1e-10; rank_errors.append(error)
        for seed in frozen['candidate_seeds']:
            candidate=recovered/'full/candidates'/name/f'seed-{seed}.json'
            assert sha256(candidate)==decision['candidate_record_sha256'][candidate.name]
        native=json.loads((recovered/'full/candidates'/name/f'seed-{decision["seed"]}.json').read_text())
        selected=recovered/'selection-only'/name
        assert sha256(selected/'selected.png')==native['artifacts'][f'seed-{decision["seed"]}.png']
        assert sha256(selected/'mask.png')==native['artifacts'][f'seed-{decision["seed"]}-mask.png']
        for filename,expected in row['selected_artifacts'].items():
            assert sha256(recovered/'metrics/selected'/name/filename)==expected
        assert np.array_equal(rgb(selected/'selected.png'),rgb(recovered/'metrics/selected'/name/'surface.png'))
        for subdir,hashes in [('inputs',case['input_hashes']),('evaluation_only',case['evaluation_hashes'])]:
            for filename,expected in hashes.items():
                assert sha256(args.experiment_dir/subdir/name/filename)==expected
        hidden=mask(args.experiment_dir/'evaluation_only'/name/'hidden_core.png')
        truth=rgb(args.experiment_dir/'evaluation_only'/name/'truth.png')
        calculated=float(np.abs(rgb(selected/'selected.png')[hidden].astype(float)-truth[hidden]).mean()/255)
        error=abs(calculated-row['raw'][index]['hidden_core_mae']); assert error<1e-12; mae_errors.append(error)
        rows.append(row); panels.append(render_panel(case,row,args.experiment_dir,recovered,args.output_dir))
    values={}; controls={name:{} for name in ('seed_11_raw','uniform_valid_raw','semantic_only_raw','local_surface_raw')}
    sources=[row['source'] for row in rows]
    for metric in rows[0]['raw'][0]:
        chosen=[row['raw'][row['selection']['trained_surface']][metric] for row in rows]
        uniform=[float(np.mean([value[metric] for value in row['raw']])) for row in rows]
        result=independent_summary(uniform,chosen,sources,metric in HIGHER)
        recorded=summary['metrics'][metric]['trained_surface_raw']['uniform_raw']
        for key in ('baseline','method','relative_gain_percent','relative_ci_percent'):
            np.testing.assert_allclose(result[key],recorded[key],rtol=0,atol=1e-10)
        assert recorded['n']==128 and recorded['excluded_pairs']==0
        values[metric]=result
        for control in controls:
            if control=='uniform_valid_raw':
                baseline=[float(np.mean([row['raw'][i][metric] for i in (row['valid_indices'] or [0])])) for row in rows]
            else:
                policy=control.removesuffix('_raw')
                baseline=[row['raw'][row['selection'][policy]][metric] for row in rows]
            value=independent_summary(baseline,chosen,sources,metric in HIGHER)
            recorded_control=summary['metrics'][metric]['trained_surface_raw'][control]
            for key in ('relative_gain_percent','relative_ci_percent'):
                np.testing.assert_allclose(value[key],recorded_control[key],rtol=0,atol=1e-10)
            controls[control][metric]=value
    audit=dict(passed=True,created_at=datetime.now(timezone.utc).isoformat(),images=128,
        sources=dict(Counter(sources)),freeze_sha256=expected_digest,
        archive_member_checks=archive_checks,candidate_record_checks=128*8,rank_decision_checks=128,
        max_rank_loss_error=max(rank_errors),selected_mae_checks=128,max_selected_mae_error=max(mae_errors),
        native_replay_passed=True,neural_metrics_recomputed_locally=False,
        generation_failures=sum(row['generation_failures'] for row in rows),
        candidate_errors=sum(row['candidate_error_count'] for row in rows),
        point_target_met=all(values[k]['relative_gain_percent']>=3 for k in PRIMARY),
        lower_nominal_95_bounds_at_least3=all(values[k]['relative_ci_percent'][0]>=3 for k in PRIMARY),
        all_recorded_metric_points_at_least3=all(value['relative_gain_percent']>=3 for value in values.values()),
        uniform_comparison=values,control_comparisons=controls,
        source_sha256=sha256(Path(__file__)),scope='Independent one-step artificial-occlusion confirmation; full progressive result remains separate.')
    assert audit['generation_failures']==0 and audit['candidate_errors']==0
    write_json(args.output_dir/'audit.json',audit)
    write_json(args.output_dir/'gallery_cases.json',panels)
    publish(args.output_dir,audit,panels)
    write_json(args.output_dir/'report-manifest.json',{p.relative_to(args.output_dir).as_posix():sha256(p)
        for p in sorted(args.output_dir.rglob('*')) if p.is_file() and p.name!='report-manifest.json'})
    print(json.dumps({key:audit[key] for key in ('passed','images','archive_member_checks','point_target_met','lower_nominal_95_bounds_at_least3','all_recorded_metric_points_at_least3','max_selected_mae_error')},indent=2))


if __name__=='__main__':
    main()
