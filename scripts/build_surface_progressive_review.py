#!/usr/bin/env python3
"""Publish the audited full progressive result separately from controlled gains."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json


def main():
    output=ROOT/'output/surface-progressive-review-20261111'
    recovered=ROOT/'output/surface-job-recovery-20261110/progressive/output'
    comparison_dir=recovered/'surface-followup-20261110/progressive-comparison'
    comparison=json.loads((comparison_dir/'comparison.json').read_text())
    audit=json.loads((output/'audit.json').read_text())
    if not audit['passed'] or audit['targets']!=2565:
        raise ValueError('Require completed independent audit')
    confirmation=json.loads((ROOT/'output/surface-confirmation128-20261110/report/audit.json').read_text())
    counts,reasons,passes=Counter(),Counter(),Counter()
    for path in (recovered/'pdmc-surface-progressive-20261110').glob('targets/*/*/result.json'):
        row=json.loads(path.read_text())
        n=len(row.get('diagnostics',{}).get('surface_iterations',[]))
        counts['reached' if n else 'not_reached']+=1
        passes[n]+=1
        if not n:
            reasons[row.get('failure_reason',row['status'])]+=1
    write_json(output/'sampling_coverage.json',dict(counts=counts,not_reached_reasons=reasons,passes=passes))
    for name in ('comparison.json','REPORT.md','baseline-verification.json','surface-verification.json'):
        destination=output/'original-comparison'/name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(comparison_dir/name,destination)
    shutil.copyfile(recovered/'surface-followup-20261110-design/progressive-experiments.csv',output/'registered-experiments.csv')
    lines=['# Full progressive Surface-PMC result','',
        '**All four full-benchmark metrics improve, but the requested 3% threshold is not met.**','',
        'The matched control and frozen corrected Surface-PMC method each evaluated all 2,565 targets. '
        'Inputs, reference bytes, model checkpoints, metric operations and installed packages match. '
        'The surface method uses eight candidates per progressive pass, so its generation compute is higher.','',
        '| Metric | Paired control | Surface | Improvement | Nominal 95% interval | n |',
        '| --- | ---: | ---: | ---: | --- | ---: |']
    for name,row in audit['metrics'].items():
        lo,hi=row['relative_ci_percent']
        lines.append(f'| {name} | {row["baseline"]:.6f} | {row["method"]:.6f} | '
            f'+{row["relative_gain_percent"]:.3f}% | [{lo:+.3f}%, {hi:+.3f}%] | {row["n"]} |')
    lines += ['', 'CLIP uses 946 shared successful outputs. LPIPS, VGG feature similarity and SSIM include '
        'all 2,565 targets and the original failure penalties. Intervals use 5,000 source-stratified '
        'image-cluster resamples. There is no hidden-RGB MAE ground truth for this natural-occlusion benchmark.', '',
        '## Coverage and diagnosis','',
        'Saved outputs increased from **951 to 958**: 946 shared, 12 gained and 5 lost. Failures decreased '
        'from 1,614 to 1,607. Iteration-limited saved outputs decreased from 152 to 140; this does not certify visual completeness.','',
        '**1,599 targets never reached surface sampling; 966 did.** The unchanged early outcomes comprise '
        '877 label mismatches, 251 ambiguous targets, 57 no-detection cases, 69 small masks, one unknown '
        'detected class, and 344 cases where upstream found no occlusion. These remain in the benchmark.','',
        'Among the 946 shared successful outputs, LPIPS improves **4.165%**. Across the full inventory it '
        'improves **0.835%**. The full LPIPS absolute gain of 0.006134 consists of +0.004375 from shared '
        'successes, +0.003024 from gained outputs, and -0.001265 from lost outputs. The shared-success '
        'result is a diagnostic, not a replacement success criterion.','',
        '## Independent controlled confirmation remains a separate result','',
        'The unchanged raw selector, without RGB correction or visible restoration, was independently '
        'confirmed on 128 new artificially occluded VG/COCOA images:','',
        '| Metric | Improvement | Nominal 95% interval |','| --- | ---: | --- |']
    labels=dict(hidden_core_mae='Hidden RGB MAE',object_lpips='Object LPIPS',dreamsim='DreamSim',clip_text_cosine='CLIP cosine')
    for key,label in labels.items():
        row=confirmation['uniform_comparison'][key];lo,hi=row['relative_ci_percent']
        lines.append(f'| {label} | +{row["relative_gain_percent"]:.3f}% | [{lo:+.3f}%, {hi:+.3f}%] |')
    lines += ['', 'All four individual lower bounds exceed 3%, versus expected uniform selection from the '
        'same eight candidates. SSIM improves only 1.668%. The controlled raw selector and the full '
        'corrected method have different interventions and reference protocols; their gains cannot '
        'be pooled or substituted. CLIP is also an explicit selector input.','',
        '## Verification and follow-up','',
        'Both full runs passed remote full/provenance verification and were registered. Local audit '
        'verified 14,123 recovered files, every target identity, cache linkage, complete-run result '
        'hashes, and 4,921 pre-resume records (2,565 control plus 2,356 surface). It independently '
        'reproduced the means, failure accounting and bootstrap intervals. Native PNGs remain on the '
        'cluster; local audit did not rerun neural metrics. The original timeout snapshot remains '
        'unfinished, with the successful resume recorded separately.','',
        'A frozen diagnostic is queued as **39404**, followed by finalizer **39407**. It replaces only '
        'the final corrected RGB with the already-selected raw candidate, preserving masks, geometry, '
        'failures and preceding trajectories. All controls are rescored together. This is exploratory '
        'and cannot establish the result of an end-to-end raw progressive run. No new diffusion or '
        'hyperparameter search is involved. The original five-hour window was exceeded.','',
        '[Numerical audit](audit.json) | [Every paired target](paired_records.json) | '
        '[Original full comparison](original-comparison/REPORT.md) | [One-page PDF](REPORT.pdf)','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
    styles=getSampleStyleSheet()
    flow=[Paragraph('Surface-PMC: full benchmark result',styles['Title']),
        Paragraph('All four metrics improve; the full-benchmark 3% target is not met.',styles['Heading2']),
        Paragraph('2,565 matched targets. Frozen corrected progressive selection, eight candidates per pass.',styles['BodyText']),Spacer(1,14)]
    def table(headers,rows):
        value=Table([headers,*rows],hAlign='LEFT')
        value.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8edf4')),
            ('GRID',(0,0),(-1,-1),.3,colors.HexColor('#a0acb8')),('FONTSIZE',(0,0),(-1,-1),9),
            ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
        return value
    values=[]
    for name,row in audit['metrics'].items():
        lo,hi=row['relative_ci_percent']
        values.append([name,f'{row["baseline"]:.6f}',f'{row["method"]:.6f}',
                       f'+{row["relative_gain_percent"]:.2f}%',f'[{lo:.2f}, {hi:.2f}]%',str(row['n'])])
    flow += [table(['Metric','Control','Surface','Gain','95% interval','n'],values),Spacer(1,12)]
    paragraphs=[
        'CLIP uses 946 shared successes. Other metrics retain all targets and failure penalties. '
        'Intervals resample source images with all their targets. Natural occlusions have no hidden RGB MAE truth.',
        'Coverage: 951 control outputs, 958 surface outputs; 12 gained and 5 lost. Of 2,565 targets, '
        '1,599 never reached surface sampling. Shared-success LPIPS improves 4.16%, while full LPIPS improves 0.83%. '
        'The subset does not replace the full criterion.',
        'Independent raw-selection confirmation: 128 new artificial-occlusion cases; '
        'MAE +7.64%, LPIPS +11.33%, DreamSim +16.77%, CLIP +4.07%. All four individual nominal 95% '
        'lower bounds exceed 3%. SSIM is only +1.67%. This uses a different intervention and reference '
        'protocol from the full corrected run.',
        'Verified: remote full/provenance checks and registration; local checks of 14,123 recovered '
        'files, all target identities, 4,921 pre-resume record hashes, all means and bootstrap intervals. '
        'Native PNGs remain on the cluster. Local audit did not rerun neural metrics.',
        'Next diagnostic: job 39404, finalizer 39407. Reuse saved final raw selections while fixing '
        'masks, geometry, failures and previous trajectories. All controls are rescored together. '
        'No new diffusion or parameter search. This does not constitute an end-to-end raw progressive run.',
        'The original five-hour deadline was missed. See REPORT.md for full tables, evidence and scope.'
    ]
    for text in paragraphs:
        flow += [Paragraph(text,styles['BodyText']),Spacer(1,12)]
    SimpleDocTemplate(str(output/'REPORT.pdf'),pagesize=A4,leftMargin=38,rightMargin=38,topMargin=38,bottomMargin=38).build(flow)
    shutil.copyfile(Path(__file__),output/Path(__file__).name)
    files={p.relative_to(output).as_posix():sha256(p) for p in output.rglob('*')
           if p.is_file() and p.suffix not in ('.zip',) and p.name not in ('package-receipt.json','MANIFEST.json')}
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),files=files,
                  scope='Reading package and numerical audit; native images and full recovered metadata remain separate')
    write_json(output/'MANIFEST.json',manifest)
    package=output/'surface-progressive-review.zip'
    with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(files):
            bundle.write(output/name,name)
        bundle.write(output/'MANIFEST.json','MANIFEST.json')
    with zipfile.ZipFile(package) as bundle:
        assert bundle.testzip() is None
        for name,expected in files.items():
            assert hashlib.sha256(bundle.read(name)).hexdigest()==expected
    write_json(output/'package-receipt.json',dict(passed=True,sha256=sha256(package),
        bytes=package.stat().st_size,checked_payloads=len(files)))
    print(json.dumps(dict(report=str(output/'REPORT.pdf'),package=str(package),bytes=package.stat().st_size)))


if __name__ == '__main__':
    main()
