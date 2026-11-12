#!/usr/bin/env python3
"""Publish the audited negative ablation and saved-pool ceilings with their scope."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json


def main():
    output = ROOT/'output/surface-lastpass-review-20261111'
    audit = json.loads((output/'audit.json').read_text())
    headroom = json.loads((output/'headroom-audit.json').read_text())
    if not audit['passed'] or not headroom['passed'] or not (audit['targets'] == headroom['targets'] == 2565):
        raise ValueError('Both complete independent audits are required')
    metrics = ('CLIP', 'LPIPS', 'Feature Similarity', 'SSIM')
    below = headroom['raw_ceiling_below3_metrics']
    conclusion = ('Even reference-informed selection from the eight saved raw final candidates misses 3% for '+
        ', '.join(below)+'. Final raw RGB reranking alone cannot reach all four targets on these fixed outputs.' if below else
        'Each metric individually has at least 3% raw-pool headroom. These different oracle choices do not establish a jointly achievable result or an observation-only selector.')
    lines = ['# Surface-PMC: completed follow-up diagnostics', '',
        '**The full-benchmark 3% target remains unmet.** Removing final corrections trades a small CLIP gain for worse reconstruction metrics.', '',
        'Jobs 39404 and 39407 completed the fixed-mask final-RGB ablation. Job 39411 then scored all eight saved raw candidates for every successful surface target. '
        'Both diagnostics retain all 2,565 targets, the 958 surface outputs, the final masks and geometry, original references, all failure penalties, and preceding progressive trajectories. '
        'No diffusion generation, fitting, or hyperparameter search was performed.', '',
        '## Corrected versus raw final RGB', '',
        '| Metric | Corrected gain vs control | Raw-final gain vs control | Raw-final change vs corrected (95% interval) |',
        '| --- | ---: | ---: | ---: |']
    for metric in metrics:
        corrected = audit['contrasts']['surface_vs_baseline'][metric]['relative_gain_percent']
        raw = audit['contrasts']['raw_final_vs_baseline'][metric]['relative_gain_percent']
        direct = audit['contrasts']['raw_final_vs_surface'][metric]
        low, high = direct['relative_ci_percent']
        lines.append(f'| {metric} | {corrected:+.3f}% | {raw:+.3f}% | {direct["relative_gain_percent"]:+.3f}% [{low:+.3f}, {high:+.3f}] |')
    lines += ['', 'Positive gains are better for every metric. CLIP comparisons with the matched control use 946 shared successful targets; '
        'the direct raw-versus-corrected CLIP comparison uses all 958 surface successes. LPIPS, VGG feature similarity and SSIM use all 2,565 targets. '
        'Intervals are nominal 95% paired source-stratified image-cluster bootstrap intervals (5,000 resamples).', '',
        '## Saved final-pool headroom', '', '**'+conclusion+'**', '',
        'Oracle choices use evaluation-reference scores and choose separately for each metric. They are optimistic diagnostics, not deployable methods or an achieved combined result. '
        'All eight raw images receive the corrected output mask, even when the original selector would reject that candidate. These are bounds for final RGB selection with fixed masks and trajectories only.', '',
        '| Policy | CLIP gain | LPIPS gain | VGG gain | SSIM gain |', '| --- | ---: | ---: | ---: | ---: |']
    for policy, values in headroom['contrasts'].items():
        label = {'frozen_raw':'Frozen surface selector', 'semantic_only':'Semantic-only selector',
                 'uniform_raw':'Expected uniform raw', 'corrected':'Existing corrected output'}.get(policy, policy.replace('oracle_', 'Oracle for '))
        lines.append('| '+label+' | '+' | '.join(f'{values[m]["relative_gain_percent"]:+.3f}%' for m in metrics)+' |')
    b = audit['contrasts']['surface_vs_baseline']['LPIPS']['baseline']
    s = audit['contrasts']['surface_vs_baseline']['LPIPS']['method']
    n, k = 2565, 958
    saved_mean = (s*n-(n-k))/k
    requirements = []
    for target in (3, 5):
        required = (b*(1-target/100)*n-(n-k))/k
        requirements.append(dict(target_percent=target, required_saved_mean=required,
            further_saved_relative_reduction_percent=100*(saved_mean-required)/saved_mean))
    write_json(output/'lpips_failure_accounting.json', dict(failures=n-k, targets=n, fixed_failure_floor=(n-k)/n,
        corrected_saved_mean=saved_mean, requirements=requirements))
    lines += ['', 'With the 1,607 failures fixed, the corrected successful-output LPIPS mean is '
        f'{saved_mean:.6f}. Reaching 3% and 5% full-benchmark gains would require successful-output means '
        f'{requirements[0]["required_saved_mean"]:.6f} and {requirements[1]["required_saved_mean"]:.6f}: '
        f'another {requirements[0]["further_saved_relative_reduction_percent"]:.2f}% and '
        f'{requirements[1]["further_saved_relative_reduction_percent"]:.2f}% reduction from the corrected outputs.', '',
        '## Separate confirmed result', '',
        'The independently frozen 128-image artificial-occlusion test remains positive: hidden MAE +7.644%, object LPIPS +11.328%, DreamSim +16.766%, and CLIP cosine +4.068% '
        'against expected uniform selection from identical eight-candidate pools. All four individual nominal 95% lower bounds exceed 3%; SSIM (+1.668%) does not. '
        'This is one-step raw selection, not the full natural-occlusion benchmark. Hidden RGB MAE cannot be computed for naturally occluded objects without ground truth.', '',
        '## Verification and decision', '',
        f'The ablation audit verified {audit["recovered_member_checks"]:,} recovered files and {audit["original_member_checks"]:,} original-run files, '
        'all target identities, prior metric caches, candidate lineage, complete counts and paired intervals. '
        f'The pool audit verified {headroom["recovered_member_checks"]:,} recovered files, 2,565 parent links, 7,664 candidate score tuples, '
        '3,832 per-metric oracle choices, observable controls, and every summary mean and interval. Selected raw image hashes match the parent ablation. '
        'Original-score replay drift is at most 1.79e-7; selected-pool replay drift is '
        f'{max(headroom["selected_score_max_absolute_drift"].values()):.3g}. Native PNGs remain on the cluster; local neural metrics were not rerun.', '',
        'Final RGB removal is not supported as the route to the target. '+conclusion+' '
        'A future method must be evaluated with a separately frozen protocol and preserve these negative results. '
        'No full progressive rerun is launched from this diagnostic. All current diagnostic jobs are complete. The original five-hour deadline was exceeded.', '',
        '[Ablation audit](audit.json) | [Headroom audit](headroom-audit.json) | [Every ablation target](paired_records.json) | '
        '[Failure accounting](lpips_failure_accounting.json)', '']
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    for name in ('surface-lastpass-ablation-20261111', 'surface-candidate-headroom-20261111'):
        source = ROOT/'output'/(name+'-recovery')/'evidence/output'/name/'analysis'
        destination = output/('lastpass-original' if 'lastpass' in name else 'headroom-original')
        destination.mkdir(exist_ok=True)
        for filename in ('REPORT.md', 'summary.json'):
            shutil.copyfile(source/filename, destination/filename)
    payload = [p for p in output.rglob('*') if p.is_file() and p.suffix not in ('.zip',)
               and p.name not in ('delivery-manifest.json', 'delivery-receipt.json')]
    manifest = {p.relative_to(output).as_posix():sha256(p) for p in payload}
    write_json(output/'delivery-manifest.json', manifest)
    archive = output/'surface-followup-diagnostics.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for p in [*payload, output/'delivery-manifest.json']:
            bundle.write(p, p.relative_to(output).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        import hashlib
        for name, expected in manifest.items():
            assert hashlib.sha256(bundle.read(name)).hexdigest() == expected
    write_json(output/'delivery-receipt.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        bytes=archive.stat().st_size, sha256=sha256(archive), payload_members=len(manifest), all_payload_hashes_verified=True))
    print(conclusion)
    print(output/'REPORT.md')


if __name__ == '__main__':
    main()
