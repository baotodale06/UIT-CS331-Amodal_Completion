#!/usr/bin/env python3
"""Post-hoc explanations of frozen decisions; never fit or change a selector."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.surface import METHODS, SurfaceConfig, score_surface, select_candidate
from amodal_benchmark.surface_evaluation import stratified_bootstrap


def decomposition(errors, losses, selected):
    """Exactly separate filtering, ranking and exclusion of the best candidate."""
    errors = np.asarray(errors, float)
    valid = [i for i, loss in enumerate(losses) if loss is not None] or [0]
    uniform, guarded = float(errors.mean()), float(errors[valid].mean())
    best, best_valid, chosen = float(errors.min()), float(errors[valid].min()), float(errors[selected])
    result = dict(total_gain=uniform - chosen, filter_gain=uniform - guarded,
                  ranking_gain=guarded - chosen, available_gain=uniform - best,
                  available_valid_gain=guarded - best_valid,
                  ranking_regret=chosen - best_valid, exclusion_regret=best_valid - best,
                  total_regret=chosen - best, valid_count=sum(v is not None for v in losses))
    assert abs(result['total_gain'] - result['filter_gain'] - result['ranking_gain']) < 1e-12
    assert abs(result['total_regret'] - result['exclusion_regret'] - result['ranking_regret']) < 1e-12
    assert abs(result['total_gain'] - result['available_gain'] + result['total_regret']) < 1e-12
    return result


def pair_concordance(errors, losses):
    """Equal weight per image; tied scores get half credit, tied errors omitted."""
    pairs = []
    for i in range(len(errors)):
        for j in range(i):
            if losses[i] is None or losses[j] is None or abs(errors[i] - errors[j]) <= 1e-12:
                continue
            difference = losses[i] - losses[j]
            pairs.append(.5 if abs(difference) <= 1e-12 else float(difference * (errors[i] - errors[j]) > 0))
    return float(np.mean(pairs)) if pairs else None


def fixed_support_truth_check(experiment, row, scales):
    """Evaluation-only: compare truth and chosen RGB using identical support."""
    case_id = row['case_id']
    inp, truth_dir = experiment / 'inputs' / case_id, experiment / 'evaluation_only' / case_id
    rgb = lambda path: np.array(Image.open(path).convert('RGB'))
    mask = lambda path: np.array(Image.open(path).convert('L')) > 0
    observed, visible, occlusion = rgb(inp / 'observed.png'), mask(inp / 'visible.png'), mask(inp / 'occlusion.png')
    truth, support = rgb(truth_dir / 'truth.png'), mask(truth_dir / 'full.png')
    seed = row['metrics'][row['selected']['surface_full']['index']]['seed']
    selected = rgb(experiment / 'full/candidates' / case_id / f'seed-{seed}.png')
    protocol = json.loads((experiment / 'protocol.json').read_text())
    config = SurfaceConfig(radius=protocol['boundary_radius'], min_region_pixels=protocol['min_region_pixels'],
                           min_visible_recall=protocol['min_visible_recall'])
    scores = [score_surface(observed, visible, occlusion, image, support, config) for image in (truth, selected)]
    losses = select_candidate(scores, METHODS['surface_full'], scales)['losses']
    return dict(truth_loss=losses[0], chosen_loss_on_identical_support=losses[1],
                truth_minus_chosen=(losses[0] - losses[1]) if all(v is not None for v in losses) else None,
                truth_terms=scores[0], chosen_terms_on_identical_support=scores[1])


def summarize(rows):
    source = [r['source'] for r in rows]
    strata = [r['cohort'] + '/' + r['source'] for r in rows]
    keys = [k for k in rows[0]['decomposition'] if k != 'valid_count']
    result = dict(n=len(rows), decomposition={k: stratified_bootstrap([r['decomposition'][k] for r in rows], source) for k in keys},
                  valid_pool_sizes=dict(Counter(r['decomposition']['valid_count'] for r in rows)))
    result['source_and_cohort_bootstrap_sensitivity'] = {
        k: stratified_bootstrap([r['decomposition'][k] for r in rows], strata)
        for k in ('total_gain', 'filter_gain', 'ranking_gain')}
    result['pair_concordance'] = {}
    for method in METHODS:
        eligible = [r for r in rows if r['concordance'][method] is not None]
        result['pair_concordance'][method] = stratified_bootstrap(
            [r['concordance'][method] for r in eligible], [r['source'] for r in eligible]) if eligible else None
    result['exploratory_paired_contrasts'] = {
        k: stratified_bootstrap([r['contrasts'][k] for r in rows], source) for k in rows[0]['contrasts']}
    checks = [r for r in rows if r.get('fixed_support_truth', {}).get('truth_minus_chosen') is not None]
    if checks:
        result['fixed_support_truth'] = dict(n=len(checks),
            erroneous_selected_rgb_scores_better_than_truth=sum(r['fixed_support_truth']['truth_minus_chosen'] > 1e-10 for r in checks),
            ties=sum(abs(r['fixed_support_truth']['truth_minus_chosen']) <= 1e-10 for r in checks),
            note='Post-hoc compatibility diagnostic on the same evaluation-only original SAM support. No selection is changed. This is not a human label or complete-object amodal ground truth.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--replication', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--include-truth-check', action='store_true')
    args = parser.parse_args()
    freeze_path = args.pilot / 'analysis/development_freeze.json'
    freeze = json.loads(freeze_path.read_text())
    rows, input_hashes = [], {}
    for cohort, experiment in [('pilot', args.pilot), ('replication', args.replication)]:
        path = experiment / 'analysis/test_cases.json'
        input_hashes[str(path)] = sha256(path)
        for row in json.loads(path.read_text()):
            errors = np.array([v['hidden_core_mae'] for v in row['metrics']])
            full = row['selected']['surface_full']
            error = lambda method: float(errors[row['selected'][method]['index']])
            diagnostic = dict(case_id=row['case_id'], source=row['source'], cohort=cohort, label=row['label'],
                decomposition=decomposition(errors, full['losses'], full['index']),
                concordance={name: pair_concordance(errors, row['selected'][name]['losses']) for name in METHODS},
                contrasts={
                    'local_gain_over_full': error('surface_full') - error('local_surface'),
                    'local_gain_over_visible_preservation': error('visible_preservation') - error('local_surface'),
                    'boundary_gain_over_full': error('surface_full') - error('boundary'),
                    'full_gain_over_visible_preservation': error('visible_preservation') - error('surface_full')})
            if args.include_truth_check:
                diagnostic['fixed_support_truth'] = fixed_support_truth_check(experiment, row, freeze['scales'])
            rows.append(diagnostic)
    summaries = {group: summarize([r for r in rows if group == 'pooled' or r['cohort'] == group])
                 for group in ('pilot', 'replication', 'pooled')}
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'cases.json', rows)
    write_json(output / 'summary.json', summaries)
    write_json(output / 'manifest.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        status='post-hoc explanatory analysis; introduced after pilot errors, before replication errors; no tuning',
        source_sha256=sha256(Path(__file__)), frozen_selection_sha256=sha256(freeze_path),
        analysis_inputs=input_hashes, include_truth_check=args.include_truth_check,
        case_diagnostics_sha256=sha256(output / 'cases.json'), summary_sha256=sha256(output / 'summary.json')))
    print(json.dumps(summaries['pooled'], indent=2))


if __name__ == '__main__':
    main()
