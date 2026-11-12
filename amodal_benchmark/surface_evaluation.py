"""Held-out diagnostics, deliberately separate from observable-only selection."""
import math
from pathlib import Path
import json

import numpy as np
from PIL import Image

from .dataset import sha256
from .surface import METHODS, select_candidate


COMPONENTS = ('boundary', 'color', 'texture', 'local_boundary', 'local_texture')


def load_pool(experiment, run, case, seeds, verify=True):
    folder = Path(run) / 'candidates' / case['case_id']
    records = []
    for seed in seeds:
        path = folder / f'seed-{seed}.json'
        if not path.exists():
            raise ValueError(f'Incomplete pool: {path}')
        record = json.loads(path.read_text())
        if record['case_id'] != case['case_id'] or record['seed'] != seed:
            raise ValueError('Candidate identity mismatch')
        if verify:
            if record['input_hashes'] != case['input_hashes']:
                raise ValueError('Candidate input identity mismatch')
            for filename, expected in record['artifacts'].items():
                if sha256(folder / filename) != expected:
                    raise ValueError('Candidate artifact changed: ' + filename)
        records.append(record)
    return records


def evaluate_pool(experiment, run, case, records):
    """No prediction mask can restrict RGB-error support."""
    evaluation = Path(experiment) / 'evaluation_only' / case['case_id']
    for filename, expected in case['evaluation_hashes'].items():
        if sha256(evaluation / filename) != expected:
            raise ValueError('Evaluation truth changed: ' + filename)
    truth = np.array(Image.open(evaluation / 'truth.png').convert('RGB'))
    hidden = np.array(Image.open(evaluation / 'hidden.png').convert('L')) > 0
    core = np.array(Image.open(evaluation / 'hidden_core.png').convert('L')) > 0
    visible = np.array(Image.open(Path(experiment) / 'inputs' / case['case_id'] / 'visible.png').convert('L')) > 0
    folder = Path(run) / 'candidates' / case['case_id']
    result = []
    for record in records:
        seed = record['seed']
        rgb_name, mask_name = f'seed-{seed}.png', f'seed-{seed}-mask.png'
        metrics = dict(seed=seed, hidden_core_mae=1., hidden_mae=1., visible_mae=1.,
                       hidden_mask_recall=0., visible_mask_recall=0., generation_missing=True)
        if rgb_name in record['artifacts']:
            candidate = np.array(Image.open(folder / rgb_name).convert('RGB'))
            if candidate.shape != truth.shape:
                raise ValueError('Evaluation RGB frame mismatch')
            error = np.abs(candidate.astype(np.float64) - truth) / 255
            metrics.update(hidden_core_mae=float(error[core].mean()), hidden_mae=float(error[hidden].mean()),
                           visible_mae=float(error[visible].mean()), generation_missing=False)
        if mask_name in record['artifacts']:
            predicted = np.array(Image.open(folder / mask_name).convert('L')) > 0
            metrics.update(hidden_mask_recall=float(predicted[core].mean()),
                           visible_mask_recall=float(predicted[visible].mean()))
        result.append(metrics)
    return result


def normalization(pools):
    scales = {}
    for key in COMPONENTS:
        values = [record['surface'][key] for records in pools for record in records
                  if record.get('surface', {}).get('valid') and
                  record['surface'].get(key) is not None and record['surface'][key] > 1e-8]
        scales[key] = float(np.median(values)) if values else 1.
    return scales


def selections(records, metrics, scales, development_method=None):
    scores = [record.get('surface', {'valid': False}) for record in records]
    chosen = {name: select_candidate(scores, weights, scales) for name, weights in METHODS.items()}
    chosen['visible_preservation'] = select_candidate(scores, {'visible_mae': 1.})
    chosen['fixed_seed_11'] = dict(index=0, fallback=False, losses=None)
    # Oracle is isolated here in the evaluation module, never in the scorer.
    chosen['oracle_hidden_mae'] = dict(index=min(range(len(metrics)), key=lambda i: metrics[i]['hidden_core_mae']),
                                     fallback=False, losses=None)
    if development_method is not None:
        chosen['development_selected'] = dict(chosen[development_method])
    return chosen


def stratified_bootstrap(values, sources, seed=20261108, repetitions=5000):
    values = np.asarray(values, float)
    sources = np.asarray(sources)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError('Bootstrap requires finite, nonempty observations')
    rng = np.random.default_rng(seed)
    total = np.zeros(repetitions)
    for source in sorted(set(sources)):
        group = values[sources == source]
        total += rng.choice(group, (repetitions, len(group)), replace=True).sum(1)
    boot = total / len(values)
    low, high = np.percentile(boot, [2.5, 97.5])
    return dict(mean=float(values.mean()), ci_low=float(low), ci_high=float(high), n=len(values))


def summarize_cases(rows, split):
    rows = [row for row in rows if row['split'] == split]
    sources = [row['source'] for row in rows]
    names = list(rows[0]['selected']) + ['uniform_random', 'uniform_valid_surface']
    summary = {}
    for method in names:
        values, fixed_gains, gains, valid_gains, recalls, visible, oracle_regret = [], [], [], [], [], [], []
        fallbacks = 0
        for row in rows:
            metrics = row['metrics']
            errors = np.array([m['hidden_core_mae'] for m in metrics])
            valid_indices = [i for i, value in enumerate(row['selected']['surface_full']['losses']) if value is not None] or [0]
            valid_uniform = float(errors[valid_indices].mean())
            if method in ('uniform_random', 'uniform_valid_surface'):
                indices = list(range(len(metrics))) if method == 'uniform_random' else valid_indices
                error = float(errors[indices].mean())
                recall = float(np.mean([metrics[i]['hidden_mask_recall'] for i in indices]))
                visible_error = float(np.mean([metrics[i]['visible_mae'] for i in indices]))
            else:
                selection = row['selected'][method]
                picked = metrics[selection['index']]
                error, recall, visible_error = picked['hidden_core_mae'], picked['hidden_mask_recall'], picked['visible_mae']
                fallbacks += selection['fallback']
            values.append(error)
            gains.append(float(errors.mean() - error))
            valid_gains.append(valid_uniform - error)
            fixed_gains.append(float(errors[0] - error))
            recalls.append(recall)
            visible.append(visible_error)
            oracle_regret.append(error - float(errors.min()))
        summary[method] = dict(hidden_core_mae=stratified_bootstrap(values, sources),
            gain_vs_uniform=stratified_bootstrap(gains, sources),
            gain_vs_valid_uniform=stratified_bootstrap(valid_gains, sources),
            gain_vs_fixed=stratified_bootstrap(fixed_gains, sources),
            hidden_mask_recall=float(np.mean(recalls)), visible_mae=float(np.mean(visible)),
            oracle_regret=float(np.mean(oracle_regret)), fallbacks=int(fallbacks),
            wins=int(sum(g > 1e-10 for g in gains)), ties=int(sum(abs(g) <= 1e-10 for g in gains)),
            losses=int(sum(g < -1e-10 for g in gains)),
            by_source={source: dict(n=sum(s == source for s in sources),
                hidden_core_mae=float(np.mean([v for v, s in zip(values, sources) if s == source])),
                gain_vs_uniform=float(np.mean([v for v, s in zip(gains, sources) if s == source])))
                for source in sorted(set(sources))})
    return summary
