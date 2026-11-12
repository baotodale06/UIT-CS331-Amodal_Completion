#!/usr/bin/env python3
"""Choose a small surface ranker on old images with leave-source-out validation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.surface_features import DEEP_FEATURE_NAMES
from amodal_benchmark.surface_refinement import FEATURE_NAMES, POOL_FEATURE_NAMES, rank_features, predict_rank
from develop_surface_followup import load_development

ENDPOINTS = ('hidden_core_mae', 'object_lpips', 'dreamsim', 'clip_text_cosine')


def fit(rows, names, regularization, weights):
    arrays = [np.stack([rank_features(score, names) for score in row['scores']]) for row in rows]
    combined = np.concatenate(arrays)
    medians = np.nan_to_num(np.nanmedian(combined, axis=0))
    combined = np.where(np.isfinite(combined), combined, medians)
    center, scale = combined.mean(0), np.maximum(combined.std(0), .001)
    normalizers = {key: np.mean([item[key] for row in rows for item in row['evaluation']['refined']])
                   for key in weights}
    x, y = [], []
    for row, features in zip(rows, arrays):
        features = (np.where(np.isfinite(features), features, medians) - center) / scale
        target = np.array([sum(weight * item[key] / normalizers[key]
                               for key, weight in weights.items())
                           for item in row['evaluation']['refined']])
        x.append(features - features.mean(0))
        y.append(target - target.mean())
    x, y = np.concatenate(x), np.concatenate(y)
    coef = np.linalg.solve(x.T @ x + regularization * np.eye(x.shape[1]), x.T @ y)
    return dict(feature_names=list(names), medians=medians.tolist(), center=center.tolist(),
                scale=scale.tolist(), coefficients=coef.tolist(), regularization=regularization,
                target_weights=weights, target_normalizers=normalizers, valid_only=True)


def summarize(rows, predictions, source_counts=None):
    gains, metrics = {}, {}
    sample_weights = np.ones(len(rows), float)
    if source_counts is not None:
        counts = {source: sum(row['source'] == source for row in rows) for source in source_counts}
        if any(number > 0 and counts[source] == 0 for source, number in source_counts.items()):
            raise ValueError('Target source has no development cases')
        sample_weights = np.array([source_counts.get(row['source'], 0) / max(counts.get(row['source'], 0), 1)
                                   for row in rows], float)
        if (sample_weights < 0).any() or not sample_weights.sum():
            raise ValueError('Target source counts must define a nonempty population')
    for metric in (*ENDPOINTS, 'hidden_spatial_lpips'):
        baseline = float(np.average([np.mean([m[metric] for m in row['evaluation']['raw']]) for row in rows],
                                    weights=sample_weights))
        value = float(np.average([row['evaluation']['refined'][predictions[row['case_id']]][metric] for row in rows],
                                 weights=sample_weights))
        gain = 100 * (value - baseline) / baseline * (1 if metric == 'clip_text_cosine' else -1)
        gains[metric] = gain
        metrics[metric] = dict(baseline=baseline, method=value, relative_gain_percent=gain)
    primary = [gains[key] for key in ENDPOINTS]
    # Reward a balanced improvement; one strong metric cannot hide a regression.
    utility = min(primary) + .25 * float(np.mean(primary))
    return dict(utility=utility, metrics=metrics)


def fit_boosted(rows, names, depth, minimum_leaf, weights):
    """Small nonlinear comparison model, exported without a runtime sklearn dependency."""
    from sklearn.ensemble import GradientBoostingRegressor
    model = fit(rows, names, 100., weights)
    x, y = [], []
    for row in rows:
        features = np.stack([rank_features(score, names) for score in row['scores']])
        features = (np.where(np.isfinite(features), features, model['medians']) - model['center']) / model['scale']
        target = np.array([sum(weight * item[key] / model['target_normalizers'][key]
                               for key, weight in weights.items()) for item in row['evaluation']['refined']])
        x.append(features - features.mean(0))
        y.append(target - target.mean())
    estimator = GradientBoostingRegressor(n_estimators=100, learning_rate=.05,
        max_depth=depth, min_samples_leaf=minimum_leaf, random_state=20261110)
    estimator.fit(np.concatenate(x), np.concatenate(y))
    model.update(kind='gradient_boosting', learning_rate=.05,
        initial_prediction=float(estimator.init_.constant_[0, 0]),
        depth=depth, minimum_leaf=minimum_leaf, trees=[])
    for item in estimator.estimators_[:, 0]:
        tree = item.tree_
        model['trees'].append(dict(children_left=tree.children_left.tolist(), children_right=tree.children_right.tolist(),
            feature=tree.feature.tolist(), threshold=tree.threshold.tolist(), value=tree.value[:, 0, 0].tolist()))
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--design-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--deep-features', action='store_true')
    p.add_argument('--boosted', action='store_true', help='Also compare a small pre-test nonlinear ranker grid')
    p.add_argument('--target-source-counts', type=Path,
                   help='Pre-test cohort design JSON with source_case_limits; no test outcomes')
    p.add_argument('--ensemble-design-dir', type=Path, help='Joint eight-pool development correction metrics')
    p.add_argument('--ensemble-variant', help='A measured, predeclared correction configuration')
    a = p.parse_args()
    source_counts = json.loads(a.target_source_counts.read_text())['source_case_limits'] if a.target_source_counts else None
    if bool(a.ensemble_design_dir) != bool(a.ensemble_variant):
        p.error('Joint correction search requires a directory and variant together')
    rows = load_development()
    provenance = {}
    for row in rows:
        cohort = 87 if 'replication' in row['experiment'] else 96
        metric_path = a.design_dir / f'dev-metrics-{cohort}' / (row['case_id'] + '.json')
        decision_path = metric_path.parent / 'decisions' / metric_path.name
        row['evaluation'] = json.loads(metric_path.read_text())
        if a.ensemble_design_dir:
            correction_path = a.ensemble_design_dir / f'ensemble-eight-{cohort}' / metric_path.name
            correction = json.loads(correction_path.read_text())
            if correction['contract'].get('development_only') is not True or not correction['contract'].get('all_candidates'):
                raise ValueError('Joint search requires complete development-only correction pools')
            if (correction['case_id'] != row['case_id'] or
                    correction['contract']['candidate_seeds'] != row['evaluation']['candidate_seeds']):
                raise ValueError('Correction and candidate metric inventories differ')
            row['evaluation']['refined'] = correction['candidate_metrics'][a.ensemble_variant]
            provenance[str(correction_path)] = sha256(correction_path)
        row['scores'] = json.loads(decision_path.read_text())['observable_scores']
        provenance[str(metric_path)] = sha256(metric_path)
        provenance[str(decision_path)] = sha256(decision_path)
        if a.deep_features:
            features_path = a.design_dir / f'dev-features-{cohort}' / metric_path.name
            features = json.loads(features_path.read_text())
            if features['contract']['truth_access'] is not False:
                raise ValueError('Features must use observation only')
            provenance[str(features_path)] = sha256(features_path)
            for score, extra in zip(row['scores'], features['values']):
                score.update(extra)
    feature_sets = {'surface_consensus': FEATURE_NAMES + POOL_FEATURE_NAMES}
    budgets = {len(row['scores']) for row in rows}
    if len(budgets) != 1 or any(len(row['evaluation']['raw']) != len(row['scores']) for row in rows):
        raise ValueError('Development pools must share a complete candidate budget')
    candidate_budget = budgets.pop()
    if a.deep_features:
        feature_sets['surface_deep'] = FEATURE_NAMES + POOL_FEATURE_NAMES + DEEP_FEATURE_NAMES
    candidates = []
    for family, names in feature_sets.items():
        for ridge in (10., 100., 1000.):
            for semantic in (0., .5, 1., 2., 4., 8., 16.):
                for appearance in ('pixel_perceptual', 'all_appearance'):
                    weights = dict(hidden_core_mae=1., hidden_spatial_lpips=1.)
                    if appearance == 'all_appearance':
                        weights.update(object_lpips=1., dreamsim=1.)
                    if semantic:
                        weights['clip_text_cosine'] = -semantic
                    predictions = {}
                    for source in sorted({row['source'] for row in rows}):
                        model = fit([row for row in rows if row['source'] != source], names, ridge, weights)
                        for row in rows:
                            if row['source'] == source:
                                predictions[row['case_id']] = predict_rank(row['scores'], model)[0]
                    result = summarize(rows, predictions, source_counts)
                    candidates.append(dict(family=family, ridge=ridge, weights=weights,
                        predictions=predictions, **result))
        if family == 'surface_deep':
            for ridge in (10.,100.,1000.):
                for top_k in range(2, candidate_budget + 1):
                    for appearance in ('pixel_perceptual','all_appearance'):
                        weights = dict(hidden_core_mae=1.,hidden_spatial_lpips=1.)
                        if appearance == 'all_appearance':
                            weights.update(object_lpips=1.,dreamsim=1.)
                        rule = dict(kind='semantic_within_surface_topk',top_k=top_k)
                        predictions = {}
                        for source in sorted({row['source'] for row in rows}):
                            model = fit([row for row in rows if row['source'] != source],names,ridge,weights)
                            model['selection_rule'] = rule
                            for row in rows:
                                if row['source'] == source:
                                    predictions[row['case_id']] = predict_rank(row['scores'],model)[0]
                        candidates.append(dict(family=family,ridge=ridge,weights=weights,
                            selection_rule=rule,predictions=predictions,**summarize(rows,predictions,source_counts)))
    if a.boosted:
        if not a.deep_features:
            p.error('The boosted comparison requires the same observable deep features')
        names = feature_sets['surface_deep']
        for depth in (2, 3):
            for minimum_leaf in (24, 48):
                for semantic in (0., 2., 4., 8.):
                    weights = dict(hidden_core_mae=1., hidden_spatial_lpips=1., object_lpips=1., dreamsim=1.)
                    if semantic:
                        weights['clip_text_cosine'] = -semantic
                    fitted = {}
                    for source in sorted({row['source'] for row in rows}):
                        fitted[source] = fit_boosted([row for row in rows if row['source'] != source],
                            names, depth, minimum_leaf, weights)
                    for top_k in ((None, 2, 4, 6) if not semantic else (None,)):
                        predictions = {}
                        for row in rows:
                            model = dict(fitted[row['source']])
                            if top_k:
                                model['selection_rule'] = dict(kind='semantic_within_surface_topk', top_k=top_k)
                            predictions[row['case_id']] = predict_rank(row['scores'], model)[0]
                        candidate = dict(family='surface_deep', model_kind='gradient_boosting',
                            depth=depth, minimum_leaf=minimum_leaf, ridge=100., weights=weights,
                            predictions=predictions, **summarize(rows, predictions, source_counts))
                        if top_k:
                            candidate['selection_rule'] = dict(kind='semantic_within_surface_topk', top_k=top_k)
                        candidates.append(candidate)
    for candidate in candidates:
        candidate['all_primary_gains_at_least_three_percent'] = all(
            candidate['metrics'][key]['relative_gain_percent'] >= 3 for key in ENDPOINTS)
    candidates.sort(key=lambda item: (not item['all_primary_gains_at_least_three_percent'], -item['utility']))
    best = candidates[0]
    model = (fit_boosted(rows, feature_sets[best['family']], best['depth'], best['minimum_leaf'], best['weights'])
             if best.get('model_kind') == 'gradient_boosting' else
             fit(rows, feature_sets[best['family']], best['ridge'], best['weights']))
    if best.get('selection_rule'):
        model['selection_rule'] = best['selection_rule']
    write_json(a.output_dir / 'ranker.json', model)
    write_json(a.output_dir / 'model_search.json', dict(
        created_at=datetime.now(timezone.utc).isoformat(), development_images=len(rows),
        candidate_budget=candidate_budget,
        target_source_counts=source_counts,
        ensemble_variant=a.ensemble_variant,
        target_source_counts_sha256=sha256(a.target_source_counts) if a.target_source_counts else None,
        split='leave one source out; all 183 old images are development',
        selection_utility='Prefer candidates with all four primary gains >=3%, then minimum relative gain plus 0.25 mean gain',
        candidates=candidates, input_hashes=provenance,
        source_sha256=sha256(Path(__file__))))
    for candidate in candidates[:8]:
        print(json.dumps({k:v for k,v in candidate.items() if k != 'predictions'}))


if __name__ == '__main__':
    main()
