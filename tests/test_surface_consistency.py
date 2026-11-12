"""Behavioral safeguards for observable-only candidate selection."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.surface import score_surface, select_candidate, SurfaceConfig, METHODS
from scripts.prepare_surface_experiment import make_case
from scripts.run_surface_candidates import SeededPipeline
from amodal_benchmark.surface_evaluation import normalization, stratified_bootstrap, summarize_cases, evaluate_pool
from amodal_benchmark.dataset import sha256


class SurfaceTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((64, 64, 3), (170, 50, 40), np.uint8)
        self.visible = np.zeros((64, 64), bool)
        self.visible[8:56, 8:32] = True
        self.occlusion = np.zeros((64, 64), bool)
        self.occlusion[:, 32:] = True
        self.predicted = np.zeros((64, 64), bool)
        self.predicted[8:56, 8:56] = True

    def score(self, candidate, observed=None, mask=None):
        return score_surface(self.rgb if observed is None else observed, self.visible,
            self.occlusion, candidate, self.predicted if mask is None else mask)

    def test_matching_solid_beats_wrong_color(self):
        wrong = self.rgb.copy()
        wrong[:, 32:] = (30, 60, 190)
        good, bad = self.score(self.rgb), self.score(wrong)
        self.assertTrue(good['valid'])
        self.assertLess(good['boundary'], bad['boundary'])
        self.assertLess(good['color'], bad['color'])
        self.assertEqual(select_candidate([bad, good], METHODS['surface_full'])['index'], 1)

    def test_repainting_visible_candidate_cannot_change_surface_terms(self):
        changed = self.rgb.copy()
        changed[self.visible] = 255
        first, second = self.score(self.rgb), self.score(changed)
        for term in ('boundary', 'color', 'texture', 'local_boundary', 'local_texture'):
            self.assertEqual(first[term], second[term], term)
        self.assertGreater(second['visible_mae'], first['visible_mae'])

    def test_unobserved_rgb_never_affects_selection(self):
        changed = self.rgb.copy()
        rng = np.random.default_rng(8)
        changed[~self.visible] = rng.integers(0, 256, (int((~self.visible).sum()), 3), dtype=np.uint8)
        first, second = self.score(self.rgb), self.score(self.rgb, observed=changed)
        for term in ('boundary', 'color', 'texture', 'local_boundary', 'local_texture'):
            self.assertAlmostEqual(first[term], second[term], places=6, msg=term)

    def test_empty_and_shrunken_predictions_are_invalid(self):
        score = self.score(self.rgb, mask=np.zeros_like(self.predicted))
        self.assertFalse(score['valid'])
        self.assertIn('insufficient_generated_support', score['invalid_reasons'])
        self.assertIn('lost_visible_target', score['invalid_reasons'])

    def test_disconnected_hidden_region_is_not_a_perfect_seam(self):
        predicted = self.visible.copy()
        predicted[8:56, 48:56] = True
        score = self.score(self.rgb, mask=predicted)
        self.assertFalse(score['valid'])
        self.assertIn('missing_boundary_support', score['invalid_reasons'])

    def test_invalid_pool_falls_back_explicitly(self):
        selected = select_candidate([{'valid': False}, {'valid': False}], {'boundary': 1})
        self.assertTrue(selected['fallback'])
        self.assertEqual(selected['index'], 0)
        self.assertEqual(selected['losses'], [None, None])

    def test_missing_term_does_not_invalidate_unrelated_ablation(self):
        score = {'valid': True, 'boundary': .2, 'texture': None}
        self.assertFalse(select_candidate([score], {'boundary': 1})['fallback'])
        self.assertTrue(select_candidate([score], {'texture': 1})['fallback'])

    def test_ties_are_stable(self):
        score = {'valid': True, 'boundary': .2}
        self.assertEqual(select_candidate([score, score], {'boundary': 1})['index'], 0)

    def test_global_statistics_cannot_resolve_stripe_phase(self):
        from scripts.surface_counterexamples import build_fixtures
        full, visible, occlusion, fixtures = build_fixtures()
        _, truth, candidates, _ = fixtures[1]
        observed = truth.copy()
        observed[occlusion] = 127
        scores = [score_surface(observed, visible, occlusion, image, full) for image in candidates]
        for term in ('boundary', 'color', 'texture'):
            self.assertEqual(scores[0][term], scores[1][term], term)
        self.assertEqual(select_candidate(scores, METHODS['surface_full'])['index'], 0)
        self.assertGreater(scores[0]['local_boundary'], scores[1]['local_boundary'])
        self.assertEqual(select_candidate(scores, METHODS['local_surface'])['index'], 1)

    def test_rejects_empty_pool_bad_scale_and_ambiguous_rgb(self):
        with self.assertRaises(ValueError):
            select_candidate([], {'boundary': 1})
        with self.assertRaises(ValueError):
            select_candidate([{'valid': True, 'boundary': 1}], {'boundary': 1}, {'boundary': 0})
        with self.assertRaises(ValueError):
            self.score(self.rgb.astype(float) / 255)

    def test_preparation_removes_hidden_rgb_after_resize(self):
        mask = np.zeros((100, 140), bool)
        mask[20:80, 25:115] = True
        image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (100, 140, 3), dtype=np.uint8))
        case = make_case(image, mask, direction=0, fraction=.35, resolution=128)
        self.assertTrue(np.all(case['observed'][case['occlusion']] == 127))
        self.assertTrue(np.array_equal(case['observed'][case['visible']], case['truth'][case['visible']]))
        self.assertFalse((case['visible'] & case['hidden']).any())
        self.assertTrue(np.array_equal(case['visible'] | case['hidden'], case['full']))

    def test_explicit_generator_reaches_pipeline(self):
        calls = []
        class Generator:
            def __init__(self, device): self.device = device
            def manual_seed(self, seed): self.seed = seed; return self
        with patch.dict(sys.modules, {'torch': types.SimpleNamespace(Generator=Generator)}):
            wrapper = SeededPipeline(lambda **kw: calls.append(kw), 22, 50)
            wrapper(prompt='striped shirt')
        self.assertEqual(wrapper.calls, 1)
        self.assertEqual(calls[0]['generator'].seed, 22)
        self.assertEqual(calls[0]['generator'].device, 'cuda')
        self.assertEqual(calls[0]['num_inference_steps'], 50)

    def test_normalization_is_based_on_valid_positive_development_terms(self):
        pools = [[{'surface': {'valid': True, 'boundary': 0.}},
                  {'surface': {'valid': True, 'boundary': .2}},
                  {'surface': {'valid': True, 'boundary': .6}},
                  {'surface': {'valid': False, 'boundary': 100.}}]]
        scales = normalization(pools)
        self.assertAlmostEqual(scales['boundary'], .4)
        self.assertEqual(scales['color'], 1.)

    def test_paired_gain_and_oracle_decomposition(self):
        metrics = [dict(hidden_core_mae=e, hidden_mask_recall=.8, visible_mae=.02) for e in (.4, .2, .3, .1)]
        row = dict(split='test', source='VG', metrics=metrics, selected={
            'surface_full': {'index': 1, 'fallback': False, 'losses': [4., 2., 3., 1.]},
            'fixed_seed_11': {'index': 0, 'fallback': False},
            'oracle_hidden_mae': {'index': 3, 'fallback': False}})
        summary = summarize_cases([row], 'test')
        surface = summary['surface_full']
        self.assertAlmostEqual(summary['uniform_random']['hidden_core_mae']['mean'], .25)
        self.assertAlmostEqual(surface['gain_vs_uniform']['mean'], .05)
        self.assertAlmostEqual(surface['gain_vs_fixed']['mean'], .2)
        self.assertAlmostEqual(surface['oracle_regret'], .1)
        self.assertAlmostEqual(summary['oracle_hidden_mae']['gain_vs_uniform']['mean'] - surface['oracle_regret'],
                               surface['gain_vs_uniform']['mean'])

    def test_bootstrap_is_reproducible_and_preserves_source_counts(self):
        a = stratified_bootstrap([1, 1, 5, 5], ['A', 'A', 'B', 'B'])
        b = stratified_bootstrap([1, 1, 5, 5], ['A', 'A', 'B', 'B'])
        self.assertEqual(a, b)
        self.assertEqual(a['ci_low'], 3.)
        self.assertEqual(a['ci_high'], 3.)

    def test_guarded_control_separates_filtering_from_ranking_gain(self):
        metrics = [dict(hidden_core_mae=e, hidden_mask_recall=.8, visible_mae=.02) for e in (.9, .1, .2, .3)]
        row = dict(split='test', source='VG', metrics=metrics, selected={
            'surface_full': {'index': 1, 'fallback': False, 'losses': [None, .1, .2, None]}})
        summary = summarize_cases([row], 'test')
        self.assertAlmostEqual(summary['uniform_random']['hidden_core_mae']['mean'], .375)
        self.assertAlmostEqual(summary['uniform_valid_surface']['hidden_core_mae']['mean'], .15)
        self.assertAlmostEqual(summary['surface_full']['gain_vs_uniform']['mean'], .275)
        self.assertAlmostEqual(summary['surface_full']['gain_vs_valid_uniform']['mean'], .05)

    def test_prediction_mask_cannot_hide_rgb_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / 'evaluation_only/case'
            inputs = root / 'inputs/case'
            folder = root / 'run/candidates/case'
            for path in (evaluation, inputs, folder): path.mkdir(parents=True)
            hidden = self.predicted & self.occlusion
            truth = self.rgb.copy()
            candidate = truth.copy()
            candidate[hidden] = 0
            Image.fromarray(truth).save(evaluation / 'truth.png')
            for name in ('hidden', 'hidden_core'):
                Image.fromarray(hidden.astype(np.uint8) * 255).save(evaluation / (name + '.png'))
            Image.fromarray(self.visible.astype(np.uint8) * 255).save(inputs / 'visible.png')
            Image.fromarray(candidate).save(folder / 'seed-11.png')
            Image.fromarray(np.zeros_like(hidden, np.uint8)).save(folder / 'seed-11-mask.png')
            case = dict(case_id='case', evaluation_hashes={p.name: sha256(p) for p in evaluation.iterdir()})
            records = [dict(seed=11, artifacts={p.name: sha256(p) for p in folder.iterdir()})]
            metrics = evaluate_pool(root, root / 'run', case, records)[0]
            self.assertGreater(metrics['hidden_core_mae'], .3)
            self.assertEqual(metrics['hidden_mask_recall'], 0.)
            self.assertFalse(metrics['generation_missing'])


if __name__ == '__main__':
    unittest.main()
