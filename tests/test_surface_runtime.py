"""Test original-observation anchoring and seed flow through progressive sampling."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from amodal_benchmark.surface_runtime import (install_progressive_surface,
    isolated_dino_imports, progressive_seed, project_original_observation, surface_contract)
from evaluate_open_world import parse_args, worker_command


class SurfaceRuntimeTests(unittest.TestCase):
    def test_dino_imports_restore_instaorder_modules_even_after_error(self):
        original = SimpleNamespace(owner='InstaOrder')
        child = SimpleNamespace(owner='InstaOrder.child')
        dino = SimpleNamespace(owner='DINO')
        with patch.dict(sys.modules, {'utils': original, 'utils.child': child}):
            for fail in (False, True):
                try:
                    with isolated_dino_imports():
                        self.assertNotIn('utils', sys.modules)
                        self.assertNotIn('utils.child', sys.modules)
                        sys.modules['utils'] = dino
                        sys.modules['utils.dino_child'] = dino
                        if fail:
                            raise RuntimeError('failed initialization')
                except RuntimeError:
                    self.assertTrue(fail)
                self.assertIs(sys.modules['utils'], original)
                self.assertIs(sys.modules['utils.child'], child)
                self.assertNotIn('utils.dino_child', sys.modules)

    def test_projection_cannot_observe_unknown_pixels_even_at_resize_boundaries(self):
        y, x = np.indices((30, 40))
        visible = (x > 3) & (x < 23) & (y > 4)
        rgb = np.stack([x + 30, y + 60, x + y + 80], axis=-1).astype(np.uint8)
        poisoned = rgb.copy()
        poisoned[~visible] = 255
        first = project_original_observation(rgb, visible, (59, 47))
        second = project_original_observation(poisoned, visible, (59, 47))
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[0].shape, (47, 59, 3))

    def test_progressive_loop_retains_original_anchor_and_explicit_candidate_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            seeds, anchors, mixed_calls, native_checks = [], [], [], []
            class Generator:
                def __init__(self, device):
                    self.device = device
                def manual_seed(self, seed):
                    self.seed = seed
                    return self
            torch = SimpleNamespace(initial_seed=lambda: 123, Generator=Generator,
                                    inference_mode=nullcontext)
            upstream = SimpleNamespace(run_sam=None, transform_image=lambda image: image)
            def crop(inputs, *args):
                return [value[4:36, 8:40].copy() for value in inputs], 4, 36, 8, 40
            def pipeline(**kwargs):
                seeds.append(kwargs['generator'].seed)
                return SimpleNamespace(images=[Image.new('RGB', (24, 24), (90, 80, 70))])
            def mixed(query_obj, sd_inpaint_model, lama_model, sd_img, sd_modal_mask,
                      sd_occ_mask, sd_prompt, clean_bkgd_img, mc_timestep=35, sd_target_size=512):
                mixed_calls.append(query_obj.iter_id)
                return sd_inpaint_model(prompt=sd_prompt).images[0]
            def iteration(query_obj):
                if query_obj.iter_id == 0:
                    query_obj.init_img_canvas = np.full((40, 48, 3), 100, np.uint8)
                    query_obj.init_query_mask_canvas = np.zeros((40, 48), bool)
                    query_obj.init_query_mask_canvas[:, :24] = True
                    query_obj.img_canvas = query_obj.init_img_canvas.copy()
                    query_obj.query_mask_canvas = query_obj.init_query_mask_canvas.copy()
                    query_obj.benchmark_diagnostics = {}
                current = [query_obj.img_canvas, query_obj.query_mask_canvas]
                (rgb, visible), *coords = upstream.crop(current, None, None)
                query_obj.amodal_completion = upstream.run_mixed_context_diffusion(
                    query_obj, pipeline, None, rgb, visible, ~visible, 'chair', rgb).resize((32, 32))
                upstream.check_occlusion(query_obj.amodal_completion)
                query_obj.img_canvas[:] = 240
                query_obj.query_mask_canvas[:] = True
                query_obj.iter_id += 1
                return query_obj
            def segment(*args):
                image = args[3]
                return np.array(image), np.ones((1, image.height, image.width), bool), ['chair'], [1.]
            upstream.crop, upstream.run_iteration = crop, iteration
            upstream.run_mixed_context_diffusion, upstream.segment = mixed, segment
            def occlusion(amodal_completion):
                native_checks.append(np.array(amodal_completion))
            upstream.check_occlusion = occlusion
            def select(observed, visible, occlusion, images, masks, freeze, **kwargs):
                anchors.append((observed.copy(), visible.copy()))
                return images[1], masks[1], dict(index=1, seed=22, fallback=False)
            query = SimpleNamespace(iter_id=0, output_img_dir=temporary)
            with patch.dict(sys.modules, {'torch': torch}), patch(
                    'amodal_benchmark.surface_runtime.apply_frozen_surface', side_effect=select):
                install_progressive_surface(upstream, [None], 'sam', dict(candidate_seeds=[11, 22]))
                # Annotated scene segmentation is installed after the surface
                # bridge. It must never replace the frozen target-only query.
                upstream.segment = lambda *a, **kw: (_ for _ in ()).throw(
                    AssertionError('Candidate incorrectly used expanded scene queries'))
                upstream.run_iteration(query)
                upstream.run_iteration(query)
                with self.assertRaises(ValueError):
                    install_progressive_surface(upstream, [None], 'sam', {})
            self.assertEqual(mixed_calls, [0, 1])
            self.assertEqual(seeds, [progressive_seed(123, iteration, seed)
                                    for iteration in (0, 1) for seed in (11, 22)])
            np.testing.assert_array_equal(anchors[0][0], anchors[1][0])
            np.testing.assert_array_equal(anchors[0][1], anchors[1][1])
            self.assertFalse(anchors[1][1].all())
            for image in native_checks:
                np.testing.assert_array_equal(image[:, :16], np.full((32, 16, 3), 100, np.uint8))
                self.assertTrue((image[:, 20:, 0] == 90).all())
            self.assertEqual(query.iter_id, 2)
            self.assertEqual(len(query.benchmark_diagnostics['surface_iterations']), 2)
            decisions = list(Path(temporary).glob('surface/*/decision.json'))
            self.assertEqual(len(decisions), 2)
            self.assertTrue(all(json.loads(path.read_text())['inference_truth_access'] is False for path in decisions))

    def test_surface_cli_is_opt_in_and_provisional_smoke_is_bounded(self):
        base = ['--dataset-root', 'data', '--output-dir', 'out', '--repo-dir', 'repo',
                '--pipeline-profile', 'released-sam']
        args = parse_args(base)
        self.assertIsNone(args.surface_freeze)
        extended = base + ['--surface-freeze', 'freeze.json', '--surface-dreamsim-cache', 'cache',
                           '--surface-development-review', '--limit', '8']
        command = worker_command(parse_args(extended))
        self.assertIn('--surface-freeze', command)
        self.assertIn('--surface-development-review', command)
        with self.assertRaises(SystemExit):
            parse_args(extended[:-2])
        with self.assertRaises(SystemExit):
            parse_args(extended + ['--reuse-results-from', 'old'])

    def test_provisional_policy_cannot_be_used_for_full_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'freeze.json'
            path.write_text(json.dumps(dict(status='provisional_development_review')))
            with self.assertRaises(ValueError):
                surface_contract(path, Path(temporary))

    def test_changed_implementation_cannot_silently_use_a_frozen_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'freeze.json'
            path.write_text(json.dumps(dict(status='frozen_before_fresh_test',
                source_hashes={'amodal_benchmark/surface.py': '0' * 64})))
            with self.assertRaisesRegex(ValueError, 'implementation differs'):
                surface_contract(path, Path(temporary))


if __name__ == '__main__':
    unittest.main()
