"""Reproduce the 24-GiB-per-card LISA OOM without allocating model tensors."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GIB = 1024 ** 3
FIXED = ["model.embed_tokens", "model.norm", "model.vision_tower", "model.mm_projector",
         "model.visual_model", "model.text_hidden_fcs", "lm_head"]


class LisaPlacementTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("amodal_benchmark.placement"),
                             "The runner still places the entire LISA model on GPU 0")
        from amodal_benchmark.placement import plan_lisa_devices
        self.plan = plan_lisa_devices

    def test_eight_24g_cards_share_decoder_layers_and_keep_segmentation_on_zero(self):
        layer_bytes = [640 * 1024 ** 2] * 40
        budgets = {0: 15 * GIB, **{i: 19 * GIB for i in range(1, 8)}}
        mapping, loads = self.plan(layer_bytes, 4 * GIB, FIXED, budgets)
        self.assertEqual(set(mapping.values()), set(range(8)))
        for name in FIXED:
            self.assertEqual(mapping[name], 0)
        assignment = [mapping[f"model.layers.{i}"] for i in range(40)]
        self.assertEqual(assignment, sorted(assignment))  # contiguous blocks minimize transfers
        for gpu, used in loads.items():
            self.assertLessEqual(used, budgets[gpu])
        self.assertLess(loads[0], 15 * GIB)

    def test_two_24g_cards_fit_unquantized_weights_without_full_copy_on_zero(self):
        mapping, loads = self.plan([640 * 1024 ** 2] * 40, 4 * GIB, FIXED,
                                  {0: 15 * GIB, 1: 19 * GIB})
        # Hand-derived optimum: 4 + 17*0.625 = 14.625 GiB, remaining 23*0.625 = 14.375.
        self.assertEqual(sum(mapping[f"model.layers.{i}"] == 0 for i in range(40)), 17)
        self.assertEqual(loads, {0: int(14.625 * GIB), 1: int(14.375 * GIB)})

    def test_accelerate_map_has_no_parent_that_would_temporarily_copy_all_weights(self):
        mapping, _ = self.plan([GIB] * 4, GIB, FIXED, {0: 3 * GIB, 1: 4 * GIB})
        self.assertNotIn("", mapping)
        self.assertNotIn("model", mapping)
        self.assertNotIn("model.layers", mapping)

    def test_one_visible_24g_card_fails_before_gpu_transfer_with_actionable_message(self):
        with self.assertRaisesRegex(ValueError, "visible GPU"):
            self.plan([640 * 1024 ** 2] * 40, 4 * GIB, FIXED, {0: 15 * GIB})

    def test_fixed_segmentation_weights_must_fit_the_main_gpu(self):
        with self.assertRaisesRegex(ValueError, "GPU 0"):
            self.plan([GIB] * 4, 5 * GIB, FIXED, {0: 4 * GIB, 1: 20 * GIB})


if __name__ == "__main__":
    unittest.main()
