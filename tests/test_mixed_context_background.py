"""Exercise the pinned Mixed Context code with CPU substitutes for model calls."""
import ast
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import warnings

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark import inference


class BackgroundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT / "tests/fixtures/pdmc_main.py"
        raw = source.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == inference.UPSTREAM_MAIN_SHA256
        # Preserve the pinned source's old regex spelling; its unrelated
        # invalid-escape warning appears only on newer Python test hosts.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(raw)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "run_mixed_context_diffusion")
        namespace = {"np": np, "Image": Image,
                     "run_lama_inpainter": lambda image, mask, model: np.zeros_like(image)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        cls.legacy = staticmethod(namespace["run_mixed_context_diffusion"])

    def wrapped(self):
        factory = getattr(inference, "size_mixed_context_background", None)
        self.assertTrue(callable(factory), "Mixed Context background sizing has not been implemented")
        return factory(self.legacy)

    @staticmethod
    def pipeline(**kwargs):
        # Only GPU diffusion is replaced: return its actual synthetic input so
        # the test can inspect the upstream mask/background composition.
        return SimpleNamespace(images=[kwargs["image"]])

    def inputs(self, shape):
        image = np.full((*shape, 3), [10, 20, 30], dtype=np.uint8)
        mask = np.zeros(shape, dtype=np.uint8)
        mask[2:5, 2:5] = 1
        return image, mask, np.zeros(shape, dtype=np.uint8)

    def test_pinned_function_reproduces_logged_1200_pixel_index_error(self):
        image, mask, occlusion = self.inputs((1201, 12))
        background = np.full((1200, 20, 3), 100, dtype=np.uint8)
        with self.assertRaisesRegex(IndexError, "1200"):
            self.legacy(None, self.pipeline, None, image, mask, occlusion, "car", background)

    def test_oversized_height_completes_without_losing_visible_query_pixels(self):
        image, mask, occlusion = self.inputs((1201, 12))
        background = np.full((1200, 20, 3), [100, 110, 120], dtype=np.uint8)
        original = background.copy()
        result = self.wrapped()(None, self.pipeline, None, image, mask, occlusion, "car", background)
        array = np.array(result)
        self.assertEqual(array.shape, image.shape)
        np.testing.assert_array_equal(array[mask == 1], image[mask == 1])
        np.testing.assert_array_equal(array[mask == 0], np.tile([100, 110, 120], ((mask == 0).sum(), 1)))
        np.testing.assert_array_equal(background, original)

    def test_oversized_width_and_keyword_arguments_work(self):
        image, mask, occlusion = self.inputs((12, 33))
        background = np.full((20, 32, 3), 100, dtype=np.uint8)
        result = self.wrapped()(query_obj=None, sd_inpaint_model=self.pipeline, lama_model=None,
                                sd_img=image, sd_modal_mask=mask, sd_occ_mask=occlusion,
                                sd_prompt="car", clean_bkgd_img=background, mc_timestep=20)
        self.assertEqual(result.size, (33, 12))

    def test_supported_crops_keep_the_original_background_and_output(self):
        image, mask, occlusion = self.inputs((12, 16))
        background = np.random.default_rng(0).integers(0, 256, (20, 32, 3), dtype=np.uint8)
        original = self.legacy(None, self.pipeline, None, image, mask, occlusion, "car", background)
        resized = self.wrapped()(None, self.pipeline, None, image, mask, occlusion, "car", background)
        np.testing.assert_array_equal(np.array(original), np.array(resized))

    def test_large_crop_remains_unchanged_when_only_the_query_extends_beyond_backdrop(self):
        image, mask, occlusion = self.inputs((1201, 12))
        mask[1200:, :] = 1  # All out-of-bounds positions belong to the visible object.
        background = np.random.default_rng(1).integers(0, 256, (1200, 20, 3), dtype=np.uint8)
        original = self.legacy(None, self.pipeline, None, image, mask, occlusion, "car", background)
        resized = self.wrapped()(None, self.pipeline, None, image, mask, occlusion, "car", background)
        np.testing.assert_array_equal(np.array(original), np.array(resized))

    def test_unrelated_model_errors_are_not_swallowed(self):
        def broken_pipeline(**kwargs):
            raise IndexError("unrelated diffusion failure")
        image, mask, occlusion = self.inputs((12, 16))
        background = np.full((20, 32, 3), 100, dtype=np.uint8)
        with self.assertRaisesRegex(IndexError, "unrelated diffusion failure"):
            self.wrapped()(None, broken_pipeline, None, image, mask, occlusion, "car", background)


if __name__ == "__main__":
    unittest.main()
