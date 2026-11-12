"""Model-source selection preserves the upstream pipeline and CLI behavior."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


MAIN = '''
import argparse
import json
import sys
class torch:
    float16 = "float16"
class StableDiffusionInpaintPipeline:
    @classmethod
    def from_pretrained(cls, model, **kwargs):
        return dict(pipeline=cls.__name__, model=model, **kwargs)
class StableDiffusionInpaintMixedContextPipeline(StableDiffusionInpaintPipeline):
    pass
def load_models(mixed):
    if not mixed:
        return StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16)
    return StableDiffusionInpaintMixedContextPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--input_dir", required=True)
    args = parser.parse_args()
    result = load_models(args.mixed)
    result.update(input_dir=args.input_dir, filename=__file__, argv0=sys.argv[0])
    print(json.dumps(result))
'''


@unittest.skipUnless(os.name == "posix", "Bash and executable symlinks require Linux/WSL")
class ModelSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="amodal-model-source-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "checkout with spaces"
        self.repo.mkdir()
        self.main = self.repo / "main.py"
        self.main.write_text(MAIN)
        self.environment = self.root / "environment"
        (self.environment / "bin").mkdir(parents=True)
        (self.environment / "bin/python").symlink_to(sys.executable)
        self.helper = Path(__file__).with_name("test_run_amodal_slurm.sh")

    def launch(self, model, *arguments):
        return subprocess.run(
            ["bash", str(self.helper), "--launch-upstream-child", str(self.repo),
             str(self.environment), model, "--input_dir", "images with spaces", *arguments],
            capture_output=True, text=True,
        )

    def test_both_pipeline_classes_keep_dtype_and_arguments(self):
        for arguments, pipeline in (
            ((), "StableDiffusionInpaintPipeline"),
            (("--mixed",), "StableDiffusionInpaintMixedContextPipeline"),
        ):
            with self.subTest(pipeline=pipeline):
                result = self.launch("sd2-community/stable-diffusion-2-inpainting", *arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads(result.stdout)
                self.assertEqual(data["model"], "sd2-community/stable-diffusion-2-inpainting")
                self.assertEqual(data["pipeline"], pipeline)
                self.assertEqual(data["torch_dtype"], "float16")
                self.assertEqual(data["input_dir"], "images with spaces")
                self.assertEqual(data["filename"], str(self.main))
                self.assertEqual(data["argv0"], str(self.main))
        self.assertEqual(self.main.read_text(), MAIN)

    def test_local_model_path_is_data_not_python_source(self):
        model = str(self.root / "local model 'quoted' directory")
        result = self.launch(model, "--mixed")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["model"], model)

    def test_changed_upstream_is_rejected_before_execution(self):
        self.main.write_text(MAIN.replace("stabilityai/stable-diffusion-2-inpainting", "another/model", 1))
        result = self.launch("selected/model")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Expected two SD2 model-loading calls", result.stderr)

    def test_local_config_preflight_checks_sd2_architecture(self):
        model = self.root / "model"
        configs = {
            "model_index.json": {"unet": ["diffusers", "UNet2DConditionModel"]},
            "unet/config.json": {"in_channels": 9, "out_channels": 4, "cross_attention_dim": 1024},
            "text_encoder/config.json": {"hidden_size": 1024},
            "vae/config.json": {"latent_channels": 4},
        }
        for filename, content in configs.items():
            path = model / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(content))
        stub = self.root / "stub"
        stub.mkdir()
        (stub / "huggingface_hub.py").write_text(
            "def hf_hub_download(*args, **kwargs):\n"
            "    raise AssertionError('Local model must not contact the Hub')\n"
        )
        environment = dict(os.environ, PYTHONPATH=str(stub))
        command = ["bash", str(self.helper), "--sd-config-child", str(self.environment),
                   str(model), str(self.root / "cache")]
        success = subprocess.run(command, env=environment, capture_output=True, text=True)
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertIn("Model configs OK", success.stdout)
        configs["unet/config.json"]["in_channels"] = 4
        (model / "unet/config.json").write_text(json.dumps(configs["unet/config.json"]))
        failure = subprocess.run(command, env=environment, capture_output=True, text=True)
        self.assertNotEqual(failure.returncode, 0)
        self.assertIn("9-channel inpainting UNet", failure.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
