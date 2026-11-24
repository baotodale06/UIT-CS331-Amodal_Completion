"""CPU checks of v2 provenance, original-observation geometry, and PMC hooks."""
import copy
from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.protocol import ensure_run_config
from amodal_benchmark import shape_v2_runtime as runtime
from evaluate_open_world import parse_args, validate_shape_v2_scoring, worker_command


def receipt_fixture(folder):
    checkpoint = folder / "best.pt"
    checkpoint.write_bytes(b"synthetic checkpoint; never a trained model")
    config = json.loads((runtime.ROOT / runtime.CONFIG).read_bytes())
    epochs = config["training"]["epochs"]
    receipt = dict(schema=2, architecture="resnet34_unet_5channel", platform="Google Colab",
        complete=True, holdout_used=False, epochs_completed=epochs, seed=config["seed"],
        selection=config["training"]["selection"], best_sha256=sha256(checkpoint),
        contract=dict(config_sha256=sha256(runtime.ROOT / runtime.CONFIG),
            data_sha256=runtime.TRAINING_DATA_SHA256,
            sources={name: sha256(runtime.ROOT / name) for name in runtime.TRAINING_SOURCES}),
        history=[dict(epoch=epoch, validation=dict(hidden_iou=.5, full_iou=.7)) for epoch in range(1, epochs+1)],
        runs=[dict(expected_gpu="H100", gpu="NVIDIA H100", cuda_fft_round_trip=True)])
    write_json(folder / "training.json", receipt)
    return checkpoint, receipt


class ShapeV2RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.base = ["--dataset-root", "data", "--output-dir", "out", "--repo-dir", "repo",
                     "--pipeline-profile", "released-sam"]

    def test_explicit_v2_cli_forwards_workers_and_rejects_combined_flags(self):
        args = parse_args(self.base + ["--shape-v2-checkpoint", "best.pt", "--workers", "4"])
        command = worker_command(args)
        self.assertEqual(command[command.index("--shape-v2-checkpoint")+1], str(Path("best.pt").resolve()))
        self.assertNotIn("--shape-checkpoint", command)
        for extra in (["--shape-checkpoint", "old.pt"], ["--surface-freeze", "freeze.json"],
                      ["--surface-dreamsim-cache", "cache"], ["--surface-development-review"],
                      ["--reuse-results-from", "old"], ["--pipeline-profile", "benchmark"]):
            with self.subTest(extra=extra), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(self.base + ["--shape-v2-checkpoint", "best.pt", *extra])
        # Legacy combined remains available only through its historical opt-in.
        legacy = parse_args(self.base + ["--shape-checkpoint", "old.pt", "--surface-freeze", "freeze.json",
                                        "--surface-dreamsim-cache", "cache"])
        self.assertIsNone(legacy.shape_v2_checkpoint)

    def test_contract_rejects_changed_checkpoint_data_sources_or_unfinished_training(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            checkpoint, receipt = receipt_fixture(folder)
            contract = runtime.shape_v2_contract(checkpoint)
            self.assertEqual(contract["selected_epoch"], 1)
            self.assertIsNone(contract["surface"])
            self.assertFalse(contract["rgb_correction"])
            variants = [dict(complete=False), dict(platform="Slurm"), dict(holdout_used=True),
                        dict(best_sha256="0"*64), dict(epochs_completed=59), dict(history=[]), dict(runs=[]),
                        dict(runs=[dict(expected_gpu="H100", gpu="NVIDIA A100", cuda_fft_round_trip=True)])]
            for change in variants:
                with self.subTest(change=change):
                    write_json(folder/"training.json", {**receipt, **change})
                    with self.assertRaises(ValueError):
                        runtime.training_receipt(checkpoint)
            for key in ("data_sha256", "config_sha256", "sources"):
                changed = copy.deepcopy(receipt)
                changed["contract"][key] = "changed"
                write_json(folder/"training.json", changed)
                with self.assertRaisesRegex(ValueError, "data, config, or implementation differs"):
                    runtime.training_receipt(checkpoint)

    def test_slurm_receipt_requires_separate_contract_and_all_gpu_evidence(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            checkpoint, receipt = receipt_fixture(folder)
            config = json.loads((runtime.ROOT / runtime.SLURM_CONFIG).read_bytes())
            receipt.update(platform="Slurm", preflight=False)
            receipt["contract"].update(config_sha256=sha256(runtime.ROOT / runtime.SLURM_CONFIG),
                sources={name: sha256(runtime.ROOT / name) for name in runtime.SLURM_TRAINING_SOURCES})
            devices = [dict(rank=rank, gpu="Quadro RTX 6000", cuda_fft_round_trip=True) for rank in range(4)]
            receipt["runs"] = [dict(job_id="synthetic-test", world_size=4, expected_gpu="Quadro RTX 6000",
                gpu="Quadro RTX 6000", cuda_fft_round_trip=True, devices=devices)]
            write_json(folder/"training.json", receipt)
            self.assertEqual(runtime.training_receipt(checkpoint)["platform"], "Slurm")
            for change in (dict(preflight=True), dict(complete=False),
                           dict(runs=[{**receipt["runs"][0], "devices":devices[:3]}])):
                write_json(folder/"training.json", {**receipt, **change})
                with self.assertRaises(ValueError):
                    runtime.training_receipt(checkpoint)

    def test_changed_contract_cannot_resume_and_score_only_needs_no_model(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            checkpoint, receipt = receipt_fixture(folder)
            policy = runtime.shape_v2_contract(checkpoint)
            saved = dict(shape_policy=policy)
            ensure_run_config(folder/"run_config.json", saved)
            ensure_run_config(folder/"run_config.json", saved)
            with self.assertRaisesRegex(ValueError, "different run configuration"):
                ensure_run_config(folder/"run_config.json", dict(shape_policy={**policy, "checkpoint_sha256":"changed"}))
            args = parse_args(self.base + ["--score-only"])
            with patch.object(runtime, "load_prior", side_effect=AssertionError("Must not load a model")):
                validate_shape_v2_scoring(args, saved)
                args.shape_v2_checkpoint = checkpoint
                validate_shape_v2_scoring(args, saved)
                with self.assertRaisesRegex(ValueError, "differs from the saved"):
                    validate_shape_v2_scoring(args, {})
                args.shape_v2_checkpoint = None
                args.shape_checkpoint = Path("old.pt")
                with self.assertRaisesRegex(ValueError, "legacy or Surface"):
                    validate_shape_v2_scoring(args, saved)

    def test_payload_must_match_validation_selected_epoch_before_model_loading(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU torch is optional in the core environment")
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            checkpoint, receipt = receipt_fixture(folder)
            state = dict(weight=torch.ones(1))
            saved = dict(model=state, contract=receipt["contract"], epoch=1, validation=receipt["history"][0]["validation"])
            def save_payload(value):
                torch.save(value, checkpoint)
                receipt["best_sha256"] = sha256(checkpoint)
                write_json(folder/"training.json", receipt)
            fake_model = SimpleNamespace()
            fake_model.load_state_dict = lambda value, strict: self.assertTrue(torch.equal(value["weight"], state["weight"]))
            fake_model.cuda = lambda: fake_model
            fake_model.eval = lambda: fake_model
            with patch.object(torch.cuda, "is_available", return_value=True), \
                    patch.object(runtime, "make_network", return_value=fake_model) as factory:
                save_payload({**saved, "epoch": 2})
                with self.assertRaisesRegex(ValueError, "payload differs"):
                    runtime.load_prior(checkpoint)
                factory.assert_not_called()
                save_payload(saved)
                self.assertIs(runtime.load_prior(checkpoint), fake_model)
                factory.assert_called_once_with(pretrained=False)

    def test_original_crop_matches_validation_and_neutralizes_hidden_rgb(self):
        visible = np.zeros((20, 24), bool)
        visible[:6, :4] = True
        rgb = np.full((20, 24, 3), 55, np.uint8)
        changed = rgb.copy()
        changed[~visible] = 255
        inputs = []
        def predict(model, observed, support):
            inputs.append((observed.copy(), support.copy()))
            return np.full(support.shape, .25, np.float32)
        with patch.object(runtime, "predict_probability", side_effect=predict):
            first, box, _, _ = runtime.original_prior(None, rgb, visible)
            second, _, _, _ = runtime.original_prior(None, changed, visible)
        self.assertEqual(box, runtime.visible_box(visible, 2.0))
        self.assertLess(box[0], 0)
        np.testing.assert_array_equal(inputs[0][0], inputs[1][0])
        self.assertTrue((inputs[0][0][~inputs[0][1]] == 127).all())
        np.testing.assert_array_equal(first, second)
        self.assertTrue((first[visible] == 1).all())
        self.assertTrue((first[12:] == 0).all())
        prior, support = runtime.project_prior(first, visible, (0, 16, 0, 20), (31, 27))
        self.assertEqual(prior.shape, (27, 31))
        self.assertTrue((prior[support] == 1).all())
        with self.assertRaises(ValueError):
            runtime.project_prior(first, visible, (-1, 16, 0, 20), (31, 27))

    def test_empty_candidate_exclusion_stable_ties_fallback_and_frame_checks(self):
        self.assertEqual(runtime.choose([None, .5, .5]), (1, False))
        self.assertEqual(runtime.choose([None]*8), (0, True))
        with self.assertRaises(ValueError):
            runtime.choose([float("nan")])
        visible = np.zeros((8, 8), bool)
        visible[:4] = True
        selected = runtime.match_candidate(np.stack([~visible, visible]), visible)
        np.testing.assert_array_equal(selected, visible)
        self.assertFalse(runtime.match_candidate(None, visible).any())
        self.assertFalse(runtime.match_candidate([~visible], visible).any())
        with self.assertRaises(ValueError):
            runtime.match_candidate([np.ones((7, 8), bool)], visible)

    def test_progressive_hooks_anchor_original_inputs_preserve_rgb_and_replay_decisions(self):
        with TemporaryDirectory() as temporary:
            seeds, anchors, outputs = [], [], []
            class Generator:
                def __init__(self, device):
                    self.device = device
                def manual_seed(self, seed):
                    self.seed = seed
                    return self
            torch = SimpleNamespace(initial_seed=lambda: 123, Generator=Generator,
                                    inference_mode=lambda: (lambda function: function))
            upstream = SimpleNamespace(run_sam=None, transform_image=lambda image: image)
            def crop(inputs):
                return [value[2:18, 2:22].copy() for value in inputs], 2, 18, 2, 22
            def pipeline(**kwargs):
                seeds.append(kwargs["generator"].seed)
                return SimpleNamespace(images=[Image.new("RGB", (24, 24), (64+(len(seeds)-1)%8, 80, 90))])
            def mixed(query_obj, sd_inpaint_model, sd_occ_mask, sd_prompt):
                return sd_inpaint_model(prompt=sd_prompt).images[0]
            def iteration(query_obj):
                images, *_ = upstream.crop([query_obj.img_canvas, query_obj.query_mask_canvas])
                result = upstream.run_mixed_context_diffusion(query_obj, pipeline, ~images[1], "chair")
                outputs.append(np.array(result))
                query_obj.img_canvas[:] = 255
                query_obj.query_mask_canvas[:] = True
                query_obj.iter_id += 1
                return query_obj
            def segment(*args):
                self.assertEqual(args[-1], ["chair"])
                image = args[3]
                # Second pass has no eligible masks and must fall back to slot 0.
                masks = np.ones((1, image.height, image.width), bool) if len(seeds) <= 8 else None
                return np.array(image), masks, ["chair"], [1.]
            def predict(model, observed, support):
                anchors.append((observed.copy(), support.copy()))
                return np.ones(support.shape, np.float32)
            original_check = lambda image: image
            upstream.crop, upstream.run_iteration, upstream.segment = crop, iteration, segment
            upstream.run_mixed_context_diffusion, upstream.check_occlusion = mixed, original_check
            visible = np.zeros((20, 24), bool)
            visible[5:15, 4:10] = True
            rgb = np.full((20, 24, 3), 100, np.uint8)
            query = SimpleNamespace(iter_id=0, output_img_dir=temporary, init_img_canvas=rgb,
                init_query_mask_canvas=visible, img_canvas=rgb.copy(), query_mask_canvas=visible.copy(),
                benchmark_diagnostics={})
            with patch.dict(sys.modules, {"torch": torch}), patch.object(runtime, "load_prior", return_value=object()), \
                    patch.object(runtime, "predict_probability", side_effect=predict):
                runtime.install_shape_v2(upstream, [None], "sam", "synthetic.pt")
                upstream.segment = lambda *args: self.fail("Candidate used expanded scene detector")
                upstream.run_iteration(query)
                upstream.run_iteration(query)
                self.assertIs(upstream.check_occlusion, original_check)
                with self.assertRaises(ValueError):
                    runtime.install_shape_v2(upstream, [None], "sam", "synthetic.pt")
                with self.assertRaisesRegex(ValueError, "no matching progressive crop"):
                    upstream.run_mixed_context_diffusion(query, pipeline, visible, "chair")
            self.assertEqual(seeds, [runtime.progressive_seed(123, iteration, slot)
                                    for iteration in (0, 1) for slot in runtime.SEED_SLOTS])
            np.testing.assert_array_equal(anchors[0][0], anchors[1][0])
            np.testing.assert_array_equal(anchors[0][1], anchors[1][1])
            self.assertEqual(len(query.benchmark_diagnostics["shape_v2_iterations"]), 2)
            for number in (0, 1):
                folder = Path(temporary)/"shape-v2"/f"iteration-{number:02d}"
                decision = json.loads((folder/"decision.json").read_bytes())
                self.assertEqual(decision["fallback"], bool(number))
                self.assertEqual(decision["index"], 0)
                self.assertFalse(decision["inference_truth_access"])
                self.assertEqual((folder/"selected.png").read_bytes(), (folder/"candidate-0.png").read_bytes())
                for name, digest in decision["artifacts"].items():
                    self.assertEqual(sha256(folder/name), digest)
                prior = np.load(folder/"prior.npy", allow_pickle=False)
                masks = [np.array(Image.open(folder/f"mask-{i}.png")) > 0 for i in range(8)]
                self.assertEqual(runtime.score_masks(prior, masks), decision["scores"])
                np.testing.assert_array_equal(outputs[number], np.array(Image.open(folder/"selected.png")))

    def test_real_evaluator_workers_resume_and_scoring_preserve_v2_provenance(self):
        from test_parallel_evaluation import ParallelEvaluationTests
        from amodal_benchmark.verify import verify_run
        harness = ParallelEvaluationTests()
        harness.setUp()
        self.addCleanup(harness.doCleanups)
        checkpoint, receipt = receipt_fixture(harness.root)
        # Replace only installation of a real CUDA model; coordinator, contracts,
        # workers, scoring, output reuse and provenance are the real code paths.
        shim = harness.root / "runtime/sitecustomize.py"
        with shim.open("a") as stream:
            stream.write('''
from amodal_benchmark import shape_v2_runtime
def shape_v2_install(upstream, models, sam_path, checkpoint):
    path = Path(os.environ["PDMC_TEST_TRACE"]) / f"v2-{os.getpid()}.json"
    path.write_text(json.dumps(dict(checkpoint=str(checkpoint), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))))
shape_v2_runtime.install_shape_v2 = shape_v2_install
if "--worker-index" in sys.argv and os.environ.get("PDMC_TEST_V2_DRIFT"):
    shape_v2_runtime.SEED_SLOTS = (1, 2)
''')
        output = harness.root / "v2-output"
        flags = ["--pipeline-profile", "released-sam", "--shape-v2-checkpoint", str(checkpoint)]
        first = harness.evaluate(output, 2, limit=2, extra=flags, inference_only=False)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        saved_config = (output/"run_config.json").read_bytes()
        kept = {row["id"]: (output/"targets"/row["id"]/"result.json").read_bytes() for row in harness.rows[:2]}
        resumed = harness.evaluate(output, 2, extra=flags, inference_only=False)
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        self.assertEqual((output/"run_config.json").read_bytes(), saved_config)
        for identity, data in kept.items():
            self.assertEqual((output/"targets"/identity/"result.json").read_bytes(), data)
        installed = [json.loads(p.read_bytes()) for p in harness.trace.glob("v2-*.json")]
        self.assertEqual({row["gpu"] for row in installed}, {"GPU-A", "GPU-B"})
        self.assertTrue(all(row["checkpoint"] == str(checkpoint) for row in installed))
        self.assertEqual(verify_run(output, harness.dataset, require_provenance=True)["selected_targets"], 6)
        sources = list(output.glob("provenance/*/source"))
        self.assertTrue(sources)
        for source in sources:
            self.assertTrue((source/"scripts/train_shape_v2.py").is_file())
            self.assertTrue((source/runtime.CONFIG).is_file())
            self.assertTrue((source/"scripts/train_shape_v2_slurm.py").is_file())
            self.assertTrue((source/runtime.SLURM_CONFIG).is_file())
        mismatch = harness.evaluate(harness.root/"drift", 2, extra=flags,
                                    env={**harness.env, "PDMC_TEST_V2_DRIFT":"1"})
        self.assertNotEqual(mismatch.returncode, 0)
        logs = list((harness.root/"drift").glob("workers/*/worker-*.log"))
        self.assertTrue(any("differ from the coordinator" in p.read_text() for p in logs))
        checkpoint.unlink()
        (harness.root/"training.json").unlink()
        scored = harness.evaluate(output, 1, extra=["--pipeline-profile", "released-sam", "--score-only"],
                                  inference_only=False)
        self.assertEqual(scored.returncode, 0, scored.stdout + scored.stderr)
        self.assertEqual((output/"run_config.json").read_bytes(), saved_config)
        self.assertEqual(verify_run(output, harness.dataset, require_provenance=True)["selected_targets"], 6)


if __name__ == "__main__":
    unittest.main()
