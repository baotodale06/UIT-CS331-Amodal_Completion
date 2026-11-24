"""Native admission failures and actual multiworker evaluator integration."""
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

from amodal_benchmark import shape_calibrated_runtime as runtime
from amodal_benchmark.dataset import sha256, write_json
from evaluate_open_world import parse_args, worker_command, validate_shape_calibrated_scoring


def publish(folder):
    """Rebind synthetic metadata to test policy checks beyond simple checksums."""
    candidate = folder / "candidate"
    policy = json.loads((candidate / "candidate.json").read_bytes())
    policy["files"] = {name: sha256(candidate / name) for name in
        ("best.pt", "training.json", "diagnostic-replay.json", "selected-native.json", "selected-masks.zip")}
    write_json(candidate / "candidate.json", policy)
    for family in ("local", "quadro"):
        contract = runtime.read(folder / (family + "-contract.json"))
        contract["candidate_sha256"] = sha256(candidate / "candidate.json")
        write_json(folder / (family + "-contract.json"), contract)
        result = runtime.read(folder / (family + "-result.json"))
        result["contract_sha256"] = sha256(folder / (family + "-contract.json"))
        write_json(folder / (family + "-result.json"), result)
    write_json(folder / "native-replay.json", dict(verified=True, objects=1264, images=217,
        candidate_sha256=sha256(candidate / "candidate.json"),
        local_result_sha256=sha256(folder / "local-result.json"), quadro_result_sha256=sha256(folder / "quadro-result.json"),
        selected_geometry=policy["native_selected"], replay_source_sha256=sha256(runtime.ROOT / runtime.REPLAY_SOURCE),
        benchmark_success=False))
    write_json(folder / "admission.json", dict(schema=1, method=runtime.METHOD,
        checkpoint_sha256=policy["checkpoint_sha256"], files=runtime.evidence_files(folder),
        sources=runtime.admission_sources(), benchmark_success=False))


def fixture(folder):
    """Test-only synthetic admission; never a trained model or scientific result."""
    candidate = folder / "candidate"
    candidate.mkdir(parents=True)
    for name in ("best.pt", "training.json", "diagnostic-replay.json", "selected-masks.zip"):
        (candidate / name).write_bytes(b"SYNTHETIC TEST ONLY")
    baseline, selected = dict(full_iou=.81, hidden_iou=.37), dict(full_iou=.82, hidden_iou=.39)
    rows = [dict(id=f"test-{i}", image_id=i % 217, prediction_sha256=f"{i:064x}") for i in range(1264)]
    write_json(candidate / "selected-native.json", dict(rows=rows, summary=selected))
    write_json(candidate / "candidate.json", dict(schema=1, method=runtime.METHOD,
        checkpoint_epoch=7, logit_bias=-2., checkpoint_sha256=sha256(candidate / "best.pt"),
        native_baseline=baseline, native_selected=selected, holdout_access=False, benchmark_access=False,
        benchmark_admitted=False, benchmark_success=False,
        arithmetic=dict(cudnn_deterministic=True, cudnn_benchmark=False, cudnn_allow_tf32=False,
                        matmul_allow_tf32=False, autocast=False),
        sources={name: sha256(runtime.ROOT / name) for name in
                 ("amodal_benchmark/shape_calibrated_prior.py", "amodal_benchmark/shape_v2.py")}))
    for family in ("local", "quadro"):
        write_json(folder / (family + "-contract.json"), dict(device_family=family, torch="1.13.1", cuda="11.7",
            cuda_fft_round_trip=True, preflight=False, holdout_access=False, benchmark_access=False,
            benchmark_admitted=False, benchmark_success=False,
            gpu="Quadro RTX 6000" if family == "quadro" else "GeForce RTX 3060",
            sources={name: sha256(runtime.ROOT / name) for name in runtime.GPU_SOURCES}))
        write_json(folder / (family + "-result.json"), dict(verified=True, complete=True, objects=1264, images=217,
            different_pixels=0, inference_gpu_validated=family == "quadro", benchmark_admitted=False,
            benchmark_success=False, rows=[dict(id=row["id"], image_id=row["image_id"], different_pixels=0,
                expected_pixel_sha256=row["prediction_sha256"], actual_pixel_sha256=row["prediction_sha256"]) for row in rows]))
    publish(folder)
    return folder


class CalibratedRuntimeTests(unittest.TestCase):
    base = ["--dataset-root", "data", "--output-dir", "out", "--repo-dir", "repo", "--pipeline-profile", "released-sam"]

    def test_cli_isolation_and_worker_propagation(self):
        flags = ["--shape-calibrated-package", "admission"]
        args = parse_args(self.base + flags)
        command = worker_command(args)
        self.assertEqual(command[command.index(flags[0])+1], str(Path("admission").resolve()))
        for extra in (["--shape-constrained-checkpoint", "old.pt"], ["--shape-v2-checkpoint", "v2.pt"],
                      ["--shape-checkpoint", "old.pt"], ["--surface-freeze", "freeze.json"],
                      ["--surface-dreamsim-cache", "cache"], ["--surface-development-review"],
                      ["--reuse-results-from", "old"], ["--pipeline-profile", "benchmark"]):
            with self.subTest(extra=extra), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(self.base + flags + extra)

    def test_native_admission_rejects_preflight_pixel_drift_and_ineligible_policy(self):
        with TemporaryDirectory() as temporary:
            folder = fixture(Path(temporary))
            self.assertEqual(runtime.admitted_policy(folder)["logit_bias"], -2.)
            changes = [("quadro-result.json", "complete", False), ("quadro-result.json", "objects", 3),
                ("local-result.json", "verified", False), ("quadro-contract.json", "preflight", True),
                ("quadro-contract.json", "gpu", "Tesla T4"), ("quadro-contract.json", "cuda_fft_round_trip", False),
                ("quadro-result.json", "inference_gpu_validated", False)]
            for name, key, value in changes:
                original = runtime.read(folder / name)
                write_json(folder / name, dict(original, **{key: value}))
                publish(folder)
                with self.subTest(name=name, key=key), self.assertRaises(ValueError):
                    runtime.admitted_policy(folder)
                write_json(folder / name, original)
                publish(folder)
            original = runtime.read(folder / "quadro-result.json")
            for mutation in ("pixels", "duplicate", "identity"):
                changed = copy.deepcopy(original)
                if mutation == "pixels":
                    changed["rows"][0]["actual_pixel_sha256"] = "different-pixels-same-IoU"
                elif mutation == "duplicate":
                    changed["rows"][0] = changed["rows"][1]
                else:
                    changed["rows"][0]["image_id"] = -1
                write_json(folder / "quadro-result.json", changed)
                publish(folder)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    runtime.admitted_policy(folder)
            write_json(folder / "quadro-result.json", original)
            policy = runtime.read(folder / "candidate/candidate.json")
            policy["native_selected"]["full_iou"] = .80
            write_json(folder / "candidate/candidate.json", policy)
            publish(folder)
            with self.assertRaisesRegex(ValueError, "geometry floors"):
                runtime.admitted_policy(folder)

    def test_score_only_rejects_another_selector(self):
        saved = dict(shape_policy=dict(method=runtime.METHOD))
        validate_shape_calibrated_scoring(parse_args(self.base), saved)
        for name in ("--shape-checkpoint", "--shape-v2-checkpoint", "--shape-constrained-checkpoint"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "substitute"):
                validate_shape_calibrated_scoring(parse_args(self.base + [name, "other.pt"]), saved)

    def test_real_workers_resume_provenance_and_package_free_scoring(self):
        from test_parallel_evaluation import ParallelEvaluationTests
        from amodal_benchmark.verify import verify_run
        harness = ParallelEvaluationTests()
        harness.setUp()
        self.addCleanup(harness.doCleanups)
        package = fixture(harness.root / "admission")
        with (harness.root / "runtime/sitecustomize.py").open("a") as stream:
            stream.write('''
from amodal_benchmark import shape_calibrated_runtime
def calibrated_install(upstream, models, sam_path, package):
    path = Path(os.environ["PDMC_TEST_TRACE"]) / f"calibrated-{os.getpid()}.json"
    path.write_text(json.dumps(dict(package=str(package), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))))
shape_calibrated_runtime.install_shape_calibrated = calibrated_install
if "--worker-index" in sys.argv and os.environ.get("PDMC_TEST_CALIBRATED_DRIFT"):
    shape_calibrated_runtime.SEED_SLOTS = (1, 2)
''')
        output = harness.root / "calibrated-output"
        flags = ["--pipeline-profile", "released-sam", "--shape-calibrated-package", str(package)]
        first = harness.evaluate(output, 2, limit=2, extra=flags, inference_only=False)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        saved = (output / "run_config.json").read_bytes()
        kept = {row["id"]: (output / "targets" / row["id"] / "result.json").read_bytes() for row in harness.rows[:2]}
        resumed = harness.evaluate(output, 2, extra=flags, inference_only=False)
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        self.assertEqual((output / "run_config.json").read_bytes(), saved)
        for key, value in kept.items():
            self.assertEqual((output / "targets" / key / "result.json").read_bytes(), value)
        installed = [runtime.read(p) for p in harness.trace.glob("calibrated-*.json")]
        self.assertEqual({row["gpu"] for row in installed}, {"GPU-A", "GPU-B"})
        self.assertTrue(all(row["package"] == str(package) for row in installed))
        self.assertEqual(verify_run(output, harness.dataset, require_provenance=True)["selected_targets"], 6)
        mismatch = harness.evaluate(harness.root / "drift", 2, extra=flags,
            env={**harness.env, "PDMC_TEST_CALIBRATED_DRIFT": "1"})
        self.assertNotEqual(mismatch.returncode, 0)
        logs = list((harness.root / "drift").glob("workers/*/worker-*.log"))
        self.assertTrue(any("differ from the coordinator" in p.read_text() for p in logs))
        for path in package.rglob("*"):
            if path.is_file():
                path.unlink()
        scored = harness.evaluate(output, 1, extra=["--pipeline-profile", "released-sam", "--score-only"], inference_only=False)
        self.assertEqual(scored.returncode, 0, scored.stdout + scored.stderr)
        self.assertEqual((output / "run_config.json").read_bytes(), saved)
        for source in output.glob("provenance/*/source"):
            for name in runtime.admission_sources():
                self.assertTrue((source / name).is_file(), name)

    def test_progressive_selector_keeps_original_observation_seeds_rgb_and_fallback(self):
        with TemporaryDirectory() as temporary:
            seeds, anchors, outputs = [], [], []
            class Generator:
                def __init__(self, device): pass
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
                outputs.append(np.array(upstream.run_mixed_context_diffusion(query_obj, pipeline, ~images[1], "chair")))
                query_obj.img_canvas[:] = 255
                query_obj.query_mask_canvas[:] = True
                query_obj.iter_id += 1
            def segment(*args):
                self.assertEqual(args[-1], ["chair"])
                image = args[3]
                masks = np.ones((1, image.height, image.width), bool) if len(seeds) <= 8 else None
                return np.array(image), masks, ["chair"], [1.]
            def prior(model, rgb, support):
                anchors.append((rgb.copy(), support.copy()))
                return np.ones(support.shape, np.float32), (0, 0, 24, 20), rgb.copy(), support.copy()
            original_gate = lambda image: image
            upstream.crop, upstream.run_iteration, upstream.segment = crop, iteration, segment
            upstream.run_mixed_context_diffusion, upstream.check_occlusion = mixed, original_gate
            support = np.zeros((20, 24), bool)
            support[5:15, 4:10] = True
            rgb = np.full((20, 24, 3), 100, np.uint8)
            query = SimpleNamespace(iter_id=0, output_img_dir=temporary, init_img_canvas=rgb,
                init_query_mask_canvas=support, img_canvas=rgb.copy(), query_mask_canvas=support.copy(), benchmark_diagnostics={})
            with patch.dict(sys.modules, {"torch": torch}), patch.object(runtime, "load_prior", return_value=object()), \
                    patch.object(runtime, "original_prior", side_effect=prior):
                runtime.install_shape_calibrated(upstream, [None], "sam", "synthetic")
                upstream.segment = lambda *args: self.fail("Candidate used expanded scene detector")
                upstream.run_iteration(query)
                upstream.run_iteration(query)
                self.assertIs(upstream.check_occlusion, original_gate)
                with self.assertRaises(ValueError):
                    runtime.install_shape_calibrated(upstream, [None], "sam", "synthetic")
            self.assertEqual(seeds, [runtime.progressive_seed(123, iteration, slot)
                                    for iteration in (0, 1) for slot in runtime.SEED_SLOTS])
            np.testing.assert_array_equal(anchors[0][0], anchors[1][0])
            np.testing.assert_array_equal(anchors[0][1], anchors[1][1])
            for number in (0, 1):
                folder = Path(temporary) / runtime.METHOD / f"iteration-{number:02d}"
                decision = runtime.read(folder / "decision.json")
                self.assertEqual(decision["index"], 0)
                self.assertEqual(decision["fallback"], bool(number))
                self.assertEqual(decision["logit_bias"], -2.)
                for name, digest in decision["artifacts"].items():
                    self.assertEqual(sha256(folder / name), digest)
                self.assertEqual((folder / "selected.png").read_bytes(), (folder / "candidate-0.png").read_bytes())
                np.testing.assert_array_equal(outputs[number], np.array(Image.open(folder / "selected.png")))


if __name__ == "__main__":
    unittest.main()
