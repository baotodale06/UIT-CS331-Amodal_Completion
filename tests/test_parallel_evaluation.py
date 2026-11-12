"""Run the real evaluator/coordinator with lightweight substitutes for GPU models."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import SOURCES, read_official

# Model construction and CUDA are the expensive boundary. Dataset verification,
# seeding, target selection, completion writes, resume, workers and summaries stay real.
CPU_RUNTIME = '''import os, random, re, sys, types
from pathlib import Path
import json
import numpy as np
from PIL import Image
cuda = types.SimpleNamespace(device_count=lambda: len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")),
                             manual_seed_all=lambda seed: None, empty_cache=lambda: None)
sys.modules["torch"] = types.SimpleNamespace(cuda=cuda, manual_seed=lambda seed: None,
    backends=types.SimpleNamespace(cudnn=types.SimpleNamespace()))
from amodal_benchmark import inference
class Upstream:
    run_sam = None
    def load_models(self, *args):
        if os.environ.get("PDMC_TEST_FAIL_GPU") == os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise RuntimeError("simulated GPU failure")
        tokenizer = lambda caption, **kwargs: {"input_ids": [0, *re.findall(r"\\w+|[^\\w\\s]", caption), 1]}
        return [types.SimpleNamespace(tokenizer=tokenizer, max_text_len=256), None, None, None]
    def transform_image(self, image): return image
    def segment(self, gdino, sam, checkpoint, image, tensor, vocabulary, device="cuda"):
        if os.environ.get("PDMC_TEST_SEGMENT_TRACE"):
            path = Path(os.environ["PDMC_TEST_TRACE"]) / f"segments-{os.getpid()}.txt"
            with path.open("a") as stream: print(int(np.random.randint(2 ** 31)), file=stream)
        vocabulary = [word for word in vocabulary if word != os.environ.get("PDMC_TEST_SKIP_LABEL")]
        if os.environ.get("PDMC_TEST_SINGLE_MASK"): vocabulary = vocabulary[:1]
        if os.environ.get("PDMC_TEST_DUPLICATE_MASKS"): vocabulary = vocabulary * 2
        return (np.asarray(image).copy(), np.ones((len(vocabulary), 12, 12), dtype=np.uint8),
                vocabulary, [.9] * len(vocabulary))
    def QueryObject(self, image_path, rgb, pil, target, mask, output):
        trace = {"gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "pid": os.getpid()}
        (Path(os.environ["PDMC_TEST_TRACE"]) / (Path(output).name + ".json")).write_text(json.dumps(trace))
        return types.SimpleNamespace(run_iter=True, iter_id=0, mask_id=target, query_mask_canvas=mask,
                                     init_outpaint_mask_canvas=mask)
    def run_iteration(self, query, *args, **kwargs):
        if os.environ.get("PDMC_TEST_CHECK_EXPANDED_VOCAB"):
            if args[3][query.mask_id] not in args[2]:
                raise AssertionError("Added detector class was missing from query eligibility vocabulary")
            result = self.segment(args[5], self.run_sam, args[6], Image.new("RGB", (12, 12)), None, args[2])
            if args[3][query.mask_id] not in result[2]:
                raise AssertionError("Completion segmentation lost the added annotation category")
        query.iter_id += 1
        query.run_iter = bool(os.environ.get("PDMC_TEST_ITERATION_LIMIT"))
        color = tuple(random.randrange(256) for _ in range(3))
        query.amodal_completion = Image.new("RGB", (12, 12), color)
        query.amodal_segmentation = np.ones((12, 12), dtype=np.uint8)
        return query
    def compute_offset(self, *args): return (0, 0)
def load_upstream(*args):
    expected_resize = os.environ.get("PDMC_TEST_BACKGROUND_AUTO") == "1"
    if os.environ.get("PDMC_TEST_EXPECT_RELEASED") and args[2:] != ("released", expected_resize):
        raise AssertionError("Native profile must forward released boundaries and selected backdrop sizing")
    if os.environ.get("PDMC_TEST_REAL_EMPTY_COMPLETION"):
        sys.path.insert(0, str(Path(inference.__file__).resolve().parents[1] / "tests"))
        from test_inference_protocol import cpu_upstream
        module = cpu_upstream()
        stub = Upstream()
        module.load_models = stub.load_models
        def segment(gdino, sam, checkpoint, image, tensor, vocabulary, device="cuda"):
            if image.size != (12, 12):
                return None, None, [], []
            return stub.segment(gdino, sam, checkpoint, image, tensor, vocabulary, device=device)
        module.segment = segment
        inference.install_runtime_hooks(module, args[2])
        return module
    if os.environ.get("PDMC_TEST_LARGE_CROP"):
        sys.path.insert(0, str(Path(inference.__file__).resolve().parents[1] / "tests"))
        from test_mixed_context_background import BackgroundTests
        BackgroundTests.setUpClass()
        case = BackgroundTests()
        image, mask, occlusion = case.inputs((1201, 12))
        background = np.full((1200, 20, 3), 100, dtype=np.uint8)
        operation = BackgroundTests.legacy
        if args[3]: operation = inference.size_mixed_context_background(operation)
        original_iteration = Upstream.run_iteration
        def large_iteration(self, query, *values, **kwargs):
            operation(None, case.pipeline, None, image, mask, occlusion, "car", background)
            return original_iteration(self, query, *values, **kwargs)
        Upstream.run_iteration = large_iteration
    return Upstream()
inference.load_upstream = load_upstream
from amodal_benchmark import metrics
class CPUMetrics:
    def __init__(self):
        if os.environ.get("PDMC_TEST_FAIL_METRICS"):
            raise AssertionError("Cached scoring attempted to construct metric models")
    def score(self, *args):
        return {"CLIP": 25., "LPIPS": .25, "Feature Similarity": .5, "SSIM": .75}
    def clip_score(self, *args): return 25.
metrics.MetricModels = CPUMetrics
if os.environ.get("PDMC_TEST_INTERRUPT_REUSE"):
    from amodal_benchmark import reuse
    original_write_json = reuse.write_json
    def interrupted_manifest(path, value):
        if Path(path).name == "reused_results.json" and value.get("target_ids"):
            raise RuntimeError("simulated interruption after result publication")
        return original_write_json(path, value)
    reuse.write_json = interrupted_manifest
if os.environ.get("PDMC_TEST_INTERRUPT_REFERENCE"):
    from amodal_benchmark import reuse
    original_copy = reuse.shutil.copy2
    original_write_reference = reuse.write_json
    def is_reference(path):
        return Path(path).suffix == ".json" and "records" in Path(path).parts
    def interrupted_reference_copy(source, destination):
        if is_reference(destination):
            Path(destination).write_bytes(b"{")
            raise RuntimeError("simulated interrupted reference publication")
        return original_copy(source, destination)
    def interrupted_reference_write(path, value):
        if is_reference(path):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.with_suffix(".json.tmp").write_bytes(b"{")
            raise RuntimeError("simulated interrupted reference publication")
        return original_write_reference(path, value)
    reuse.shutil.copy2 = interrupted_reference_copy
    reuse.write_json = interrupted_reference_write
if "--worker-index" in sys.argv and os.environ.get("PDMC_TEST_NO_WORKER_HASH"):
    from amodal_benchmark import dataset
    original_sha256 = dataset.sha256
    def guarded_hash(path):
        if str(path).endswith((".pth", ".tar", ".ckpt")):
            raise RuntimeError("A worker repeated the coordinator's checkpoint hashing")
        return original_sha256(path)
    dataset.sha256 = guarded_hash
if "--worker-index" in sys.argv and os.environ.get("PDMC_TEST_MATCHER_DRIFT"):
    from amodal_benchmark.matching import MATCHING_CONFIG
    MATCHING_CONFIG["version"] = -1
if "--worker-index" in sys.argv and os.environ.get("PDMC_TEST_DETECTOR_DRIFT"):
    from amodal_benchmark.detection import DETECTION_CONFIG
    DETECTION_CONFIG["version"] = -1
'''


class ParallelEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.trace = self.root / "trace"
        self.trace.mkdir()
        rows, _ = read_official(ROOT / "tests/fixtures")
        self.rows = rows[:6]
        for source, (filename, *_) in SOURCES.items():
            path = self.dataset / "official" / source / filename
            path.parent.mkdir(parents=True)
            path.write_bytes((ROOT / "tests/fixtures/official" / source / filename).read_bytes())
        for row in self.rows:
            path = self.dataset / row["image"]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (12, 12), "white").save(path)
        (self.repo / "classes.txt").write_text("\n".join(dict.fromkeys(r["label"] for r in self.rows)))
        for name in ["main.py", "Grounded-Segment-Anything/groundingdino_swint_ogc.pth",
                     "Grounded-Segment-Anything/sam_vit_h_4b8939.pth",
                     "InstaOrder/InstaOrder_ckpt/InstaOrder_InstaOrderNet_od.pth.tar",
                     "lama/big-lama/models/best.ckpt"]:
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("CPU model boundary fixture")
        runtime = self.root / "runtime"
        runtime.mkdir()
        (runtime / "sitecustomize.py").write_text(CPU_RUNTIME, encoding="utf-8")
        self.env = os.environ.copy()
        self.env.update(CUDA_VISIBLE_DEVICES="GPU-A,GPU-B", PDMC_TEST_TRACE=str(self.trace),
                        OMP_NUM_THREADS="4",
                        PYTHONPATH=os.pathsep.join([str(runtime), str(ROOT), self.env.get("PYTHONPATH", "")]))

    def evaluate(self, output, workers, limit=6, extra=(), env=None, inference_only=True):
        return subprocess.run([sys.executable, str(ROOT / "evaluate_open_world.py"),
            "--dataset-root", str(self.dataset), "--repo-dir", str(self.repo),
            "--output-dir", str(output), "--target-selection", "label",
            "--workers", str(workers), *(["--limit", str(limit)] if limit is not None else []),
            *(["--inference-only"] if inference_only else []), *extra],
            env=env or self.env, capture_output=True, text=True, timeout=30)

    def test_frozen_cohort_is_forwarded_verified_and_resumes_without_changing_queries(self):
        from amodal_benchmark.verify import verify_run
        output = self.root / 'cohort'
        manifest = self.root / 'cohort.json'
        selected = [self.rows[index] for index in (0, 2, 4)]
        manifest.write_text(json.dumps(dict(target_ids=[row['id'] for row in selected])))
        settings = ['--pipeline-profile', 'released-sam', '--target-selection', 'label-semantic',
                    '--detector-vocabulary', 'annotated']
        result = self.evaluate(output, 2, limit=None, extra=settings + ['--target-manifest', str(manifest)],
                               inference_only=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = verify_run(output, self.dataset, require_provenance=True)
        self.assertEqual(report['selected_targets'], 3)
        self.assertFalse(report['full_evaluation'])
        records = {row['id']: (output / 'targets' / row['id'] / 'result.json').read_bytes() for row in selected}
        before = (output / 'run_config.json').read_bytes()
        snapshot = self.root / 'snapshot'
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/snapshot_annotated_surface.py'),
            '--run-dir', str(output), '--output-dir', str(snapshot), '--dataset-root', str(self.dataset),
            '--target-manifest', str(manifest)], env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(verify_run(snapshot, self.dataset, require_provenance=True)['selected_targets'], 3)
        result = self.evaluate(output, 2, extra=settings, inference_only=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((output / 'run_config.json').read_bytes(), before)
        self.assertEqual(verify_run(output, self.dataset, require_provenance=True)['selected_targets'], 6)
        for identifier, data in records.items():
            self.assertEqual((output / 'targets' / identifier / 'result.json').read_bytes(), data)
        self.assertEqual(verify_run(snapshot, self.dataset, require_provenance=True)['selected_targets'], 3)

    def test_parallel_resume_matches_serial_completions_seeds_and_full_summary(self):
        baseline, resumed = self.root / "serial", self.root / "resumed"
        serial = self.evaluate(baseline, 1)
        self.assertEqual(serial.returncode, 0, serial.stderr)
        first = self.evaluate(resumed, 1, limit=2)
        self.assertEqual(first.returncode, 0, first.stderr)
        saved = {r["id"]: (resumed / "targets" / r["id"] / "result.json").read_bytes()
                 for r in self.rows[:2]}
        parallel = self.evaluate(resumed, "auto")
        self.assertEqual(parallel.returncode, 0, parallel.stdout + parallel.stderr)
        self.assertEqual(json.loads((resumed / "run_config.json").read_text()),
                         json.loads((baseline / "run_config.json").read_text()))
        for row in self.rows:
            path = Path("targets") / row["id"]
            self.assertEqual((resumed / path / "completion.png").read_bytes(),
                             (baseline / path / "completion.png").read_bytes())
            self.assertEqual(json.loads((resumed / path / "result.json").read_text()),
                             json.loads((baseline / path / "result.json").read_text()))
        for identifier, content in saved.items():
            self.assertEqual((resumed / "targets" / identifier / "result.json").read_bytes(), content)
        summary = json.loads((resumed / "summary.json").read_text())
        self.assertEqual(summary["selected_targets"], 6)
        self.assertEqual(summary["statistics"]["Overall"]["successes"], 6)
        with (resumed / "per_target.csv").open() as stream:
            self.assertEqual([row["id"] for row in csv.DictReader(stream)], [r["id"] for r in self.rows])
        traces = [json.loads((self.trace / (Path(r["id"]).name + ".json")).read_text())
                  for r in self.rows[2:]]
        self.assertEqual({r["gpu"] for r in traces}, {"GPU-A", "GPU-B"})
        self.assertEqual(len({r["pid"] for r in traces}), 2)
        # Switching back to one worker must reuse results too.
        again = self.evaluate(resumed, 1)
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertNotIn("PD-MC worker", again.stdout)

    def test_worker_failure_returns_nonzero_without_writing_a_combined_report(self):
        env = {**self.env, "PDMC_TEST_FAIL_GPU": "GPU-B"}
        output = self.root / "failed"
        result = self.evaluate(output, 2, env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("worker-01.log", result.stderr)
        self.assertFalse((output / "summary.json").exists())
        executions = list(output.glob("workers/*/execution.json"))
        self.assertEqual(len(executions), 1)
        self.assertEqual(json.loads(executions[0].read_text())["status"], "failed")

    def test_parallel_scoring_preserves_success_failure_counts_and_metric_denominators(self):
        references = self.root / "references"
        for row in self.rows:
            path = references / row["reference"]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGBA", (12, 12), (255, 255, 255, 255)).save(path)
        env = {**self.env, "PDMC_TEST_SKIP_LABEL": self.rows[0]["label"]}
        self.assertEqual(sum(row["label"] == self.rows[0]["label"] for row in self.rows), 1)
        reports = []
        for workers in (1, 2):
            output = self.root / f"scored-{workers}"
            result = self.evaluate(output, workers, env=env, inference_only=False,
                extra=("--reference-dir", str(references), "--metric-alignment", "original-frame"))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["evaluation_performed"])
            stat = summary["statistics"]["Overall"]
            self.assertEqual((stat["targets"], stat["successes"], stat["failures"]), (6, 5, 1))
            self.assertEqual((stat["CLIP_n"], stat["LPIPS_n"]), (5, 6))
            self.assertAlmostEqual(stat["LPIPS"], .375)
            self.assertAlmostEqual(stat["Feature Similarity"], 5 / 12)
            self.assertAlmostEqual(stat["SSIM"], .625)
            self.assertEqual(len(list(output.glob("targets/*/*/metrics.json"))), 5)
            reports.append((summary, (output / "table.md").read_text()))
        self.assertEqual(reports[0], reports[1])

    def test_cpu_check_only_accepts_auto_without_loading_cuda_or_creating_outputs(self):
        output = self.root / "check"
        result = self.evaluate(output, "auto", extra=("--check-only",))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(output.exists())

    def test_only_the_coordinator_hashes_shared_checkpoint_files(self):
        env = {**self.env, "PDMC_TEST_NO_WORKER_HASH": "1"}
        result = self.evaluate(self.root / "hash-once", 2, env=env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_repeated_image_targets_share_a_worker_and_match_serial_detection_seeds(self):
        self.rows = read_official(ROOT / "tests/fixtures")[0][:12]
        self.assertEqual(len({r["image"] for r in self.rows}), 11)
        for row in self.rows[6:]:
            path = self.dataset / row["image"]
            Image.new("RGB", (12, 12), "white").save(path)
        (self.repo / "classes.txt").write_text("\n".join(dict.fromkeys(r["label"] for r in self.rows)))
        seeds = []
        outputs = []
        for workers in (1, 2):
            trace = self.root / f"trace-{workers}"
            trace.mkdir()
            env = {**self.env, "PDMC_TEST_SEGMENT_TRACE": "1", "PDMC_TEST_TRACE": str(trace)}
            output = self.root / f"repeated-{workers}"
            result = self.evaluate(output, workers, limit=12, env=env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            samples = [int(line) for path in trace.glob("segments-*.txt") for line in path.read_text().splitlines()]
            self.assertEqual(len(samples), 11)
            seeds.append(sorted(samples))
            outputs.append(output)
            if workers == 2:
                repeated = [json.loads((trace / (Path(row["id"]).name + ".json")).read_text())
                            for row in self.rows[-2:]]
                self.assertEqual(repeated[0], repeated[1])
        self.assertEqual(seeds[0], seeds[1])
        for row in self.rows:
            relative = Path("targets") / row["id"] / "completion.png"
            self.assertEqual((outputs[0] / relative).read_bytes(), (outputs[1] / relative).read_bytes())

    def test_default_reference_selection_keeps_empty_lisa_references_out_of_failure_counts(self):
        import numpy as np
        from amodal_benchmark.dataset import sha256
        from amodal_benchmark.references import save_reference
        references = self.root / "prepared"
        for index, row in enumerate(self.rows[:2]):
            mask = np.ones((12, 12), dtype=bool) if index else None
            save_reference(references, row, np.full((12, 12, 3), 255, dtype=np.uint8),
                           mask, sha256(self.dataset / row["image"]), {"seed": 0})
        env = {**self.env, "PDMC_TEST_SINGLE_MASK": "1"}
        reports = []
        for workers in (1, 2):
            output = self.root / f"reference-{workers}"
            result = self.evaluate(output, workers, limit=2, env=env, inference_only=False,
                extra=("--target-selection", "auto", "--reference-dir", str(references)))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            config = json.loads((output / "run_config.json").read_text())
            self.assertEqual(config["target_selection"], "reference-best")
            summary = json.loads((output / "summary.json").read_text())
            stat = summary["statistics"]["Overall"]
            self.assertEqual((stat["targets"], stat["successes"], stat["failures"], stat["reference_failures"]),
                             (2, 1, 0, 1))
            self.assertEqual((stat["CLIP_n"], stat["LPIPS_n"]), (1, 1))
            reports.append(summary)
        self.assertEqual(reports[0], reports[1])

    def test_clip_only_success_is_cached_verified_and_rescored_without_model_loading(self):
        import numpy as np
        from amodal_benchmark.dataset import sha256
        from amodal_benchmark.references import save_reference
        from amodal_benchmark.verify import verify_run
        references = self.root / "empty-reference"
        for index, row in enumerate(self.rows[:2]):
            save_reference(references, row, np.full((12, 12, 3), 255, dtype=np.uint8),
                           np.ones((12, 12), dtype=bool) if index else None,
                           sha256(self.dataset / row["image"]), {})
        output = self.root / "clip-only"
        extra = ("--reference-dir", str(references), "--target-selection", "label")
        result = self.evaluate(output, 1, limit=2, extra=extra, inference_only=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = verify_run(output, self.dataset, references, require_provenance=True)
        self.assertEqual(report["clip_only_targets"], 1)
        self.assertEqual(report["uncached_clip_only_targets"], 0)
        self.assertEqual(report["statistics"]["Overall"]["LPIPS_n"], 1)
        result = self.evaluate(output, 1, limit=2, extra=extra + ("--score-only",), inference_only=False,
            env={**self.env, "PDMC_TEST_FAIL_METRICS": "1", "PDMC_TEST_FAIL_GPU": self.env["CUDA_VISIBLE_DEVICES"]})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_corrected_configuration_cannot_silently_resume_released_boundary_outputs(self):
        output = self.root / "legacy"
        result = self.evaluate(output, 1, extra=("--boundary-check", "released"))
        self.assertEqual(result.returncode, 0, result.stderr)
        config = json.loads((output / "run_config.json").read_text())
        self.assertEqual(config["adapter_version"], 3)
        self.assertNotIn("boundary_check", config)
        result = self.evaluate(output, 2)
        self.assertEqual(result.returncode, 2)
        self.assertIn("different run configuration", result.stderr)
        result = self.evaluate(output, 2, extra=("--boundary-check", "released"))
        self.assertEqual(result.returncode, 0, result.stderr)
        corrected = self.root / "corrected"
        result = self.evaluate(corrected, 2)
        self.assertEqual(result.returncode, 0, result.stderr)
        config = json.loads((corrected / "run_config.json").read_text())
        self.assertEqual((config["adapter_version"], config["boundary_check"]), (4, "paper"))
        references = self.root / "score-legacy"
        for row in self.rows:
            path = references / row["reference"]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGBA", (12, 12), "white").save(path)
        result = self.evaluate(output, 1, inference_only=False, extra=("--score-only",
            "--reference-dir", str(references), "--metric-alignment", "original-frame"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["boundary_check"], "released")

    def test_score_only_paper_report_keeps_the_saved_inference_seed(self):
        output = self.root / "saved-seed"
        result = self.evaluate(output, 1, limit=2, extra=("--seed", "9"))
        self.assertEqual(result.returncode, 0, result.stderr)
        references = self.root / "score-references"
        for row in self.rows[:2]:
            path = references / row["reference"]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGBA", (12, 12), "white").save(path)
        result = self.evaluate(output, 1, limit=2, inference_only=False,
            extra=("--score-only", "--reference-dir", str(references),
                   "--metric-alignment", "original-frame"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("seed: 9.", (output / "paper_comparison.md").read_text())

    def test_best_reference_matching_survives_duplicate_masks_and_records_candidates(self):
        import numpy as np
        from amodal_benchmark.dataset import sha256
        from amodal_benchmark.references import save_reference
        references = self.root / "duplicate-references"
        for row in self.rows[:2]:
            save_reference(references, row, np.full((12, 12, 3), 255, dtype=np.uint8),
                           np.ones((12, 12), dtype=bool), sha256(self.dataset / row["image"]), {})
        for selection, successes in (("reference", 0), ("reference-best", 2)):
            output = self.root / selection
            result = self.evaluate(output, 2, limit=2, extra=("--reference-dir", str(references),
                                   "--target-selection", selection))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["statistics"]["Overall"]["successes"], successes)
            record = json.loads((output / "targets" / self.rows[0]["id"] / "result.json").read_text())
            self.assertEqual(record["selection_diagnostics"]["best_iou"], 1.)
            self.assertGreater(len(record["selection_diagnostics"]["candidates"]), 1)

    def test_released_sam_profile_scores_without_lisa_and_counts_detection_failures(self):
        env = {**self.env, "PDMC_TEST_SKIP_LABEL": self.rows[0]["label"], "PDMC_TEST_EXPECT_RELEASED": "1"}
        summaries = []
        for workers in (1, 2):
            output = self.root / f"sam-{workers}"
            result = self.evaluate(output, workers, env=env, inference_only=False,
                                   extra=("--pipeline-profile", "released-sam"))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            config = json.loads((output / "run_config.json").read_text())
            self.assertEqual(config["pipeline_profile"], "released-sam")
            self.assertEqual(config["boundary_check"], "released")
            self.assertEqual(config["target_selection"], "label")
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["reference_backend"], "grounded-sam")
            stat = summary["statistics"]["Overall"]
            self.assertEqual((stat["successes"], stat["failures"], stat["reference_failures"]), (5, 1, 0))
            self.assertEqual((stat["CLIP_n"], stat["LPIPS_n"]), (5, 6))
            self.assertAlmostEqual(stat["LPIPS"], .375)
            for row in self.rows:
                reference = output / "visible_references/records" / Path(row["reference"]).with_suffix(".json")
                self.assertEqual(json.loads(reference.read_text())["provenance"]["backend"], "grounded-sam")
            ref_config = json.loads((output / "visible_references/reference_config.json").read_text())
            self.assertEqual(len(ref_config["checkpoints"]), 2)
            comparison = json.loads((output / "paper_comparison.json").read_text())
            self.assertEqual(comparison["sources"]["Overall"]["metrics"]["LPIPS"]["comparison_status"],
                             "different_reference_backend")
            summaries.append(summary)
        self.assertEqual(summaries[0], summaries[1])

    def test_sam_profile_resumes_and_scores_without_loading_inference_models(self):
        output = self.root / "sam-resume"
        extra = ("--pipeline-profile", "released-sam")
        result = self.evaluate(output, 1, limit=2, extra=extra)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        saved = (output / "targets" / self.rows[0]["id"] / "result.json").read_bytes()
        result = self.evaluate(output, 2, extra=extra)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((output / "targets" / self.rows[0]["id"] / "result.json").read_bytes(), saved)
        # Any attempt to load inference models now fails, including in workers.
        env = {**self.env, "PDMC_TEST_FAIL_GPU": self.env["CUDA_VISIBLE_DEVICES"]}
        result = self.evaluate(output, 1, env=env, inference_only=False, extra=extra + ("--score-only",))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads((output / "summary.json").read_text())["statistics"]["Overall"]["LPIPS_n"], 6)
        reference = output / "visible_references/masks" / self.rows[0]["reference"]
        reference.write_bytes(b"changed SAM reference")
        result = self.evaluate(output, 1, extra=extra)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("artifact missing or changed", result.stderr)

    def test_sam_cpu_preflight_does_not_need_references_or_create_outputs(self):
        output = self.root / "sam-check"
        result = self.evaluate(output, "auto", inference_only=False,
                               extra=("--pipeline-profile", "released-sam", "--check-only"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(output.exists())

    def test_lisa_profile_rejects_sam_reference_directory_before_inference(self):
        sam = self.root / "sam-reference-source"
        result = self.evaluate(sam, 1, limit=2, extra=("--pipeline-profile", "released-sam"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output = self.root / "wrong-lisa-protocol"
        result = self.evaluate(output, 1, limit=2, inference_only=False,
            extra=("--reference-dir", str(sam / "visible_references"), "--check-only"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("SAM references define a separate protocol", result.stderr)
        self.assertFalse(output.exists())

    def test_sam_profile_rejects_lisa_references_and_existing_benchmark_outputs(self):
        references = self.root / "lisa-references"
        references.mkdir()
        (references / "LISA.txt").write_text("existing LISA artifacts")
        result = self.evaluate(self.root / "sam", 1, limit=2,
            extra=("--pipeline-profile", "released-sam", "--reference-dir", str(references)))
        self.assertEqual(result.returncode, 2)
        self.assertIn("use a new reference directory", result.stderr)
        self.assertEqual([p.name for p in references.iterdir()], ["LISA.txt"])
        output = self.root / "old-benchmark"
        result = self.evaluate(output, 1, limit=2)
        self.assertEqual(result.returncode, 0, result.stderr)
        saved = (output / "run_config.json").read_bytes()
        result = self.evaluate(output, 1, limit=2, extra=("--pipeline-profile", "released-sam"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("different run configuration", result.stderr)
        self.assertEqual((output / "run_config.json").read_bytes(), saved)

    def test_sam_reference_regeneration_rejects_new_detections_without_overwriting_artifacts(self):
        output = self.root / "frozen-sam"
        extra = ("--pipeline-profile", "released-sam")
        result = self.evaluate(output, 1, limit=2, extra=extra)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        from amodal_benchmark.references import reference_paths
        paths = reference_paths(output / "visible_references", self.rows[0])
        saved = {name: path.read_bytes() for name, path in paths.items()}
        (output / "targets" / self.rows[0]["id"] / "result.json").unlink()
        env = {**self.env, "PDMC_TEST_SKIP_LABEL": self.rows[0]["label"]}
        result = self.evaluate(output, 1, limit=2, extra=extra, env=env)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Regenerated SAM reference changed", result.stderr)
        self.assertEqual({name: path.read_bytes() for name, path in paths.items()}, saved)

    def test_standalone_pdmc_evaluator_enforces_native_profile(self):
        output = self.root / "standalone"
        result = subprocess.run([sys.executable, str(ROOT / "evaluate_pdmc_open_world.py"),
            "--dataset-root", str(self.dataset), "--repo-dir", str(self.repo),
            "--output-dir", str(output), "--workers", "2", "--limit", "2",
            "--pipeline-profile", "benchmark"], env={**self.env, "PDMC_TEST_EXPECT_RELEASED": "1",
                                                     "PDMC_TEST_BACKGROUND_AUTO": "1"},
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["reference_backend"], "grounded-sam")
        self.assertTrue(summary["evaluation_performed"])
        self.assertEqual(summary["target_selection"], "label-semantic")
        self.assertTrue(summary["resize_background"])

    def test_sam_profile_rejects_reference_matching_and_repaired_boundary_options(self):
        for option in (("--target-selection", "reference-best"), ("--boundary-check", "paper")):
            with self.subTest(option=option):
                result = self.evaluate(self.root / "invalid", 1, limit=2,
                    extra=("--pipeline-profile", "released-sam", *option))
                self.assertEqual(result.returncode, 2)
                self.assertIn("released-sam requires", result.stderr)

    def test_standalone_can_resume_legacy_matching_and_background_settings(self):
        output = self.root / "standalone-legacy"
        command = [sys.executable, str(ROOT / "evaluate_pdmc_open_world.py"),
            "--dataset-root", str(self.dataset), "--repo-dir", str(self.repo),
            "--output-dir", str(output), "--limit", "2",
            "--target-selection", "label", "--background-sizing", "released", "--inference-only"]
        result = subprocess.run(command, env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        config = json.loads((output / "run_config.json").read_text())
        self.assertEqual(config["target_selection"], "label")
        self.assertFalse(config["resize_background"])
        self.assertNotIn("label_matching", config)
        # Score-only uses the saved settings even when the wrapper has new defaults.
        command = command[:command.index("--target-selection")] + ["--score-only"]
        result = subprocess.run(command, env={**self.env, "PDMC_TEST_FAIL_GPU": self.env["CUDA_VISIBLE_DEVICES"]},
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["target_selection"], "label")
        self.assertFalse(summary["resize_background"])

    def test_sam_background_auto_repairs_the_pinned_1200_pixel_failure_in_serial_and_workers(self):
        profile = ("--pipeline-profile", "released-sam")
        env = {**self.env, "PDMC_TEST_LARGE_CROP": "1", "PDMC_TEST_EXPECT_RELEASED": "1"}
        failed = self.root / "released-crash"
        result = self.evaluate(failed, 1, limit=2, extra=profile, env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("index 1200", result.stderr)
        self.assertFalse((failed / "summary.json").exists())
        reports = []
        for workers in (1, 2):
            output = self.root / f"background-auto-{workers}"
            result = self.evaluate(output, workers, limit=2, inference_only=False,
                extra=profile + ("--background-sizing", "auto"),
                env={**env, "PDMC_TEST_BACKGROUND_AUTO": "1"})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            config = json.loads((output / "run_config.json").read_text())
            self.assertTrue(config["resize_background"])
            self.assertEqual(config["boundary_check"], "released")
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["resize_background"])
            self.assertEqual(summary["statistics"]["Overall"]["CLIP_n"], 2)
            reports.append(summary)
            # Scoring restores the saved repair mode without loading PD-MC.
            result = self.evaluate(output, 1, limit=2, inference_only=False,
                extra=profile + ("--score-only",),
                env={**self.env, "PDMC_TEST_FAIL_GPU": self.env["CUDA_VISIBLE_DEVICES"]})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(json.loads((output / "summary.json").read_text())["resize_background"])
        self.assertEqual(reports[0], reports[1])

    def test_sam_background_repair_reuses_valid_completed_targets_in_a_new_directory(self):
        profile = ("--pipeline-profile", "released-sam")
        source = self.root / "original-sam"
        result = self.evaluate(source, 1, limit=2, extra=profile)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        source_config = (source / "run_config.json").read_bytes()
        saved = {row["id"]: (source / "targets" / row["id"] / "result.json").read_bytes()
                 for row in self.rows[:2]}
        output = self.root / "repaired-sam"
        result = self.evaluate(output, 2, inference_only=False,
            extra=profile + ("--background-sizing", "auto", "--reuse-results-from", str(source)),
            env={**self.env, "PDMC_TEST_LARGE_CROP": "1", "PDMC_TEST_BACKGROUND_AUTO": "1",
                 "PDMC_TEST_EXPECT_RELEASED": "1"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((source / "run_config.json").read_bytes(), source_config)
        for identifier, content in saved.items():
            self.assertEqual((output / "targets" / identifier / "result.json").read_bytes(), content)
            self.assertEqual((output / "targets" / identifier / "completion.png").read_bytes(),
                             (source / "targets" / identifier / "completion.png").read_bytes())
        manifest = json.loads((output / "reused_results.json").read_text())
        self.assertEqual(set(manifest["target_ids"]), set(saved))
        self.assertEqual(json.loads((output / "summary.json").read_text())["statistics"]["Overall"]["successes"], 6)
        result = self.evaluate(source, 1, limit=2,
            extra=profile + ("--background-sizing", "auto"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("different run configuration", result.stderr)

    def test_sam_background_reuse_rejects_different_experiments_and_changed_completions(self):
        profile = ("--pipeline-profile", "released-sam")
        source = self.root / "reuse-source"
        result = self.evaluate(source, 1, limit=2, extra=profile)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = profile + ("--background-sizing", "auto", "--reuse-results-from", str(source))
        result = self.evaluate(self.root / "wrong-seed", 1, limit=2, extra=args + ("--seed", "9"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("differ", result.stderr)
        (source / "targets" / self.rows[0]["id"] / "completion.png").write_bytes(b"changed pixels")
        result = self.evaluate(self.root / "changed-completion", 1, limit=2, extra=args)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Saved completion is missing or changed", result.stderr)

    def test_sam_background_reuse_recovers_provenance_after_interrupted_import(self):
        profile = ("--pipeline-profile", "released-sam")
        source, output = self.root / "source", self.root / "interrupted-import"
        result = self.evaluate(source, 1, limit=2, extra=profile)
        self.assertEqual(result.returncode, 0, result.stderr)
        extra = profile + ("--background-sizing", "auto", "--reuse-results-from", str(source))
        result = self.evaluate(output, 1, limit=2, extra=extra,
                               env={**self.env, "PDMC_TEST_INTERRUPT_REUSE": "1"})
        self.assertEqual(result.returncode, 2)
        self.assertIn("simulated interruption", result.stderr)
        self.assertTrue((output / "targets" / self.rows[0]["id"] / "result.json").is_file())
        result = self.evaluate(output, 1, limit=2, extra=extra,
                               env={**self.env, "PDMC_TEST_FAIL_GPU": self.env["CUDA_VISIBLE_DEVICES"]})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads((output / "reused_results.json").read_text())
        self.assertEqual(set(manifest["target_ids"]), {row["id"] for row in self.rows[:2]})
        self.assertEqual(manifest["pending_target_ids"], [])

    def test_sam_background_reuse_recovers_after_interrupted_reference_publication(self):
        profile = ("--pipeline-profile", "released-sam")
        source, output = self.root / "source", self.root / "interrupted-reference"
        result = self.evaluate(source, 1, limit=2, extra=profile)
        self.assertEqual(result.returncode, 0, result.stderr)
        extra = profile + ("--background-sizing", "auto", "--reuse-results-from", str(source))
        result = self.evaluate(output, 1, limit=2, extra=extra,
                               env={**self.env, "PDMC_TEST_INTERRUPT_REFERENCE": "1"})
        self.assertEqual(result.returncode, 2)
        self.assertIn("simulated interrupted reference publication", result.stderr)
        result = self.evaluate(output, 1, limit=2, extra=extra,
                               env={**self.env, "PDMC_TEST_FAIL_GPU": self.env["CUDA_VISIBLE_DEVICES"]})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads((output / "reused_results.json").read_text())
        self.assertEqual(set(manifest["target_ids"]), {row["id"] for row in self.rows[:2]})
        self.assertEqual(manifest["pending_target_ids"], [])

    def test_semantic_sam_matching_recovers_aliases_and_preserves_detected_prompt_serial_and_parallel(self):
        (self.repo / "classes.txt").write_text("dining table")
        reports = []
        extra = ("--pipeline-profile", "released-sam", "--target-selection", "label-semantic")
        for workers in (1, 2):
            output = self.root / f"semantic-{workers}"
            result = self.evaluate(output, workers, inference_only=False, extra=extra)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            config = json.loads((output / "run_config.json").read_text())
            self.assertEqual(config["adapter_version"], 6)
            self.assertIn("label_matching", config)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["statistics"]["Overall"]["successes"], 1)
            record = json.loads((output / "targets" / self.rows[5]["id"] / "result.json").read_text())
            self.assertEqual(record["label"], "brown table")
            self.assertEqual(record["detected_class"], "dining table")
            self.assertEqual(record["selection_diagnostics"]["selected_index"], 0)
            self.assertEqual(len(record["selection_diagnostics"]["candidates"]), 1)
            reports.append(summary)
        self.assertEqual(reports[0], reports[1])

    def test_semantic_sam_matching_deduplicates_and_refuses_old_outputs_and_reference_roots(self):
        (self.repo / "classes.txt").write_text("car")
        legacy = self.root / "legacy-selection"
        env = {**self.env, "PDMC_TEST_DUPLICATE_MASKS": "1"}
        profile = ("--pipeline-profile", "released-sam")
        result = self.evaluate(legacy, 1, limit=2, extra=profile, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        semantic = profile + ("--target-selection", "label-semantic")
        result = self.evaluate(legacy, 1, limit=2, extra=semantic, env=env)
        self.assertEqual(result.returncode, 2)
        self.assertIn("different run configuration", result.stderr)
        result = self.evaluate(self.root / "wrong-references", 1, limit=2, env=env,
            extra=semantic + ("--reference-dir", str(legacy / "visible_references")))
        self.assertEqual(result.returncode, 2)
        self.assertIn("different run configuration", result.stderr)
        output = self.root / "deduplicated"
        result = self.evaluate(output, 2, limit=2, extra=semantic, env=env, inference_only=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["statistics"]["Overall"]["successes"], 1)
        record = json.loads((output / "targets" / self.rows[1]["id"] / "result.json").read_text())
        self.assertEqual(record["selection_diagnostics"]["duplicate_count"], 1)
        result = self.evaluate(output, 1, limit=2, inference_only=False, extra=profile + ("--score-only",),
            env={**self.env, "PDMC_TEST_FAIL_GPU": self.env["CUDA_VISIBLE_DEVICES"]})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads((output / "summary.json").read_text())["target_selection"], "label-semantic")

    def test_iteration_limited_outputs_are_reported_separately_and_csv_roundtrips(self):
        from amodal_benchmark.reproduction import load_run
        output = self.root / "iteration-limited"
        result = self.evaluate(output, 2, limit=2, inference_only=False,
            extra=("--pipeline-profile", "released-sam", "--target-selection", "label-semantic"),
            env={**self.env, "PDMC_TEST_ITERATION_LIMIT": "1"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((output / "summary.json").read_text())
        stat = summary["statistics"]["Overall"]
        self.assertEqual((stat["successes"], stat["completed_outputs"], stat["iteration_limited_outputs"]), (2, 0, 2))
        self.assertEqual(load_run(output)["summary"]["statistics"], summary["statistics"])

    def test_semantic_score_only_rejects_changed_reference_matching_configuration(self):
        output = self.root / "semantic-reference-identity"
        profile = ("--pipeline-profile", "released-sam", "--target-selection", "label-semantic")
        result = self.evaluate(output, 1, limit=2, extra=profile)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        marker = output / "visible_references/reference_config.json"
        original = marker.read_bytes()
        summary = (output / "summary.json").read_bytes()
        for key, value in (("target_selection", "label"), ("label_matching", {"version": -1}),
                           ("seed", 999)):
            with self.subTest(key=key):
                config = json.loads(original)
                config[key] = value
                marker.write_text(json.dumps(config))
                result = self.evaluate(output, 1, limit=2, inference_only=False,
                    extra=profile + ("--score-only",))
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("reference configuration", result.stderr)
                self.assertEqual((output / "summary.json").read_bytes(), summary)

    def test_semantic_matching_cannot_import_legacy_background_only_results(self):
        legacy = self.root / "legacy-background-import"
        profile = ("--pipeline-profile", "released-sam")
        result = self.evaluate(legacy, 1, limit=2, extra=profile)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        saved = (legacy / "run_config.json").read_bytes()
        result = self.evaluate(self.root / "invalid-semantic-import", 1, limit=2,
            extra=profile + ("--target-selection", "label-semantic", "--background-sizing", "auto",
                             "--reuse-results-from", str(legacy)))
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("differ beyond enabling background sizing", result.stderr)
        self.assertEqual((legacy / "run_config.json").read_bytes(), saved)

    def test_workers_reject_a_different_semantic_matching_version(self):
        output = self.root / "worker-matcher-drift"
        result = self.evaluate(output, 2, limit=2,
            extra=("--pipeline-profile", "released-sam", "--target-selection", "label-semantic"),
            env={**self.env, "PDMC_TEST_MATCHER_DRIFT": "1"})
        self.assertNotEqual(result.returncode, 0)
        logs = list(output.glob("workers/*/worker-*.log"))
        self.assertTrue(any("Worker settings differ" in path.read_text() for path in logs))
        self.assertFalse((output / "summary.json").exists())
        self.assertFalse(list(output.glob("targets/*/*/result.json")))

    def test_previous_semantic_version_can_be_scored_but_cannot_resume_inference(self):
        output = self.root / "previous-semantic-version"
        profile = ("--pipeline-profile", "released-sam", "--target-selection", "label-semantic")
        result = self.evaluate(output, 1, limit=2, extra=profile)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        previous_matching = {
            "version": 1,
            "aliases_sha256": "b8158502f25150a4992699ed2b5ae9ee453288f67e132eeca69f44240128709a",
            "annotation_rule": "longest-object-suffix-before-spatial-description-v1",
            "deduplication": "exact-binary-mask",
            "duplicate_representative": "highest-detector-score/stable-first-index",
            "implementation_sha256": "984b113885ea6543f60d25cd826bdadcae1dced46149a17ab44d926b44c0213c",
        }
        # Simulate a completed run bearing the previous matcher identity.
        markers = [output / "run_config.json", output / "visible_references/reference_config.json"]
        for path in markers:
            config = json.loads(path.read_text())
            config["label_matching"] = previous_matching
            path.write_text(json.dumps(config))
        saved = {path: path.read_bytes() for path in markers}
        records = {path: path.read_bytes() for path in output.glob("targets/*/*/result.json")}
        for workers in (1, 2):
            result = self.evaluate(output, workers, limit=2, extra=profile)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("different run configuration", result.stderr)
        result = self.evaluate(output, 1, limit=2, inference_only=False,
            extra=profile + ("--score-only",),
            env={**self.env, "PDMC_TEST_FAIL_GPU": self.env["CUDA_VISIBLE_DEVICES"]})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads((output / "summary.json").read_text())["label_matching"], previous_matching)
        for path, content in {**saved, **records}.items():
            self.assertEqual(path.read_bytes(), content)

    def test_annotation_vocabulary_recovers_missing_categories_serial_and_four_workers(self):
        (self.repo / "classes.txt").write_text("cat")
        profile = ("--pipeline-profile", "released-sam", "--target-selection", "label-semantic")
        baseline = self.root / "original-vocabulary"
        result = self.evaluate(baseline, 1, extra=profile)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((baseline / "summary.json").read_text())
        self.assertEqual(summary["statistics"]["Overall"]["successes"], 0)
        self.assertNotIn("detector_vocabulary", json.loads((baseline / "run_config.json").read_text()))
        extra = profile + ("--detector-vocabulary", "annotated")
        env = {**self.env, "CUDA_VISIBLE_DEVICES": "GPU-A,GPU-B,GPU-C,GPU-D",
               "PDMC_TEST_CHECK_EXPANDED_VOCAB": "1"}
        reports = []
        for workers in (1, 4):
            output = self.root / f"expanded-{workers}"
            result = self.evaluate(output, workers, extra=extra, env=env, inference_only=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            config = json.loads((output / "run_config.json").read_text())
            self.assertEqual(config["adapter_version"], 7)
            self.assertEqual(config["detector_vocabulary"]["mode"], "annotated")
            reference = json.loads((output / "visible_references/reference_config.json").read_text())
            self.assertEqual(reference["detector_vocabulary"], config["detector_vocabulary"])
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["statistics"]["Overall"]["successes"], 6)
            self.assertEqual(summary["detector_vocabulary"], config["detector_vocabulary"])
            record = json.loads((output / "targets" / self.rows[0]["id"] / "result.json").read_text())
            self.assertEqual(record["detected_class"], "floor lamp")
            self.assertEqual(record["detection_diagnostics"]["mode"], "annotated")
            self.assertTrue(any(q["annotation_query"] for q in record["detection_diagnostics"]["queries"]))
            reports.append(summary)
            if workers == 4:
                self.assertEqual(len(list(output.glob("workers/*/worker-*.log"))), 4)
        self.assertEqual(reports[0], reports[1])
        # Saved settings restore score-only mode without running any models.
        result = self.evaluate(output, 1, extra=profile + ("--score-only",), inference_only=False,
            env={**self.env, "PDMC_TEST_FAIL_GPU": self.env["CUDA_VISIBLE_DEVICES"]})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads((output / "summary.json").read_text()), reports[1])

    def test_annotation_vocabulary_resume_keeps_subset_queries_and_rejects_old_reference_roots(self):
        (self.repo / "classes.txt").write_text("cat")
        profile = ("--pipeline-profile", "released-sam", "--target-selection", "label-semantic")
        expanded = profile + ("--detector-vocabulary", "annotated")
        output = self.root / "expanded-resume"
        result = self.evaluate(output, 1, limit=2, extra=expanded)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        config_bytes = (output / "run_config.json").read_bytes()
        first = (output / "targets" / self.rows[0]["id"] / "result.json").read_bytes()
        result = self.evaluate(output, 2, extra=expanded)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((output / "run_config.json").read_bytes(), config_bytes)
        self.assertEqual((output / "targets" / self.rows[0]["id"] / "result.json").read_bytes(), first)
        result = self.evaluate(output, 1, extra=profile)
        self.assertEqual(result.returncode, 2)
        self.assertIn("different run configuration", result.stderr)
        baseline = self.root / "baseline-references"
        result = self.evaluate(baseline, 1, extra=profile)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        baseline_config = (baseline / "run_config.json").read_bytes()
        result = self.evaluate(baseline, 1, extra=expanded)
        self.assertEqual(result.returncode, 2)
        self.assertEqual((baseline / "run_config.json").read_bytes(), baseline_config)
        result = self.evaluate(self.root / "wrong-expanded-references", 1,
            extra=expanded + ("--reference-dir", str(baseline / "visible_references")))
        self.assertEqual(result.returncode, 2)
        self.assertIn("different run configuration", result.stderr)

    def test_annotation_detector_configuration_is_checked_in_workers_and_scoring(self):
        expanded = ("--pipeline-profile", "released-sam", "--target-selection", "label-semantic",
                    "--detector-vocabulary", "annotated")
        output = self.root / "expanded-worker-drift"
        result = self.evaluate(output, 2, limit=2, extra=expanded,
                               env={**self.env, "PDMC_TEST_DETECTOR_DRIFT": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((output / "summary.json").exists())
        self.assertTrue(any("Worker settings differ" in p.read_text() for p in output.glob("workers/*/worker-*.log")))
        output = self.root / "expanded-reference-drift"
        result = self.evaluate(output, 1, limit=2, extra=expanded)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary_bytes = (output / "summary.json").read_bytes()
        marker = output / "visible_references/reference_config.json"
        reference = json.loads(marker.read_text())
        reference["detector_vocabulary"]["version"] = -1
        marker.write_text(json.dumps(reference))
        result = self.evaluate(output, 1, limit=2, extra=expanded + ("--score-only",), inference_only=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("reference configuration", result.stderr)
        self.assertEqual((output / "summary.json").read_bytes(), summary_bytes)

    def test_empty_completion_resume_keeps_completed_results_and_counts_failed_targets(self):
        (self.repo / "classes.txt").write_text("cat")
        gray = self.repo / "images/gray_wallpaper.jpeg"
        gray.parent.mkdir()
        Image.new("RGB", (12, 12)).save(gray)
        profile = ("--pipeline-profile", "released-sam", "--target-selection", "label-semantic",
                   "--detector-vocabulary", "annotated")
        env = {**self.env, "PDMC_TEST_REAL_EMPTY_COMPLETION": "1",
               "CUDA_VISIBLE_DEVICES": "GPU-A,GPU-B,GPU-C,GPU-D"}
        reports = []
        for workers in (1, 4):
            output = self.root / f"resume-empty-completion-{workers}"
            result = self.evaluate(output, 1, limit=2, extra=profile)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            config_bytes = (output / "run_config.json").read_bytes()
            existing = {path: path.read_bytes() for path in output.glob("targets/*/*/*") if path.is_file()}
            result = self.evaluate(output, workers, extra=profile, env=env, inference_only=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((output / "run_config.json").read_bytes(), config_bytes)
            for path, content in existing.items():
                self.assertEqual(path.read_bytes(), content)
            summary = json.loads((output / "summary.json").read_text())
            stat = summary["statistics"]["Overall"]
            self.assertEqual((stat["successes"], stat["failures"]), (2, 4))
            self.assertEqual(stat["failure_reasons"], {"completion_not_detected": 4})
            self.assertEqual(stat["failure_stage_counts"]["completion_segmentation"], 4)
            for row in self.rows[2:]:
                record = json.loads((output / "targets" / row["id"] / "result.json").read_text())
                self.assertEqual(record["status"], "no_completion")
                self.assertEqual(record["iterations"], 1)
                self.assertEqual(record["diagnostics"]["iterations"][0]["completion_segmentation_failure"],
                                 "empty_mask_array")
                self.assertFalse((output / "targets" / row["id"] / "completion.png").exists())
            reports.append(summary)
        self.assertEqual(reports[0], reports[1])


if __name__ == "__main__":
    unittest.main()
