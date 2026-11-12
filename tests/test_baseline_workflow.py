"""The baseline must stop before expensive work or registration when a gate fails."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("baseline_workflow", ROOT / "scripts/run_baseline_workflow.py")
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


class BaselineWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = workflow.arguments(["--cache-dir", "/scratch/amodal-cache",
                                        "--output-dir", str(self.root / "full"), "--run-only"])

    def test_full_and_smoke_expand_the_declared_contract_and_four_workers(self):
        args = workflow.arguments(["--cache-dir", "/scratch/amodal-cache",
                                   "--output-dir", "output/test-baseline", "--run-only"])
        stages = dict(workflow.steps(args))
        self.assertEqual(stages["gpu-inventory"][0], "/scratch/amodal-cache/conda-env/bin/python")
        for name in ("smoke", "full"):
            command = stages[name]
            self.assertEqual(command[:3], ["bash", "run_amodal_slurm.sh", "--local"])
            self.assertEqual(command[command.index("--benchmark-workers") + 1], "4")
            self.assertEqual(command[command.index("--benchmark-target-selection") + 1], "label-semantic")
            self.assertEqual(command[command.index("--benchmark-detector-vocabulary") + 1], "released")
            self.assertIn("--run-only", command)
        self.assertNotIn("--benchmark-limit", stages["full"])
        self.assertEqual(stages["smoke"][stages["smoke"].index("--benchmark-limit") + 1], "8")
        self.assertIn("--require-full", stages["verify-full"])
        self.assertIn("--require-provenance", stages["verify-full"])

    @unittest.skipIf(os.name == "nt", "The real Bash parser requires Linux/WSL")
    def test_generated_commands_pass_real_runner_parser_for_both_cache_modes(self):
        for run_only in (False, True):
            options = ["--cache-dir", "/scratch/amodal-cache", "--output-dir", "output/test-baseline"]
            if run_only:
                options.append("--run-only")
            stages = dict(workflow.steps(workflow.arguments(options)))
            self.assertEqual("setup" in stages, not run_only)
            for name, command in stages.items():
                if command[:2] != ["bash", "run_amodal_slurm.sh"]:
                    continue
                with self.subTest(run_only=run_only, stage=name):
                    result = subprocess.run(
                        ["bash", "-c", 'source "$1"; shift; parse_args "$@"',
                         "baseline-parser", str(ROOT / "run_amodal_slurm.sh"), *command[2:]],
                        cwd=ROOT, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_run_only_starts_with_gpu_validation_without_provisioning(self):
        stages = workflow.steps(self.args)
        self.assertEqual(stages[0][0], "gpu-inventory")
        self.assertNotIn("setup", dict(stages))
        for _, command in stages:
            self.assertNotIn("--setup-only", command)

    def write_smoke(self, *, successes=1, complete_metrics=True):
        folder = Path(self.args.smoke_output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "summary.json").write_text(json.dumps({"evaluation_performed": True,
            "statistics": {"Overall": {"successes": successes, "CLIP_n": successes}}}))
        target = folder / "targets/VG/example"
        target.mkdir(parents=True, exist_ok=True)
        values = {"CLIP": 25., "LPIPS": .4, "Feature Similarity": .5, "SSIM": .6}
        if not complete_metrics:
            values["LPIPS"] = None
        (target / "metrics.json").write_text(json.dumps({"metrics": values}))

    def test_smoke_requires_generated_completion_and_all_four_metrics(self):
        self.write_smoke(successes=0)
        with self.assertRaisesRegex(ValueError, "did not generate"):
            workflow.require_scored_completion(self.args.smoke_output_dir)
        self.write_smoke(complete_metrics=False)
        with self.assertRaisesRegex(ValueError, "all four metrics"):
            workflow.require_scored_completion(self.args.smoke_output_dir)
        self.write_smoke()
        workflow.require_scored_completion(self.args.smoke_output_dir)

    @unittest.skipIf(os.name == "nt", "Workflow uses the cluster's POSIX run locks")
    def test_failed_gpu_validation_never_starts_inference(self):
        names = []
        def fake_stage(name, command, log, environment):
            names.append(name)
            log.write_text("test stage\n")
            if name == "cuda-extension":
                raise RuntimeError("CUDA failed")
        with patch.object(workflow, "run_stage", side_effect=fake_stage):
            with self.assertRaisesRegex(RuntimeError, "CUDA failed"):
                workflow.run_workflow(self.args)
        self.assertEqual(names, ["gpu-inventory", "cuda-extension"])
        records = list((self.root / "full/workflow").glob("*/workflow.json"))
        self.assertEqual(json.loads(records[0].read_text())["status"], "failed")

    @unittest.skipIf(os.name == "nt", "Workflow uses the cluster's POSIX run locks")
    def test_failed_full_verification_never_registers(self):
        names = []
        def fake_stage(name, command, log, environment):
            names.append(name)
            log.write_text("test stage\n")
            if name == "smoke":
                self.write_smoke()
            if name == "verify-full":
                raise RuntimeError("Full artifact mismatch")
        with patch.object(workflow, "run_stage", side_effect=fake_stage):
            with self.assertRaisesRegex(RuntimeError, "Full artifact mismatch"):
                workflow.run_workflow(self.args)
        self.assertIn("full", names)
        self.assertNotIn("register", names)

    @unittest.skipIf(os.name == "nt", "Workflow uses the cluster's POSIX run locks")
    def test_no_success_smoke_prevents_full_run(self):
        names = []
        def fake_stage(name, command, log, environment):
            names.append(name)
            log.write_text("test stage\n")
            if name == "smoke":
                self.write_smoke(successes=0)
        with patch.object(workflow, "run_stage", side_effect=fake_stage):
            with self.assertRaisesRegex(ValueError, "did not generate"):
                workflow.run_workflow(self.args)
        self.assertNotIn("full", names)
        self.assertNotIn("register", names)

    @unittest.skipIf(os.name == "nt", "Workflow uses the cluster's POSIX run locks")
    def test_success_preserves_stage_evidence_and_registers_last(self):
        names = []
        def fake_stage(name, command, log, environment):
            names.append(name)
            log.write_text("test stage\n")
            if name == "smoke":
                self.write_smoke()
        with patch.object(workflow, "run_stage", side_effect=fake_stage):
            workflow.run_workflow(self.args)
        self.assertEqual(names[-2:], ["verify-full", "register"])
        record = json.loads(next((self.root / "full/workflow").glob("*/workflow.json")).read_text())
        self.assertEqual(record["status"], "complete")
        self.assertEqual(len(record["stages"]), 9)
        self.assertTrue(all(stage["log_sha256"] for stage in record["stages"]))


if __name__ == "__main__":
    unittest.main()
