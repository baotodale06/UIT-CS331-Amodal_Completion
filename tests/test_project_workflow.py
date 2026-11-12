"""Contracts, offline smoke, independent verification, and durable research records."""
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from amodal_benchmark.config import effective_config, load_config, validate_config
from amodal_benchmark.dataset import read_official, write_json
from amodal_benchmark.knowledge import init_notes
from amodal_benchmark.planning import render_plan, slurm_command
from amodal_benchmark.parallel import output_lock
from amodal_benchmark.provenance import environment_snapshot, execution_snapshot
from amodal_benchmark.registry import compare_baseline, register_run
from amodal_benchmark.smoke import build_smoke_run
from amodal_benchmark.verify import verify_artifacts, verify_run

ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_presets_resolve_and_expansion_changes_one_scientific_factor(self):
        baseline = load_config(ROOT / "configs/baseline.json")
        expanded = load_config(ROOT / "configs/annotated.json")
        changed = [key for key in baseline["inference"] if baseline["inference"][key] != expanded["inference"][key]]
        self.assertEqual(changed, ["detector_vocabulary"])
        self.assertEqual(expanded["execution"]["workers"], 4)
        for path in (ROOT / "configs").glob("*.json"):
            self.assertIsInstance(load_config(path), dict)

    def test_unknown_fields_unsupported_tasks_and_false_full_coverage_are_rejected(self):
        base = load_config(ROOT / "configs/baseline.json")
        for section, field, value in [("project", "task", "binary-classification"),
                                      ("dataset", "limit", 2), ("dataset", "allow_incomplete", True),
                                      ("inference", "seed", True), ("execution", "workers", 0),
                                      ("dataset", "sources", ["VG"]), ("inference", "mc_timestep", 50)]:
            config = copy.deepcopy(base)
            config[section][field] = value
            with self.subTest(section=section, field=field), self.assertRaises(ValueError):
                validate_config(config)
        base["inference"]["typo"] = 1
        with self.assertRaisesRegex(ValueError, "inference"):
            validate_config(base)

    def test_duplicate_json_keys_and_inheritance_cycles_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"schema": 1, "schema": 1}')
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_config(path)
            path.write_text('{"extends": "config.json"}')
            with self.assertRaisesRegex(ValueError, "cycle"):
                load_config(path)

    def test_cli_overrides_are_recorded_and_cannot_mislabel_a_full_run(self):
        from evaluate_open_world import parse_args
        args = parse_args(["--config", str(ROOT / "configs/smoke.json"), "--output-dir", "test-output",
                           "--seed", "9", "--workers", "2"])
        config = effective_config(load_config(args.config), args)
        self.assertEqual(config["inference"]["seed"], 9)
        self.assertEqual(config["execution"]["workers"], 2)
        with self.assertRaisesRegex(ValueError, "Full evaluation"):
            parse_args(["--config", str(ROOT / "configs/baseline.json"), "--output-dir", "test-output", "--limit", "2"])

    def test_cluster_plan_requests_four_gpus_before_script_and_four_workers_after(self):
        config = load_config(ROOT / "configs/annotated.json")
        command = slurm_command(config, "configs/annotated.json", "output/run with spaces", "/scratch/cache")
        script_index = command.index("run_amodal_slurm.sh")
        self.assertLess(command.index("--gres=gpu:4"), script_index)
        self.assertGreater(command.index("--benchmark-workers"), script_index)
        self.assertEqual(command[command.index("--benchmark-workers") + 1], "4")
        self.assertGreater(command.index("--cpus"), script_index)
        self.assertEqual(command[command.index("--cpus") + 1], "16")
        self.assertIn("'output/run with spaces'", render_plan(command))
        with self.assertRaisesRegex(ValueError, "Linux paths"):
            slurm_command(config, "C:/configs/baseline.json", "output/run", "/cache")

    def test_notes_initialization_preserves_existing_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plan.md").write_text("Existing hypothesis")
            created = init_notes(root, "research-project")
            self.assertNotIn("plan.md", created)
            self.assertEqual((root / "plan.md").read_text(), "Existing hypothesis")
            self.assertEqual(init_notes(root, "research-project"), [])


class ArtifactWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.root = Path(cls.fixture.name)
        build_smoke_run(cls.root)
        cls.rows = json.loads((cls.root / "synthetic-targets.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.case = Path(self.temporary.name)
        shutil.copytree(self.root, self.case, dirs_exist_ok=True)
        self.run = self.case / "run"

    def verify(self, **kwargs):
        return verify_artifacts(self.run, self.case / "data", self.rows, **kwargs)

    def test_smoke_is_independently_verified_and_resume_keeps_results(self):
        report = self.verify(require_provenance=True)
        self.assertTrue(report["verified"])
        self.assertFalse(report["full_evaluation"])
        self.assertEqual(report["statistics"]["Overall"]["LPIPS"], .625)
        execution = json.loads((self.run / "execution.json").read_text())
        self.assertEqual(execution["new_or_updated_target_ids"], [])

    def test_changed_input_image_is_rejected(self):
        from PIL import Image
        Image.new("RGB", (32, 32), "red").save(self.case / "data" / self.rows[0]["image"])
        with self.assertRaisesRegex(ValueError, "input"):
            self.verify()

    def test_changed_completion_is_rejected(self):
        (self.run / "targets" / self.rows[0]["id"] / "completion.png").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "completion"):
            self.verify()

    def test_sam_reference_artifacts_cannot_be_relabelled_as_lisa(self):
        path = self.run / "run_config.json"
        config = json.loads(path.read_text())
        config["pipeline_profile"] = "benchmark"
        write_json(path, config)
        path = self.run / "summary.json"
        summary = json.loads(path.read_text())
        summary["reference_backend"] = "lisa"
        write_json(path, summary)
        with self.assertRaisesRegex(ValueError, "SAM references"):
            verify_artifacts(self.run, self.case / "data", self.rows,
                             reference_dir=self.run / "visible_references")

    def test_metric_cache_disagreement_is_rejected(self):
        path = self.run / "targets" / self.rows[0]["id"] / "metrics.json"
        data = json.loads(path.read_text())
        data["metrics"]["LPIPS"] = .8
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, "metric"):
            self.verify()

    def test_reference_inventory_cannot_omit_artifacts(self):
        path = self.run / "visible_references/records" / Path(self.rows[0]["reference"]).with_suffix(".json")
        data = json.loads(path.read_text())
        data["artifacts"] = {}
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.verify()

    def test_resolved_contract_must_match_actual_inference(self):
        path = self.run / "resolved_config.json"
        config = json.loads(path.read_text())
        config["inference"]["max_iter"] = 1
        write_json(path, config)
        with self.assertRaisesRegex(ValueError, "max_iter"):
            self.verify()

    def test_source_snapshot_corruption_is_rejected(self):
        execution = json.loads((self.run / "execution.json").read_text())
        path = self.run / "provenance" / execution["id"] / "source/amodal_benchmark/inference.py"
        path.write_text("changed source")
        with self.assertRaisesRegex(ValueError, "Source snapshot"):
            self.verify()

    def test_attempt_configuration_corruption_is_rejected(self):
        execution = json.loads((self.run / "execution.json").read_text())
        path = self.run / "provenance" / execution["id"] / "resolved_config.json"
        config = json.loads(path.read_text())
        config["execution"]["workers"] = 99
        write_json(path, config)
        with self.assertRaisesRegex(ValueError, "configuration snapshot"):
            self.verify()

    def test_stale_summary_failure_statistics_are_rejected_before_normalization(self):
        path = self.run / "summary.json"
        saved = json.loads(path.read_text())
        for key, value in (("failure_rate", 0), ("status_counts", {"success": 2}),
                           ("failure_reasons", {"invented": 1})):
            with self.subTest(key=key):
                changed = copy.deepcopy(saved)
                changed["statistics"]["Overall"][key] = value
                write_json(path, changed)
                with self.assertRaisesRegex(ValueError, "summary Overall/" + key):
                    self.verify()
        write_json(path, saved)

    def test_synthetic_results_cannot_pass_real_dataset_verification_or_registration(self):
        with self.assertRaisesRegex(ValueError, "full"):
            self.verify(require_full=True)
        with self.assertRaisesRegex(ValueError, "outside"):
            verify_run(self.run, ROOT / "tests/fixtures")
        with self.assertRaisesRegex(ValueError, "Smoke"):
            register_run(self.run, self.case / "data", self.case / "experiments.csv")
        self.assertFalse((self.case / "experiments.csv").exists())

    def test_adopting_wrong_config_preserves_existing_resolved_contract(self):
        config = load_config(self.run / "resolved_config.json")
        config["inference"]["seed"] = 1
        before = (self.run / "resolved_config.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "differs"):
            with execution_snapshot(self.run, SimpleNamespace(), self.rows, config):
                self.fail("Wrong config was adopted")
        self.assertEqual((self.run / "resolved_config.json").read_bytes(), before)


class ProvenanceAndRegistryTests(unittest.TestCase):
    def test_registration_cannot_race_an_active_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with output_lock(root), self.assertRaisesRegex(ValueError, "already using output"):
                register_run(root, ROOT / "tests/fixtures", root / "experiments.csv")
            self.assertFalse((root / "experiments.csv").exists())

    def test_environment_snapshot_does_not_capture_credentials(self):
        with patch.dict(os.environ, {"HF_TOKEN": "secret-sentinel", "AWS_SECRET_ACCESS_KEY": "another-secret"}):
            record = json.dumps(environment_snapshot())
        self.assertNotIn("secret-sentinel", record)
        self.assertNotIn("another-secret", record)
        self.assertNotIn("HF_TOKEN", record)

    def test_failed_full_execution_is_retained_and_registration_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "failed"
            output.mkdir()
            rows, _ = read_official(ROOT / "tests/fixtures")
            config = load_config(ROOT / "configs/baseline.json")
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                with execution_snapshot(output, SimpleNamespace(workers="4"), rows, config):
                    raise RuntimeError("fixture failure")
            entry = register_run(output, ROOT / "tests/fixtures", root / "experiments.csv")
            self.assertEqual(entry["status"], "failed")
            register_run(output, ROOT / "tests/fixtures", root / "experiments.csv")
            with (root / "experiments.csv").open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)
            self.assertIn("failed", (root / "experiment_report.md").read_text())
            with self.assertRaisesRegex(ValueError, "different metadata"):
                register_run(output, ROOT / "tests/fixtures", root / "experiments.csv", hypothesis="changed")

    def test_complete_registration_requires_full_artifact_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, _ = read_official(ROOT / "tests/fixtures")
            write_json(root / "execution.json", {"id": "test", "status": "complete", "target_ids": [r["id"] for r in rows]})
            write_json(root / "resolved_config.json", load_config(ROOT / "configs/baseline.json"))
            with patch("amodal_benchmark.registry.verify_run", side_effect=ValueError("missing completion")) as verify:
                with self.assertRaisesRegex(ValueError, "missing completion"):
                    register_run(root, ROOT / "tests/fixtures", root / "experiments.csv")
                self.assertTrue(verify.call_args.kwargs["require_full"])
                self.assertTrue(verify.call_args.kwargs["require_provenance"])
            self.assertFalse((root / "experiments.csv").exists())

    def test_failed_attempt_cannot_replace_full_coverage_with_duplicate_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "failed"
            output.mkdir()
            rows, _ = read_official(ROOT / "tests/fixtures")
            rows[-1] = rows[0]
            with self.assertRaises(RuntimeError):
                with execution_snapshot(output, SimpleNamespace(), rows,
                                        load_config(ROOT / "configs/baseline.json")):
                    raise RuntimeError("fixture failure")
            with self.assertRaisesRegex(ValueError, "official target"):
                register_run(output, ROOT / "tests/fixtures", root / "experiments.csv")
            self.assertFalse((root / "experiments.csv").exists())

    def test_failed_attempt_registration_checks_preserved_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "failed"
            output.mkdir()
            rows, _ = read_official(ROOT / "tests/fixtures")
            with self.assertRaises(RuntimeError):
                with execution_snapshot(output, SimpleNamespace(), rows,
                                        load_config(ROOT / "configs/baseline.json")):
                    raise RuntimeError("fixture failure")
            execution = json.loads((output / "execution.json").read_text())
            execution["error"] = "silently rewritten reason"
            write_json(output / "execution.json", execution)
            with self.assertRaisesRegex(ValueError, "Execution snapshot differs"):
                register_run(output, ROOT / "tests/fixtures", root / "experiments.csv")


class BaselineComparisonTests(unittest.TestCase):
    def test_lisa_comparison_requires_frozen_references_but_sam_labels_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous, current = root / "baseline", root / "current"
            # Isolate comparison policy after each independent run has passed
            # verification; there is no need to fabricate 2,565 model outputs.
            first = {"data_identity": {"targets_sha256": "targets", "inputs_sha256": "inputs",
                                       "references_sha256": "new-references"}}
            second = {"data_identity": {**first["data_identity"], "references_sha256": "old-references"}}
            for preset in ("lisa-reference.json", "baseline.json"):
                config = load_config(ROOT / "configs" / preset)
                write_json(previous / "resolved_config.json", config)
                config["inference"]["seed"] = 1
                config["experiment"]["changed_factor"] = "inference.seed"
                write_json(current / "resolved_config.json", config)
                for folder, seed in ((previous, 0), (current, 1)):
                    write_json(folder / "run_config.json", {"seed": seed})
                    write_json(folder / "metric_config.json", {"alignment": "released"})
                with self.subTest(preset=preset), patch("amodal_benchmark.registry.verify_run",
                                                       side_effect=[first, second]):
                    if preset == "lisa-reference.json":
                        with self.assertRaisesRegex(ValueError, "frozen references"):
                            compare_baseline(current, previous, root)
                    else:
                        self.assertEqual(compare_baseline(current, previous, root),
                                         "same_inputs_different_references")


if __name__ == "__main__":
    unittest.main()
