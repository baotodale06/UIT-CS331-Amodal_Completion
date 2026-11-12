"""Keep paper comparisons and controlled runs honest about coverage and seeds."""
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sample_run(seed=0):
    stat = {"targets": 2, "successes": 1, "failures": 1,
            "missing_images": 0, "reference_failures": 0,
            "CLIP": 30., "CLIP_n": 1, "LPIPS": .625, "LPIPS_n": 2,
            "Feature Similarity": .25, "Feature Similarity_n": 2,
            "SSIM": .375, "SSIM_n": 2}
    return {"run_dir": f"seed-{seed}",
            "run_config": {"seed": seed, "boundary_check": "paper",
                           "target_selection": "reference-best", "mc_timestep": 35},
            "metric_config": {"alignment": "released"},
            "summary": {"selected_targets": 2, "full_image_coverage": False,
                        "evaluation_performed": True, "metric_alignment": "released",
                        "data_identity": {"targets_sha256": "targets", "inputs_sha256": "inputs",
                                          "references_sha256": "references"},
                        "statistics": {"VG": copy.deepcopy(stat), "Overall": copy.deepcopy(stat)}}}


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("amodal_benchmark.reproduction"),
                             "Paper comparison and seed reporting are missing")
        from amodal_benchmark import reproduction
        self.r = reproduction

    def test_comparison_keeps_clip_denominator_and_marks_partial_coverage(self):
        report = self.r.paper_comparison(sample_run()["summary"])
        vg = report["sources"]["VG"]
        self.assertFalse(vg["complete_source"])
        self.assertEqual(vg["metrics"]["CLIP_success"]["n"], 1)
        self.assertEqual(vg["metrics"]["CLIP_success"]["comparison_status"], "paper_denominator_unknown")
        self.assertEqual(vg["metrics"]["LPIPS"]["comparison_status"], "subset")
        self.assertAlmostEqual(vg["metrics"]["LPIPS"]["delta"], .047)
        self.assertAlmostEqual(vg["failure_rate"]["delta_pp"], 10.)
        self.assertFalse(report["exact_table_reproduction"])

    def test_complete_source_still_exposes_missing_reference_denominators(self):
        summary = sample_run()["summary"]
        vg = summary["statistics"]["VG"]
        vg.update(targets=1294, successes=823, failures=470, reference_failures=1,
                  LPIPS_n=1293)
        report = self.r.paper_comparison(summary)["sources"]["VG"]
        self.assertTrue(report["complete_source"])
        self.assertEqual(report["metrics"]["LPIPS"]["comparison_status"], "excluded_targets")
        self.assertEqual(report["failure_rate"]["n"], 1293)

    def test_inference_only_and_alternate_alignment_are_not_table_scores(self):
        summary = sample_run()["summary"]
        summary["evaluation_performed"] = False
        report = self.r.paper_comparison(summary)
        self.assertIsNone(report["sources"]["VG"]["metrics"]["LPIPS"]["observed"])
        summary["evaluation_performed"] = True
        summary["metric_alignment"] = "original-frame"
        report = self.r.paper_comparison(summary)
        self.assertEqual(report["sources"]["VG"]["metrics"]["LPIPS"]["comparison_status"],
                         "different_alignment")

    def test_seed_report_uses_sample_std_and_does_not_pool_protocols(self):
        first, second, different = sample_run(0), sample_run(1), sample_run(2)
        second["summary"]["statistics"]["VG"]["LPIPS"] = .875
        different["run_config"]["boundary_check"] = "released"
        groups = self.r.seed_statistics([first, second, different])
        self.assertEqual(len(groups), 2)
        group = next(g for g in groups if g["seeds"] == [0, 1])
        stat = group["statistics"]["VG"]["LPIPS"]
        self.assertEqual(stat["mean"], .75)
        self.assertAlmostEqual(stat["std"], .1767766952966369)
        self.assertEqual(stat["n"], 2)

    def test_seed_report_requires_the_same_inputs_references_and_unique_seeds(self):
        first, second = sample_run(0), sample_run(1)
        second["summary"]["data_identity"]["references_sha256"] = "changed-reference"
        self.assertEqual(len(self.r.seed_statistics([first, second])), 2)
        second = sample_run(1)
        second["summary"].pop("data_identity")
        self.assertEqual(len(self.r.seed_statistics([first, second])), 2)
        with self.assertRaisesRegex(ValueError, "seed"):
            self.r.seed_statistics([first, sample_run(0)])

    def test_identity_is_order_independent_and_detects_reference_changes(self):
        rows = [{"id": "VG/a", "label": "cat", "source": "VG", "input_sha256": "one"},
                {"id": "VG/b", "label": "dog", "source": "VG", "input_sha256": "two"}]
        audit = {"reference_records": {r["id"]: {"artifacts": {"mask": r["id"], "crop": "crop"}}
                                       for r in rows}}
        original = self.r.data_identity(rows, audit)
        self.assertEqual(original, self.r.data_identity(list(reversed(rows)), audit))
        audit["reference_records"]["VG/a"]["artifacts"]["crop"] = "changed"
        self.assertNotEqual(original["references_sha256"], self.r.data_identity(rows, audit)["references_sha256"])
        self.assertIsNone(self.r.data_identity(rows, {"reference_records": {}})["references_sha256"])

    def test_loader_rejects_stale_summary_and_duplicate_target_rows(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            run = sample_run()
            for name, value in (("summary", run["summary"]), ("run_config", run["run_config"]),
                                ("metric_config", run["metric_config"])):
                (folder / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
            rows = [["VG/a", "VG", "success", "30", ".25", ".5", ".75"],
                    ["VG/b", "VG", "not_detected", "", "1", "0", "0"]]
            def save_csv():
                with (folder / "per_target.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["id", "source", "status", "CLIP", "LPIPS", "Feature Similarity", "SSIM"])
                    writer.writerows(rows)
            save_csv()
            loaded = self.r.load_run(folder)
            self.assertEqual(loaded["summary"]["statistics"]["Overall"]["LPIPS"], .625)
            rows[0][4] = ".75"
            save_csv()
            with self.assertRaisesRegex(ValueError, "summary"):
                self.r.load_run(folder)
            rows[1][0] = "VG/a"
            save_csv()
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                self.r.load_run(folder)

    def test_report_cli_audits_saved_results_without_modifying_them(self):
        script = ROOT / "compare_open_world.py"
        self.assertTrue(script.exists(), "Saved-run comparison CLI is missing")
        source = ROOT / "output/pdmc-paper-v2"
        if not source.is_dir():
            self.skipTest("Saved full-run reports are not installed")
        before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
        with tempfile.TemporaryDirectory() as td:
            result = subprocess.run([sys.executable, str(script), "--run-dir", str(source),
                                     "--output-dir", td], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((Path(td) / "comparison.json").read_text(encoding="utf-8"))
            overall = report["runs"][0]["comparison"]["sources"]["Overall"]
            self.assertEqual(overall["metrics"]["CLIP_success"]["n"], 1413)
            self.assertEqual(overall["metrics"]["LPIPS"]["n"], 2564)
            self.assertIn("CLIP_success", (Path(td) / "comparison.md").read_text(encoding="utf-8"))
        self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()})

    def test_loader_validates_completion_flags_and_failure_stages(self):
        from amodal_benchmark.protocol import aggregate
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            rows = [{"id": "VG/a", "source": "VG", "status": "success",
                     "iteration_limit_reached": False},
                    {"id": "VG/b", "source": "VG", "status": "success",
                     "iteration_limit_reached": True},
                    {"id": "VG/c", "source": "VG", "status": "success"},
                    {"id": "VG/d", "source": "VG", "status": "not_detected",
                     "failure_reason": "segmentation_unavailable"}]
            summary = {"selected_targets": len(rows), "statistics": aggregate(rows, detailed=True)}
            (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (folder / "run_config.json").write_text("{}", encoding="utf-8")
            fields = ["id", "source", "status", "failure_reason", "iteration_limit_reached"]
            def save_csv():
                with (folder / "per_target.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fields)
                    writer.writeheader()
                    writer.writerows(rows)
            save_csv()
            stat = self.r.load_run(folder)["summary"]["statistics"]["Overall"]
            self.assertEqual((stat["completed_outputs"], stat["iteration_limited_outputs"],
                              stat["completion_state_unknown"]), (1, 1, 1))
            self.assertEqual(stat["initial_segmentation_failures"], 1)
            rows[0]["iteration_limit_reached"] = "True"
            save_csv()
            with self.assertRaisesRegex(ValueError, "completed_outputs"):
                self.r.load_run(folder)
            rows[0]["iteration_limit_reached"] = "invalid"
            save_csv()
            with self.assertRaisesRegex(ValueError, "iteration-limit flag"):
                self.r.load_run(folder)
            fields.remove("iteration_limit_reached")
            for row in rows:
                row.pop("iteration_limit_reached", None)
            save_csv()
            with self.assertRaisesRegex(ValueError, "summary"):
                self.r.load_run(folder)


class ControlPlanTests(unittest.TestCase):
    def setUp(self):
        self.script = ROOT / "run_open_world_controls.py"
        self.assertTrue(self.script.exists(), "Controlled-run launcher is missing")

    def test_plan_is_dry_by_default_and_freezes_references_across_seeds(self):
        with tempfile.TemporaryDirectory(prefix="controls with spaces ") as td:
            root = Path(td)
            output = root / "experiments"
            result = subprocess.run([sys.executable, str(self.script), "--dataset-root", str(root / "dataset"),
                "--reference-dir", str(root / "fixed references"), "--cache-dir", str(root / "cache"),
                "--output-root", str(output), "--seeds", "0", "1", "--plan-dir", str(root / "plan")],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads((root / "plan/plan.json").read_text(encoding="utf-8"))
            self.assertEqual(len(plan["runs"]), 4)
            self.assertEqual({r["boundary_check"] for r in plan["runs"]}, {"paper", "released"})
            self.assertEqual(len({r["output_dir"] for r in plan["runs"]}), 4)
            for run in plan["runs"]:
                command = run["command"]
                self.assertEqual(command[command.index("--reference-dir") + 1], str(root / "fixed references"))
                self.assertEqual(command[command.index("--mc-timestep") + 1], "35")
                self.assertEqual(command[command.index("--max-iter") + 1], "5")
                self.assertNotIn("--benchmark-generate-references", command)
            self.assertFalse(output.exists())

    def test_execute_refuses_missing_references_before_launching_jobs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = subprocess.run([sys.executable, str(self.script), "--dataset-root", str(root),
                "--reference-dir", str(root / "missing"), "--cache-dir", str(root / "cache"),
                "--output-root", str(root / "experiments"), "--plan-dir", str(root / "plan"), "--execute"],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("reference", result.stderr.lower())
            self.assertFalse((root / "experiments").exists())

    @unittest.skipIf(os.name == "nt", "Execute the real Bash/Slurm submission boundary on Linux or WSL")
    def test_execute_chains_submitted_jobs_and_stops_after_submission_failure(self):
        import numpy as np
        from PIL import Image
        from amodal_benchmark.dataset import SOURCES, read_official, sha256
        from amodal_benchmark.references import save_reference
        with tempfile.TemporaryDirectory(prefix="control submission ") as td:
            root = Path(td)
            dataset, references = root / "dataset", root / "references"
            for source, (filename, *_) in SOURCES.items():
                path = dataset / "official" / source / filename
                path.parent.mkdir(parents=True)
                path.write_bytes((ROOT / "tests/fixtures/official" / source / filename).read_bytes())
            for row in read_official(dataset)[0][:2]:
                path = dataset / row["image"]
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (12, 12), "white").save(path)
                save_reference(references, row, np.full((12, 12, 3), 255, dtype=np.uint8),
                               np.ones((12, 12), dtype=bool), sha256(path), {})
            binary = root / "bin/sbatch"
            binary.parent.mkdir()
            binary.write_text("#!/usr/bin/env python3\nimport json, os, sys\nfrom pathlib import Path\n"
                "p=Path(os.environ['CONTROL_SUBMISSION_LOG'])\n"
                "rows=json.loads(p.read_text()) if p.exists() else []\n"
                "rows.append(sys.argv[1:]); p.write_text(json.dumps(rows))\n"
                "sys.exit(7) if os.environ.get('CONTROL_FAIL_SUBMISSION') == str(len(rows)) else None\n"
                "print(str(100+len(rows))+';test-cluster')\n", encoding="utf-8")
            binary.chmod(0o755)
            env = {**os.environ, "PATH": str(binary.parent) + os.pathsep + os.environ["PATH"],
                   "CONTROL_SUBMISSION_LOG": str(root / "submissions.json")}
            env.pop("SLURM_JOB_ID", None)
            command = [sys.executable, str(self.script), "--dataset-root", str(dataset),
                       "--reference-dir", str(references), "--cache-dir", str(root / "cache"),
                       "--output-root", str(root / "experiments"), "--plan-dir", str(root / "plan"),
                       "--experiments", "boundary-released", "matching-strict", "corrected", "--limit", "2", "--execute"]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            submissions = json.loads((root / "submissions.json").read_text())
            self.assertEqual(len(submissions), 3)
            self.assertNotIn("--dependency", submissions[0])
            self.assertIn("--dependency", submissions[1])
            self.assertEqual(submissions[1][submissions[1].index("--dependency") + 1], "afterok:101")
            self.assertEqual(submissions[2][submissions[2].index("--dependency") + 1], "afterok:102")
            env.update(CONTROL_FAIL_SUBMISSION="2", CONTROL_SUBMISSION_LOG=str(root / "failed-submissions.json"))
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
            plan = json.loads((root / "plan/plan.json").read_text())
            self.assertEqual([r["status"] for r in plan["runs"]], ["submitted", "failed", "planned"])


if __name__ == "__main__":
    unittest.main()
