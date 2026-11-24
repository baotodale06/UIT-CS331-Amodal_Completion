"""A timeout may resume immutable results; application failures must not loop."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from amodal_benchmark.dataset import write_json
from scripts import continue_shape_v2 as controller


class ShapeContinuationTests(unittest.TestCase):
    def invoke(self, study, state, submit, method="shape-v2"):
        previous = dict(job_id="123", state=state, exit_code="0:0", elapsed="1:30:00", allocation="gres/gpu=4")
        with patch("sys.argv", ["continue", "--study", str(study), "--job-id", "123"]), \
                patch.object(controller, "accounting", return_value=previous), \
                patch.object(controller, "verify_frozen_source"), \
                patch.object(controller, "submit", submit), patch("builtins.print"):
            controller.main(method)

    def test_calibrated_timeout_preserves_closed_artifacts_and_uses_own_runners(self):
        with TemporaryDirectory() as temporary:
            study = Path(temporary)
            write_json(study/"benchmark/targets/VG/one/result.json", dict(status="success"))
            image = study/"benchmark/targets/VG/one/shape-calibrated-probability/iteration-00/selected.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"synthetic retained pixels")
            write_json(study/"benchmark/targets/VG/incomplete/partial.json", dict(complete=False))
            write_json(study/"benchmark/run_config.json", dict(method="shape-calibrated"))
            write_json(study/"policy-freeze.json", dict(frozen=True))
            calls = []
            def submit(command, cwd):
                calls.append(command)
                return str(456+len(calls))
            self.invoke(study, "TIMEOUT", submit, "shape-calibrated")
            self.assertIn("--nodelist=gebru", calls[0])
            self.assertIn("scripts/run_shape_calibrated_evaluation.sh", calls[0])
            self.assertIn("scripts/run_shape_calibrated_continuation.sh", calls[1])
            retained = json.loads((study/"preserved-before-resume-123.json").read_bytes())["files"]
            self.assertEqual(len(retained), 2)
            self.assertIn(image.relative_to(study/"benchmark").as_posix(), retained)
            self.invoke(study, "TIMEOUT", submit, "shape-calibrated")
            self.assertEqual(len(calls), 2)

    def test_timeout_preserves_records_and_submits_only_one_continuation(self):
        with TemporaryDirectory() as temporary:
            study = Path(temporary)
            write_json(study/"benchmark/targets/A/1/result.json", dict(status="success"))
            write_json(study/"benchmark/run_config.json", dict(method="shape-v2"))
            write_json(study/"policy-freeze.json", dict(frozen=True))
            calls = []
            def submit(command, cwd):
                calls.append(command)
                return str(456+len(calls))
            self.invoke(study, "TIMEOUT", submit)
            self.assertEqual(len(calls), 2)
            self.assertIn("--gres=gpu:quadro_rtx_6000:4", calls[0])
            self.assertEqual(calls[0][-3:], ["--benchmark-workers", "4", "full"])
            self.assertIn("--dependency=afterany:457", calls[1])
            receipt = json.loads((study/"preserved-before-resume-123.json").read_bytes())
            self.assertEqual(len(receipt["files"]), 1)
            self.invoke(study, "TIMEOUT", submit)
            self.assertEqual(len(calls), 2)

    def test_failed_job_does_not_automatically_retry_or_cancel_other_work(self):
        with TemporaryDirectory() as temporary:
            study = Path(temporary)
            def submit(*args):
                self.fail("Application failures must not submit jobs")
            self.invoke(study, "FAILED", submit)
            receipt = json.loads((study/"continuations.json").read_bytes())
            self.assertEqual(receipt["events"][0]["action"], "needs_review")

    def test_completed_benchmark_with_failed_gain_gate_is_not_repeated(self):
        with TemporaryDirectory() as temporary:
            study = Path(temporary)
            write_json(study/"full-audit.json", dict(verified=True, targets=2565,
                gate=dict(eligible_for_success_claim=False)))
            self.invoke(study, "COMPLETED", lambda *args:self.fail("Do not rerun completed benchmarks"))
            receipt = json.loads((study/"continuations.json").read_bytes())
            self.assertFalse(receipt["events"][0]["four_metric_gate"])


if __name__ == "__main__":
    unittest.main()
