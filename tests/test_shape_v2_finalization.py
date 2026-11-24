"""The export follower must not rerun inference or publish partial evidence."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from amodal_benchmark.dataset import sha256, write_json
from scripts import finalize_shape_v2 as finalizer


class FinalizationTests(unittest.TestCase):
    def test_resumed_job_follows_new_controller_once_without_gpu_request(self):
        with TemporaryDirectory() as folder:
            study = Path(folder)
            write_json(study / "continuations.json", dict(events=[dict(
                previous=dict(job_id="1"), action="resumed", next_job_id="3", controller_job_id="4")]))
            with patch.object(finalizer, "accounting", return_value=dict(state="COMPLETED", exit_code="0:0")), \
                    patch.object(finalizer, "submit", return_value="5") as submit:
                result = finalizer.finalize(study, "1", "2")
                self.assertEqual(finalizer.finalize(study, "1", "2"), result)
                submit.assert_called_once()
                command, cwd = submit.call_args.args
                self.assertIn("--dependency=afterany:4", command)
                self.assertFalse(any("gres" in argument for argument in command))
                self.assertEqual(command[-4:], ["--job-id", "3", "--controller-id", "4"])
                self.assertEqual(cwd, study / "operational-source")

    def test_application_failure_requires_review_without_export_or_submission(self):
        with TemporaryDirectory() as folder:
            study = Path(folder)
            write_json(study / "continuations.json", dict(events=[dict(
                previous=dict(job_id="1"), action="needs_review", reason="Application failure")]))
            with patch.object(finalizer, "accounting", return_value=dict(state="COMPLETED", exit_code="0:0")), \
                    patch.object(finalizer, "export") as export, patch.object(finalizer, "submit") as submit:
                self.assertEqual(finalizer.finalize(study, "1", "2")["action"], "needs_review")
                export.assert_not_called()
                submit.assert_not_called()

    def test_complete_failed_gain_gate_is_exported_but_not_rerun(self):
        with TemporaryDirectory() as folder:
            study = Path(folder)
            write_json(study / "continuations.json", dict(events=[dict(previous=dict(job_id="1"), action="finished")]))
            write_json(study / "policy-freeze.json", {})
            write_json(study / "postprocessing-source-manifest.json", {})
            write_json(study / "full-audit.json", dict(verified=True, targets=2565,
                freeze_sha256=sha256(study / "policy-freeze.json"), gate=dict(eligible_for_success_claim=False)))
            write_json(study / "benchmark/execution.json", dict(status="complete", target_ids=list(range(2565))))
            (study / "registry.csv").write_text("registered\n")
            with patch.object(finalizer, "accounting", return_value=dict(state="COMPLETED", exit_code="0:0")), \
                    patch.object(finalizer, "verify_frozen_source"), patch.object(finalizer, "verify_postprocessing"), \
                    patch.object(finalizer, "submit") as submit, \
                    patch.object(finalizer, "export", return_value=dict(targets=2565)) as export:
                result = finalizer.finalize(study, "1", "2")
                self.assertFalse(result["four_metric_gate"])
                self.assertEqual(result["action"], "exported")
                export.assert_called_once()
                submit.assert_not_called()

    def test_controller_failure_does_not_use_an_old_continuation_receipt(self):
        with TemporaryDirectory() as folder, patch.object(finalizer, "accounting", return_value=dict(
                state="FAILED", exit_code="1:0")), patch.object(finalizer, "submit") as submit:
            with self.assertRaisesRegex(ValueError, "controller"):
                finalizer.finalize(Path(folder), "1", "2")
            submit.assert_not_called()

    def test_uncertain_submission_is_not_automatically_repeated(self):
        with TemporaryDirectory() as folder, patch.object(finalizer, "submit") as submit:
            study = Path(folder)
            write_json(study / "finalization-control/1-submission.json", dict(command=["sbatch"]))
            with self.assertRaisesRegex(ValueError, "Uncertain"):
                finalizer.finalize(study, "1", "2")
            submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
