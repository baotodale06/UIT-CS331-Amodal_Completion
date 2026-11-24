"""Incomplete or failed endpoints must not become a successful report."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from amodal_benchmark.dataset import write_json
from scripts import build_shape_v2_report as report


class ShapeReportTests(unittest.TestCase):
    def test_live_incomplete_study_cannot_publish_a_report(self):
        with TemporaryDirectory() as temporary:
            study = Path(temporary)
            write_json(study/"collection-status.json", dict(disposition="waiting", counts=dict(success=4)))
            with self.assertRaises(FileNotFoundError):
                report.build(study, study/"report")
            self.assertFalse((study/"report").exists())

    def test_failed_endpoint_remains_failure_in_report_and_machine_receipt(self):
        with TemporaryDirectory() as temporary:
            study = Path(temporary)
            geometry = dict(control=.4, method=.42, relative_gain_percent=5.,
                            relative_ci_percent=[-2., 12.], control_n=321, method_n=321)
            metrics = {name:dict(control=1., method=1.02, relative_gain_percent=2.,
                relative_ci_percent=[1., 3.], control_n=2565, method_n=2565,
                pass_point_threshold=False) for name in report.METRICS}
            gate = dict(eligible_for_success_claim=False, clip_coverage_not_reduced=False, metrics=metrics)
            replay = dict(gate=gate, full=metrics, unseen=metrics, saved=1393, failures=1172,
                          freeze_sha256="fixture", native_audit_sha256="fixture", method_execution="fixture")
            write_json(study/"full-metadata-replay.json", replay)
            fixture = dict(snapshot=study/"recovery/fixture", replay=replay,
                audit=dict(geometry={name:dict(full_iou=geometry, hidden_iou=geometry)
                    for name in ("full", "historical_unseen")}, selector=dict(pools=1, candidates=8, fallbacks=0)),
                cost=dict(reserved_gpu_hours_by_role=dict(full=10., training=1.), total_reserved_gpu_hours=11.),
                prior_geometry={split:dict(objects=10, images=5, full_iou=.8,
                    visible_only_full_iou=.7, hidden_iou=.4, hidden_iou_images=4) for split in ("validation", "holdout")})
            fixture["audit"]["geometry"]["occlusion_strata"] = {
                "fixture stratum": dict(objects=321, full_iou=geometry, hidden_iou=geometry)}
            with patch.object(report, "evidence", return_value=fixture), patch.object(report, "draw_gains"):
                result = report.build(study, study/"report")
            self.assertFalse(result["full_four_metric_goal_passed"])
            text = (study/"report/REPORT.md").read_text(encoding="utf8")
            self.assertIn("was not achieved", text)
            self.assertIn("CLIP coverage gate: **fail**", text)
            self.assertIn("| fixture stratum | 321 | hidden_iou |", text)
            self.assertFalse(json.loads((study/"report/report-receipt.json").read_bytes())["full_four_metric_goal_passed"])
            replay["method"] = "shape-calibrated-probability"
            with patch.object(report, "evidence", return_value=fixture), patch.object(report, "draw_gains"):
                calibrated = report.build(study, study/"calibrated-report")
            self.assertFalse(calibrated["full_four_metric_goal_passed"])
            text = (study/"calibrated-report/REPORT.md").read_text(encoding="utf8")
            self.assertIn("Calibrated shape", text)
            self.assertIn("not untouched confirmation", text)
            self.assertNotIn("holdout was read only after policy freezing", text)


if __name__ == "__main__":
    unittest.main()
