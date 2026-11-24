"""Calibrated export must preserve seed decisions, geometry and terminal evidence."""
import argparse
import copy
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.provenance import execution_snapshot
from amodal_benchmark.shape_v2_runtime import progressive_seed, SEED_SLOTS
from scripts.export_shape_metadata import export
from scripts import verify_shape_v2_metadata as verifier
from scripts import finalize_shape_v2 as finalizer
from scripts.recover_shape_calibrated import disposition, validate_export
from scripts.audit_shape_v2 import paired_estimate

METHOD = "shape-calibrated-probability"


class CalibratedPostprocessingTests(unittest.TestCase):
    def test_real_export_binds_calibrated_decisions_and_rejects_policy_or_seed_drift(self):
        with TemporaryDirectory() as temporary:
            root, tid = Path(temporary), "VG/test"
            run = root / "run"
            target = run / "targets" / tid
            path = target / METHOD / "iteration-00/decision.json"
            decision = dict(schema=2, method=METHOD, logit_bias=-2., iteration=0, target_seed=12,
                seed_slots=list(SEED_SLOTS), seeds=[progressive_seed(12, 0, slot) for slot in SEED_SLOTS],
                scores=[.2, .8, None, .1, .3, .2, .1, .2], index=1, fallback=False,
                rgb_correction=False, inference_truth_access=False)
            def close():
                with execution_snapshot(run, argparse.Namespace(), [dict(id=tid)]):
                    write_json(run / "run_config.json", dict(shape_policy=dict(method=METHOD)))
                    write_json(path, decision)
                    write_json(target / "metrics.json", dict(inputs=dict(completion_sha256="rgb", reference_sha256="crop")))
                    write_json(target / "result.json", dict(id=tid, seed=12, status="success", completion_sha256="rgb",
                        evaluation_reference_artifacts=dict(crop="crop"), diagnostics=dict(shape_calibrated_iterations=[
                            dict(decision_sha256=sha256(path), index=1, fallback=False)])))
            close()
            result = export(run, root / "valid.tar.gz")
            self.assertEqual(result["targets"], 1)
            with tarfile.open(root / "valid.tar.gz") as bundle:
                self.assertIn(path.relative_to(run).as_posix(), bundle.getnames())
            original = copy.deepcopy(decision)
            for key, changed in (("logit_bias", -1.5), ("method", "shape-v2"), ("target_seed", 13)):
                decision[key] = changed
                close()
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "target policy"):
                    export(run, root / (key + ".tar.gz"))
                decision = copy.deepcopy(original)
            close()
            path.unlink()
            with self.assertRaisesRegex(ValueError, "inventory"):
                export(run, root / "missing.tar.gz")

    def test_geometry_replay_rejects_duplicate_inventory_changed_strata_and_statistics(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            official = [dict(id=f"VG/{i}", source="VG", original_id=str(i//11)) for i in range(321)]
            write_json(folder / "manifest.json", dict(rows=[dict(id=row["id"], bin="test") for row in official]))
            checksum = sha256(folder / "manifest.json")
            rows = [dict(id=row["id"], source=row["source"], image_identity="VG/"+row["original_id"], occlusion_bin="test",
                control=dict(full_iou=.4, hidden_iou=.2), method=dict(full_iou=.42, hidden_iou=.21)) for row in official]
            values = {metric: paired_estimate(rows, metric) for metric in ("full_iou", "hidden_iou")}
            geometry = dict(objects=321, annotation_manifest_sha256=checksum, rows=rows,
                full=values, historical_unseen=copy.deepcopy(values), occlusion_strata=dict(test=dict(objects=321, **values)))
            with patch.object(verifier, "GEOMETRY_SHA256", checksum):
                self.assertTrue(verifier.verify_benchmark_geometry(geometry, official, {row["id"] for row in official}, folder)["verified"])
                for change in ("duplicate", "stratum", "value", "statistic"):
                    altered = copy.deepcopy(geometry)
                    if change == "duplicate":
                        altered["rows"][0] = altered["rows"][1]
                    elif change == "stratum":
                        altered["rows"][0]["occlusion_bin"] = "other"
                    elif change == "value":
                        altered["rows"][0]["method"]["full_iou"] = float("nan")
                    else:
                        altered["full"]["full_iou"]["relative_gain_percent"] = 4.
                    with self.subTest(change=change), self.assertRaises(ValueError):
                        verifier.verify_benchmark_geometry(altered, official, {row["id"] for row in official}, folder)

    def test_calibrated_follower_uses_calibrated_runner_without_new_inference(self):
        with TemporaryDirectory() as temporary:
            study = Path(temporary)
            write_json(study / "continuations.json", dict(events=[dict(previous=dict(job_id="1"), action="resumed",
                next_job_id="3", controller_job_id="4")]))
            with patch.object(finalizer, "accounting", return_value=dict(state="COMPLETED", exit_code="0:0")), \
                    patch.object(finalizer, "submit", return_value="5") as submit:
                receipt = finalizer.finalize(study, "1", "2", method="shape-calibrated")
                self.assertEqual(finalizer.finalize(study, "1", "2", method="shape-calibrated"), receipt)
                command = submit.call_args.args[0]
                self.assertIn("scripts/run_shape_calibrated_finalization.sh", command)
                self.assertIn("--dependency=afterany:4", command)
                self.assertFalse(any("gres" in argument for argument in command))
                submit.assert_called_once()

    def test_collection_waits_for_export_and_does_not_hide_failed_prerequisites(self):
        report = dict(jobs=dict(native="1", full="2", finalizer="3"), accounting={
            "1": ["COMPLETED"], "2": ["COMPLETED"], "3": ["PENDING"]}, audit_ready=True, export_ready=False)
        self.assertEqual(disposition(report), "waiting")
        report["accounting"]["3"] = ["COMPLETED"]
        self.assertEqual(disposition(report), "needs_review")
        report["export_ready"] = True
        self.assertEqual(disposition(report), "export_ready")
        report["accounting"]["1"] = ["FAILED"]
        self.assertEqual(disposition(report), "needs_review")
        report["accounting"].pop("3")
        with self.assertRaises(ValueError):
            disposition(report)

    def test_changed_native_audit_or_partial_export_cannot_be_recovered_as_full(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for name in ("policy-freeze.json", "full-audit.json", "postprocessing-source-manifest.json"):
                write_json(folder / name, dict(test_only=True))
            from scripts.recover_shape_calibrated import REMOTE_STUDY
            receipt = dict(targets=2565, archive=REMOTE_STUDY + "/benchmark-metadata.tar.gz",
                freeze_sha256=sha256(folder / "policy-freeze.json"), audit_sha256=sha256(folder / "full-audit.json"),
                postprocessing_manifest_sha256=sha256(folder / "postprocessing-source-manifest.json"))
            validate_export(receipt, folder)
            with self.assertRaises(ValueError):
                validate_export(dict(receipt, targets=2564), folder)
            write_json(folder / "full-audit.json", dict(changed=True))
            with self.assertRaises(ValueError):
                validate_export(receipt, folder)


if __name__ == "__main__":
    unittest.main()
