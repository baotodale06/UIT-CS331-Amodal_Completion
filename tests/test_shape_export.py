"""Verify that a local Shape metadata export preserves evidence tied to its execution."""
import argparse
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.provenance import execution_snapshot
from scripts import collect_shape_results
from scripts.export_shape_metadata import export
from scripts.replay_shape_outlines import original_frame_overlap
from scripts.replay_shape_slide_tables import allocated_hours, method_values
from scripts.verify_shape_metadata import reconcile


class ShapeExportTests(unittest.TestCase):
    def test_slide_replay_counts_failures_and_elapsed_time_for_interrupted_runs(self):
        scores = {"a": {"LPIPS": .25, "SSIM": .8, "status": "success"},
                  "b": {"LPIPS": 1., "SSIM": 0., "status": "not_detected"}}
        outlines = {"a": {"success": True, "amodal_iou": .9}, "b": {"success": False, "amodal_iou": 0.}}
        values = method_values(scores, outlines, [["a", "b"]])
        self.assertEqual(values["bins"], [dict(lpips=.625, saved=1, amodal_iou=.45)])
        self.assertEqual((values["saved"], values["LPIPS"], values["SSIM"]), (1, .625, .4))
        outlines["b"]["success"] = True
        with self.assertRaisesRegex(ValueError, "coverage"):
            method_values(scores, outlines, [["a", "b"]])
        accounting = {"1": ("COMPLETED", 900, "gres/gpu=4"), "2": ("TIMEOUT", 3600, "gres/gpu=4"),
                      "3": ("CANCELLED", 0, ""), "4": ("COMPLETED", 1800, "gres/gpu=4")}
        self.assertEqual(allocated_hours(accounting, accounting), 7.0)
        accounting["4"] = ("RUNNING", 400, "gres/gpu=4")
        with self.assertRaisesRegex(ValueError, "still RUNNING"):
            allocated_hours(accounting, accounting)

    def test_outline_replay_clips_native_alpha_to_original_coordinates(self):
        outline = Image.new("L", (3, 2))
        outline.putdata([3, 2, 0, 0, 1, 1])
        completion = Image.new("RGBA", (3, 3))
        alpha = Image.new("L", completion.size)
        alpha.putdata([255, 255, 255, 9, 255, 0, 0, 255, 255])
        completion.putalpha(alpha)
        self.assertEqual(original_frame_overlap(completion, outline, {
            "original_size": [3, 2], "offset_rc": [-1, 0],
        }), (3, 4))
        self.assertEqual(original_frame_overlap(completion, outline, {
            "original_size": [3, 2], "offset_rc": [2, -1],
        }), (0, 3))
        with self.assertRaisesRegex(ValueError, "frame"):
            original_frame_overlap(completion, outline, {"original_size": [2, 3], "offset_rc": [0, 0]})

    def test_collector_replays_each_closed_polygon_table_once(self):
        shape = {"shape-full-pairs.json": "pairs", "shape-full-summary.json": "summary",
                 "progressive-l40s/shape/execution.json": "execution"}
        result = dict(execution_sha256="execution", pairs_sha256="pairs", targets=321)
        with TemporaryDirectory() as temp, patch.object(collect_shape_results, "OUT", Path(temp)), patch.object(
            collect_shape_results, "replay", return_value=result
        ) as replay, redirect_stdout(StringIO()):
            checked = {}
            for hashes in ({}, shape, shape):
                collect_shape_results.check_closed_outlines(hashes, checked)
            replay.assert_called_once_with("shape", cohort=False, fetch=True)
            self.assertEqual(json.loads((Path(temp)/"shape-full-cocoa-native-replay.json").read_bytes()), result)
            with self.assertRaisesRegex(ValueError, "differ"):
                collect_shape_results.check_closed_outlines(shape | {"shape-full-pairs.json": "other"}, checked)

    def test_collection_uses_native_archive_receipt_and_waits_for_both_exports(self):
        with TemporaryDirectory() as temp, patch.object(Path, "home", return_value=Path(temp)):
            root = Path(temp)/"Vu_Hung/cv_nangcao/output/shape-20261121"
            study = root/"progressive-l40s"
            write_json(root/"controlled-audit.json", {})
            write_json(study/"shape/execution.json", {"id": "full-shape"})
            (study/"submissions.tsv").write_text("")
            (study/"report-jobs.txt").write_text("full 10\n")
            (study/"metadata-export-jobs.tsv").write_text("12\tshape\t8\tsbatch\n13\tcombined\t9\tsbatch\n")
            archive = "progressive-l40s/shape-full-metadata.tar.gz"
            receipt = dict(archive="output/shape-20261121/"+archive, sha256="a"*64,
                           targets=2565, execution="full-shape")
            log = study/"shape-export-12.log"
            write_json(log, receipt)
            states = "10|PENDING|0|0:0\n12|COMPLETED|2|0:0\n13|RUNNING|1|0:0"
            with patch("subprocess.check_output", return_value=states), redirect_stdout(StringIO()) as output:
                exec(collect_shape_results.PROBE, {})
                state = json.loads(output.getvalue())
                write_json(log, dict(receipt, execution="old-shape"))
                with self.assertRaisesRegex(ValueError, "receipt"):
                    exec(collect_shape_results.PROBE, {})
            self.assertEqual(state["hashes"][archive], receipt["sha256"])
            self.assertNotIn("progressive-l40s/combined-full-metadata.tar.gz", state["hashes"])
            state["observed_at"] = "now"
            state["jobs"]["10"][0] = "COMPLETED"
            for arm, job in state["export_jobs_by_arm"].items():
                for name in (f"progressive-l40s/{arm}/execution.json", f"{arm}-full-pairs.json",
                             f"{arm}-full-summary.json", f"progressive-l40s/{arm}-export-{job}.log",
                             f"progressive-l40s/{arm}-full-metadata.tar.gz"):
                    state["hashes"].setdefault(name, "hash")
            final = dict(state, jobs=state["jobs"] | {"13": ["COMPLETED", "2", "0:0"]})
            with patch.object(sys, "argv", ["collect", "--watch"]), patch.object(
                collect_shape_results, "collect", side_effect=[state, final]
            ), patch.object(collect_shape_results, "verify_collected", return_value={"targets": 2565}) as verify, patch.object(
                collect_shape_results, "check_closed_outlines"
            ), patch.object(collect_shape_results, "replay_full_slide_tables") as slides, patch.object(
                collect_shape_results, "OUT", Path(temp)
            ), patch("time.sleep") as sleep, redirect_stdout(StringIO()):
                collect_shape_results.main()
                sleep.assert_called_once_with(180)
                slides.assert_called_once_with()
                self.assertEqual([c.args[1:] for c in verify.call_args_list], [("shape", "12"), ("combined", "13")])
                for arm in ("shape", "combined"):
                    self.assertEqual(json.loads((Path(temp)/f"{arm}-full-metadata-replay.json").read_bytes()), {"targets": 2565})

    def test_local_archive_check_rejects_changed_paired_failure_score_or_extra_target(self):
        with TemporaryDirectory() as temp:
            run = Path(temp)
            tid = "VG/1_dog"
            pair = dict(id=tid, status={"shape": "not_detected"}, seed=12, input_sha256="input",
                        method_reference={"crop": "crop"}, shape={"CLIP": None, "LPIPS": 1., "SSIM": 0., "Feature Similarity": 0.})
            with execution_snapshot(run, argparse.Namespace(), [{"id": tid}]):
                write_json(run/"run_config.json", {"shape_policy": {"method":"shape","seeds":list(range(11,89,11))}})
                write_json(run/"targets"/tid/"result.json", dict(id=tid, status="not_detected", seed=12, input_sha256="input",
                                                                 evaluation_reference_artifacts=pair["method_reference"]))
                (run/"per_target.csv").write_text("id,status,seed,CLIP,LPIPS,SSIM,Feature Similarity\n"
                                                  + tid + ",not_detected,12,,1,0,0\n")
            members = {p.relative_to(run).as_posix(): sha256(p) for p in run.rglob("*") if p.is_file()}
            members["member-sha256.json"] = "manifest"
            self.assertEqual(reconcile(run, "shape", [tid], [pair], members)["targets"], 1)
            with self.assertRaisesRegex(ValueError, "membership"):
                reconcile(run, "shape", [tid], [pair], members | {"targets/VG/extra/result.json":"unused"})
            pair["shape"]["LPIPS"] = .4
            with self.assertRaisesRegex(ValueError, "metrics differ"):
                reconcile(run, "shape", [tid], [pair], members)

    def test_export_validates_decision_and_score_before_publishing(self):
        with TemporaryDirectory() as temp:
            run = Path(temp) / "shape"
            run.mkdir()
            tid = "VG/1_dog"
            target = run / "targets" / tid
            decision = target / "shape/iteration-00/decision.json"
            score = target / "metrics.json"
            with execution_snapshot(run, argparse.Namespace(), [{"id": tid}]):
                write_json(decision, {"index": 3})
                write_json(score, {"inputs": {"completion_sha256": "rgb", "reference_sha256": "crop"}})
                write_json(target / "result.json", {
                    "id": tid, "status": "success", "completion_sha256": "rgb",
                    "evaluation_reference_artifacts": {"crop": "crop"},
                    "diagnostics": {"shape_iterations": [{"decision_sha256": sha256(decision)}]},
                })
            archive = Path(temp) / "metadata.tar.gz"
            original_decision = decision.read_bytes()
            write_json(decision, {"index": 4})
            with self.assertRaisesRegex(ValueError, "Native metadata"):
                export(run, archive)
            self.assertFalse(archive.exists())
            decision.write_bytes(original_decision)
            original_score = score.read_bytes()
            write_json(score, {"inputs": {"completion_sha256": "rgb", "reference_sha256": "other"}})
            with self.assertRaisesRegex(ValueError, "Score cache inputs"):
                export(run, archive)
            self.assertFalse(archive.exists())
            score.write_bytes(original_score)
            receipt = export(run, archive)
            self.assertEqual(sha256(archive), receipt["sha256"])
            with tarfile.open(archive) as bundle:
                members = json.load(bundle.extractfile("member-sha256.json"))
                self.assertEqual(set(bundle.getnames()), {"member-sha256.json", *members})
                for name, digest in members.items():
                    self.assertEqual(hashlib.sha256(bundle.extractfile(name).read()).hexdigest(), digest)
                    self.assertEqual(sha256(run / name), digest)
            with self.assertRaises(FileExistsError):
                export(run, archive)
            inputs = {"completion_sha256":"rgb", "reference_sha256":"full", "clip_only":True}
            other = run / "targets" / tid / "result.json"
            with execution_snapshot(run, argparse.Namespace(), [{"id": tid}]):
                record = json.loads(other.read_bytes())
                record["evaluation_reference_artifacts"]["full"] = "full"
                write_json(other, record)
                write_json(score, {"inputs":inputs})
            clip_archive = Path(temp)/"clip-only.tar.gz"
            write_json(score, {"inputs":dict(inputs, reference_sha256="crop")})
            with self.assertRaisesRegex(ValueError, "Score cache inputs"):
                export(run, clip_archive)
            write_json(score, {"inputs":inputs})
            self.assertEqual(export(run, clip_archive)["sha256"], sha256(clip_archive))

