from contextlib import redirect_stdout
import csv
import hashlib
from io import StringIO
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image

from amodal_benchmark.shape import choose, overlap, score_masks
from scripts import build_shape_report, collect_shape_results, resume_shape_timeout, snapshot_shape_timeout
from amodal_benchmark.dataset import read_official, sha256, write_json
from amodal_benchmark.provenance import digest
from amodal_benchmark.references import reference_paths
from amodal_benchmark.protocol import METRICS
from amodal_benchmark.registry import register_run
from scripts.audit_shape_controlled_cocoa import native_mask
from scripts.replay_shape_slide_tables import match_report, method_values
from scripts.replay_shape_reference_intervals import reference_groups
from scripts.compare_shape_progressive import (
    compare_first_pass, natural_geometry, verify_cohort_reuse, verify_full_registration, verify_reference_transfers,
    verify_selector_snapshot, verify_vg_scores,
)


class ShapeTests(unittest.TestCase):
    def test_vg_lpips_uses_signed_changes_and_replaces_cpu_scores_after_the_full_audit(self):
        row = dict(id="VG/a", status="success", LPIPS=.4)
        hashes = {"baseline": "pmc", "surface": "surface"}
        stats = dict(n=1, images=1, baseline=.5, method=.4, relative_gain_percent=20.,
                     relative_ci_percent=[18., 21.])
        replay = dict(rows=[row], control_csv_sha256=hashes,
                      comparisons={"baseline": stats, "surface": stats})
        ssim = dict(rows=[dict(id="VG/a", status="success")], control_csv_sha256=hashes)
        summary = build_shape_report.vg_lpips_paragraph(replay, ssim)
        self.assertIn("-20,00%", summary)
        self.assertIn("-21,00% đến -18,00%", summary)
        closed = [dict(id="VG/a", source="VG", image_identity="VG/1", status={"shape": "success"},
                       baseline={"LPIPS": .5}, surface={"LPIPS": .4}, shape={"LPIPS": .45})]
        exact = build_shape_report.vg_lpips_paragraph(replay, ssim, closed)
        self.assertIn("0,4500 cho Shape", exact)
        self.assertIn("shape-full-pairs.json", exact)
        with self.assertRaisesRegex(ValueError, "closed VG scores"):
            build_shape_report.vg_lpips_paragraph(replay, ssim, [{**closed[0], "id": "VG/b"}])

    def test_severe_plate_hidden_recall_uses_only_in_frame_amodal_minus_visible(self):
        target = "COCOA/COCO_train2014_000000060172_plate"
        with TemporaryDirectory() as temp, patch.object(build_shape_report, "OUT", Path(temp)):
            study = Path(temp)
            shape = study / "progressive-l40s/shape/targets" / target
            shape.mkdir(parents=True)
            labels = study / "evaluation_only/cocoa"
            labels.mkdir(parents=True)
            Image.fromarray(np.array([[3, 1], [1, 2]], dtype=np.uint8)).save(labels / "plate.png")
            write_json(labels / "manifest.json", {"rows": [
                dict(id=target, case="plate.png", sha256=sha256(labels / "plate.png"))]})
            native = Image.new("RGBA", (3, 2))
            native.putalpha(Image.fromarray(np.array([[255, 0, 1], [0, 0, 255]], dtype=np.uint8)))
            native.save(shape / "completion.png")
            write_json(shape / "result.json", dict(id=target, status="success",
                       geometry=dict(original_size=[2, 2], offset_rc=[0, -1]),
                       completion_sha256=sha256(shape / "completion.png")))
            digest = sha256(labels / "manifest.json")
            summary = " ".join(build_shape_report.severe_plate_section(digest))
            self.assertIn("IoU 0,2500", summary)
            self.assertIn("1 trên 2 pixel (50,00%)", summary)
            self.assertNotIn("Shape + Surface", summary)
            combined = study / "progressive-l40s/combined/targets" / target
            write_json(combined / "result.json", dict(id=target, status="no_completion"))
            self.assertIn("nhận IoU bằng 0", " ".join(build_shape_report.severe_plate_section(digest)))
            result = shape / "result.json"
            record = json.loads(result.read_bytes())
            record["diagnostics"] = {"shape_iterations": [{"decision_sha256": "pass-one"}]}
            write_json(result, record)
            figure = study / "shape-plate-first-pass.png"
            native.save(figure)
            replay = dict(target=target, record_sha256=sha256(result), manifest_sha256=digest,
                          annotation_sha256=sha256(labels / "plate.png"), hidden_pixels=2,
                          figure_sha256=sha256(figure), passes=[dict(decision_sha256="pass-one",
                              selected=1, highest_iou_index=0, prior={"hidden_recall": 0.},
                              available={"hidden_recall": 1.}, candidates=[
                                  {"iou": .8, "hidden_recall": 1.}, {"iou": .4, "hidden_recall": 0.}])])
            write_json(study / "shape-plate-trace.json", replay)
            summary = " ".join(build_shape_report.severe_plate_section(digest))
            self.assertIn("Shape chọn ảnh số 2, có IoU 0,400", summary)
            self.assertIn("Ảnh số 1 đạt IoU 0,800", summary)
            replay["passes"][0]["decision_sha256"] = "different"
            write_json(study / "shape-plate-trace.json", replay)
            with self.assertRaisesRegex(ValueError, "Plate selection figure or trace"):
                build_shape_report.severe_plate_section(digest)
            native.putalpha(0)
            native.save(shape / "completion.png")
            with self.assertRaisesRegex(ValueError, "recorded hash"):
                build_shape_report.severe_plate_section(digest)

    def test_vg_ssim_shared_means_use_only_targets_saved_by_both_methods(self):
        shape = [{"id": "VG/a", "status": "success", "SSIM": .7},
                 {"id": "VG/b", "status": "success", "SSIM": .5},
                 {"id": "VG/c", "status": "no_completion", "SSIM": 0.}]
        replay = dict(targets=3, rows=shape, means=dict(shape=.4, surface=.5, baseline=.3))
        surface = {"VG/a": dict(status="success", SSIM=.9), "VG/b": dict(status="no_completion", SSIM=0.),
                   "VG/c": dict(status="success", SSIM=.6)}
        paragraph = build_shape_report.vg_ssim_paragraph(replay, surface)
        self.assertIn("Trên 1 đối tượng", paragraph)
        self.assertIn("0,7000 cho Shape và 0,9000 cho Surface", paragraph)
        with self.assertRaisesRegex(ValueError, "in order"):
            build_shape_report.vg_ssim_paragraph({**replay, "rows": shape[::-1]}, surface)

    def test_controlled_cocoa_appearance_uses_the_polygon_selection_and_first_four_seeds(self):
        with TemporaryDirectory() as directory, patch.object(build_shape_report, "OUT", Path(directory)):
            path = Path(directory) / "controlled/case/result.json"
            write_json(path, dict(target_id="target", indices={"shape4": 1},
                                  candidate_metrics=[{"hidden_spatial_lpips": n} for n in (1., 0., 1., 0., 8.)]))
            row = dict(id="target", case_id="case", image="image", controlled_result_sha256=sha256(path),
                       amodal_iou={"shape4": .8}, candidate_amodal_iou=[.2, .8, .6, .3])
            dream_row = dict(id="target", decision_sha256=sha256(path), candidate_distances=[.8, .2, .8, .2, 4.])
            cocoa = dict(cases_sha256="cases", rows=[row])
            dream = dict(cases_sha256="cases", rows=[dream_row])
            stats = build_shape_report.cocoa_hidden_appearance(cocoa, dream)
            self.assertEqual((stats["LPIPS"]["baseline"], stats["LPIPS"]["method"]), (.5, 0.))
            self.assertEqual((stats["DreamSim"]["baseline"], stats["DreamSim"]["method"]), (.5, .2))
            dream_row["decision_sha256"] = "another-decision"
            with self.assertRaisesRegex(ValueError, "different candidate selections"):
                build_shape_report.cocoa_hidden_appearance(cocoa, dream)

    def test_native_reference_groups_distinguish_equal_crops_from_equal_masks(self):
        official = [dict(id=f"VG/{tid}", source="VG", original_id="1")
                    for tid in ("1_a", "1_b", "1_c")]
        with TemporaryDirectory() as directory:
            runs = {}
            for arm in ("baseline", "surface", "shape", "combined"):
                run = Path(directory) / arm
                hashes = {}
                scores = {}
                for i, row in enumerate(official):
                    tid = row["id"]
                    refs = dict(mask="mask", full="full", crop="crop")
                    if arm in ("shape", "combined") and i > 0:
                        refs["mask"] = "other-mask"
                        if i == 2:
                            refs["crop"] = "other-crop"
                    path = run / "targets" / tid / "result.json"
                    path.parent.mkdir(parents=True)
                    write_json(path, dict(id=tid, evaluation_reference_artifacts=refs))
                    hashes[tid] = sha256(path)
                    scores[tid] = {"LPIPS": 1.0}
                runs[arm] = run, {"result_artifacts": hashes}, scores
            early = {"VG/1_b": dict(all_files=False, scoring_crop=True)}
            groups = reference_groups(official, runs, early)
            self.assertEqual([r["id"] for r in groups["same-reference"]], ["VG/1_a"])
            self.assertEqual([r["id"] for r in groups["same-scoring-crop"]], ["VG/1_a", "VG/1_b"])
            self.assertEqual(groups["same-scoring-crop"][1]["image_identity"], "VG/1")
            early["VG/1_b"]["scoring_crop"] = False
            with self.assertRaisesRegex(ValueError, "PNG audit"):
                reference_groups(official, runs, early)

    def test_final_execution_preserves_archived_selector_records_in_official_order(self):
        ids = ["VG/2_bed", "VG/1_bus", "VG/3_cup"]
        hashes = dict(zip(ids, ("bed-hash", "bus-hash", "cup-hash")))
        snapshot = dict(targets=2, target_ids_sha256=digest(ids[:2]),
                        result_artifacts_sha256=digest({tid: hashes[tid] for tid in ids[:2]}))
        execution = dict(status="complete", target_ids=ids, result_artifacts=hashes)
        self.assertEqual(verify_selector_snapshot(execution, snapshot), snapshot)
        with self.assertRaisesRegex(ValueError, "target order"):
            verify_selector_snapshot(dict(execution, target_ids=sorted(ids)), snapshot)
        with self.assertRaisesRegex(ValueError, "closed execution"):
            verify_selector_snapshot(dict(execution, status="running"), snapshot)
        with self.assertRaisesRegex(ValueError, "missing"):
            verify_selector_snapshot(dict(execution, result_artifacts={ids[0]: hashes[ids[0]]}), snapshot)
        with self.assertRaisesRegex(ValueError, "differ"):
            verify_selector_snapshot(dict(execution, result_artifacts=hashes | {ids[1]: "changed"}), snapshot)

    def test_vg_replay_checks_each_score_and_both_inputs(self):
        with TemporaryDirectory() as directory:
            run = Path(directory)
            hashes = {}
            coverage, replay = [], []
            for tid, status, value in (("VG/1_bus", "success", .75), ("VG/2_cat", "not_detected", 0.)):
                record = dict(id=tid, status=status)
                early = dict(id=tid, status=status, SSIM=value)
                if status == "success":
                    record.update(completion_sha256="image", evaluation_reference_artifacts={"crop": "ref"})
                    early.update(completion_sha256="image", crop_sha256="ref")
                path = run / "targets" / tid / "result.json"
                write_json(path, record)
                hashes[tid] = sha256(path)
                coverage.append(dict(id=tid, shape=status, result_sha256=hashes[tid]))
                replay.append(early)
            path = run / "per_target.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, ["id", "status", *METRICS], extrasaction="ignore")
                writer.writeheader()
                writer.writerows(replay)
            execution = dict(status="complete", target_ids=list(hashes) + ["COCOA/3_cup"],
                             result_artifacts=hashes, artifacts={"per_target.csv": sha256(path)})
            result = verify_vg_scores(run, execution, coverage, replay)
            self.assertEqual((result["targets"], result["saved"]), (2, 1))
            for key, changed in (("SSIM", .72), ("completion_sha256", "other-pred"), ("crop_sha256", "other-ref")):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "VG/1_bus"):
                    verify_vg_scores(run, execution, coverage, [dict(replay[0], **{key: changed}), replay[1]])
            with self.assertRaisesRegex(ValueError, "native result"):
                verify_vg_scores(run, execution, [coverage[0], dict(coverage[1], result_sha256="other")], replay)
            with self.assertRaisesRegex(ValueError, "target order"):
                verify_vg_scores(run, execution, coverage, replay[:1])

    def test_full_audit_rejects_stale_registration_and_changed_scores(self):
        with TemporaryDirectory() as directory:
            run = Path(directory) / "shape"
            run.mkdir()
            registry = run.parent / "shape-registry.csv"
            execution = {"id": "finished", "status": "complete", "target_ids": list(range(2565))}
            write_json(run / "execution.json", execution)
            for name in ("run_config.json", "metric_config.json", "summary.json"):
                write_json(run / name, {})
            overall = dict(targets=2565, successes=2000, failures=565, failure_rate=565 / 2565)
            for metric in METRICS:
                overall[metric] = .75
                overall[metric + "_n"] = 2000 if metric == "CLIP" else 2565
            report = dict(run_dir=str(run.resolve()), full_evaluation=True, selected_targets=2565,
                          verified=True, provenance_verified=True, uncached_clip_only_targets=0,
                          data_identity=dict(inputs_sha256="inputs", references_sha256="refs", targets_sha256="ids"),
                          statistics={"Overall": overall})
            with patch("amodal_benchmark.registry.verify_run", return_value=report):
                original = register_run(run, run.parent, registry, hypothesis="Test Shape")
            receipt = verify_full_registration(run, registry, report)
            self.assertEqual(receipt, dict(entry_id=original["entry_id"], execution_id="finished",
                                           registry_sha256=sha256(registry)))
            for key, changed in (("entry_id", "previous"), ("run_dir", "different-run"),
                                 ("references_sha256", "other-refs"), ("protocol_sha256", "other-config"),
                                 ("summary_sha256", "other-summary"), ("failures", 564),
                                 ("CLIP_n", 2565), ("LPIPS", .76)):
                with self.subTest(key=key):
                    with registry.open("w", encoding="utf8", newline="") as stream:
                        writer = csv.DictWriter(stream, original.keys())
                        writer.writeheader()
                        writer.writerow({**original, key: changed})
                    message = "Expected one registration" if key == "entry_id" else key
                    with self.assertRaisesRegex(ValueError, message):
                        verify_full_registration(run, registry, report)

    def test_recovery_submits_only_after_timeout_and_retains_the_live_child_jobs(self):
        with TemporaryDirectory() as directory, patch.object(resume_shape_timeout, "ROOT", Path(directory)):
            study = Path(directory) / resume_shape_timeout.STUDY
            study.mkdir(parents=True)
            ledger = study / "submissions.tsv"
            ledger.write_text("100\tshape\tfull\tsbatch --parsable -p gpu --gres=gpu:nvidia_l40s:4 "
                              "--dependency=afterok:99 scripts/run_shape_progressive.sh --benchmark-workers 4 shape full\n"
                              "101\taudit-shape\tfull\tsbatch ...\n102\tcombined\tfull\tsbatch ...\n")
            args = ("shape", "100", ["101", "102"], "103")
            for state in ("COMPLETED", "RUNNING", "PENDING", "FAILED", "CANCELLED"):
                with self.subTest(state=state), patch("subprocess.check_output", return_value=f"100|{state}\n"), \
                        patch.object(resume_shape_timeout, "submit") as submit:
                    if state == "COMPLETED":
                        resume_shape_timeout.resume(*args)
                    else:
                        with self.assertRaisesRegex(ValueError, "confirmed timeout"):
                            resume_shape_timeout.resume(*args)
                    submit.assert_not_called()
            write_json(study / "shape-timeout-98.json", {"completed": {"VG/1": "original"}, "cached_scores": 0})
            write_json(study / "shape-timeout-100.json", {"completed": {"VG/1": "original"}, "cached_scores": 1})
            waiting = f"JobState=PENDING WorkDir={directory} Dependency=afterok:103(unfulfilled)"
            with patch("subprocess.check_output", side_effect=["100|TIMEOUT\n", waiting, waiting]), \
                    patch.object(resume_shape_timeout, "submit", side_effect=["104", "105"]) as submit, \
                    patch("subprocess.run") as run:
                resume_shape_timeout.resume(*args)
            gpu, monitor = [call.args[0] for call in submit.call_args_list]
            self.assertEqual(gpu[-5:], ["scripts/run_shape_progressive.sh", "--benchmark-workers", "4", "shape", "full"])
            self.assertIn("--dependency=afterok:103", gpu)
            self.assertNotIn("--dependency=afterok:99", gpu)
            self.assertIn("--dependency=afterany:104", monitor)
            self.assertEqual(monitor[-1].split()[-4:], ["shape", "104", "101", "102"])
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(commands[0][1:], ["scripts/snapshot_shape_timeout.py", "shape", "100"])
            self.assertEqual(commands[1:], [["scontrol", "update", "JobId=" + child,
                                               "Dependency=afterok:104?afterok:105"] for child in args[2]])
            self.assertIn("104\tshape\tfull\t", ledger.read_text())
            with patch("subprocess.check_output", return_value="100|TIMEOUT\n"), \
                    patch.object(resume_shape_timeout, "submit") as duplicate:
                with self.assertRaisesRegex(ValueError, "current recorded job IDs"):
                    resume_shape_timeout.resume(*args)
                duplicate.assert_not_called()

    def test_timeout_archives_only_partial_targets_and_audits_completed_bytes_after_retry(self):
        rows, _ = read_official(Path(__file__).resolve().parent / "fixtures")
        ids = [row["id"] for row in rows]
        with TemporaryDirectory() as directory, patch.object(snapshot_shape_timeout, "ROOT", Path(directory)):
            study = Path(directory) / "output/shape-20261121/progressive-l40s"
            run = study / "shape"
            run.mkdir(parents=True)
            (study / "submissions.tsv").write_text("123 shape full\n")
            job_args = ["snapshot_shape_timeout.py", "shape", "123"]
            receipt = study / "shape-timeout-123.json"
            with patch.object(sys, "argv", job_args), patch("subprocess.check_output", return_value="123|RUNNING\n"):
                snapshot_shape_timeout.main()
            self.assertFalse(receipt.exists())
            env = run / "provenance/attempt/environment.json"
            write_json(env, {"execution_environment": {"SLURM_JOB_ID": "123"}})
            write_json(run / "execution.json", dict(id="attempt", target_ids=ids, environment_sha256=sha256(env)))
            write_json(run / "run_config.json", {"seed": 1})
            completed = {}
            for tid in ids[:205]:
                result = run / "targets" / tid / "result.json"
                write_json(result, dict(id=tid))
                completed[tid] = sha256(result)
            score = run / "targets" / ids[204] / "metrics.json"
            write_json(score, {"metrics": {"LPIPS": 0.25}})
            original_score = score.read_bytes()
            write_json(study / "shape-cohort/execution.json", dict(result_artifacts=dict(list(completed.items())[:204])))
            partial = run / "targets" / ids[205]
            write_json(partial / "iteration-00/decision.json", {"index": 3})
            archived = receipt.with_suffix("") / ids[205]
            with patch.object(sys, "argv", job_args), patch("subprocess.check_output", return_value="123|TIMEOUT\n"), \
                    patch.object(snapshot_shape_timeout, "read_official", return_value=(rows, {})):
                with patch.object(Path, "rename", side_effect=OSError("archive interrupted")):
                    with self.assertRaisesRegex(OSError, "archive interrupted"):
                        snapshot_shape_timeout.main()
                expected_receipt = sha256(receipt)
                self.assertTrue(partial.is_dir())
                snapshot_shape_timeout.main()
                snapshot_shape_timeout.main()
                self.assertEqual(sha256(receipt), expected_receipt)
            self.assertFalse(partial.exists())
            self.assertEqual(json.loads((archived / "iteration-00/decision.json").read_bytes()), {"index": 3})
            resumed = dict(new_or_updated_target_ids=[ids[205]], result_artifacts={**completed, ids[205]: "new"})
            snapshot = json.loads(receipt.read_bytes())
            self.assertEqual(snapshot["cached_scores"], 1)
            self.assertEqual(snapshot["cached_score_sha256"], {ids[204]: sha256(score)})
            write_json(run / "targets" / ids[205] / "metrics.json", {"metrics": {"LPIPS": 0.125}})
            self.assertEqual(snapshot_shape_timeout.verify_timeout_snapshots(run, resumed), {"123": 205})
            write_json(score, {"metrics": {"LPIPS": 0.5}})
            with self.assertRaisesRegex(ValueError, "changed the pre-timeout score"):
                snapshot_shape_timeout.verify_timeout_snapshots(run, resumed)
            score.write_bytes(original_score)
            resumed["result_artifacts"][ids[204]] = "changed"
            with self.assertRaisesRegex(ValueError, "changed the pre-timeout record"):
                snapshot_shape_timeout.verify_timeout_snapshots(run, resumed)
            resumed["result_artifacts"].update(completed)
            write_json(archived / "iteration-00/decision.json", {"index": 4})
            with self.assertRaisesRegex(ValueError, "Archived partial target differs"):
                snapshot_shape_timeout.verify_timeout_snapshots(run, resumed)
            write_json(archived / "iteration-00/decision.json", {"index": 3})
            del snapshot["cached_score_sha256"]
            write_json(receipt, snapshot)
            self.assertEqual(snapshot_shape_timeout.verify_timeout_snapshots(run, resumed), {"123": 205})

    def test_visibility_screen_checks_full_annotation_and_both_image_and_crop_edges(self):
        mask = Image.new("L", (6, 5))
        mask.paste(3, (1, 1, 5, 4))
        self.assertEqual(build_shape_report.outline_status(mask, (1, 1, 5, 4)), "eligible")
        self.assertEqual(build_shape_report.outline_status(mask, (1, 1, 4, 4)), "clipped by crop")
        for point in ((0, 2), (5, 2), (2, 0), (2, 4)):
            with self.subTest(point=point):
                touching = mask.copy()
                touching.putpixel(point, 3)
                self.assertEqual(build_shape_report.outline_status(touching, (-1, -1, 7, 6)),
                                 "touches image border")
        mask.putpixel((2, 2), 1)
        self.assertEqual(build_shape_report.outline_status(mask, (0, 0, 6, 5)), "visible mask differs")

    def test_controlled_occlusion_counts_overlapping_hidden_pixels_once_inside_crop(self):
        # Five object pixels enter the crop, with two already hidden. The
        # occluder overlaps one hidden pixel and hides one more visible pixel.
        labels = Image.fromarray(np.array([[3, 3, 1, 1, 3, 3]], dtype=np.uint8))
        crop = (-1, 0, 5, 1)
        occluder = Image.fromarray(np.array([[255, 255, 0, 255, 0, 0]], dtype=np.uint8))
        enlarged = occluder.resize((12, 2), Image.Resampling.NEAREST)
        self.assertEqual(build_shape_report.controlled_hidden_pixels(labels, crop, enlarged), (12, 20))
        self.assertEqual(build_shape_report.controlled_hidden_pixels(labels, crop, Image.new("L", (6, 1), 255)), (5, 5))

    def test_controlled_mask_uses_original_frame_and_discards_crop_padding(self):
        mask = Image.fromarray(np.array([[255, 0], [0, 255]], dtype=np.uint8))
        restored = native_mask(mask, (-1, 1, 3, 5), (5, 4))
        expected = np.zeros((4, 5), bool)
        expected[1:3, 0] = True
        expected[3, 1:3] = True
        np.testing.assert_array_equal(restored, expected)

    def test_collector_recovers_score_tables_from_the_completed_run_directories(self):
        with TemporaryDirectory() as directory, patch.object(Path, "home", return_value=Path(directory)):
            root = Path(directory)/"Vu_Hung/cv_nangcao/output/shape-20261121"
            study = root/"progressive-l40s"
            study.mkdir(parents=True)
            (root/"controlled-audit.json").write_text("{}")
            (study/"submissions.tsv").write_text("99 combined full\n100 shape cohort\n101 shape full\n102 combined full\n"
                                                 "104 audit-reference-intervals full\n")
            (study/"report-jobs.txt").write_text("full 103\n")
            (study/"shape-cohort-verification.json").write_text("{}")
            (study/"shape-refs-cohort.json").write_text("cohort transfer")
            (study/"combined-full-verification.json").write_text("{}")
            (study/"combined-registry.csv").write_text("registered")
            (study/"combined-refs-full-99.json").write_text("first full transfer")
            (study/"combined-timeout-99.json").write_text("verified timeout inventory")
            (study/"combined-refs-full-102.json").write_text("full transfer")
            (root/"full-reference-interval-replay.json").write_text("intervals")
            (study/"shape-reference-intervals-104.log").write_text("verified")
            names = ("execution.json", "per_target.csv", "summary.json", "run_config.json", "metric_config.json")
            for folder in (study/"shape-cohort", study/"combined"):
                folder.mkdir()
                for name in names:
                    (folder/name).write_text(folder.name + name)
            states = "99|TIMEOUT|24:00:00|0:1\n100|COMPLETED|1:00|0:0\n101|RUNNING|0:10|0:0\n102|COMPLETED|2:00|0:0\n103|PENDING|0:00|0:0"
            states += "\n104|COMPLETED|0:15|0:0"
            output = StringIO()
            with patch("subprocess.check_output", return_value=states), redirect_stdout(output):
                exec(collect_shape_results.PROBE, {})
            state = json.loads(output.getvalue())
            hashes = state["hashes"]
            self.assertEqual(state["interval_job"], "104")
            for name in ("full-reference-interval-replay.json", "progressive-l40s/shape-reference-intervals-104.log"):
                self.assertEqual(hashes[name], sha256(root / name))
            for folder in ("shape-cohort", "combined"):
                for name in names:
                    key = f"progressive-l40s/{folder}/{name}"
                    self.assertEqual(hashes[key], hashlib.sha256((root/key).read_bytes()).hexdigest())
            self.assertNotIn("progressive-l40s/shape/per_target.csv", hashes)
            for name in ("shape-refs-cohort.json", "combined-refs-full-99.json", "combined-refs-full-102.json",
                         "combined-timeout-99.json"):
                self.assertEqual(hashes[f"progressive-l40s/{name}"], hashlib.sha256((study/name).read_bytes()).hexdigest())

    def test_collector_follows_replacement_jobs_and_keeps_old_accounting(self):
        with TemporaryDirectory() as directory, patch.object(Path, "home", return_value=Path(directory)):
            root = Path(directory)/"Vu_Hung/cv_nangcao/output/shape-20261121"
            study = root/"progressive-l40s"
            study.mkdir(parents=True)
            (root/"controlled-audit.json").write_text("{}")
            (study/"submissions.tsv").write_text("100 shape full\n101 report full\n102 shape full\n")
            (study/"report-jobs.txt").write_text("full 103\n")
            account = ("100|TIMEOUT|24:00:00|0:0\n101|CANCELLED by 0|00:00:00|0:0\n"
                       "102|RUNNING|00:01:00|0:0\n103|PENDING|00:00:00|0:0")
            printed = StringIO()
            with patch("subprocess.check_output", return_value=account), redirect_stdout(printed):
                exec(collect_shape_results.PROBE, {})
            state = json.loads(printed.getvalue())
            self.assertEqual(state["current_job_ids"], ["102", "103"])
            self.assertEqual(set(state["jobs"]), {"100", "101", "102", "103"})
            with (study/"submissions.tsv").open("a") as stream:
                stream.write("104 resume-shape full\n")
            repairing = StringIO()
            timed_out = account.replace("102|RUNNING", "102|TIMEOUT") + "\n104|PENDING|00:00|0:0"
            with patch("subprocess.check_output", return_value=timed_out), redirect_stdout(repairing):
                exec(collect_shape_results.PROBE, {})
            ongoing = json.loads(repairing.getvalue())
            self.assertEqual(ongoing["recovering"], ["102"])
            ongoing["observed_at"] = "during recovery"
        state["observed_at"] = "now"
        terminal = dict(state, jobs=state["jobs"] | {"102": ["TIMEOUT", "24:00:00", "0:0"]})
        with patch.object(sys, "argv", ["collect", "--watch"]), patch.object(
            collect_shape_results, "collect", side_effect=[state, ongoing, terminal]
        ), patch("time.sleep") as sleep, redirect_stdout(StringIO()):
            with self.assertRaisesRegex(SystemExit, r"102 \(TIMEOUT\)"):
                collect_shape_results.main()
            self.assertEqual([call.args for call in sleep.call_args_list], [(180,), (180,)])

    def test_collector_waits_for_reference_intervals_after_report_finishes(self):
        state = dict(observed_at="now", counts={}, hashes={}, export_jobs_by_arm={}, export_jobs=[],
                     full_report_job="103", interval_job="104", current_job_ids=["103", "104"])
        waiting = dict(state, jobs={"103": ["COMPLETED"], "104": ["PENDING"]})
        finished = dict(state, jobs={"103": ["COMPLETED"], "104": ["COMPLETED"]})
        with patch.object(sys, "argv", ["collect", "--watch"]), patch.object(
            collect_shape_results, "collect", side_effect=[waiting, finished]
        ), patch.object(collect_shape_results, "check_interval_receipt") as verify, patch.object(
            collect_shape_results, "replay_full_slide_tables"
        ) as slides, patch("time.sleep") as sleep, redirect_stdout(StringIO()):
            collect_shape_results.main()
        sleep.assert_called_once_with(180)
        verify.assert_called_once_with(collect_shape_results.OUT)
        slides.assert_called_once_with()

    def test_appearance_table_retains_penalties_and_counts_paired_clip_scores(self):
        saved = {"CLIP": 25., "LPIPS": .5, "Feature Similarity": .4, "SSIM": .6}
        failed = {"CLIP": None, "LPIPS": 1., "Feature Similarity": 0., "SSIM": 0.}
        rows = [dict(source="VG", image_identity="a", baseline=saved, surface=saved),
                dict(source="VG", image_identity="b", baseline=failed, surface=saved)]
        estimates = {metric: build_shape_report.estimate(rows, metric) for metric in METRICS}
        lines = build_shape_report.appearance_table(estimates)
        self.assertIn("| CLIP ↑ | 1 | 25.000000 | 25.000000 | +0.000% | [+0.000%, +0.000%] |", lines)
        self.assertTrue(any(line.startswith("| LPIPS ↓ | 2 | 0.750000 | 0.500000 | +33.333%") for line in lines))
        self.assertIn("| VGG feature similarity ↑ | 2 | 0.200000 | 0.400000 | +100.000% | — |", lines)
        estimates["CLIP"] = None
        self.assertIn("| CLIP ↑ | 0 | — | — | — | — |", build_shape_report.appearance_table(estimates))

    def test_source_results_include_failed_targets_and_use_paired_comparators(self):
        rows = []
        for source, image, scores, statuses in [
            ("VG", "a", (.5, 1., .25), ("success", "no_completion", "success")),
            ("VG", "b", (1., .5, .5), ("no_completion", "success", "success")),
            ("COCOA", "c", (0., 0., 0.), ("success", "success", "success")),
        ]:
            arms = ("baseline", "shape", "combined")
            row = dict(source=source, image_identity=image, status=dict(zip(arms, statuses)))
            row.update({arm: {"LPIPS": score} for arm, score in zip(arms, scores)})
            rows.append(row)
        lines = build_shape_report.source_table(rows, historical=True)
        self.assertIn("comparisons against PMC are cross-device", "\n".join(lines))
        self.assertTrue(lines[-2].startswith("| COCOA | 1 | 1 / 1 / 1 | undefined [undefined]"))
        cells = [cell.strip() for cell in lines[-1].strip("|").split("|")]
        self.assertEqual(cells[:3], ["VG", "2", "1 / 1 / 2"])
        gains = [float(cell.split("%")[0]) for cell in cells[3:]]
        self.assertEqual(gains, [0., 50., 50.])

    def test_severity_table_counts_saved_outputs_from_status(self):
        records, annotations = [], []
        for group, size in enumerate([203, 54, 39, 21, 4]):
            for index in range(size):
                tid = f"{group}/{index}"
                ratio = 1.0 if group == 4 else group / 5
                annotations.append({"id": tid, "ratio": ratio})
                records.append({"id": tid, "shape": {"LPIPS": 1.0},
                                "status": {"shape": "success" if index == 0 else "no_completion"}})
        bins = build_shape_report.severity_table(records, annotations, ["shape"], include_saved=True)
        self.assertEqual([row["shape_saved"] for row in bins], [1] * 5)
        self.assertEqual([row["shape"] for row in bins], [1.0] * 5)
        annotations[-1] = annotations[-2]
        with self.assertRaisesRegex(ValueError, "Duplicate COCOA annotation"):
            build_shape_report.severity_table(records, annotations, ["shape"], include_saved=True)

    def test_shape_severity_is_available_while_combined_is_pending(self):
        annotations, rows = [], []
        scores = {"baseline": .6, "shape": .5, "surface": .4, "combined": .3}
        for group, size in enumerate((203, 54, 39, 21, 4)):
            for index in range(size):
                tid = f"{group}/{index}"
                annotations.append({"id": tid, "ratio": group / 5})
                rows.append({"id": tid, "status": dict.fromkeys(scores, "success"),
                             **{arm: {"LPIPS": value, "amodal_iou": 1 - value} for arm, value in scores.items()}})
        with patch.object(build_shape_report, "read", return_value={"rows": annotations}), patch.object(
                build_shape_report, "render_severity") as render:
            early = build_shape_report.severity_section(rows, ["baseline", "surface", "shape"])
            render.assert_not_called()
            self.assertIn("### Slide 20 · COCOA occlusion severity", early)
            self.assertIn("Full results pending verification for Shape + Surface · L40S.", early)
            self.assertIn("| 80–100% | 4 | 0.6000 (4 saved) | 0.5000 (4 saved) | 0.4000 (4 saved) |", early)
            self.assertIn("| 80–100% | 4 | 0.4000 | 0.5000 | 0.6000 |", early)
            complete = build_shape_report.severity_section(rows, list(scores))
            render.assert_called_once()
            self.assertEqual([entry["n"] for entry in render.call_args.args[0]], [203, 54, 39, 21, 4])
            self.assertIn("| 80–100% | 4 | 0.4000 | 0.5000 | 0.6000 | 0.7000 |", complete)
            groups = [[a["id"] for a in annotations if a["ratio"] == group / 5] for group in range(5)]
            values = {}
            for arm in scores:
                native = {row["id"]: dict(LPIPS=row[arm]["LPIPS"], SSIM=.2, status="success") for row in rows}
                masks = {row["id"]: dict(success=True, amodal_iou=row[arm]["amodal_iou"]) for row in rows}
                values[arm] = dict(method_values(native, masks, groups), gpu_hours=2.0)
            report = "\n".join(["### 2,565 natural targets",
                "| Method | GPU | K per pass | Failure-inclusive LPIPS ↓ | SSIM ↑ | Saved | GPU-hours allocated |",
                "| PMC | Quadro RTX 6000 (historical) | 1 | 0.600000 | 0.200000 | 321 | 2.000 |",
                "| Surface | Quadro RTX 6000 (historical) | 8 | 0.400000 | 0.200000 | 321 | 2.000 |",
                "| Shape | L40S | 8 | 0.500000 | 0.200000 | 321 | 2.000 |",
                "| Shape + Surface | L40S | 8 | 0.300000 | 0.200000 | 321 | 2.000 |", *complete])
            exported = render.call_args.args[0]
            match_report(values, groups, report, exported)
            match_report(values, groups, report.replace("### Slide 20 ·", "### Slide 18 ·"), exported)
            with self.assertRaisesRegex(ValueError, "Full benchmark values"):
                match_report(values, groups, report.replace("2.000 |", "4.000 |"), exported)
            with self.assertRaisesRegex(ValueError, "method columns"):
                match_report(values, groups, report.replace("Shape · L40S", "PMC · Quadro"), exported)
            with self.assertRaisesRegex(ValueError, "COCOA severity report"):
                match_report(values, groups, report.replace("0.5000 (4 saved)", "0.5000 (3 saved)"), exported)
            with self.assertRaisesRegex(ValueError, "CSV differs"):
                match_report(values, groups, report, [*exported[:-1], dict(exported[-1], combined=.9)])

    def test_unseen_polygon_report_uses_paired_images_and_arm_specific_audits(self):
        rows = [dict(source="COCOA", image_identity=f"COCOA/{i % 202}",
                     baseline={"amodal_iou": .5}, surface={"amodal_iou": .375}) for i in range(211)]
        rows.append(dict(source="VG", image_identity="VG/1", baseline={}, surface={}))
        summaries = {arm: {"metrics": {
            control + "-unseen": {"amodal_iou": dict(n=211, images=202, method=score, baseline=prior)}
            for control, prior in (("baseline", .4), ("surface", .45))}}
            for arm, score in (("shape", .5), ("combined", .375))}
        paragraph = build_shape_report.unseen_geometry_section(rows, summaries)
        self.assertIn("211 đối tượng có đường bao COCOA thuộc 202 ảnh", paragraph)
        self.assertIn("0,4000 với PMC và 0,4500 với Surface", paragraph)
        self.assertIn("-0,1250 (KTC 95% từ -0,1250 đến -0,1250)", paragraph)
        summaries["combined"]["metrics"]["surface-unseen"]["amodal_iou"]["baseline"] = .4
        with self.assertRaisesRegex(ValueError, "Historical surface scores differ"):
            build_shape_report.unseen_geometry_section(rows, summaries)

    def test_vietnamese_conclusion_converts_gain_to_lpips_change_with_ordered_interval(self):
        full = dict(n=2565, baseline=.60, method=.594, relative_gain_percent=1.,
                    relative_ci_percent=[-.5, 2.3])
        unseen = dict(n=2017, baseline=.61, method=.59)
        geometry = dict(n=321, baseline=.45, method=.47)
        conclusion = " ".join(build_shape_report.full_conclusion_vi(
            full, unseen, geometry, {"shape": 60.25, "combined": 70.}))
        self.assertIn("thay đổi -1,00% (KTC 95% từ -2,30% đến 0,50%)", conclusion)
        self.assertIn("60,25 giờ-GPU L40S", conclusion)
        with self.assertRaisesRegex(ValueError, "denominators"):
            build_shape_report.full_conclusion_vi({**full, "n": 204}, unseen, geometry, {})

    def test_full_control_table_uses_separate_appearance_and_outline_estimates(self):
        labels = {"shape": "Shape", "combined": "Shape + Surface", "baseline": "PMC", "surface": "Surface"}
        lpips = dict(n=2565, absolute_gain=-.01, absolute_ci=[-.02, -.005])
        geometry = dict(n=321, absolute_gain=.02, absolute_ci=[-.001, .04])
        metrics = {"baseline-all": {"LPIPS": lpips, "amodal_iou": geometry},
                   "surface-all": {"LPIPS": dict(lpips, absolute_gain=-.015), "amodal_iou": geometry}}
        summary = dict(historical_controls=True, controls=["baseline", "surface"], metrics=metrics)
        summaries = {"shape": summary}
        table = build_shape_report.historical_control_table_vi(summaries, labels)
        self.assertEqual(table[-2], "| Shape | PMC | -0,0100 (-0,0200 đến -0,0050) | +0,0200 (-0,0010 đến +0,0400) |")
        self.assertIn("| Shape | Surface | -0,0150", table[-1])
        self.assertFalse(any("Shape + Surface" in line for line in table))
        summaries["combined"] = summary
        self.assertIn("| Shape + Surface | PMC |", build_shape_report.historical_control_table_vi(summaries, labels)[-2])
        self.assertEqual([], build_shape_report.historical_control_table_vi({"shape": {"historical_controls": False}}, labels))
        hidden = dict(n=230, images=220, method=.24, baseline=.23, absolute_gain=.01, absolute_ci=[-.01, .03])
        metrics["surface-all"]["hidden_iou"] = hidden
        table = build_shape_report.historical_control_table_vi(summaries, labels)
        self.assertIn("| Shape | Surface | 0,2400 | 0,2300 | +0,0100 (-0,0100 đến +0,0300) |", table)
        with self.assertRaisesRegex(ValueError, "hidden-region"):
            metrics["surface-all"]["hidden_iou"] = dict(hidden, n=321)
            build_shape_report.historical_control_table_vi(summaries, labels)
        del metrics["surface-all"]["hidden_iou"]
        with self.assertRaisesRegex(ValueError, "amodal_iou target count"):
            metrics["baseline-all"]["amodal_iou"] = dict(geometry, n=22)
            build_shape_report.historical_control_table_vi(summaries, labels)

    def test_coverage_contributions_charge_each_group_to_the_full_denominator(self):
        scores = [("success", "success", .5, .25), ("success", "success", .25, .25),
                  ("no_completion", "success", 1., .5), ("success", "not_detected", .25, 1.),
                  ("ambiguous_target", "ambiguous_target", 1., 1.)]
        rows = [dict(status=dict(baseline=a, shape=b), baseline={"LPIPS": base}, shape={"LPIPS": method})
                for a, b, base, method in scores]
        self.assertEqual(build_shape_report.coverage_contributions(rows, "shape", 0.),
                         {(True, True): (2, .05), (False, True): (1, .1),
                          (True, False): (1, -.15), (False, False): (1, 0.)})
        self.assertEqual(build_shape_report.coverage_contributions(rows, "baseline", 0., comparator="shape"),
                         {(True, True): (2, -.05), (False, True): (1, .15),
                          (True, False): (1, -.1), (False, False): (1, 0.)})
        with self.assertRaisesRegex(ValueError, "audited all-target"):
            build_shape_report.coverage_contributions(rows, "shape", .1)

    def test_cross_device_coverage_uses_the_archived_surface_scores(self):
        rows = [dict(status=dict(combined="no_completion", surface="success", baseline="no_completion"),
                     combined={"LPIPS": 1.}, surface={"LPIPS": .25}, baseline={"LPIPS": 1.})]
        label = {"surface": "Surface", "combined": "Shape + Surface"}
        self.assertEqual(build_shape_report.coverage_row(rows, "combined", "surface", -.75, label, cross_device=True),
                         "| Shape + Surface | Surface (Quadro, cross-device) | 0 (+0.000000) | "
                         "0 (+0.000000) | 1 (-0.750000) | 0 (+0.000000) |")

    def test_surface_polygon_comparison_pairs_by_target_and_retains_failed_completions(self):
        common = dict(source="COCOA", image_identity="same-image")
        archived = [dict(id="b", **common, surface={"amodal_iou": .4}),
                    dict(id="a", **common, surface={"amodal_iou": 0.})]
        pairs = [dict(id="a", **common, shape={"amodal_iou": .6}),
                 dict(id="b", **common, shape={"amodal_iou": 0.}),
                 dict(id="outside", **common, shape={})]
        result = build_shape_report.surface_cohort_geometry(pairs, "shape", archived)
        self.assertEqual((result["n"], result["images"]), (2, 1))
        self.assertAlmostEqual(result["baseline"], .2)
        self.assertAlmostEqual(result["method"], .3)
        self.assertAlmostEqual(result["absolute_gain"], .1)
        with self.assertRaisesRegex(ValueError, "membership"):
            build_shape_report.surface_cohort_geometry(pairs, "shape", archived[:1])
        archived[0]["image_identity"] = "different-image"
        with self.assertRaisesRegex(ValueError, "identity"):
            build_shape_report.surface_cohort_geometry(pairs, "shape", archived)

    def test_first_pass_selection_requires_identical_candidates_and_observations(self):
        with TemporaryDirectory() as directory:
            shape, combined = (Path(directory)/arm for arm in ("shape","combined"))
            self.assertEqual(compare_first_pass(shape, combined), "neither_sampled")
            files = ["observed.png", "visible.png", "occlusion.png", "prior.png"]
            files += [f"{kind}-{i}.png" for kind in ("candidate","mask") for i in range(8)]
            decision = dict(iteration=0, crop_rc=[0,4,0,4], target_seed=8, seeds=list(range(8)),
                            index=1, artifacts={name:name for name in files})
            for arm in (shape, combined):
                (arm/"iteration-00").mkdir(parents=True)
            (shape/"iteration-00/decision.json").write_text(json.dumps(decision))
            self.assertEqual(compare_first_pass(shape, combined), "shape_only")
            result = combined/"iteration-00/decision.json"
            result.write_text(json.dumps(decision))
            self.assertEqual(compare_first_pass(shape, combined), "same_slot")
            decision["index"] = 3
            result.write_text(json.dumps(decision))
            self.assertEqual(compare_first_pass(shape, combined), "changed_slot")
            decision["artifacts"]["mask-7.png"] = "changed"
            result.write_text(json.dumps(decision))
            self.assertEqual(compare_first_pass(shape, combined), "different_inputs_or_pool")

    def test_reference_transfer_replays_original_and_resumed_receipts_against_native_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root/"shape", root/"combined"
            row = {"reference": "VG/lamp.png"}
            paths = [Path("reference_config.json"), *reference_paths(Path("."), row).values()]
            for arm in (source, destination):
                for relative in paths:
                    target = arm/"visible_references"/relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(relative.as_posix().encode())
            files = {p.as_posix(): sha256(source/"visible_references"/p) for p in paths}
            receipt = dict(verified=True, source=str(source), destination=str(destination),
                           source_targets=1, files=files)
            first, last = root/"combined-refs-full-1.json", root/"combined-refs-full-2.json"
            write_json(first, dict(receipt, new_files=5))
            write_json(last, dict(receipt, new_files=0))
            result = verify_reference_transfers(source, destination, [row], [first, last])
            self.assertEqual(result, dict(targets=1, files=5, receipts={
                first.name: {"sha256": sha256(first), "new_files": 5},
                last.name: {"sha256": sha256(last), "new_files": 0},
            }))
            write_json(first, dict(receipt, files={}, new_files=5))
            with self.assertRaisesRegex(ValueError, first.name):
                verify_reference_transfers(source, destination, [row], [first, last])
            (destination/"visible_references"/paths[-1]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Copied reference differs"):
                verify_reference_transfers(source, destination, [row], [last])
            (source/"visible_references"/paths[-1]).unlink()
            with self.assertRaisesRegex(ValueError, "missing files"):
                verify_reference_transfers(source, destination, [row], [last])

    def test_full_run_keeps_each_frozen_cohort_record_through_a_continuation(self):
        with TemporaryDirectory() as directory:
            run = Path(directory)/"full"
            snap = Path(directory)/"cohort"
            run.mkdir()
            snap.mkdir()
            previous = dict(status="complete", result_artifacts={"existing":"original"})
            full = dict(status="complete", result_artifacts={"existing":"original", "new":"added"},
                        new_or_updated_target_ids=["new"])
            header = "id,reference_status,CLIP,LPIPS,Feature Similarity,SSIM\n"
            for folder, execution, values in (
                (snap, previous, "existing,ready,,1,0,0\n"),
                (run, full, "new,ready,22,.2,.3,.4\nexisting,ready,,1.0,0.0,0.0\n"),
            ):
                scores = folder/"per_target.csv"
                scores.write_text(header + values, encoding="utf8")
                execution["artifacts"] = {"per_target.csv": hashlib.sha256(scores.read_bytes()).hexdigest()}
            (snap/"execution.json").write_text(json.dumps(previous))
            (run/"execution.json").write_text(json.dumps(full))
            reuse = verify_cohort_reuse(run, snap, {"existing"})
            self.assertEqual((reuse["preserved_target_records"], reuse["preserved_score_rows"]), (1, 1))
            scores = run/"per_target.csv"
            scores.write_text(header + "new,ready,22,.2,.3,.4\nexisting,ready,,.9,0,0\n", encoding="utf8")
            with self.assertRaisesRegex(ValueError, "verified execution"):
                verify_cohort_reuse(run, snap, {"existing"})
            full["artifacts"]["per_target.csv"] = hashlib.sha256(scores.read_bytes()).hexdigest()
            (run/"execution.json").write_text(json.dumps(full))
            with self.assertRaisesRegex(ValueError, "existing / LPIPS"):
                verify_cohort_reuse(run, snap, {"existing"})
            scores.write_text(header + "new,ready,22,.2,.3,.4\nexisting,missing,,1,0,0\n", encoding="utf8")
            full["artifacts"]["per_target.csv"] = hashlib.sha256(scores.read_bytes()).hexdigest()
            (run/"execution.json").write_text(json.dumps(full))
            with self.assertRaisesRegex(ValueError, "existing / reference_status"):
                verify_cohort_reuse(run, snap, {"existing"})
            full["result_artifacts"]["existing"] = "replacement"
            (run/"execution.json").write_text(json.dumps(full))
            with self.assertRaisesRegex(ValueError, "changed completed cohort results"):
                verify_cohort_reuse(run, snap, {"existing"})

    def test_natural_geometry_checks_native_alpha_and_clips_an_outpainted_crop(self):
        truth = np.zeros((4, 6), dtype=bool)
        truth[:2, 3:5] = True
        mask = Image.new("L", (5, 3), 255)
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            mask.save(folder/"mask.png")
            image = Image.new("RGBA", mask.size)
            image.putalpha(mask)
            image.save(folder/"completion.png")
            record = dict(id="COCOA/test", status="success", geometry=dict(original_size=[6,4], offset_rc=[-1,3]),
                          completion_sha256=hashlib.sha256((folder/"completion.png").read_bytes()).hexdigest())
            self.assertAlmostEqual(natural_geometry(folder, record, truth), 4/6)
            visible = np.zeros_like(truth)
            visible[0, 3:6] = True
            self.assertAlmostEqual(natural_geometry(folder, record, truth, exclude=visible), 2/3)
            visible[1, 3] = True
            self.assertEqual(natural_geometry(folder, record, truth, exclude=visible), 1/2)
            self.assertEqual(natural_geometry(folder, {"status": "no_completion"}, truth, exclude=visible), 0.)
            self.assertIsNone(natural_geometry(folder, record, truth, exclude=truth))
            self.assertIsNone(natural_geometry(folder, {"status": "no_completion"}, truth, exclude=truth))
            with self.assertRaisesRegex(ValueError, "annotation frame"):
                natural_geometry(folder, record, truth, exclude=visible[:1])
            mask.putpixel((0, 1), 0)
            mask.save(folder/"mask.png")
            with self.assertRaisesRegex(ValueError, "completion alpha"):
                natural_geometry(folder, record, truth)
            self.assertEqual(natural_geometry(folder, {"status": "no_completion"}, truth), 0.)

    def test_shape_uses_mask_geometry_and_falls_back_on_empty_pool(self):
        truth = np.eye(4, dtype=bool)
        empty = np.zeros_like(truth)
        self.assertEqual(score_masks(truth, [truth, empty, np.flipud(truth)]), [1., None, 0.])
        self.assertEqual(choose([None, .7, .7]), (1, False))
        self.assertEqual(choose([None, None]), (0, True))
        with self.assertRaises(ValueError):
            overlap(truth, empty[0])

    def test_joint_rule_uses_intersection_of_valid_masks_and_stable_midranks(self):
        self.assertEqual(choose([1., .6, .7, None], [None, 3., 1., 0.]), (2, False))
        self.assertEqual(choose([.9, .9, .6], [1., 0., 0.]), (1, False))
        self.assertEqual(choose([.9, .7], [1., 0.]), (0, False))

    def test_joint_report_requires_identical_native_references_even_if_historical_control_agrees(self):
        row = dict(id="VG/example", source="VG", image_identity="VG/1", gt_bin=None,
                   input_sha256="same-input", seed=11, reference_equal=True,
                   baseline={"LPIPS": .5}, method_reference={"mask": "original"})
        with TemporaryDirectory() as directory, patch.object(build_shape_report, "OUT", Path(directory)):
            summaries = {}
            for arm in ("shape", "combined"):
                candidate = dict(row, status={arm: "success", "baseline": "success"})
                candidate[arm] = {"LPIPS": .25}
                if arm == "combined":
                    candidate["first_pass_comparison"] = "changed_slot"
                path = Path(directory)/f"{arm}-cohort-vs-pmc-pairs.json"
                path.write_text(json.dumps([candidate]), encoding="utf8")
                summaries[arm] = dict(controls=["baseline"], targets=1,
                                      pairs_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            summaries["combined"]["first_pass_comparison"] = {"changed_slot": 1}
            matched = build_shape_report.matched_shape_rows("cohort", summaries)
            self.assertEqual(matched[0]["combined"]["LPIPS"], .25)
            summaries["combined"]["first_pass_comparison"] = {"same_slot": 1}
            with self.assertRaisesRegex(ValueError, "First-pass counts"):
                build_shape_report.matched_shape_rows("cohort", summaries)
            summaries["combined"]["first_pass_comparison"] = {"changed_slot": 1}
            candidate["status"]["baseline"] = "no_completion"
            path.write_text(json.dumps([candidate]), encoding="utf8")
            summaries["combined"]["pairs_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "Control status"):
                build_shape_report.matched_shape_rows("cohort", summaries)
            candidate["status"]["baseline"] = "success"
            candidate["method_reference"] = {"mask": "different"}
            path.write_text(json.dumps([candidate]), encoding="utf8")
            summaries["combined"]["pairs_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "method_reference"):
                build_shape_report.matched_shape_rows("cohort", summaries)

    def test_exact_reference_interval_uses_paired_subset_and_names_both_devices(self):
        result = dict(n=166, images=159, method=.74218, baseline=.72843,
                      relative_gain_percent=-1.88731, relative_ci_percent=[-3.87298, -.61615])
        sentence = build_shape_report.exact_reference_lpips(result, "Shape", "Surface")
        self.assertIn("166 targets from 159 images", sentence)
        self.assertIn("0.742180 for Shape on L40S and 0.728430 for Surface on Quadro", sentence)
        self.assertIn("-1.887% (nominal 95% interval -3.873% to -0.616%)", sentence)
        self.assertIn("raw SAM masks, full RGBA images and crops", sentence)
        self.assertIn("cropped RGBA scoring images", build_shape_report.exact_reference_lpips(
            result, "Shape", "Surface", crops_only=True))

    def test_scoring_crop_comparison_includes_changed_raw_masks_and_failure_penalties(self):
        pairs, refs, scores = [], [], {}
        for tid, crop in (("same-crop", "old"), ("different-crop", "new")):
            current = {"crop": crop, "mask": "new-mask", "full": "eroded"}
            pairs.append(dict(id=tid, seed=11, source="VG", image_identity=tid, input_sha256=tid,
                              method_reference=current, baseline={"LPIPS": 1.}, shape={"LPIPS": .5},
                              combined={"LPIPS": .25}))
            refs.append(dict(id=tid, input_sha256=tid, current_artifacts=current,
                             historical_artifacts={"crop": "old", "mask": "old-mask", "full": "eroded"}))
            scores[tid] = {"source": "VG", "seed": "11", "LPIPS": .25}
        controls = {"baseline": scores, "surface": scores}
        lines = build_shape_report.cohort_scoring_crop_table(pairs, refs, controls)
        self.assertIn("| PMC | PMC | 1 | 1.000000 | 0.250000 | -300.000% | [-300.000%, -300.000%] |", lines)
        refs[0]["input_sha256"] = "different-input"
        with self.assertRaisesRegex(ValueError, "Scoring reference mismatch"):
            build_shape_report.cohort_scoring_crop_table(pairs, refs, controls)

    def test_archived_surface_cohort_row_includes_penalties_and_labels_hardware(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"surface-cohort").mkdir()
            (root/"report").mkdir()
            path = root/"surface-cohort/per_target.csv"
            path.write_text("status,LPIPS,SSIM\n" + "success,0.25,0.625\n" * 102
                            + "no_completion,1.0,0.0\n" * 102, encoding="utf8")
            cost = {"surface-cohort": {"job_id": "39487", "state": "COMPLETED",
                                       "reserved_gpu_hours": 7.3044444}}
            (root/"report/cohort-compute.json").write_text(json.dumps(cost), encoding="utf8")
            paired = {"surface_scores_sha256": sha256(path), "all": {"n": 204, "baseline": .625}}
            self.assertEqual(build_shape_report.archived_surface_cohort_row(root, paired),
                             "| Surface | Quadro RTX 6000 (historical) | 8 | 0.625000 | 0.312500 | 102 | 7.304 |")
            path.write_text(path.read_text().replace("0.625", "0.524"), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "scores differ"):
                build_shape_report.archived_surface_cohort_row(root, paired)

    def test_gpu_hours_include_timed_out_work_and_reused_cohort_once(self):
        arms = ("baseline", "shape", "surface", "combined")
        submissions = []
        records = {}
        for stage in ("cohort", "full"):
            for arm in arms:
                job = str(len(submissions) + 1)
                submissions.append(f"{job}\t{arm}\t{stage}\n")
                records[job] = "COMPLETED|900|gres/gpu:nvidia_l40s=4"
        submissions.append("9\tshape\tfull\n")
        records["6"] = "TIMEOUT|1800|gres/gpu:nvidia_l40s=4"
        records["9"] = "COMPLETED|450|gres/gpu:nvidia_l40s=4"

        def sacct(command, text):
            jobs = command[command.index("-j") + 1].split(",")
            return "\n".join(f"{job}|{records[job]}" for job in jobs if job in records)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            receipts = root / "progressive-l40s"
            receipts.mkdir()
            (receipts / "submissions.tsv").write_text("".join(submissions), encoding="utf8")
            with patch.object(build_shape_report, "OUT", root), patch.object(
                build_shape_report.subprocess, "check_output", side_effect=sacct
            ):
                self.assertEqual(build_shape_report.allocation("cohort"), dict.fromkeys(arms, 1.))
                self.assertEqual(build_shape_report.allocation("full"),
                                 dict(baseline=2., shape=3.5, surface=2., combined=2.))
                del records["6"]
                with self.assertRaisesRegex(ValueError, "Incomplete Slurm accounting"):
                    build_shape_report.allocation("full")

