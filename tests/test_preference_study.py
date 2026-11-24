"""Regression checks for study integrity and actual human-response arithmetic."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_preference_study import align, build, common_bounds
from scripts.summarize_preference_study import summarize, validate
from scripts.validate_preference_study import validate as validate_packet


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class PreferenceStudyTests(unittest.TestCase):
    def fixture(self, root):
        methods = [{"id": "baseline", "name": "Baseline", "run": "runs/baseline"},
                   {"id": "surface", "name": "Surface", "run": "runs/surface"}]
        for method in methods:
            run = root / method["run"]
            table = []
            for i in range(5):
                target = f"VG/target{i}"
                source = root / f"data/images/{i}.png"
                source.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 24), (60 + i, 80, 90)).save(source)
                folder = run / "targets" / target
                folder.mkdir(parents=True)
                visible = run / f"visible_references/{i}.png"
                visible.parent.mkdir(parents=True, exist_ok=True)
                im = Image.new("RGBA", (32, 24))
                im.paste((100, 90, 80, 255), (8, 8, 15, 20))
                im.save(visible)
                result = {"id": target, "label": "chair", "source": "VG", "image": f"images/{i}.png",
                          "input_sha256": sha(source), "status": "success",
                          "geometry": {"original_size": [32, 24], "offset_rc": [-3, -5]},
                          "reference": f"{i}.png", "evaluation_reference_artifacts": {"full": sha(visible)}}
                output = Image.new("RGBA", (40, 30), (0, 0, 0, 0))
                output.paste((110 + i, 50, 80, 255), (0, 0, 30, 25))
                output.save(folder / "completion.png")
                result["completion_sha256"] = sha(folder / "completion.png")
                if method["id"] == "surface" and i == 4:
                    result["status"] = "not_detected"
                write(folder / "result.json", result)
                table.append({"id": target, "status": result["status"]})
            write(run / "run_config.json", {"detector_vocabulary": {"mode": "annotated"}})
            with (run / "per_target.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, ["id", "status"])
                writer.writeheader(); writer.writerows(table)
            write(run / "execution.json", {"id": method["id"] + "-execution", "status": "complete",
                  "target_ids": [r["id"] for r in table],
                  "artifacts": {name: sha(run / name) for name in ("run_config.json", "per_target.csv")}})
        config = {"round": "test", "seed": 10, "title": "Test", "expected_targets": 5,
                  "dataset_root": "data", "methods": methods}
        write(root / "config.json", config)
        return root / "config.json"

    def response(self, study):
        rows = []
        for c in study["cases"]:
            rows.append({"case_id": c["id"], "status": "rated",
                         "judgements": {"overall": {"kind": "preference", "winners": ["A"]}, "shape": None, "surface": None},
                         "note": "", "confidence": "", "skip_reason": "",
                         "created_at": "2026-11-22T00:00:00Z", "updated_at": "2026-11-22T00:00:00Z"})
        return {"schema": "amodal-preference-response-v1", "study_id": study["study_id"],
                "round": study["round"], "complete": True, "exported_at": "2026-11-22T00:00:00Z",
                "reviewer": {"code": "TEST_ONLY", "role": "researcher", "prior_exposure": "yes"}, "records": rows}

    def test_common_frame_preserves_negative_offset_support(self):
        im = Image.new("RGBA", (20, 10), (1, 2, 3, 255))
        bounds = common_bounds([(im, (-4, -7)), (im, (6, 10))])
        self.assertEqual(bounds, (-15, -12, 38, 24))
        mapped = align(im, (-4, -7), bounds)
        self.assertEqual(mapped.getchannel("A").getbbox(), (8, 8, 28, 18))

    def test_builder_freezes_all_shared_outputs_without_metric_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = self.fixture(root)
            study = build(root, config, root / "packet", 2)
            self.assertEqual((study["eligible"], study["excluded"]), (4, 1))
            result = validate_packet(root / "packet")
            self.assertEqual(result["images"], 20)
            self.assertEqual(result["position_counts"]["A"], {"baseline": 2, "surface": 2})
            for case in study["cases"]:
                self.assertEqual(set(case), {"id", "label", "files"})
            repeat = build(root, config, root / "repeat", 2)
            self.assertEqual(study, repeat)
            with self.assertRaisesRegex(ValueError, "overwritten"):
                build(root, config, root / "packet")
            asset = root / "packet/dist" / study["cases"][0]["files"]["A"]
            asset.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Image changed"):
                validate_packet(root / "packet")

    def test_builder_rejects_changed_native_image_and_partial_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = self.fixture(root)
            path = root / "runs/baseline/targets/VG/target0/completion.png"
            path.write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "Artifact hash mismatch"):
                build(root, config, root / "bad")
            execution = root / "runs/surface/execution.json"
            value = json.loads(execution.read_bytes()); value["status"] = "running"; write(execution, value)
            with self.assertRaisesRegex(ValueError, "not complete"):
                build(root, config, root / "unfinished")

    def test_ties_unknown_skips_and_incomplete_have_distinct_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = self.fixture(root)
            study = build(root, config, root / "packet", 2)
            key = json.loads((root / "packet/COORDINATOR_ONLY/key.json").read_bytes())
            response = self.response(study)
            response["records"][1]["judgements"]["overall"]["winners"] = ["A", "B"]
            response["records"][2]["judgements"]["overall"] = {"kind": "unassessable", "winners": []}
            response["records"][3].update(status="skipped", skip_reason="Cannot identify")
            response["records"][3]["judgements"]["overall"] = None
            summary, _ = summarize(response, study, key)
            result = summary["criteria"]["overall"]
            self.assertEqual((result["assessable"], result["cannot_judge"], summary["skipped"]), (2, 1, 1))
            self.assertEqual(sorted(m["preference_percent"] for m in result["methods"].values()), [25, 75])
            response["records"].pop(); response["complete"] = False
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                validate(response, study)
            self.assertEqual(len(validate(response, study, allow_partial=True)), 3)

    def test_three_method_nonwinners_are_not_invented_ties(self):
        study = {"study_id": "x", "round": "r", "slots": ["A", "B", "C"],
                 "cases": [{"id": "P0001"}], "total_targets": 1, "excluded": 0}
        key = {"study_id": "x", "cases": [{"case_id": "P0001", "target_id": "VG/x", "image": "x",
               "assignment": {"A": "one", "B": "two", "C": "three"}}],
               "methods": [{"id": name} for name in ("one", "two", "three")], "coverage": {}}
        summary, _ = summarize(self.response(study), study, key)
        pair = next(p for p in summary["criteria"]["overall"]["pairwise"] if p["method_a"] == "two")
        self.assertEqual((pair["ties"], pair["not_compared"]), (0, 1))

    def test_response_rejects_wrong_round_duplicate_unknown_and_fake_completion(self):
        study = {"study_id": "x", "round": "r", "slots": ["A", "B"], "cases": [{"id": "P0001"}]}
        original = self.response(study)
        mutations = [lambda r: r.update(round="wrong"),
                     lambda r: r["records"].append(copy.deepcopy(r["records"][0])),
                     lambda r: r["records"][0]["judgements"]["overall"].update(winners=["C"]),
                     lambda r: r.update(complete=False),
                     lambda r: r["records"][0].update(status="skipped", skip_reason="later")]
        for change in mutations:
            response = copy.deepcopy(original); change(response)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(response, study)


if __name__ == "__main__":
    unittest.main()
