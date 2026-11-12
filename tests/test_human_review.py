"""Human review must preserve blinding, missingness and response identity."""
import importlib.util
import csv
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


prepare = module("prepare_human_review")
summarizer = module("summarize_human_review")


class HumanReviewTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {"pack_id": "test", "version": prepare.VERSION}
        self.mapping = [{"case_id": "H0001"}, {"case_id": "H0002"}]
        self.response = {**self.protocol, "reviewer_code": "TEST_ONLY",
                         "involved": "no", "exposed": "no", "disclosure": "",
                         "understood": True, "responses": {}}
        self.rating = {**{key: "0" for key in summarizer.FAILURES}, "complete": True,
                       "confounds": ["none_observed"], "evidence": "Synthetic test, not a real rating.",
                       "cooccurrence_subtype": "", "cooccurrence_output_scope": ""}

    def full(self):
        self.response["responses"] = {r["case_id"]: dict(self.rating) for r in self.mapping}
        return self.response

    def validate(self, partial=False):
        return summarizer.validate_response(self.response, self.protocol, self.mapping, partial)

    def test_random_order_reproducible_and_independent_of_input_order_or_scores(self):
        rows = [{"id": str(i), "status": "success", "CLIP": i} for i in range(30)]
        a = [r["id"] for r in prepare.ordered_cases(rows, 3)]
        b = [r["id"] for r in prepare.ordered_cases(
            [{**r, "CLIP": 1000 - r["CLIP"]} for r in reversed(rows)], 3)]
        self.assertEqual(a, b)
        self.assertEqual(len(set(a)), 30)
        self.assertNotEqual(a, [r["id"] for r in prepare.ordered_cases(rows, 4)])

    def test_no_output_cases_are_not_visual_zero_ratings(self):
        self.assertEqual(prepare.ordered_cases([{"id": "failure", "status": "not_detected"}], 3), [])

    def test_incomplete_assignment_rejected_by_default(self):
        self.response["responses"]["H0001"] = self.rating
        with self.assertRaisesRegex(ValueError, "Incomplete assignment"):
            self.validate()
        self.assertEqual(set(self.validate(partial=True)), {"H0001"})

    def test_draft_with_zero_scores_is_not_complete(self):
        self.full()["responses"]["H0002"]["complete"] = False
        self.assertEqual(len(self.validate(partial=True)), 1)

    def test_completed_case_with_blank_is_rejected(self):
        self.full()["responses"]["H0001"]["wrong_texture"] = ""
        with self.assertRaisesRegex(ValueError, "missing wrong_texture"):
            self.validate()

    def test_na_only_allowed_for_pattern(self):
        self.full()["responses"]["H0001"]["pattern_inconsistency"] = "NA"
        self.assertEqual(len(self.validate()), 2)
        self.response["responses"]["H0001"]["semantic_drift"] = "NA"
        with self.assertRaisesRegex(ValueError, "semantic_drift"):
            self.validate()

    def test_unknown_and_na_excluded_not_passes(self):
        result = summarizer.label_summary(["0", "1", "2", "3", "U", "NA"])
        self.assertEqual((result["positive"], result["assessable"], result["unknown"], result["not_applicable"]), (3, 4, 1, 1))
        self.assertIsNone(summarizer.label_summary(["U", "NA"])["positive_fraction"])

    def test_positive_secondary_object_needs_scope_and_subtype(self):
        self.full()["responses"]["H0001"]["cooccurring_object"] = "2"
        with self.assertRaisesRegex(ValueError, "Secondary-object detail"):
            self.validate()

    def test_conflicting_confounds_rejected(self):
        self.full()["responses"]["H0001"]["confounds"] = ["none_observed", "target_switch"]
        with self.assertRaisesRegex(ValueError, "conflicting confounds"):
            self.validate()

    def test_wrong_packet_or_unexpected_case_rejected(self):
        self.full()["pack_id"] = "other"
        with self.assertRaisesRegex(ValueError, "Wrong packet"):
            self.validate()
        self.response["pack_id"] = "test"
        self.response["responses"]["H9999"] = self.rating
        with self.assertRaisesRegex(ValueError, "Unexpected response"):
            self.validate()

    def test_exposure_is_disclosed_not_silently_declared_independent(self):
        self.full()["exposed"] = "yes"
        with self.assertRaisesRegex(ValueError, "Explain"):
            self.validate()
        self.response["disclosure"] = "Previously saw a report."
        self.assertEqual(len(self.validate()), 2)

    def test_evidence_and_guide_declaration_required(self):
        self.full()["responses"]["H0001"]["evidence"] = " "
        with self.assertRaisesRegex(ValueError, "Evidence note"):
            self.validate()
        self.full()["understood"] = False
        with self.assertRaisesRegex(ValueError, "guide declaration"):
            self.validate()

    def test_duplicate_json_ids_are_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "duplicate.json"
            path.write_text('{"responses":{"H0001":{},"H0001":{}}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                summarizer.load_json(path)

    def test_agreement_uses_joint_assessability_and_degenerate_kappa_is_null(self):
        result = summarizer.agreement([("0", "0"), ("U", "1"), ("NA", "0")])
        self.assertEqual(result["jointly_assessable"], 1)
        self.assertEqual(result["binary_agreement"], 1)
        self.assertIsNone(result["binary_cohen_kappa"])
        self.assertIsNone(summarizer.agreement([("U", "0")])["binary_agreement"])

    def test_end_to_end_import_preserves_source_and_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packet = root / "packet"
            private = packet / "COORDINATOR_ONLY"
            private.mkdir(parents=True)
            analysis = root / "analysis"
            (analysis / "review").mkdir(parents=True)
            prior = analysis / "review/annotations.csv"
            with prior.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["id", *summarizer.FAILURES])
                writer.writeheader()
                for i in (1, 2):
                    writer.writerow({"id": f"target-{i}", **{k: "0" for k in summarizer.FAILURES}})
            original_hash = summarizer.sha256(prior)
            mapping = [{"case_id": f"H{i:04d}", "id": f"target-{i}", "inclusion_probability": .5} for i in (1, 2)]
            protocol = {**self.protocol, "analysis_dir": str(analysis), "population": 4,
                        "prior_review_sha256": original_hash}
            (private / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
            (private / "identity_key.json").write_text(json.dumps({"cases": mapping}), encoding="utf-8")
            self.full()["responses"]["H0001"]["wrong_texture"] = "U"
            self.response["responses"]["H0002"]["color_inconsistency"] = "2"
            source = root / "SYNTHETIC_TEST_ONLY.json"
            source.write_text(json.dumps(self.response, indent=4) + "\n", encoding="utf-8")
            output = root / "received"
            result = summarizer.summarize(packet, source, output)
            self.assertEqual(result["complete"], 2)
            self.assertEqual(result["labels"]["wrong_texture"]["assessable"], 1)
            self.assertEqual(result["labels"]["color_inconsistency"]["positive"], 1)
            self.assertEqual((output / "response_original.json").read_bytes(), source.read_bytes())
            self.assertEqual(summarizer.sha256(prior), original_hash)
            self.assertEqual(result["scope"], "fixed random sample of saved outputs")
            with self.assertRaisesRegex(ValueError, "new output directory"):
                summarizer.summarize(packet, source, output)


if __name__ == "__main__":
    unittest.main()
