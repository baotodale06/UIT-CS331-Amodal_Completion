"""Prevent incomplete or unattributed review data from becoming a census."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from amodal_benchmark.appearance import read_review, write_csv

spec = importlib.util.spec_from_file_location("census_script", Path(__file__).resolve().parents[1] /
                                            "scripts/expanded_appearance_review.py")
census = importlib.util.module_from_spec(spec)
spec.loader.exec_module(census)


class CensusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.review = self.output / "review"
        (self.review / "batches").mkdir(parents=True)
        (self.review / "additional_batches").mkdir()
        self.rows = [{"id": f"case-{n}", "review_number": n, "proxy_bin": "[0,20)",
                      "stratum_population": 2, "stratum_sample": 2, "inclusion_probability": 1}
                     for n in (1, 2)]
        write_csv(self.review / "annotations_template.csv", self.rows)
        write_csv(self.review / "census_index.csv", self.rows)

    def compile(self):
        with contextlib.redirect_stdout(io.StringIO()):
            census.compile_review(self.output, census=True)

    def test_missing_case_cannot_silently_become_a_negative(self):
        (self.review / "batches/001.txt").write_text("1 | 0,0,NA,0,U,0 | | First inspected case.\n")
        with self.assertRaisesRegex(ValueError, "Missing 1"):
            self.compile()
        self.assertFalse((self.review / "annotations.csv").exists())

    def test_duplicate_across_batch_directories_is_rejected(self):
        line = "1 | 0,0,NA,0,U,0 | | First inspected case.\n"
        (self.review / "batches/001.txt").write_text(line)
        (self.review / "additional_batches/001.txt").write_text(line)
        with self.assertRaisesRegex(ValueError, "Duplicate or unexpected"):
            self.compile()

    def test_census_cannot_reuse_partial_sample_weights(self):
        self.rows[0]["stratum_sample"] = 1
        self.rows[0]["inclusion_probability"] = .5
        write_csv(self.review / "census_index.csv", self.rows)
        with self.assertRaisesRegex(ValueError, "unit inclusion"):
            self.compile()

    def test_explicit_revision_preserves_unknown_and_original_notes(self):
        (self.review / "batches/001.txt").write_text("1 | 0,0,NA,0,U,0 | | First inspected case.\n", encoding="utf-8-sig")
        (self.review / "additional_batches/001.txt").write_text("2 | U,0,NA,0,U,2 | | Candidate extra object.\n")
        with self.assertRaisesRegex(ValueError, "lacks checked scope"):
            self.compile()
        revisions = [{"review_number": 2, "labels": {"cooccurring_object": "0"},
                      "evidence": "Full source confirms object was already present."}]
        (self.review / "adjudications.json").write_text(json.dumps(revisions))
        self.compile()
        final = read_review(self.review / "annotations.csv", ["case-1", "case-2"])
        initial = read_review(self.review / "annotations_census_initial.csv", ["case-1", "case-2"])
        self.assertEqual(final[1]["wrong_texture"], "U")
        self.assertEqual(final[1]["cooccurring_object"], "0")
        self.assertEqual(initial[1]["cooccurring_object"], "2")
        self.assertEqual(initial[1]["evidence"], "Candidate extra object.")
        self.assertEqual(final[1]["cooccurrence_output_scope"], "")


if __name__ == "__main__":
    unittest.main()
