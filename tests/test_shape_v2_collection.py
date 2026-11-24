"""A completed export still waits for accounting and continuation closure."""
import unittest
from scripts.collect_shape_v2 import disposition


class CollectionTests(unittest.TestCase):
    def test_pending_continuation_keeps_collection_open_after_timeout(self):
        self.assertEqual(disposition(dict(jobs={"1":["TIMEOUT", "0:0"], "2":["PENDING", "0:0"]},
                                          export_ready=False)), "waiting")

    def test_export_waits_for_finalizer_to_terminate(self):
        self.assertEqual(disposition(dict(jobs={"1":["COMPLETED", "0:0"], "2":["RUNNING", "0:0"]},
                                          export_ready=True)), "waiting")

    def test_terminated_chain_without_export_needs_review(self):
        self.assertEqual(disposition(dict(jobs={"1":["FAILED", "1:0"], "2":["COMPLETED", "0:0"]},
                                          export_ready=False)), "needs_review")

    def test_completed_export_can_include_historical_timeout(self):
        self.assertEqual(disposition(dict(jobs={"1":["TIMEOUT", "0:0"], "2":["COMPLETED", "0:0"]},
                                          export_ready=True)), "ready")


if __name__ == "__main__":
    unittest.main()
