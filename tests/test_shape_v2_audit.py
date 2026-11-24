"""Keep full CLIP coverage in the point estimate and paired image resampling."""
import unittest
from scripts.audit_shape_v2 import paired_estimate


class ShapeAuditTests(unittest.TestCase):
    def test_clip_keeps_each_arms_denominator_when_outputs_are_gained(self):
        rows = [dict(source="A", image_identity="A/1", control={"CLIP":20}, method={"CLIP":22}),
                dict(source="A", image_identity="A/2", control={"CLIP":None}, method={"CLIP":30})]
        result = paired_estimate(rows, "CLIP", draws=100)
        self.assertEqual((result["control_n"], result["method_n"]), (1, 2))
        self.assertEqual(result["relative_gain_percent"], 30.)
        self.assertGreater(result["undefined_bootstrap_samples"], 0)

    def test_failure_inclusive_lpips_and_image_clusters(self):
        rows = [dict(source="A", image_identity="A/1", control={"LPIPS":.5}, method={"LPIPS":.4}),
                dict(source="A", image_identity="A/1", control={"LPIPS":1.}, method={"LPIPS":1.})]
        result = paired_estimate(rows, "LPIPS", draws=100)
        self.assertEqual(result["image_clusters"], 1)
        self.assertAlmostEqual(result["relative_gain_percent"], 100*.1/1.5)
        self.assertEqual(result["undefined_bootstrap_samples"], 0)


if __name__ == "__main__":
    unittest.main()
