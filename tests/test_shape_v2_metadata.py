"""The local replay must reject score/reference drift and omitted penalties."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from amodal_benchmark.dataset import write_json
from amodal_benchmark.protocol import failure_metrics
from scripts.verify_shape_v2_metadata import check_target, equal


class ShapeV2MetadataTests(unittest.TestCase):
    def test_rejects_different_reference_seed_or_cached_completion(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            original = dict(id="VG/1_dog", seed=12, input_sha256="input",
                            evaluation_reference_artifacts=dict(crop="crop", full="full"))
            record = dict(original, status="success", completion_sha256="generated")
            values = {"CLIP":28., "LPIPS":.2, "Feature Similarity":.8, "SSIM":.7}
            row = dict(id=original["id"], seed="12", status="success", reference_status="ready",
                       **{key:str(value) for key, value in values.items()})
            cache = dict(inputs=dict(completion_sha256="generated", reference_sha256="crop"), metrics=values)
            write_json(folder/"result.json", record)
            write_json(folder/"metrics.json", cache)
            self.assertEqual(check_target(folder, row, original), record)
            write_json(folder/"metrics.json", dict(cache, inputs=dict(cache["inputs"], completion_sha256="other")))
            with self.assertRaisesRegex(ValueError, "score inputs"):
                check_target(folder, row, original)
            write_json(folder/"metrics.json", cache)
            for key, replacement in (("seed", 13), ("evaluation_reference_artifacts", dict(crop="different", full="full"))):
                write_json(folder/"result.json", dict(record, **{key:replacement}))
                with self.assertRaisesRegex(ValueError, "matched"):
                    check_target(folder, row, original)

    def test_failure_penalties_cannot_disappear_and_clip_only_is_explicit(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            original = dict(id="VG/1_dog", seed=12, input_sha256="input",
                            evaluation_reference_artifacts=dict(crop="crop", full="full"))
            record = dict(original, status="not_detected")
            row = dict(id=original["id"], seed="12", status="not_detected", reference_status="ready",
                       **{key:str(value) if value is not None else "" for key,value in failure_metrics().items()})
            write_json(folder/"result.json", record)
            self.assertEqual(check_target(folder, row, original)["status"], "not_detected")
            with self.assertRaisesRegex(ValueError, "failure-inclusive scores"):
                check_target(folder, dict(row, LPIPS=""), original)
            values = {"CLIP":28., "LPIPS":None, "Feature Similarity":None, "SSIM":None}
            record.update(status="success", completion_sha256="generated")
            row.update(status="success", reference_status="empty_mask",
                       **{key:str(value) if value is not None else "" for key,value in values.items()})
            write_json(folder/"result.json", record)
            write_json(folder/"metrics.json", dict(metrics=values, inputs=dict(
                completion_sha256="generated", reference_sha256="full", clip_only=True)))
            self.assertEqual(check_target(folder, row, original)["status"], "success")
            with self.assertRaisesRegex(ValueError, "score inputs"):
                check_target(folder, dict(row, reference_status="ready"), original)

    def test_statistics_replay_rejects_nonfinite_or_missing_evidence(self):
        equal(dict(gain=3., interval=[2., 4.]), dict(gain=3.+1e-12, interval=[2., 4.]), "test")
        for first, second in ((dict(gain=3.), dict(gain=4.)), (float("nan"), float("nan")),
                              (True, 1), (dict(verified=True), dict(verified=True, extra=1))):
            with self.assertRaises(ValueError):
                equal(first, second, "test")


if __name__ == "__main__":
    unittest.main()
