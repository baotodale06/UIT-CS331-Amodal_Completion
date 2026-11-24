"""Reject omitted pools, substituted selections and detached target evidence."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.shape_calibrated_runtime import METHOD, SEED_SLOTS, progressive_seed, read
from amodal_benchmark.shape_v2 import score_masks
from scripts.audit_shape_v2 import audit_decisions
from scripts.prepare_shape_calibrated_evaluation import audit_target_links
from scripts.package_shape_calibrated import package


class CalibratedAuditTests(unittest.TestCase):
    def test_closed_target_binds_every_pool_seed_choice_and_hash(self):
        with TemporaryDirectory() as temporary:
            run = Path(temporary)
            target = run / "targets/VG/test"
            pool = target / METHOD / "iteration-00"
            pool.mkdir(parents=True)
            prior = np.ones((2, 3), np.float32)
            np.save(pool / "prior.npy", prior, allow_pickle=False)
            masks = [np.ones((2, 3), bool) for _ in SEED_SLOTS]
            for i in range(8):
                Image.fromarray(masks[i].astype(np.uint8)*255).save(pool / f"mask-{i}.png")
                Image.new("RGB", (3, 2), (i, 80, 60)).save(pool / f"candidate-{i}.png")
            (pool / "selected.png").write_bytes((pool / "candidate-0.png").read_bytes())
            decision = dict(method=METHOD, logit_bias=-2., target_seed=123, iteration=0, index=0, fallback=False,
                rgb_correction=False, inference_truth_access=False, seed_slots=list(SEED_SLOTS),
                seeds=[progressive_seed(123, 0, seed) for seed in SEED_SLOTS], scores=score_masks(prior, masks),
                artifacts={p.name: sha256(p) for p in pool.iterdir()})
            write_json(pool / "decision.json", decision)
            link = dict(index=0, fallback=False, decision_sha256=sha256(pool / "decision.json"))
            record = dict(id="VG/test", status="success", seed=123, diagnostics=dict(shape_calibrated_iterations=[link]))
            write_json(target / "result.json", record)
            self.assertEqual(audit_decisions(run / "targets", METHOD)["pools"], 1)
            self.assertEqual(audit_target_links(run, dict(method=METHOD))["pools"], 1)
            for change in (dict(seed=456), dict(diagnostics={})):
                write_json(target / "result.json", dict(record, **change))
                with self.assertRaises(ValueError):
                    audit_target_links(run, dict(method=METHOD))
            write_json(target / "result.json", record)
            (pool / "selected.png").write_bytes((pool / "candidate-1.png").read_bytes())
            decision["artifacts"]["selected.png"] = sha256(pool / "selected.png")
            write_json(pool / "decision.json", decision)
            with self.assertRaisesRegex(ValueError, "Selected RGB"):
                audit_decisions(run / "targets", METHOD)

    def test_missing_quadro_replay_never_creates_admitted_package(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "study/local-native/result.json", dict(complete=True, verified=True))
            with self.assertRaises(FileNotFoundError):
                package(root / "study", root / "admitted")
            self.assertFalse((root / "admitted").exists())


if __name__ == "__main__":
    unittest.main()
