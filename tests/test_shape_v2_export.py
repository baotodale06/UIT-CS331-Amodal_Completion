"""A closed shape-only v2 export must retain every target's selection evidence."""
import argparse
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest

from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.provenance import execution_snapshot
from amodal_benchmark.shape_v2_runtime import progressive_seed, SEED_SLOTS
from scripts.export_shape_metadata import export


class ShapeV2ExportTests(unittest.TestCase):
    def test_export_rejects_missing_unbound_or_changed_v2_decisions(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, tid = root/"run", "VG/1_dog"
            target = run/"targets"/tid
            path = target/"shape-v2/iteration-00/decision.json"
            decision = dict(schema=2, method="shape-v2", iteration=0, target_seed=12,
                seed_slots=list(SEED_SLOTS), seeds=[progressive_seed(12, 0, slot) for slot in SEED_SLOTS],
                scores=[.2, .9, .3, .2, .2, .2, .2, None], index=1, fallback=False,
                rgb_correction=False, inference_truth_access=False)

            def close():
                with execution_snapshot(run, argparse.Namespace(), [{"id":tid}]):
                    write_json(run/"run_config.json", {"shape_policy":{"method":"shape-v2"}})
                    write_json(path, decision)
                    write_json(target/"metrics.json", {"inputs":{
                        "completion_sha256":"rgb", "reference_sha256":"crop"}})
                    write_json(target/"result.json", dict(id=tid, seed=12, status="success",
                        completion_sha256="rgb", evaluation_reference_artifacts={"crop":"crop"},
                        diagnostics={"shape_v2_iterations":[dict(decision_sha256=sha256(path), index=1, fallback=False)]}))

            close()
            original = path.read_bytes()
            path.unlink()
            with self.assertRaisesRegex(ValueError, "inventory"):
                export(run, root/"missing.tar.gz")
            path.write_bytes(original+b"\n")
            with self.assertRaisesRegex(ValueError, "target trace"):
                export(run, root/"changed.tar.gz")
            path.write_bytes(original)
            result = export(run, root/"valid.tar.gz")
            self.assertEqual(result["targets"], 1)
            with tarfile.open(root/"valid.tar.gz") as archive:
                self.assertIn("targets/"+tid+"/shape-v2/iteration-00/decision.json", archive.getnames())
                self.assertEqual(json.load(archive.extractfile("targets/"+tid+"/shape-v2/iteration-00/decision.json")), decision)
            decision["target_seed"] = 13
            close()
            with self.assertRaisesRegex(ValueError, "target policy"):
                export(run, root/"unbound.tar.gz")
            decision["target_seed"] = 12
            decision["scores"][0] = 1.
            close()
            with self.assertRaisesRegex(ValueError, "does not replay"):
                export(run, root/"bad-choice.tar.gz")


if __name__ == "__main__":
    unittest.main()
