"""Recovery must not undercount allocation or accept altered evidence."""
import base64
import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.recover_shape_v2 import allocation_cost, save_member


class ShapeRecoveryTests(unittest.TestCase):
    def test_counts_total_gpu_tres_once_and_keeps_failed_time(self):
        accounting = ("JobIDRaw|State|ExitCode|ElapsedRaw|AllocTRES|NodeList|Start|End\n"
            "1|COMPLETED|0:0|3600|gres/gpu=4,gres/gpu:quadro_rtx_6000=4|gebru|start|end\n"
            "2|TIMEOUT|0:0|1800|gres/gpu=4|gebru|start|end\n"
            "3|FAILED|1:0|1800|gres/gpu:quadro_rtx_6000=1|gebru|start|end\n"
            "4|PENDING|0:0|0||None|Unknown|Unknown\n")
        result = allocation_cost(accounting, {"1":"full", "2":"full", "3":"runtime_check", "4":"controller"})
        self.assertEqual(result["reserved_gpu_hours_by_role"]["full"], 6.)
        self.assertEqual(result["total_reserved_gpu_hours"], 6.5)
        self.assertFalse(result["final"])

    def test_rejects_incomplete_or_duplicate_accounting(self):
        header = "JobIDRaw|State|ExitCode|ElapsedRaw|AllocTRES|NodeList|Start|End\n"
        row = "1|COMPLETED|0:0|1|gres/gpu=4|gebru|start|end\n"
        with self.assertRaises(ValueError):
            allocation_cost(header+row, {"1":"full", "2":"training"})
        with self.assertRaises(ValueError):
            allocation_cost(header+row+row, {"1":"full"})

    def test_checksum_paths_and_existing_evidence_are_protected(self):
        data = b"a completed receipt\n"
        record = dict(base64=base64.b64encode(data).decode(), bytes=len(data),
                      sha256=hashlib.sha256(data).hexdigest())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("../outside", "/outside", "C:/outside", "..\\outside"):
                with self.assertRaises(ValueError):
                    save_member(root, name, record)
            with self.assertRaises(ValueError):
                save_member(root, "bad.json", dict(record, sha256="0"*64))
            save_member(root, "training/training.json", record)
            self.assertEqual((root/"training/training.json").read_bytes(), data)
            with self.assertRaises(FileExistsError):
                save_member(root, "training/training.json", record)


if __name__ == "__main__":
    unittest.main()
