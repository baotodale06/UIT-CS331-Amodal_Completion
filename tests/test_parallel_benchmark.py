"""Exercise GPU scheduling with real CPU subprocesses at the model boundary."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ParallelSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("amodal_benchmark.parallel"),
                             "Independent multi-GPU benchmark workers are missing")
        from amodal_benchmark import parallel
        self.parallel = parallel

    def test_partition_keeps_repeated_images_together_without_losing_target_order(self):
        rows = [{"id": name, "image": image} for name, image in
                [("a1", "a.jpg"), ("a2", "a.jpg"), ("b1", "b.jpg"),
                 ("c1", "c.jpg"), ("d1", "d.jpg"), ("b2", "b.jpg")]]
        groups = self.parallel.partition_rows(rows, 2)
        self.assertEqual([[row["id"] for row in group] for group in groups],
                         [["a1", "a2", "c1"], ["b1", "d1", "b2"]])
        self.assertEqual(rows[0], {"id": "a1", "image": "a.jpg"})

    def test_auto_uses_only_visible_devices_and_caps_workers_to_image_count(self):
        self.assertEqual(self.parallel.plan_devices("auto", 4, "3,5,6,7", 2), ["3", "5"])
        self.assertEqual(self.parallel.plan_devices("2", 4, "GPU-A,GPU-B,GPU-C,GPU-D", 7),
                         ["GPU-A", "GPU-B"])
        self.assertEqual(self.parallel.plan_devices("auto", 3, None, 9), ["0", "1", "2"])

    def test_worker_requests_cannot_expose_unallocated_gpus(self):
        for request, count, mask in [("4", 2, "3,5"), ("auto", 0, "-1"),
                                     ("auto", 2, "GPU-A"), ("auto", 2, "3,3")]:
            with self.subTest(request=request, count=count, mask=mask):
                with self.assertRaises(ValueError):
                    self.parallel.plan_devices(request, count, mask, 9)
        for request in ["0", "-1", "many"]:
            with self.assertRaises(ValueError):
                self.parallel.plan_devices(request, 2, "3,5", 9)

    def test_workers_run_concurrently_with_one_allocated_gpu_and_shared_cpu_budget(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            child = root / "worker.py"
            child.write_text('''import argparse, json, os, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("root", type=Path)
p.add_argument("--worker-index", type=int)
p.add_argument("--worker-count", type=int)
a = p.parse_args()
data = {"gpu": os.environ["CUDA_VISIBLE_DEVICES"], "count": a.worker_count,
        "omp": os.environ["OMP_NUM_THREADS"], "mkl": os.environ["MKL_NUM_THREADS"]}
(a.root / f"{a.worker_index}.json").write_text(json.dumps(data))
deadline = time.monotonic() + 5
while not all((a.root / f"{i}.json").exists() for i in range(a.worker_count)):
    if time.monotonic() > deadline: raise SystemExit(8)
    time.sleep(.01)
print("worker finished", flush=True)
''', encoding="utf-8")
            self.parallel.run_workers([sys.executable, str(child), str(root)],
                                      ["GPU-A", "GPU-B"], root / "logs", cpu_threads=8)
            for index, gpu in enumerate(["GPU-A", "GPU-B"]):
                self.assertEqual(json.loads((root / f"{index}.json").read_text()),
                                 {"gpu": gpu, "count": 2, "omp": "4", "mkl": "4"})
            self.assertIn("worker finished", (root / "logs/worker-00.log").read_text())
            execution = json.loads((root / "logs/execution.json").read_text())
            self.assertEqual(execution["status"], "complete")

    @unittest.skipUnless(os.name == "posix", "POSIX process cleanup")
    def test_failed_worker_stops_siblings_and_reports_the_failed_log(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            child = root / "fail.py"
            child.write_text('''import argparse, os, time
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("root", type=Path)
p.add_argument("--worker-index", type=int); p.add_argument("--worker-count", type=int)
a=p.parse_args()
if a.worker_index == 0:
    (a.root / "pid").write_text(str(os.getpid()))
    while True: time.sleep(.1)
deadline=time.monotonic()+5
while not (a.root / "pid").exists():
    if time.monotonic() > deadline: raise SystemExit(9)
    time.sleep(.01)
print("simulated CUDA failure", flush=True)
raise SystemExit(7)
''', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "worker-01.log"):
                self.parallel.run_workers([sys.executable, str(child), str(root)],
                                          ["3", "5"], root / "logs", cpu_threads=2)
            with self.assertRaises(ProcessLookupError):
                os.kill(int((root / "pid").read_text()), 0)
            self.assertEqual(json.loads((root / "logs/execution.json").read_text())["status"],
                             "failed")

    @unittest.skipUnless(os.name == "posix", "POSIX output locking")
    def test_output_lock_rejects_other_coordinators_and_is_reusable_after_exit(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            with self.parallel.output_lock(output):
                with self.assertRaisesRegex(ValueError, "already|Another"):
                    with self.parallel.output_lock(output):
                        self.fail("A second coordinator acquired the active output")
            with self.parallel.output_lock(output):
                self.assertTrue(output.is_dir())

    @unittest.skipUnless(os.name == "posix", "POSIX Slurm cancellation")
    def test_terminating_coordinator_reaps_its_inference_workers(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            child = root / "wait.py"
            child.write_text('''import argparse, os, time
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("root", type=Path)
p.add_argument("--worker-index"); p.add_argument("--worker-count")
a=p.parse_args(); (a.root / (a.worker_index + ".pid")).write_text(str(os.getpid()))
while True: time.sleep(.1)
''', encoding="utf-8")
            code = ("from amodal_benchmark.parallel import run_workers; "
                    "import sys; run_workers([sys.executable, sys.argv[1], sys.argv[2]], "
                    "['3','5'], sys.argv[3])")
            env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])}
            coordinator = subprocess.Popen([sys.executable, "-c", code, str(child), str(root), str(root / "logs")],
                                           env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 5
                while len(list(root.glob("*.pid"))) < 2 and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertEqual(len(list(root.glob("*.pid"))), 2)
                coordinator.send_signal(signal.SIGTERM)
                coordinator.communicate(timeout=8)
                self.assertEqual(coordinator.returncode, 128 + signal.SIGTERM)
                for path in root.glob("*.pid"):
                    with self.assertRaises(ProcessLookupError):
                        os.kill(int(path.read_text()), 0)
            finally:
                if coordinator.poll() is None:
                    coordinator.kill()
                coordinator.communicate()


if __name__ == "__main__":
    unittest.main()
