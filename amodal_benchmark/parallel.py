"""Schedule independent PD-MC subprocesses within the current GPU allocation."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def partition_rows(rows, worker_count):
    """Assign each image to one worker, preserving annotation order per worker."""
    if worker_count < 1:
        raise ValueError("Worker count must be positive")
    groups = [[] for _ in range(worker_count)]
    owners = {}
    for row in rows:
        if row["image"] not in owners:
            owners[row["image"]] = len(owners) % worker_count
        groups[owners[row["image"]]].append(row)
    return groups


def plan_devices(request, device_count, visible_devices, image_count):
    """Translate logical devices back to inherited CUDA visibility tokens."""
    if request != "auto" and (not str(request).isascii() or
                              not str(request).isdigit() or int(request) < 1):
        raise ValueError("--workers must be auto or a positive integer")
    if device_count < 1:
        raise ValueError("No allocated CUDA GPUs are visible to PD-MC")
    count = device_count if request == "auto" else int(request)
    if count > device_count:
        raise ValueError(f"Requested {count} PD-MC workers, but only {device_count} CUDA GPUs "
                         "are visible. Request more GPUs from Slurm or reduce --workers.")
    tokens = ([str(index) for index in range(device_count)] if visible_devices is None
              else [token.strip() for token in visible_devices.split(",")])
    selected = tokens[:device_count]
    if len(selected) != device_count or len(set(selected)) != device_count or not all(selected):
        raise ValueError("CUDA_VISIBLE_DEVICES does not match the CUDA-visible device count")
    return selected[:min(count, image_count)]


@contextmanager
def output_lock(output):
    """Reject overlapping coordinators; the OS releases the lock after a crash."""
    with (Path(output) / ".pdmc.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            acquire = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: (stream.seek(0), msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1))
        else:
            import fcntl
            acquire = lambda: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(stream, fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as error:
            raise ValueError(f"Another evaluation is already using output directory {output}") from error
        try:
            yield
        finally:
            release()


def run_workers(command, devices, log_dir, cpu_threads=None):
    """Run disjoint inference workers, failing fast and reaping all children."""
    if not devices:
        return
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=False)
    budget = int(cpu_threads if cpu_threads is not None else os.environ.get("OMP_NUM_THREADS", "1"))
    threads = max(1, budget // len(devices))
    execution = {"status": "running", "devices": list(devices), "threads_per_worker": threads,
                 "started_utc": datetime.now(timezone.utc).isoformat(), "command": list(command)}
    metadata = log_dir / "execution.json"
    metadata.write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
    processes, handles = [], []
    previous_handler = signal.getsignal(signal.SIGTERM)

    def terminate_requested(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate_requested)
    succeeded = False
    try:
        for index, device in enumerate(devices):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device
            env["PYTHONUNBUFFERED"] = "1"
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[name] = str(threads)
            log = log_dir / f"worker-{index:02d}.log"
            handle = log.open("w", encoding="utf-8")
            handles.append(handle)
            args = [*command, "--worker-index", str(index), "--worker-count", str(len(devices))]
            processes.append(subprocess.Popen(args, env=env, stdout=handle, stderr=subprocess.STDOUT))
            print(f"PD-MC worker {index}: allocated GPU {device}, {threads} CPU thread(s); log: {log}",
                  flush=True)
        next_progress = time.monotonic() + 30
        while True:
            codes = [process.poll() for process in processes]
            for index, code in enumerate(codes):
                if code is not None and code != 0:
                    raise RuntimeError(f"PD-MC worker {index} failed (exit {code}). "
                                       f"Read {log_dir / f'worker-{index:02d}.log'}. "
                                       "Rerun with the same output directory to resume.")
            done = sum(code is not None for code in codes)
            if done == len(devices):
                succeeded = True
                break
            if time.monotonic() >= next_progress:
                print(f"PD-MC: {done}/{len(devices)} workers finished; logs: {log_dir}", flush=True)
                next_progress = time.monotonic() + 30
            time.sleep(.1)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for handle in handles:
            handle.close()
        execution.update(status="complete" if succeeded else "failed",
                         returncodes=[process.returncode for process in processes],
                         finished_utc=datetime.now(timezone.utc).isoformat())
        metadata.write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
