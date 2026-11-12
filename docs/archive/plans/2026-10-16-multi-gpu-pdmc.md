# Multi-GPU PD-MC Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans with test-driven-development. Execute in the current workspace: this snapshot has no Git metadata.

**Goal:** Run independent PD-MC image groups on allocated GPUs and preserve the existing benchmark outputs, seeds, settings, and resume behavior.

**Architecture:** The evaluation entry point remains the coordinator. It validates the shared configuration, launches isolated subprocesses with one CUDA-visible GPU each, and collects their existing per-target records. Workers own disjoint image groups and never write global configuration or summary files. The coordinator scores and summarizes only after all workers succeed.

**Tech Stack:** Python standard-library subprocesses and file locks, existing PyTorch runtime, Bash/Slurm launcher, unittest.

**Spec:** User-approved design in this conversation: process different images on multiple allocated GPUs without changing model or evaluation settings.

## Global Constraints

- Keep semantic run/metric configurations unchanged so existing completed targets resume.
- Keep original per-image and per-target seed derivation, vocabulary, model revisions, thresholds, and diffusion settings.
- Respect the inherited CUDA_VISIBLE_DEVICES, including GPU UUIDs; never expose unallocated devices.
- Apply source filters and target limits before splitting work; keep each image on one worker.
- Only one coordinator writes summaries/configuration; fail the run if any worker fails.
- Multi-GPU execution preserves the method; GPU numerical equivalence still requires cluster validation.

## Review Focus

- Sparse or UUID GPU visibility must map workers to the same allocated devices.
- Global limits and repeated images must not duplicate or omit targets.
- Resuming a previous serial run or changing worker count must reuse completed results.
- Failed or interrupted workers must be stopped and must not produce a misleading full report.
- Concurrent coordinators must not overwrite each other's files; worker logs remain reviewable.

## Tasks

### 1. Worker scheduling and process lifecycle

Files: create `amodal_benchmark/parallel.py` and `tests/test_parallel_benchmark.py`.

- [x] Write and run failing tests for grouped partitioning, visibility validation, real subprocess isolation, failure/cancellation cleanup, and output locking.
- [x] Implement `partition_rows`, `plan_devices`, `run_workers`, and `output_lock` using only standard-library imports.
- [x] Verify the tests pass with CPU subprocess fixtures.

### 2. Integrate evaluator and launcher

Files: modify `evaluate_open_world.py`, `run_amodal_slurm.sh`, and `tests/test_open_world_slurm.sh`; extend parallel tests.

- [x] Write failing tests for global-limit partitioning, worker-only writes, full aggregation, stable seeds, and serial-to-parallel resume.
- [x] Add `--workers auto|N` to the evaluator and `--benchmark-workers auto|N` to the runner (runner default: auto). Keep internal worker flags hidden.
- [x] Prepare shared configuration once; run inference workers with explicit resolved target selection/reference paths, then collect records and score centrally.
- [x] Divide the CPU thread budget among workers and record GPU placement/log paths separately from semantic configuration.
- [x] Verify worker failures propagate and parser/default/dispatch behavior remains compatible.

### 3. Documentation and verification

Files: update `OPEN_WORLD_REPRODUCTION.md` and `AMODAL_SLURM.md`.

- [x] Document local execution, allocation resource requests, worker logs, resume, and single-worker fallback.
- [x] Run the full unittest suite, every shell regression script, and Bash syntax checks.
- [x] Request an independent code review, address actionable findings, and rerun affected checks.
- [x] Report exact transfer/rerun commands and the remaining GPU validation requirement.

## Verification Record

- Full Linux/WSL unittest discovery: 54 tests passed, no skips (2026-10-16).
- All four `tests/test_*.sh` regression scripts passed; Bash syntax and Python compilation passed.
- Independent review found no remaining blockers after removing repeated worker checkpoint hashing.
- End-to-end CPU fixtures verify identical serial/parallel completion artifacts, per-image RNG samples, target seeds, repeat-image ownership, resume, full metrics, and empty-reference denominators.
- No GPU execution or Slurm job was performed locally. GPU numerical parity, throughput, and cluster-specific filesystem/visibility behavior require node validation.
