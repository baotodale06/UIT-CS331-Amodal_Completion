# Submitted provenance baseline — 2026-11-05

Quadro smoke 30436 completed successfully: eight targets, one completion scored by all four metrics, seven expected selection failures, and passing provenance verification. Full job 30426 failed before inference because the launcher combined `--run-only` and `--setup-only`. The repaired launcher skips provisioning for an existing cache and retains run-only preflight in the inference stages. Eight focused regressions pass in WSL and the actual cluster runtime, including generated commands through the real Bash parser. The two repaired files were backed up, uploaded, and SHA-256 verified. Job **31548** is running on `debug-gpu` with four Quadro GPUs, four workers, 16 CPUs, 64G RAM and a 24-hour limit. The output remains `output/pdmc-baseline-provenance-20261105`; no scientific settings changed. Runtime validation, four-worker smoke, full evaluation, provenance verification and registration remain mandatory gates. No full baseline is accepted yet.

The failed full-launch workflow and completed Quadro smoke are preserved separately. All prior failed smoke/runtime attempts remain intact. Commands, backup hashes and remote parser-test output are in `launcher-repair-20261106T052337Z.json`; the new submission is in `submission-repaired-20261106.json`. `latest-baseline-progress.json` records the latest observed state. Earlier submission/dependency receipts are retained as historical evidence.

Runtime validation, the 2,565-target source preflight and the four-worker scored/provenance-verified smoke have now passed in job 31548. Full inference started at 05:30 UTC on 2026-11-06 and is progressing across all four workers. A local hidden watcher (current process recorded in `watcher-process.json`) checks this existing job every minute; after successful full verification and registration it recovers the artifacts, checks all transfer hashes, independently verifies the full run locally, and updates `REPORT.md` and `status.json`. The watcher never submits or cancels jobs. Its current state is in `watcher-status.json`; if the local machine sleeps or loses network access, the cluster job continues, and recovery can resume later. No full acceptance is claimed while inference remains running.

The experiment uses the unchanged `configs/baseline.json`: all 2,565 targets, SAM references, semantic matcher v2, released detector vocabulary, released boundary/metric alignment, automatic background sizing, seed 0, timestep 35, five iterations, and four workers. It evaluates pretrained PD-MC; there is no training stage.

## Evidence already collected

- Full local source-image preflight: 2,565 targets, zero missing images.
- WSL Python 3.10: all 155 tests without skips, all five Bash regressions and synthetic smoke passed (`logs/baseline-final-check-wsl-20261105.log`). Six workflow gate tests also passed after checking Windows planning compatibility (`logs/baseline-workflow-tests-20261105.log`). Configuration/provenance regression tests passed (`logs/baseline-config-provenance-tests-20261105.log`).
- Source-only upload preview: 88 files; excludes AGENTS.md, research notes, data, environments, outputs, and ledgers.
- Local RTX 3060 has 6 GB; the documented released pipeline needs approximately 24 GB per worker. The cluster's debug-gpu partition has eight Quadro RTX 6000 GPUs and a 24-hour maximum allocation.
- Read-only Slurm inspection found other active user jobs in `playground-series-s6e11`, not this project. No jobs were modified. Before the source upload, the remote amodal source lacked the current provenance modules/configuration workflow; the uploaded source now includes them.
- The existing `/dev/shm/daoqm/amodal-cache` is compute-node-local and cannot be inspected from the login node. Direct compute-node SSH is denied by cluster policy. The allocated probe found it unavailable; the submitted replacement uses the persistent project cache.

The original two-target smoke covers two historical selection failures. `configs/baseline-gpu-smoke.json` uses the first eight targets and four workers; target `VG/43_television` had a saved completion historically. This is a runtime test, not a quality estimate. The job refuses to start the full baseline unless the smoke generates at least one completion with a verified cache containing all four metrics.

## Upload and submit

Local PowerShell, from the project:

```powershell
.\scripts\push.ps1 -RemotePath Vu_Hung/cv_nangcao -WhatIf
.\scripts\push.ps1 -RemotePath Vu_Hung/cv_nangcao
ssh daoqm@slurm.csse.rose-hulman.edu
```

Remote shell:

```bash
cd ~/Vu_Hung/cv_nangcao
mkdir -p logs
sbatch --partition=debug-gpu --job-name=pdmc-baseline \
  --gres=gpu:4 --cpus-per-task=16 --mem=64G --time=24:00:00 \
  --output=logs/%x-%j.out scripts/run_baseline_slurm.sh \
  --cache-dir /home/daoqm/Vu_Hung/cv_nangcao/.amodal-cache \
  --output-dir output/pdmc-baseline-provenance-20261105 \
  --benchmark-workers 4 --run-only
```

The shared runner reuses the existing dataset/checkpoints and repairs or provisions only its pinned runtime if needed. It then checks all four visible GPUs, runs the real GroundingDINO compilation and forward/backward checks, validates the complete pipeline imports, runs a scored eight-target smoke in a separate directory, verifies its provenance, and starts the full baseline. After the full run it executes `verify --require-full --require-provenance` and then `register`. Any failing stage stops the sequence; failed verification cannot register a result. The workflow has no session timeout, so the Slurm job can continue beyond the three-hour working window, subject to its 24-hour allocation limit.

The original expanded stage plan is in `workflow-plan.sh`; the current submission receipt and dependency/partition updates are in `submission.json` and `baseline-dependency-update.json`. Supplementary preparation and smoke scripts/receipts are alongside them. Scientific source/configuration snapshots are in `source-before-launch/` with hashes in `source-before-launch.json`. Runtime logs and the workflow's own source snapshot will be saved under `output/pdmc-baseline-provenance-20261105/workflow/ATTEMPT/`; evaluator execution/source/package provenance will be under `provenance/EXECUTION/`.

## Monitor and resume

```bash
squeue -u daoqm
tail -n 80 logs/pdmc-runtime-30425.out
tail -n 80 logs/pdmc-quadro-smoke-30436.out
tail -n 80 logs/pdmc-baseline-31548.out
find output/pdmc-baseline-provenance-20261105/workflow -name workflow.json -print
```

Inspect the active stage log named in the Slurm log. During inference, worker logs are under the full or smoke run's `workers/EXECUTION/`. Leave a progressing job running. If the allocation ends, inspect the recorded failure and resubmit the same command with the same output/cache/settings; completed targets are preserved. Never import historical targets into this new baseline.

After completion, the independent check can be repeated inside the prepared runtime:

```bash
/home/daoqm/Vu_Hung/cv_nangcao/.amodal-cache/conda-env/bin/python -m amodal_benchmark verify \
  --run-dir output/pdmc-baseline-provenance-20261105 \
  --dataset-root datasets/open_world_amodal --require-full --require-provenance
```

The acceptance report must retain all source counts, successful/failed targets, failure reasons, completed versus iteration-limited outputs, and each metric denominator. GPU/runtime and scored-smoke gates have passed; full acceptance remains pending until scoring, verification and registration finish.
