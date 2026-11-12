# GitHub collaborator setup

For a fresh clone of this repository, start with
[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md). It provides the locked CPU
environment, branch workflow, dataset requirements and GPU launch sequence.
The older project-specific setup notes below are retained for context.


# Set up a reproducible amodal research project

This filename is retained from the supplied starter guide. The workflow is adapted to this project's fixed image benchmark and GPU inference pipeline. It has no Kaggle submission, tabular trainer, ROC-AUC, positive-class mapping, random folds, or competition leaderboard. The runnable command reference is [README.md](README.md).

## 1. Reuse source and preserve existing experiments

For this repository, work in place and preserve `datasets/`, `input/`, `output/`, model caches and existing scientific records. For a genuinely new project, copy only:

```text
amodal_benchmark/  configs/  scripts/  tests/  cookbooks/  docs/
prepare_open_world.py  prepare_visible_references.py
evaluate_open_world.py  evaluate_pdmc_open_world.py
compare_open_world.py  run_open_world_controls.py
run_amodal_slurm.sh  run_pdmc_open_world.sh
pyproject.toml  requirements.txt  MANIFEST.in
README.md  SETUP_NEW_COMPETITION.md  OPEN_WORLD_REPRODUCTION.md  AMODAL_SLURM.md
.gitignore  .gitattributes
```

Keep `AGENTS.md` local and follow its cluster restrictions. Do not copy another project's environment, original images, downloaded weights, generated references, results, logs, registry or live notes by default. Create fresh `datasets/`, `output/` and `logs/` directories. Use a distinct remote directory for a distinct project.

The established flat package/root entry points remain because existing Slurm launches and runtime source hashes depend on them. Modules already separate data, inference, references, metrics and scheduling. New project workflow modules handle config, provenance, verification and the research ledger without importing GPU libraries during setup.

## 2. Declare the research contract

Start with `configs/baseline.json`. Use `extends` for a small experiment-specific JSON file; parent paths are config-relative. Dataset paths are working-directory-relative. Validate with:

```bash
python -m amodal_benchmark config configs/baseline.json
```

| Decision | Configuration |
| --- | --- |
| Project identity and supported task | `project.slug`, `project.task` |
| Full evaluation or functional subset | `evaluation_scope` |
| Dataset location, selected sources, limit and missing-input policy | `dataset` |
| SAM versus LISA-reference evaluation | `inference.pipeline_profile` |
| Matcher, detector vocabulary, boundary rule and backdrop sizing | `inference` |
| Seeds, Mixed Context timestep, inclusive iteration cap and diffusion model | `inference` |
| Reference/completion geometry for metrics | `metrics.alignment` |
| Allocated worker count | `execution.workers` |
| Hypothesis and single changed factor | `experiment` |

The official evaluation contract is 2,565 targets and 553 labels from four source sets, with pinned annotation checksums. This project evaluates a released model; no training folds are needed. Repeated targets from an image share a worker. A new task or dataset needs an explicit new contract and checks before evaluation; changing the slug does not change the task.

SAM references measure preservation against the pipeline's initial detection. LISA references are independent visible segmentations. They are separate appearance protocols, not interchangeable ground truth. Record the reference choice before interpreting scores.

## 3. Establish the CPU environment and offline smoke

Use Python 3.10+ for CPU tooling, preferably 3.12 for a new environment. Install the actual package:

```bash
python -m venv .venv
# Activate .venv using the command for your shell; see README.
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/smoke_check.py
```

Run Bash and symlink tests in Linux/WSL using `scripts/check.sh`. The tests carry small pinned metadata/source fixtures; all functional images are synthetic. Two optional historical integration tests skip if the original release ZIP or old saved reports are absent.

The smoke runs the actual inference adapter, semantic matching, reference generation, result publication, failure bookkeeping, scoring/cache path, resume and independent artifact verification. Model calls and metric values are substituted. It uses temporary storage and cannot enter the full experiment registry. Use a new `--output-dir` only to retain debugging artifacts.

The CPU dependencies have bounded versions. Every evaluation attempt captures exact installed versions separately. The GPU stack remains pinned by `run_amodal_slurm.sh`; do not upgrade it with the CPU requirements file.

## 4. Inspect original data before inference

Reuse the installed benchmark and model cache on the current cluster. For a fresh installation, follow [OPEN_WORLD_REPRODUCTION.md](OPEN_WORLD_REPRODUCTION.md), using `prepare_open_world.py` to import the official release and selected original images.

```bash
python prepare_visible_references.py --dataset-root datasets/open_world_amodal --check-only
python evaluate_open_world.py --config configs/smoke.json \
  --output-dir output/pdmc-smoke-001 --check-only
```

Preserve original image bytes, checksums and provenance once runs begin. LAION reconstruction is explicitly different from a preserved historical shard. Full contracts reject missing inputs; exploratory subsets must be smoke runs. Input IDs and target order come from verified annotations rather than directory enumeration.

Use `configs/smoke.json` for a small real-data GPU functional run, with a separate output directory. The offline synthetic smoke is a separate command, not a replacement dataset configuration for research results.

## 5. Establish a verified full baseline

Use `configs/baseline.json` with a new output directory. A baseline is ready when:

- Every selected official target appears exactly once in annotation order.
- Original inputs, references, completions and metric-cache signatures agree with recorded hashes.
- Finite recorded metrics, failure penalties, exclusions and denominators reproduce the summary.
- Saved outputs at the iteration cap are distinguished from completed outputs.
- The resolved config, source/environment snapshots and latest execution status are present.
- CPU verification passes with `--require-full --require-provenance`, followed by explicit registry registration.

Learned metric networks are not recomputed by the CPU verifier; it checks their saved values, input signatures and aggregates. No hidden-pixel accuracy or exact paper reproduction is inferred. Keep the protocol's limits visible in reports.

## 6. Configure and review the cluster launch

The current SSH login is `daoqm@slurm.csse.rose-hulman.edu`, project directory `~/Vu_Hung/cv_nangcao`. Run source transfers and submissions yourself. Preview `scripts/push.ps1` and `scripts/pull.ps1` with `-WhatIf` and an explicit `-RemotePath`. They do not contact any host in dry-run mode.

Use `amodal plan` from the README to expand the config into a reviewable Slurm command. Full presets request four GPUs before the script name and four workers after it. Confirm partition, memory, time and CPU limits; create `logs/` before submitting. Run from the remote project directory.

The shared runner provisions the pinned environment and performs its actual GPU/import checks. For runtime changes, run `tests/validate_cuda_extension.sh` and `tests/validate_pipeline_imports.sh` inside a suitable allocation, then a real two-target smoke. CPU tests cannot establish GPU memory fit or numerical parity. No remote execution is performed as part of this rebuild.

## 7. Preserve comparable research history

For a new hypothesis, inherit a baseline config and change one declared scientific factor. Use a new output directory. Keep original inputs fixed; for LISA controls, retain the same reference directory. Register with an explicit `--baseline-dir` when declaring `experiment.changed_factor`.

`experiments.csv` retains completed and recorded failed full attempts; `experiment_report.md` shows results in registration order without automatically selecting a winner. Smoke/inference-only/partial outputs cannot masquerade as full scored evaluations. Each execution snapshot is retained, including after resume. Register a failed attempt before resuming if it should also appear in the ledger.

Download a single run, verify it locally, and keep remote ledger snapshots separate from independently maintained local registries. Update an accepted best only after comparable full evidence; this infrastructure rebuild does not select one.

Initialize fresh notes with `python -m amodal_benchmark notes --directory . --project new-project`. Existing notes are never overwritten. Maintain `plan.md`, `progress.md`, `discoveries.md`, `closed_ideas.md` and the live `session.md`; carry reviewed `cookbooks/` procedures into future projects and revalidate them against the new task.
