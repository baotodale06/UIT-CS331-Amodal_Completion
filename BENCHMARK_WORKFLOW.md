> For a fresh GitHub clone, use [the collaborator guide](docs/GETTING_STARTED.md).
> Historical output links below require separately transferred experiment artifacts.
> Local research notes are archived under `docs/archive/local-handoff-20261112/`.

# Open-World Amodal Appearance Completion

Reproducible evaluation of the released Progressive Mixed Context diffusion (PD-MC) pipeline on the verified 2,565-target Open-World benchmark. This is a research project, with separate SAM and LISA reference protocols.

The existing root entry points and cluster cache layout remain supported. See [SETUP_NEW_COMPETITION.md](SETUP_NEW_COMPETITION.md) for the project setup workflow, [OPEN_WORLD_REPRODUCTION.md](OPEN_WORLD_REPRODUCTION.md) for the scientific protocol, and [AMODAL_SLURM.md](AMODAL_SLURM.md) for pinned GPU provisioning.

## Accepted baseline

The [2026-11-06 baseline report](output/baseline-establishment-20261105/REPORT.md) records the completed, scored and provenance-verified four-GPU run **31548**: **951 saved completions / 2,565 targets**, with **1,614 failures**. It is registered on the cluster and independently verified after recovering all 16,814 files locally.

Use `/home/daoqm/Vu_Hung/cv_nangcao/output/pdmc-baseline-provenance-20261105` as the explicit remote baseline for compatible future comparisons. The [recovered per-target results](output/baseline-31548/output/pdmc-baseline-provenance-20261105/per_target.csv) and [cluster ledger snapshot](output/baseline-31548/experiments.csv) are local. Preserve the accepted run and choose a new output directory for a new experiment.

The [appearance-failure report](output/appearance-failures-20261107/REPORT.md) analyzes this baseline with a six-type taxonomy, severity and frequency tables, annotated-occlusion plots, category comparisons and a reproducible 120-output visual audit. Open the [PDF](output/appearance-failures-20261107/REPORT.pdf) or [offline review viewer](output/appearance-failures-20261107/review/index.html). True-occlusion bins cover a matched COCOA subset; model-generated expansion is kept separate.

The [expanded-vocabulary appearance report](output/appearance-expanded-20261107/REPORT.md) analyzes all 2,565 annotated-vocabulary targets and visually reviews **all 1,390 saved outputs**. It includes the six-type failure taxonomy, severity/frequency, true-occlusion plots, category tables and source-checked examples. Open the [13-page PDF](output/appearance-expanded-20261107/REPORT.pdf) or [complete census viewer](output/appearance-expanded-20261107/review/index.html). This is separate from the baseline sample above.

## Surface consistency study

The [Surface-PMC controlled study](docs/SURFACE_CONSISTENCY_EXPERIMENT.md) tests
observable-only candidate selection using artificially hidden observed pixels,
shared four-seed pools, a development freeze and an independent replication.
Use the [reproduction cookbook](cookbooks/surface-consistency.md) for its separate
commands and `configs/research/` contracts. This mechanism study is distinct
from the full benchmark and its registry.

The completed [research handoff](output/surface-consistency-20261108/START_HERE.md)
contains a 13-page report, all 159 held-out cases, component explanations and
exact replay evidence. The full surface rule is inconclusive; the local seam
variant has an exploratory pooled benefit. See [measured results and limits](docs/SURFACE_CONSISTENCY_RESULTS.md).

The [2026-11-10 follow-up](docs/SURFACE_FOLLOWUP_20261110.md) adds an immutable
eight-candidate ranker, observation-based color correction and a progressive
runtime adapter. Use its [inference and report cookbook](cookbooks/surface-followup.md).
Fresh controlled evaluation and matched full progressive evaluation have separate
outputs; the older study above remains unchanged.

## CPU installation

Use Python 3.10 or newer for CPU tooling (Python 3.12 is a conservative choice). The GPU runner provisions its separate pinned Python 3.10 environment.

```bash
python -m venv .venv
# Linux/WSL: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests -v
python -m amodal_benchmark smoke
```

Bash and symlink regression tests require Linux/WSL:

```bash
for test_script in tests/test_*.sh; do bash "$test_script" || exit; done
```

No installation, CPU test, synthetic smoke, plan, or transfer `-WhatIf` contacts the cluster. The smoke command uses temporary synthetic images and substituted model calls. It checks the actual evaluation/artifact path; it does not measure model quality. Retain a debugging run only with `python -m amodal_benchmark smoke --output-dir output/smoke-debug-001` (the directory must be new).

## Project layout

| Path | Purpose |
| --- | --- |
| `amodal_benchmark/` | Data contract, references, detector/matcher, inference, metrics, scheduling, configuration, provenance, verification and registry |
| Root Python entry points | Backward-compatible preparation, evaluation, comparison and control commands |
| `configs/` | Explicit inherited research experiment contracts |
| `scripts/` | CPU smoke/check helpers and reviewable PowerShell source/result transfers |
| `tests/` | Python and Bash regressions, plus explicit GPU validation helpers; see the [test guide](tests/README.md) |
| `tests/fixtures/` | Small pinned annotation/source fixtures; tests need no real image download |
| `cookbooks/` | Reproducible evaluation and cluster recovery procedures |
| `docs/` | [Documentation index](docs/README.md), [codebase map](docs/CODEBASE.md), paper notes and archived implementation plans |
| `datasets/open_world_amodal/` | Existing original images, annotations and provenance; excluded from source distribution |
| `output/`, `logs/` | Experiment artifacts and execution logs; excluded from source distribution |
| `plan.md`, `progress.md`, `discoveries.md`, `closed_ideas.md`, `session.md` | Current research and handoff notes |

The package remains at the repository root to preserve cluster upload paths, old launch commands and source hashes. Editable installation makes `python -m amodal_benchmark` and `amodal` available from any working directory. Configured data paths are relative to the working directory; `extends` paths are relative to the config file.

Research papers and the original release ZIP remain at the root because the guides and optional integration tests use those paths. Historical logs are retained under `logs/`; completed implementation plans are indexed under [docs/archive/](docs/archive/README.md). The [codebase map](docs/CODEBASE.md) explains which files are generated and which must be preserved.

## Dataset and evaluation contract

The benchmark is fixed: **2,379 original image identities, 2,565 annotated targets, 553 labels**, across VG, COCOA, free images and LAION. There is no training split, submission format or inferred classification task. Repeated targets from an image stay on one worker; per-image and per-target seeds remain deterministic.

| Config | Intended use |
| --- | --- |
| `configs/baseline.json` | Full SAM-reference evaluation, semantic matcher v2, released vocabulary, four workers |
| `configs/annotated.json` | Same baseline with per-image annotated detector vocabulary; new output directory |
| `configs/lisa-reference.json` | Independent LISA references, best-reference-IoU matching and paper boundary repair; separate protocol |
| `configs/smoke.json` | First two real targets, one worker; excluded from the full registry |

These are JSON rather than YAML to avoid adding a parser to the pinned runtime. Unknown fields, duplicate JSON keys, inheritance cycles, invalid types and unsupported protocol combinations fail before model loading. A `full` contract requires every source, no target limit and complete inputs. CLI overrides are validated and captured in the resolved config.

```bash
python -m amodal_benchmark config configs/annotated.json
python -m amodal_benchmark references --dataset-root datasets/open_world_amodal --check-only
python -m amodal_benchmark run --config configs/smoke.json \
  --output-dir output/pdmc-smoke-001 --check-only
```

Reuse the existing dataset. For a new installation, use `prepare_open_world.py --help` and the [dataset installation guide](OPEN_WORLD_REPRODUCTION.md). Preserve original bytes and LAION provenance: reconstructed URLs do not prove identity with the authors' historical archive. The verifier checks recorded bytes and identities, not historical authenticity.

SAM visible references are generated during inference. LISA references require the separate preparation workflow, which the shared runner already handles. Their appearance metrics measure different reference protocols. CLIP uses successful completions only; always inspect denominators, missing inputs, reference failures and iteration-limited outputs.

## Cluster execution

The existing host is `daoqm@slurm.csse.rose-hulman.edu`; the project is `~/Vu_Hung/cv_nangcao`. Agents are authorized to perform project transfers, submissions and monitoring for requested work. In local PowerShell, inspect the source-only upload first:

```powershell
.\scripts\push.ps1 -RemotePath Vu_Hung/cv_nangcao -WhatIf
.\scripts\push.ps1 -RemotePath Vu_Hung/cv_nangcao
ssh daoqm@slurm.csse.rose-hulman.edu
```

`AGENTS.md`, current notes, environments, caches, results and the local ledger are excluded. `-IncludeData` is an explicit one-time option for a fresh remote project; the existing installation should reuse its dataset and model cache.

Generate a command locally or in an installed CPU environment. Supply your actual existing remote cache path. This example retains the configured `/dev/shm` cache; it is node-local and volatile:

```bash
python -m amodal_benchmark plan --config configs/baseline.json \
  --cache-dir /home/daoqm/Vu_Hung/cv_nangcao/.amodal-cache \
  --output-dir output/pdmc-baseline-config-v1 --run-only
```

The planner prints an executable Bash script with `sbatch --gres=gpu:4` before `run_amodal_slurm.sh` and `--benchmark-workers 4` after it. It expands the contract into all shared-runner flags and forwards `--benchmark-config` for provenance. Use the generated command from the remote project directory:

```bash
cd ~/Vu_Hung/cv_nangcao
mkdir -p logs
# Paste the reviewed command printed by amodal plan.
```

Review the plan's partition (`debug-gpu` by default), 16 CPUs, 64G memory and 24-hour limit against current cluster policy. Omit `--run-only` when provisioning the cache for the first time. The runner checks GPU availability, imports and its real CUDA extension before inference. The project package is not required in the GPU environment: the root evaluator and package directory travel together. Do not install the CPU dependency file over the pinned GPU environment.

For a baseline that validates the complete runtime before evaluation, the staged workflow runs both GPU integration checks, an eight-target/four-worker scored smoke, full evaluation, provenance-required verification, and registration:

```bash
cd ~/Vu_Hung/cv_nangcao
mkdir -p logs
sbatch --partition=debug-gpu --job-name=pdmc-baseline \
  --gres=gpu:4 --cpus-per-task=16 --mem=64G --time=24:00:00 \
  --output=logs/%x-%j.out scripts/run_baseline_slurm.sh \
  --cache-dir /home/daoqm/Vu_Hung/cv_nangcao/.amodal-cache \
  --output-dir output/pdmc-baseline-provenance-new \
  --benchmark-workers 4
```

Use `python scripts/run_baseline_workflow.py --cache-dir /home/daoqm/Vu_Hung/cv_nangcao/.amodal-cache --output-dir output/pdmc-baseline-provenance-new --plan` to inspect every command without running it. Add `--run-only` to require the prepared persistent cache: provisioning is skipped, while GPU validation and the inference preflight still run. The separate `-smoke` directory uses `configs/baseline-gpu-smoke.json`: the first two targets historically fail selection, so eight targets exercise a known completion case. Full inference starts only after smoke produces a verified completion scored by all four metrics. The final registry entry requires full provenance verification. Stage logs and launcher source hashes are preserved under `RUN/workflow/ATTEMPT/`. Resume with the same command and output paths; the workflow refuses to adopt historical targets lacking execution provenance. A job continues beyond an interactive work session, up to its Slurm time limit.

The existing direct workflow still works:

```bash
sbatch --partition=debug-gpu --gres=gpu:4 \
  --cpus-per-task=16 --mem=64G --time=24:00:00 \
  --output=logs/%x-%j.out run_pdmc_open_world.sh --run-only \
  --cache-dir /home/daoqm/Vu_Hung/cv_nangcao/.amodal-cache --benchmark-workers 4 \
  --output-dir output/pdmc-open-world-sam-semantic-v2
```

Inside the already provisioned runtime, with the shared runner's import paths configured, a direct configured evaluation is:

```bash
python evaluate_open_world.py --config configs/baseline.json \
  --repo-dir /path/to/cache/amodal --output-dir output/pdmc-baseline-config-v1
```

## Resume and inspect artifacts

Resume with the same experiment settings and output directory. A worker-count change is operational and may reuse results. Matcher, detector, boundary, seed, model or metric changes require a new directory. A configured smoke run has a fixed smoke scope; start a new full run. Historical CLI-only runs retain their existing subset-to-full resume behavior.

Every new coordinator execution that passes data preflight records:

- `execution.json`: latest attempt, status, arguments, target IDs, result hashes and newly written targets.
- `provenance/EXECUTION/`: preserved source files and hashes, exact installed package versions, selected execution environment variables and the attempt record.
- `resolved_config.json`: the configured experiment, when `--config` is used. Per-execution resolved configs retain operational overrides.
- Existing `run_config.json`, `metric_config.json`, `targets/`, `visible_references/`, `summary.json`, `per_target.csv`, and paper-comparison reports.

Caught errors and cancellation are recorded; a hard kill can leave an unresolved `running` status. Workers retain their separate logs. Historical results are reused without rewriting them. Provenance marks which targets were already present, so a new snapshot does not claim authorship of old outputs.

Score-only uses the saved inference settings and never reloads PD-MC. Successful targets with empty references now cache their CLIP-only scores, too. For a configured SAM run:

```bash
python evaluate_open_world.py --config configs/baseline.json \
  --output-dir output/pdmc-baseline-config-v1 --score-only
```

## Independent verification and research ledger

```bash
python -m amodal_benchmark verify \
  --run-dir output/pdmc-baseline-config-v1 \
  --dataset-root datasets/open_world_amodal --require-full --require-provenance

python -m amodal_benchmark register \
  --run-dir output/pdmc-baseline-config-v1 \
  --dataset-root datasets/open_world_amodal
```

Verification is read-only. It checks annotations/order, original image hashes, completion and reference artifacts, geometry, metric-cache inputs, finite metrics, failure penalties, counts, denominators, recomputed aggregates and source/environment snapshots. It does not rerun learned metric models on CPU. Older CLIP-only outputs without a cache are explicitly reported; rescore them before full registration.

Registration creates `experiments.csv` and `experiment_report.md` only on request. Completed entries must be full, scored and verified with provenance. A failed/interrupted full attempt can be registered with its recorded reason, even before any summary exists. Re-registering the same execution is idempotent; changed metadata cannot silently overwrite it. Register failed attempts before resuming to retain them in the ledger; execution snapshots are retained either way.

For a one-factor experiment, inherit the baseline config, set a hypothesis and `experiment.changed_factor`, use a new output directory, and register against the explicit baseline:

```bash
python -m amodal_benchmark register \
  --run-dir output/pdmc-annotated-config-v1 \
  --dataset-root datasets/open_world_amodal \
  --baseline-dir output/pdmc-baseline-config-v1
```

The baseline check validates one declared scientific change, unchanged checkpoint/model/matcher identities beyond that factor, identical input bytes and target identities, and reference comparability. LISA comparisons require identical frozen reference bytes. SAM-generated references may change when detection changes; the ledger labels that distinction. Use `--reference-dir` and `--baseline-reference-dir` for external LISA reference directories. Registration locks the runs while verifying them so it cannot race resumed inference. No automatic best-model selection is made from incompatible protocols or conditional metric means.

**Accepted best:** not selected by this rebuild. Existing experiment results remain historical evidence; no GPU quality result was generated here.

## Download and handoff

```powershell
.\scripts\pull.ps1 -RemotePath Vu_Hung/cv_nangcao `
  -Run pdmc-baseline-config-v1 -IncludeLedger -WhatIf
.\scripts\pull.ps1 -RemotePath Vu_Hung/cv_nangcao `
  -Run pdmc-baseline-config-v1 -IncludeLedger
```

The destination must be new. Remote ledgers go into timestamped `ledger-snapshots/`, never over the local registry. After download, verify against local original inputs. Preserve external LISA references separately when needed.

Initialize notes for a fresh project with `python -m amodal_benchmark notes --directory . --project NEW-SLUG` (use a lowercase slug). Existing files are never overwritten. Update the hypothesis before an experiment and record findings with run paths, evidence, protocol and limitations.

See [the paper-to-code map](docs/PAPER_CODE_MAP.md), [reproducible evaluation cookbook](cookbooks/reproducible-evaluation.md), and [cluster recovery cookbook](cookbooks/cluster-recovery.md).

## Independent human visual review

The [human review handoff](output/human-review-20261108/README.md) contains a ready-to-share 160-case random audit and a full 1,390-output alternative for one human rater. Both use an offline form with blank ratings, native evidence, anonymous case IDs and backup/resume. Send only the selected reviewer ZIP; keep the identity key and earlier assistant labels with the coordinator.

See [the protocol and response-import commands](docs/HUMAN_VISUAL_REVIEW.md). Human ratings are still pending; the packet itself is not evidence of completed human evaluation.
