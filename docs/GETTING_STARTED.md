# Clone, check, and run an experiment

This branch adds the maintained Open-World benchmark and Surface-PMC research
tooling to the existing PD-MC repository. Original `main.py`, `main.sh`,
`install.sh`, root `requirements.txt`, vendored model code, licenses and the two
existing PMC planning documents are preserved. The branch starts from
`14834b65a09878b20fe32267919a47f16e784c7d`.

## 1. Install the CPU development environment

Use Linux or WSL2 for the complete test suite. Install Git and Python 3.12 with
venv/ensurepip support (on an Ubuntu version providing Python 3.12:
`sudo apt install git python3.12 python3.12-venv`). The tested patch is 3.12.12;
see `.python-version`. Python's standard installer or a Python version manager
can also provide it. No CUDA or model download is needed for this step.

```bash
git clone --branch setup/reproducible-development \
  https://github.com/baotodale06/UIT-CS331-Amodal_Completion.git
cd UIT-CS331-Amodal_Completion
python3.12 scripts/setup_cpu.py
source .venv-cpu/bin/activate
bash scripts/check.sh
amodal config configs/baseline.json
```

The installer creates `.venv-cpu`, installs exact versions with verified hashes,
and installs the project editable. It refuses to modify an unrelated existing
environment. Installation needs internet; checks afterwards are offline.
Run commands from the repository root so relative dataset/config paths resolve.

On native Windows, `py -3.12 scripts/setup_cpu.py` creates
`.venv-cpu\Scripts\python.exe`. Use that Python to run
`-m unittest discover -s tests -v` and `-m amodal_benchmark smoke`.
Use a separate checkout/environment inside WSL for all Bash, symlink and POSIX
process tests; do not share a Windows virtual environment with WSL.

Checks cover Python regressions, five Bash regressions, and synthetic evaluator
output/provenance verification. Two optional tests skip when the original
release ZIP and historical scored run are absent. SciPy and scikit-learn are
included in the CPU lock so research unit tests run. Synthetic smoke does not
measure completion quality. The [CI workflow](../.github/workflows/cpu-checks.yml)
also resolves all four main configs and builds the source/wheel distributions.

## 2. Work on a personal branch

```bash
git switch -c feature/my-change
# Edit source and run checks.
bash scripts/check.sh
git add <changed-source-files>
git commit -m "Describe the change"
git push -u origin feature/my-change
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for PR targets and experiment rules.
The standard runtime code is in `amodal_benchmark/`; `configs/` defines
experiments and `tests/` contains small offline fixtures. See the
[codebase map](CODEBASE.md) and [benchmark workflow](../BENCHMARK_WORKFLOW.md).
The root upstream model trees are retained for reading and development.
**The reproducible runner executes a separate pinned upstream checkout in its
cache. Editing root `main.py` does not change that cached runtime.** Changes to
the scientific model need an explicit new source identity/experiment, with
appropriate provenance and GPU validation.

## 3. Obtain and validate the benchmark data

Git includes annotation fixtures for tests, not benchmark images, checkpoints,
PDFs, frozen research outputs or generated references. To compare with existing
team runs, obtain the team's complete `datasets/open_world_amodal/` directory,
including image checksums, manifests and acquisition provenance, and preserve
its relative paths. Use your own SSH/account permissions for any transfer.

Alternatively, retrieve the public metadata and selected VG/COCO/free images:

```bash
python prepare_open_world.py --download-metadata --download-images --allow-incomplete
```

This is a **partial installation** until the selected LAION images are present.
The downloader pins official annotation hashes and writes an explicit missing
image report. If the authors' original LAION shard is available:

```bash
python prepare_open_world.py --laion-tar /path/to/original/00000.tar
```

The historical team data uses documented LAION reconstruction; an arbitrary
archive with matching filenames is not equivalent. Preserve the team's exact
bytes for comparisons. Read [dataset identity and installation](../OPEN_WORLD_REPRODUCTION.md#verified-dataset-and-installation-status)
before replacing files. A fresh public download cannot establish byte-identical
historical reproduction on its own.

Validate copied/downloaded data before inference:

```bash
python prepare_visible_references.py --dataset-root datasets/open_world_amodal --check-only
python -m amodal_benchmark run --config configs/smoke.json \
  --output-dir output/preflight-only --check-only
```

The complete contract is 2,379 image identities, 2,565 targets and 553 labels.
Do not use `--allow-incomplete` to describe a full scored experiment.

## 4. Provision the separate GPU runtime

Use Linux, an NVIDIA GPU with roughly 24 GiB available memory per PD-MC worker,
and the compiler/toolkit requirements in [AMODAL_SLURM.md](../AMODAL_SLURM.md).
The runner provisions Python 3.10, PyTorch 1.13.1, CUDA 11.7, compatible native
extensions, the upstream commit `bfbfcdec581484586774d9ca53f5c603c97c725b`, and
model checkpoints in `.amodal-cache`. Reuse that persistent cache. LISA has
additional memory/model requirements and a separate reference protocol.

The legacy root `requirements.txt` and `install.sh` remain for upstream context.
Use the shared runner for the benchmark runtime. Never install CPU locks into
the GPU environment. Provisioning and model downloads require network access;
some model endpoints may require `HF_TOKEN` in the environment.

Inside an allocated GPU session:

```bash
bash run_amodal_slurm.sh --local --setup-only --cache-dir "$PWD/.amodal-cache"
bash tests/validate_cuda_extension.sh "$PWD/.amodal-cache/conda-env" "$PWD/.amodal-cache/amodal"
bash tests/validate_pipeline_imports.sh "$PWD/.amodal-cache/conda-env" "$PWD/.amodal-cache/amodal"
```

The checks exercise a real CUDA operator, FFT round trip and pipeline imports.
The validated team hardware is Quadro RTX 6000 in `debug-gpu`; the pinned stack
previously failed LaMa FFT on L40S. Passing basic CUDA imports alone is inadequate.
This setup branch does not claim a new GPU validation or benchmark result.

## 5. Smoke, full run, verify

On a Slurm login node, from the cloned project directory with the CPU environment
active, generate and review the exact submission command:

```bash
amodal plan --config configs/smoke.json --cache-dir "$PWD/.amodal-cache" \
  --output-dir output/pdmc-smoke-001 --partition debug-gpu \
  --output-script output-smoke.sbatch
cat output-smoke.sbatch
bash output-smoke.sbatch
```

This creates a script containing `sbatch`; use `bash` to execute the generated
script. Omit `--run-only` for initial provisioning. Choose the partition/resource
options appropriate to your cluster. To run without Slurm, inside an existing
GPU session, use the dedicated wrapper's explicit equivalent smoke flags:

```bash
bash run_pdmc_open_world.sh --local --cache-dir "$PWD/.amodal-cache" \
  --benchmark-root datasets/open_world_amodal --benchmark-limit 2 \
  --benchmark-workers 1 --output-dir output/pdmc-smoke-local-001
```

After a successful real smoke, plan a full run in a new directory:

```bash
amodal plan --config configs/baseline.json --cache-dir "$PWD/.amodal-cache" \
  --output-dir output/pdmc-baseline-001 --partition debug-gpu \
  --output-script output-baseline.sbatch
cat output-baseline.sbatch
bash output-baseline.sbatch
```

The full config expands to `sbatch --gres=gpu:4 ... run_amodal_slurm.sh ...
--benchmark-workers 4`, preserving allocation and worker count. A bare
`--benchmark-config` does not expand runner options; use `amodal plan`.
Track job IDs and logs; leave valid long jobs running. Resume the same run with
its exact config and output directory. Choose a new directory for any scientific
change, and preserve previous outputs. SAM references are generated by inference;
LISA evaluation uses `configs/lisa-reference.json` and independent references.

When all targets are scored:

```bash
amodal verify --run-dir output/pdmc-baseline-001 \
  --dataset-root datasets/open_world_amodal --require-full --require-provenance
amodal register --run-dir output/pdmc-baseline-001 \
  --dataset-root datasets/open_world_amodal
```

Record `git rev-parse HEAD` with your job. The evaluator also captures source,
runtime, model and input provenance. Keep failure counts and metric denominators;
smoke/inference-only outputs are not complete scored baselines.

## Historical research artifacts

The maintained Surface-PMC source and research configs are included for continued
development. Some research scripts are dated, experiment-specific workflows.
Their `output/...` paths refer to external saved data, learned ranker freezes,
candidate pools, cached features, reports and native runtime records. Obtain the
matching artifact bundle before using those replay commands. Historical links in
`BENCHMARK_WORKFLOW.md` and research docs will not resolve from Git alone.
Preserve freezes and source hashes; do not regenerate or retune them silently.

Use [the Surface-PMC reference](SURFACE_CONSISTENT_PMC.md),
[measured results](SURFACE_CONSISTENCY_RESULTS.md), and
[follow-up cookbook](../cookbooks/surface-followup.md) for scope and limitations.
The two original PMC proposal documents remain unchanged. The latest imported
local notes are archived under [local-handoff-20261112](archive/local-handoff-20261112/README.md);
they are historical context, not a live job queue or an instruction to resubmit.
