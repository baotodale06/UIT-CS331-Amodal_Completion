# Working together

Start with [the collaborator guide](docs/GETTING_STARTED.md). The supported
development check uses Linux/WSL, Python 3.12 and the committed CPU locks.

Create a personal branch from the shared setup branch:

```bash
git switch setup/reproducible-development
git pull --ff-only
git switch -c feature/your-change
source .venv-cpu/bin/activate
bash scripts/check.sh
git add <changed-source-files>
git commit -m "Describe the change"
git push -u origin feature/your-change
```

Open a pull request against `setup/reproducible-development` until that branch
is merged into `main`. Include the problem, resulting behavior, checks performed,
and any GPU validation still needed. Keep changes to the vendored original
pipeline separate from benchmark infrastructure changes.

Use four-space Python indentation, `snake_case`, `pathlib`, `argparse`, and
`unittest`. Quote Bash paths and retain strict error handling. Add focused
regressions for changed behavior. GitHub Actions runs the offline suite, synthetic
smoke, configuration checks and distribution build without models or datasets.

Commit source, experiment configs and small pinned fixtures. Keep datasets,
model checkpoints, caches, virtual environments, logs, generated results,
credentials and `AGENTS.md` local. Preserve third-party licenses and citations;
the vendored trees retain their existing upstream terms.

Experiments must record the Git commit, resolved configuration, seed, dataset
hashes, model identities, protocol and failure denominators. Inherit a config,
declare the scientific factor being changed, and choose a new output directory.
Resume only with the original configuration and source. Preserve frozen source
and artifacts for historical replay. Full results require `amodal verify
--require-full --require-provenance` before explicit registration. SAM and LISA
references define different protocols and must not be pooled.

To update dependencies, follow [requirements/README.md](requirements/README.md),
review the lock diff, and test in a fresh environment. CPU locks are for
development; the GPU runner owns its separate pinned runtime. Runtime/build
changes also require `tests/validate_cuda_extension.sh` and
`tests/validate_pipeline_imports.sh` in a real GPU allocation.
