# Repository handoff: 2026-11-12

Branch: `setup/reproducible-development`.
Base: `14834b65a09878b20fe32267919a47f16e784c7d`.

Start with [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md). Install the locked
CPU environment, run `bash scripts/check.sh`, and create a personal branch.
See [progress.md](progress.md) for completed checks and
[CONTRIBUTING.md](CONTRIBUTING.md) for PR and experiment conventions.

The original model code, vendored projects and two proposal documents remain
unchanged. Imported package/config/test bytes match the local source snapshot.
Datasets, caches, model weights, generated results, credentials and `AGENTS.md`
are excluded. Reproduction of saved research results requires separate artifacts;
the CPU setup is not a replacement for the pinned GPU/native replay environments.

This repository-setup task submitted no GPU jobs and changed no existing cluster
experiments. Historical job IDs in imported documents are context, not a current
queue. The original local workspace remains separate from this Git checkout.
