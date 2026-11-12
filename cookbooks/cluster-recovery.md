# Cluster handoff and recovery

Use `scripts/push.ps1 -RemotePath Vu_Hung/cv_nangcao -WhatIf` to inspect the source upload. `AGENTS.md`, environments, current research notes, generated results and the experiment ledger are excluded. Upload raw data only with an explicit `-IncludeData`; the existing cluster installation normally needs source updates only.

Run from `~/Vu_Hung/cv_nangcao`. Full presets request four GPUs before the script name and four workers after it. `amodal plan` prints the complete command without contacting Slurm. Review partition, time, CPU and memory requests against the allocation before running it.

Use the existing cache location. Paths under `/dev/shm` are node-local and volatile; a durable scratch cache is preferable for future runs, but moving a running job's cache is unnecessary. Do not overwrite a source checkout while its job is running.

After a worker failure, inspect `workers/*/worker-*.log` and `execution.json`. Preserve the failed attempt in the registry when it was a full configured run. Resume with the same output directory and settings. Snapshot records retain each attempt; a hard kill can leave `status: running`, which means termination is unconfirmed, not success.

The empty completion-mask repair remains in `amodal_benchmark/inference.py`. It preserves matcher and detector identities. Historical annotated outputs can resume with their existing CLI, without adding a new config. New experimental settings require a new output directory.

Download one completed run with `scripts/pull.ps1`; `-IncludeLedger` saves a timestamped ledger snapshot and never merges it into the local registry. Run `amodal verify` against the local original images after download. For LISA runs, supply the same fixed reference artifacts with `--reference-dir`.

GPU validation remains a separate cluster step:

```bash
bash tests/validate_cuda_extension.sh /path/to/cache/conda-env /path/to/cache/amodal
bash tests/validate_pipeline_imports.sh /path/to/cache/conda-env /path/to/cache/amodal
```

These checks need the provisioned runtime; passing CPU tests does not establish GPU memory fit or numeric equivalence.
