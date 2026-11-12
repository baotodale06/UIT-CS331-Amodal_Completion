# Reproduce the controlled Surface-PMC experiment

This is a separate one-step mechanism study. It does not use the full benchmark
registry or alter the accepted baseline. Read the [scientific contract](../docs/SURFACE_CONSISTENCY_EXPERIMENT.md)
before comparing results. Use a fresh directory for a changed experiment.

## Prepare observation-only inputs on CPU

```bash
python scripts/prepare_surface_experiment.py \
  --dataset-root datasets/open_world_amodal \
  --reference-run output/baseline-31548/output/pdmc-baseline-provenance-20261105 \
  --config configs/research/surface-consistency-v1.json \
  --output-dir output/NEW-surface-pilot
```

The preparation writes `inputs/` and `generation_manifest.json` separately from
`evaluation_only/`. The former contains observed RGB with every artificial-occluder
pixel replaced by 127, observed visible masks and occlusion masks. The latter
contains held-out RGB and masks and is never read by the generation worker.
`cases.json` and `preparation.json` retain IDs, transformations, splits and hashes.

For the prespecified replication, exclude every pilot image identity and byte hash:

```bash
python scripts/prepare_surface_experiment.py \
  --dataset-root datasets/open_world_amodal \
  --reference-run output/baseline-31548/output/pdmc-baseline-provenance-20261105 \
  --config configs/research/surface-consistency-replication-v1.json \
  --exclude-cases output/NEW-surface-pilot/cases.json \
  --output-dir output/NEW-surface-replication
```

The fixed replication has 87 cases because only 15 additional LAION images meet
the unchanged eligibility criteria. It contains no development cases. The
experiment config files in `configs/research/` use their own mechanism-study
schema; they are not inputs to `amodal config` or the full benchmark evaluator.

## Generate shared pools in the existing cluster cache

Transfer the new research scripts/package and prepared observation inputs without
overwriting historical outputs. Submit from `~/Vu_Hung/cv_nangcao`:

```bash
sbatch --partition=debug-gpu --gres=gpu:4 \
  --output=output/NEW-surface-pilot-smoke-%j.log \
  scripts/run_surface_slurm.sh --benchmark-workers 4 \
  --experiment-dir output/NEW-surface-pilot \
  --output-dir output/NEW-surface-pilot/smoke --limit 4 --candidate-limit 2
```

The smoke runs the real extension/FFT and pipeline-import validations, then loads
the frozen sampler. Inspect all eight records and images. The existing cache is
used without reinstalling CPU dependencies into the GPU environment.

```bash
sbatch --partition=debug-gpu --gres=gpu:4 \
  --export=ALL,SURFACE_VALIDATE_RUNTIME=0 \
  --output=output/NEW-surface-pilot-full-%j.log \
  scripts/run_surface_slurm.sh --benchmark-workers 4 \
  --experiment-dir output/NEW-surface-pilot \
  --output-dir output/NEW-surface-pilot/full
```

Skip repeated runtime checks only after they have passed in the same unchanged
environment. Every candidate writes PNG RGB, predicted mask, raw surface terms,
validity reasons, timing, peak allocated VRAM and hashes. Resume requires the
same source/input contract and verifies cached artifacts. Error records are
retained rather than silently retried with a different configuration. Check the
record counts and errors; a successful shell exit alone does not prove complete
model-quality coverage. Logs currently use the study's fixed
`output/surface-consistency-20261108/job-logs/` folder; the records themselves go
to the requested output directory.

Repeat with the replication experiment directory, using the same frozen sampler
and seed schedule. Record exact commands/job IDs and preserve unrelated jobs.

## Freeze choices before evaluating test truth

Recover candidate files with independently computed remote hashes. In CPU tooling:

```bash
python scripts/analyze_surface_experiment.py \
  --experiment-dir output/NEW-surface-pilot \
  --run-dir output/NEW-surface-pilot/full --stage freeze
python scripts/analyze_surface_experiment.py \
  --experiment-dir output/NEW-surface-pilot \
  --run-dir output/NEW-surface-pilot/full --stage test
python scripts/analyze_surface_experiment.py \
  --experiment-dir output/NEW-surface-replication \
  --run-dir output/NEW-surface-replication/full --stage test \
  --freeze-from output/NEW-surface-pilot/analysis/development_freeze.json
```

The first command reads development truth only and refuses to overwrite its
freeze. Test analysis validates the frozen scorer and protocol. Replication
also rejects overlap with pilot images and scientific changes beyond the declared
cohort fields. No hidden-ground-truth oracle may become a deployment selector.

## Supporting perceptual evaluation

Transfer `cases.json` and `evaluation_only/` for the separate metric worker:

```bash
sbatch --partition=debug-gpu --gres=gpu:4 \
  scripts/run_surface_metrics_slurm.sh --benchmark-workers 4 \
  --experiment-dir output/NEW-surface-pilot --run-dir output/NEW-surface-pilot/full
```

Repeat for replication. This evaluates masked spatial LPIPS using existing
AlexNet weights. It never modifies candidate selection. Keep common complete-pool
denominators and preserve missing perceptual values explicitly.

## Replay and report

```bash
python scripts/validate_surface_experiment.py \
  --experiment-dir output/NEW-surface-pilot --run-dir output/NEW-surface-pilot/full \
  --dataset-root datasets/open_world_amodal \
  --reference-run output/baseline-31548/output/pdmc-baseline-provenance-20261105 \
  --smoke-dir output/NEW-surface-pilot/smoke
python scripts/validate_surface_experiment.py \
  --experiment-dir output/NEW-surface-replication --run-dir output/NEW-surface-replication/full \
  --dataset-root datasets/open_world_amodal \
  --reference-run output/baseline-31548/output/pdmc-baseline-provenance-20261105 \
  --freeze-from output/NEW-surface-pilot/analysis/development_freeze.json
python scripts/surface_counterexamples.py output/NEW-surface-pilot/counterexamples
python scripts/explain_surface_results.py --pilot output/NEW-surface-pilot \
  --replication output/NEW-surface-replication \
  --output-dir output/NEW-surface-pilot/diagnostics --include-truth-check
python scripts/build_surface_report.py --pilot output/NEW-surface-pilot \
  --replication output/NEW-surface-replication --output-dir output/NEW-surface-pilot/report
python scripts/validate_surface_report.py --pilot output/NEW-surface-pilot \
  --replication output/NEW-surface-replication --report output/NEW-surface-pilot/report
```

Reporting additionally needs the existing CPU Matplotlib/ReportLab dependencies.
The validators replay prepared RGB/masks from source images, verify output hashes,
recompute every score, metric, selection and summary, and compare repeated smoke
seeds. **Use the recorded scorer runtime for exact raw-score replay:** Python
3.10, NumPy 1.23.5 and Pillow 9.5.0. Both completed cohorts replay with maximum
raw-score difference zero in that environment. The original development-analysis
runtime is recorded separately in `analysis/development_freeze.json`.

Float32 conversion followed by hard feature bins is numerically sensitive. The
separate Windows NumPy 2.5.4 audit found raw-term changes up to 0.04718 and one
changed pilot boundary-plus-color decision. Full/local decisions are unchanged
in both cohorts. Do not widen the validator tolerance or overwrite frozen results
to make a different numerical runtime pass. Instead, preserve a separate audit:

```bash
python scripts/audit_surface_portability.py --experiment-dir output/NEW-surface-pilot \
  --freeze output/NEW-surface-pilot/analysis/development_freeze.json \
  --output-dir output/NEW-surface-pilot/portability-NEW-runtime --workers 4
```

The optional truth-compatibility diagnostic is evaluation only and was introduced
after pilot test analysis, before replication errors were inspected. It must not
change a selector or its frozen scales. The report validator recomputes both
cohort and pooled summaries, checks perceptual input identities, and verifies
all gallery images, report links and the rendered PDF. Learned LPIPS values are
not recomputed by this CPU-only report validator.

Report the primary full-surface result even if it is negative or inconclusive.
Keep all fixed ablations, equal-budget controls, the unavailable oracle, coverage,
paired confidence intervals, visual losses and limits. The independent human
review packet is a separate pending task; do not relabel assistant judgments.
