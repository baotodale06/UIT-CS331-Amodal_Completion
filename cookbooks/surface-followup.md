# Use the frozen eight-candidate Surface-PMC method

The 2026-11-10 follow-up learns a small candidate ranker from 183 old images and
evaluates it on 166 new images. It preserves the pretrained generator and uses
observed surface continuity, visible-pixel preservation, sample agreement and
pretrained appearance features. Color calibration and bounded smooth consensus
correction follow selection. The model freeze is immutable:

`output/surface-followup-20261110-design/final-method-freeze.json`

SHA-256: `a0a626fcfddec095a1d7b677b496309fccb79ff6b4e446f5127b69368f3358e1`.

See [the research record](../docs/SURFACE_FOLLOWUP_20261110.md) for development
decisions and limitations. The original five-hour window elapsed before final
evaluation. Controlled reconstruction and full progressive results are separate.

## Observation-only inference on an existing candidate pool

Run with the recorded Python 3.10 / NumPy 1.23.5 / Pillow 9.5.0 environment and
SciPy 1.10.1. Use a new output directory. This command does not open evaluation
RGB or hidden-support files:

```bash
python scripts/select_surface_pool.py \
  --experiment-dir output/surface-followup-20261110 \
  --run-dir output/surface-followup-20261110/full \
  --extra-run-dir output/surface-followup-20261110-design/extra-pools-166/full \
  --freeze output/surface-followup-20261110-design/final-method-freeze.json \
  --features-dir output/surface-followup-20261110-design/eight/dev-features-166 \
  --output-dir output/NEW-surface-inference
```

The eight seeds are 11, 22, 33, 44, 55, 66, 77 and 88. Cached features are tied
to exact candidate bytes and observation manifests. Frozen source checks reject
unrecorded implementation changes. A fresh candidate pool needs its own matching
observation-only features; existing cached features cannot be reused for it.

## Progressive inference on the cluster

From `~/Vu_Hung/cv_nangcao`, in the prepared persistent environment:

```bash
SURFACE_CUFFT_COMPAT=1 sbatch --partition=debug-gpu --gres=gpu:4 \
  scripts/run_surface_progressive_slurm.sh --benchmark-workers 4 \
  --surface-freeze output/surface-followup-20261110-design/final-method-freeze.json \
  --output-dir output/NEW-surface-progressive
```

Add `--limit 8` for a separate smoke output. The adapter samples eight candidates
per diffusion step, anchors all later steps to original observed pixels, restores
native-resolution observed RGB, and retains upstream stopping and segmentation.
Keep the current model cache, detector vocabulary and SAM reference protocol.
For a paired full comparison, use the same allocation/runtime and copy the
control's frozen visible references, as in `scripts/run_surface_final_slurm.sh`.
Additional sampling has a substantial runtime cost.

Controlled scoring completed in job **37785**. Strict replay on a different CPU
failed before progressive work, so native replay uses **37799** on the original
generation node gus (CPU only). GPU job **37812** depends on that replay and runs
the finalized smoke, matched full control and progressive method. CPU **37813**
exports controlled evidence; **37814** verifies, compares and registers the full
progressive pair after37812 succeeds. Old pending finalizer37795 was superseded.
Check those jobs before submitting duplicate work. Resume interrupted runs only
with the same frozen method, runtime and output directory.

## Rebuild and validate the controlled report

With all saved native artifacts present, run the independent CPU replay:

For strict primitive-score reproduction, use the original generation node gus
and pinned Conda libraries. `scripts/replay_surface_followup.sh` isolates user
packages and captures a numerical diagnostic. Matching NumPy/Pillow versions
alone is insufficient across CPU dispatch paths; do not loosen the tolerance.

```bash
python scripts/validate_surface_followup.py \
  --experiment-dir output/surface-followup-20261110 \
  --run-dir output/surface-followup-20261110/full \
  --extra-run-dir output/surface-followup-20261110-design/extra-pools-166/full \
  --metrics-dir output/surface-followup-20261110/metrics-eight \
  --inference-dir output/surface-followup-20261110/inference-eight \
  --features-dir output/surface-followup-20261110-design/eight/dev-features-166 \
  --freeze output/surface-followup-20261110-design/final-method-freeze.json \
  --output output/surface-followup-20261110/analysis-eight/replay-verification.json \
  --workers 4
python scripts/build_surface_followup_report.py \
  --experiment-dir output/surface-followup-20261110 \
  --run-dir output/surface-followup-20261110/full \
  --extra-run-dir output/surface-followup-20261110-design/extra-pools-166/full \
  --metrics-dir output/surface-followup-20261110/metrics-eight \
  --freeze output/surface-followup-20261110-design/final-method-freeze.json \
  --summary output/surface-followup-20261110/analysis-eight/summary.json \
  --full-comparison output/surface-followup-20261110-design/paired_comparison.json \
  --output-dir output/surface-followup-20261110/report
python scripts/validate_surface_followup_report.py \
  --experiment-dir output/surface-followup-20261110 \
  --freeze output/surface-followup-20261110-design/final-method-freeze.json
```

The report includes every case and equal-budget controls, not only favorable
examples. CPU replay checks exact choices, corrected RGB and hidden MAE; neural
metrics require the recorded GPU metric environment. Report rendering can use a
separate CPU environment with Matplotlib and ReportLab. Add `--reuse-panels` to
the builder to use already verified display panels without rereading candidates.

The fresh study uses artificial occlusion of observed pixels, with MAE measured
on a fixed hidden core. Paper-style LPIPS/DreamSim use objects on black backgrounds;
CLIP is category-text cosine and also an explicit selector feature. These are
not numerical reproductions of the paper's COCO/Open Images dataset. Natural
occlusion in the full Open-World benchmark lacks hidden RGB ground truth, so it
cannot establish hidden MAE. Preserve its existing metric protocol and failures.
