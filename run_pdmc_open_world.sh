#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
set -Eeuo pipefail

# Reuse pinned PD-MC provisioning and Slurm handling, with no LISA stage.
PDMC_SCRIPT_DIR="${AMODAL_SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
if [[ ! -f "$PDMC_SCRIPT_DIR/run_amodal_slurm.sh" && -n "${SLURM_JOB_ID:-}" ]]; then
    PDMC_SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PDMC_SCRIPT_DIR}"
fi
[[ -f "$PDMC_SCRIPT_DIR/run_amodal_slurm.sh" ]] || {
    printf 'ERROR: Cannot locate run_amodal_slurm.sh; set AMODAL_SCRIPT_DIR to the project directory\n' >&2
    exit 1
}
for PDMC_ARG in "$@"; do
    if [[ "$PDMC_ARG" == --help || "$PDMC_ARG" == -h ]]; then
        cat <<'EOF'
Usage: bash run_pdmc_open_world.sh [shared runner options]

Run released PD-MC on OpenWorld using Grounding DINO + SAM, InstaOrder,
LaMa and Mixed Context diffusion. No LISA setup, download or inference.
Defaults: ./datasets/open_world_amodal, ./output/pdmc-open-world-sam-semantic-v2,
timestep 35, inclusive max_iter_id 5, released boundary behavior,
semantic label matching v2 and automatic backdrop sizing.
SAM appearance references are generated inside OUTPUT/visible_references.
These references define a separate evaluation protocol from the paper's LISA references.

Use --local inside an allocation, --benchmark-limit 2 for a smoke run,
--benchmark-inference-only for completions only, and --benchmark-score-only
to score saved results. All other options come from the shared runner below.

Use --benchmark-detector-vocabulary annotated to add missing per-image
annotation categories as separate detector queries. This is a new experiment;
set --output-dir ./output/pdmc-open-world-sam-annotated.

Use a new --output-dir for corrected matching; v1 results cannot be resumed
for inference or imported with the v2 matcher. Saved runs can still be scored.
Resume historical runs with --benchmark-target-selection label and their
original --benchmark-background-sizing released|auto setting. Result import
is supported only when the sole change is enabling automatic backdrop sizing.

EOF
        exec bash "$PDMC_SCRIPT_DIR/run_amodal_slurm.sh" --help
    fi
done
exec bash "$PDMC_SCRIPT_DIR/run_amodal_slurm.sh" \
    --benchmark-root "$PDMC_SCRIPT_DIR/datasets/open_world_amodal" \
    --output-dir "$PDMC_SCRIPT_DIR/output/pdmc-open-world-sam-semantic-v2" \
    --job-name pdmc-open-world-sam \
    --benchmark-target-selection label-semantic \
    --benchmark-background-sizing auto \
    "$@" --benchmark-pipeline-profile released-sam
