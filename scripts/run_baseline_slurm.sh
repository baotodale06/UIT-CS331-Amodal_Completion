#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --export=ALL
set -Eeuo pipefail

PROJECT_DIR="${AMODAL_SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}}"
cd -- "$PROJECT_DIR"
export AMODAL_SCRIPT_DIR="$PROJECT_DIR"
exec python3 -u scripts/run_baseline_workflow.py "$@"
