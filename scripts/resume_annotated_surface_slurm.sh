#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -Eeuo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the project root}"
[[ "${1:-}" == --benchmark-workers && "${2:-}" == 4 ]] || exit 2
shift 2
[[ "${1:-}" == --previous-job && "$#" == 2 && "$2" =~ ^[0-9]+$ ]] || exit 2
PREVIOUS="$2"
STATE="$(sacct -j "$PREVIOUS" --noheader --parsable2 --format=JobIDRaw,State | awk -F'|' -v id="$PREVIOUS" '$1 == id {print $2}')"
if [[ "$STATE" == COMPLETED ]]; then
    echo "Previous full surface job $PREVIOUS completed; continuation is unnecessary."
    exit 0
fi
if [[ "$STATE" != TIMEOUT ]]; then
    echo "Previous job $PREVIOUS ended as $STATE; require diagnosis instead of automatic retry." >&2
    exit 1
fi
bash scripts/run_annotated_surface_slurm.sh --benchmark-workers 4 --stage surface-full
