#!/usr/bin/env bash
# Export completed controlled evidence while the full progressive job continues.
set -Eeuo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Run inside the existing project allocation}"
EXPERIMENT=output/surface-followup-20261110
DESIGN=output/surface-followup-20261110-design
ARCHIVE="$DESIGN/fresh-controlled-evidence.tar.gz"
[[ ! -e "$ARCHIVE" ]] || { echo 'Use the existing completed evidence archive'; exit 0; }
for ATTEMPT in $(seq 1 160); do
    [[ -f "$EXPERIMENT/analysis-eight/replay-verification.json" ]] && break
    sleep 15
done
[[ -f "$EXPERIMENT/analysis-eight/replay-verification.json" ]] || exit 1
export OMP_NUM_THREADS=2 PYTHONNOUSERSITE=1
.amodal-cache/surface-metrics-env/bin/python scripts/build_surface_followup_report.py \
    --experiment-dir "$EXPERIMENT" --run-dir "$EXPERIMENT/full" \
    --extra-run-dir "$DESIGN/extra-pools-166/full" \
    --metrics-dir "$EXPERIMENT/metrics-eight" --freeze "$DESIGN/final-method-freeze.json" \
    --summary "$EXPERIMENT/analysis-eight/summary.json" \
    --output-dir "$EXPERIMENT/report" --panels-only
tar -czf "$ARCHIVE.partial" \
    "$EXPERIMENT/metrics-eight" "$EXPERIMENT/analysis-eight" \
    "$EXPERIMENT/inference-eight/contract.json" "$EXPERIMENT/inference-eight/decisions" \
    "$EXPERIMENT/report" "$DESIGN/eight/dev-features-166" \
    "$DESIGN/final-runtime-audit.log" "$DESIGN/final-runtime-audit" \
    "$DESIGN/final-runtime-audit-logs" "$DESIGN/fresh-eight-metric-logs"
mv -- "$ARCHIVE.partial" "$ARCHIVE"
sha256sum "$ARCHIVE" | tee "$ARCHIVE.sha256"
echo 'Controlled evidence export complete; full progressive work continues separately.'
