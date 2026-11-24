#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf -- "$TMP_DIR"' EXIT
CODE_DIR="$TMP_DIR/study/evaluation-source"
PROJECT="$TMP_DIR/project"
mkdir -p "$CODE_DIR" "$PROJECT/.amodal-cache/surface-metrics-env/bin"
cp "$ROOT/scripts/run_shape_v2_evaluation.sh" "$CODE_DIR/runner.sh"
cat > "$CODE_DIR/run_amodal_slurm.sh" <<'SH'
CACHE_DIR="${AMODAL_CACHE_DIR:-$PWD/.amodal-cache}"
ENV_DIR="${AMODAL_ENV_DIR:-$CACHE_DIR/conda-env}"
REPO_DIR="${AMODAL_REPO_DIR:-$CACHE_DIR/amodal}"
configure_python_imports() {
    [[ "$CACHE_DIR" == "$SHAPE_PROJECT_DIR/.amodal-cache" ]]
    [[ "$ENV_DIR" == "$CACHE_DIR/conda-env" ]]
    [[ "$REPO_DIR" == "$CACHE_DIR/amodal" ]]
    export PYTHONPATH="$REPO_DIR"
}
SH
cat > "$PROJECT/.amodal-cache/surface-metrics-env/bin/python" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
printf '%s\n' "$*" >> "$SHAPE_TEST_CALLS"
SH
chmod +x "$PROJECT/.amodal-cache/surface-metrics-env/bin/python"
export SHAPE_PROJECT_DIR="$PROJECT" SHAPE_TEST_CALLS="$TMP_DIR/calls.txt"
export SLURM_SUBMIT_DIR="$CODE_DIR" CUDA_VISIBLE_DEVICES=0,1,2,3
bash "$CODE_DIR/runner.sh" geometry
[[ "$(wc -l < "$SHAPE_TEST_CALLS")" == 3 ]]
grep -q -- '--split validation' "$SHAPE_TEST_CALLS"
grep -q -- 'freeze_shape_v2.py' "$SHAPE_TEST_CALLS"
grep -q -- '--split holdout' "$SHAPE_TEST_CALLS"
bash "$CODE_DIR/runner.sh" --benchmark-workers 4 smoke
grep -q -- "--repo-dir $PROJECT/.amodal-cache/amodal" "$SHAPE_TEST_CALLS"
grep -q -- '--shape-v2-checkpoint' "$SHAPE_TEST_CALLS"
if CUDA_VISIBLE_DEVICES=0 bash "$CODE_DIR/runner.sh" --benchmark-workers 4 smoke; then
    echo 'FAIL: accepted one GPU for four-worker benchmark' >&2
    exit 1
fi
echo 'PASS: isolated shape-v2 evaluation uses the persistent cache and enforces GPU count'
