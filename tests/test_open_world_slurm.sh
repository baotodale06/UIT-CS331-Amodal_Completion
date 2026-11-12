#!/usr/bin/env bash
set -Eeuo pipefail
test_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$test_root/run_amodal_slurm.sh"
parse_args --benchmark-root "$test_root/datasets/open_world_amodal" --benchmark-sources VG,COCOA \
    --benchmark-inference-only --benchmark-limit 2 --benchmark-seed 9 \
    --reference-dir /references --output-dir /outputs
[[ "$BENCHMARK_ROOT" == "$test_root/datasets/open_world_amodal" ]]
[[ "$BENCHMARK_SOURCES" == VG,COCOA && "$BENCHMARK_INFERENCE_ONLY" == 1 ]]
[[ "$BENCHMARK_LIMIT" == 2 && "$BENCHMARK_SEED" == 9 && "$REFERENCE_DIR" == /references ]]
printf 'PASS: benchmark options survive the shared Slurm argument parser\n'

# A full benchmark should prepare its own visible references by default.
REFERENCE_DIR=""
BENCHMARK_INFERENCE_ONLY=0
parse_args --benchmark-root /dataset --lisa-model xinlai/LISA-13B-llama2-v1 \
    --lisa-precision fp16 --lisa-device-map auto --benchmark-reference-only
[[ "$REFERENCE_DIR" == /dataset/visible_references ]]
[[ "$BENCHMARK_GENERATE_REFERENCES" == 1 && "$BENCHMARK_REFERENCE_ONLY" == 1 ]]
[[ "$LISA_MODEL" == xinlai/LISA-13B-llama2-v1 && "$LISA_PRECISION" == fp16 ]]
[[ "$LISA_DEVICE_MAP" == auto ]]
printf 'PASS: reference preparation uses the same annotated dataset and automatic reference directory\n'

[[ "$BENCHMARK_WORKERS" == auto ]]
parse_args --benchmark-workers 4
[[ "$BENCHMARK_WORKERS" == 4 ]]
parse_args --benchmark-workers 1
[[ "$BENCHMARK_WORKERS" == 1 ]]
if (parse_args --benchmark-workers 0) >/dev/null 2>&1; then
    printf 'FAIL: zero benchmark workers were accepted\n' >&2
    exit 1
fi
printf 'PASS: automatic multi-GPU default, explicit worker count, and single-GPU fallback\n'

[[ "$BENCHMARK_BOUNDARY_CHECK" == paper ]]
parse_args --benchmark-boundary-check released --benchmark-target-selection reference
[[ "$BENCHMARK_BOUNDARY_CHECK" == released && "$BENCHMARK_TARGET_SELECTION" == reference ]]
parse_args --benchmark-boundary-check paper --benchmark-target-selection reference-best
if (parse_args --benchmark-boundary-check unknown) >/dev/null 2>&1; then
    printf 'FAIL: invalid boundary protocol was accepted\n' >&2
    exit 1
fi
printf 'PASS: corrected defaults and explicit legacy protocol options\n'

# The control launcher chains jobs sharing one installation/model cache.
parse_args --dependency afterok:123 --parsable
sbatch() { printf '%s\n' "$@"; }
submission_args="$(submit_to_slurm --benchmark-root /dataset)"
[[ "$submission_args" == *$'--dependency\nafterok:123\n'* ]]
[[ "$submission_args" == *$'--parsable\n'* ]]
if (parse_args --dependency invalid) >/dev/null 2>&1; then
    printf 'FAIL: invalid Slurm dependency was accepted\n' >&2
    exit 1
fi
printf 'PASS: dependency and parsable job IDs are forwarded to Slurm\n'

# Keep the real command builder; substitute only the external Python runtime.
benchmark_test_tmp="$(mktemp -d)"
trap 'rm -f -- "$benchmark_test_tmp/bin/python"; rmdir -- "$benchmark_test_tmp/bin" "$benchmark_test_tmp"' EXIT
mkdir -- "$benchmark_test_tmp/bin"
cat > "$benchmark_test_tmp/bin/python" <<'PYTHON'
#!/usr/bin/env bash
printf '%s\n' "$@"
PYTHON
chmod +x "$benchmark_test_tmp/bin/python"
configure_model_caches() { :; }
configure_python_imports() { :; }
parse_args --benchmark-workers 4
parse_args --env-dir "$benchmark_test_tmp" --benchmark-config 'configs/experiment with spaces.json'
benchmark_test_args="$(run_benchmark --check-only)"
[[ "$benchmark_test_args" == *$'--workers\n4\n'* ]]
[[ "$benchmark_test_args" == *$'--boundary-check\npaper\n'* ]]
[[ "$benchmark_test_args" == *$'--target-selection\nreference-best\n'* ]]
[[ "$benchmark_test_args" == *$'--config\nconfigs/experiment with spaces.json\n'* ]]
printf 'PASS: the benchmark launcher forwards the selected GPU worker count to Python\n'
