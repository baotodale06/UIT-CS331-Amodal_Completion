#!/usr/bin/env bash
set -Eeuo pipefail
test_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$test_root/run_amodal_slurm.sh"
parse_args --benchmark-root /dataset --benchmark-pipeline-profile released-sam --output-dir /results
[[ "$BENCHMARK_TARGET_SELECTION" == label && "$BENCHMARK_BOUNDARY_CHECK" == released ]]
[[ "$REFERENCE_DIR" == /results/visible_references && "$BENCHMARK_GENERATE_REFERENCES" == 0 ]]
if (parse_args --benchmark-generate-references) >/dev/null 2>&1; then
    printf 'FAIL: LISA preparation was accepted by the released-sam profile\n' >&2
    exit 1
fi
printf 'PASS: native profile keeps LISA preparation disabled\n'

pdmc_test_tmp="$(mktemp -d)"
trap 'rm -f -- "$pdmc_test_tmp/bin/python"; rmdir -- "$pdmc_test_tmp/bin" "$pdmc_test_tmp"' EXIT
mkdir -- "$pdmc_test_tmp/bin"
cat > "$pdmc_test_tmp/bin/python" <<'PYTHON'
#!/usr/bin/env bash
printf '%s\n' "$@"
PYTHON
chmod +x "$pdmc_test_tmp/bin/python"
configure_model_caches() { :; }
configure_python_imports() { :; }
ENV_DIR="$pdmc_test_tmp"
ENV_DIR_EXPLICIT=1
args="$(run_benchmark --check-only)"
[[ "$args" == *$'--pipeline-profile\nreleased-sam\n'* ]]
[[ "$args" == *$'--target-selection\nlabel\n'* ]]
[[ "$args" == *$'--boundary-check\nreleased\n'* ]]
printf 'PASS: worker and preflight command forwards the native evaluation profile\n'

parse_args --benchmark-background-sizing auto --benchmark-reuse-results-from /original-results
args="$(run_benchmark --check-only)"
[[ "$args" == *$'--background-sizing\nauto\n'* ]]
[[ "$args" == *$'--reuse-results-from\n/original-results\n'* ]]
if (parse_args --benchmark-background-sizing invalid) >/dev/null 2>&1; then
    printf 'FAIL: invalid background sizing was accepted\n' >&2
    exit 1
fi
printf 'PASS: explicit backdrop repair and result reuse reach the evaluator\n'

parse_args --benchmark-target-selection label-semantic
args="$(run_benchmark --check-only)"
[[ "$args" == *$'--target-selection\nlabel-semantic\n'* ]]
printf 'PASS: corrected semantic matching is forwarded without enabling LISA\n'

[[ "$BENCHMARK_DETECTOR_VOCABULARY" == released ]]
parse_args --benchmark-detector-vocabulary annotated
args="$(run_benchmark --check-only)"
[[ "$args" == *$'--detector-vocabulary\nannotated\n'* ]]
if (parse_args --benchmark-detector-vocabulary invalid) >/dev/null 2>&1; then
    printf 'FAIL: invalid detector vocabulary was accepted\n' >&2
    exit 1
fi
if (parse_args --benchmark-target-selection label) >/dev/null 2>&1; then
    printf 'FAIL: annotation vocabulary accepted legacy matching\n' >&2
    exit 1
fi
parse_args --benchmark-detector-vocabulary released
printf 'PASS: opt-in annotation vocabulary reaches Python and requires semantic matching\n'

# Run the real shared main flow while replacing external provisioning only.
load_optional_modules() { :; }
check_gpu() { :; }
require_download_tool() { :; }
require_command() { :; }
absolute_dir() { printf '%s\n' "$1"; }
clone_repository() { :; }
create_environment() { :; }
preflight_python() { :; }
prepare_lisa_runtime() { printf 'FAIL: LISA runtime invoked\n' >&2; exit 1; }
run_reference_preparation() { printf 'FAIL: LISA generation invoked\n' >&2; exit 1; }
preflight_sd_model() { :; }
download_checkpoints() { :; }
prepare_benchmark_metrics() { :; }
run_benchmark() { printf 'native evaluator called %s\n' "$*"; }
REPO_DIR="$test_root"
ENV_DIR="$test_root"
BENCHMARK_GENERATE_REFERENCES=0
main --local --benchmark-root /dataset --benchmark-pipeline-profile released-sam --output-dir /results \
    --repo-dir "$test_root" --env-dir "$test_root"
printf 'PASS: complete native launch flow never enters LISA setup or generation\n'

# The standalone wrapper must preserve the profile when submitting the shared
# script, because the wrapper itself is no longer running on the compute node.
submission="$(
    unset SLURM_JOB_ID
    sbatch() { printf '%s\n' "$@"; }
    export -f sbatch
    bash "$test_root/run_pdmc_open_world.sh" --benchmark-limit 2 --benchmark-workers 2
)"
[[ "$submission" == *$'--benchmark-pipeline-profile\nreleased-sam'* ]]
[[ "$submission" == *$'--benchmark-workers\n2\n'* ]]
[[ "$submission" == *$'--benchmark-target-selection\nlabel-semantic\n'* ]]
[[ "$submission" == *$'--benchmark-background-sizing\nauto\n'* ]]
[[ "$submission" == *"$test_root/output/pdmc-open-world-sam-semantic-v2"* ]]
[[ "$submission" == *"$test_root/run_amodal_slurm.sh"* ]]
printf 'PASS: standalone wrapper preserves the native profile across Slurm submission\n'

expanded_submission="$(
    unset SLURM_JOB_ID
    sbatch() { printf '%s\n' "$@"; }
    export -f sbatch
    bash "$test_root/run_pdmc_open_world.sh" --benchmark-detector-vocabulary annotated \
        --benchmark-workers 4 --output-dir /expanded-results
)"
[[ "$expanded_submission" == *$'--benchmark-detector-vocabulary\nannotated\n'* ]]
[[ "$expanded_submission" == *$'--benchmark-workers\n4\n'* ]]
printf 'PASS: expanded vocabulary and four workers survive dedicated Slurm submission\n'

legacy_submission="$(
    unset SLURM_JOB_ID
    sbatch() { printf '%s\n' "$@"; }
    export -f sbatch
    bash "$test_root/run_pdmc_open_world.sh" --benchmark-target-selection label \
        --benchmark-background-sizing released
)"
[[ "${legacy_submission##*--benchmark-target-selection$'\n'}" == label$'\n'* ]]
[[ "${legacy_submission##*--benchmark-background-sizing$'\n'}" == released$'\n'* ]]
printf 'PASS: explicit legacy settings override dedicated launcher defaults\n'

if invalid_profile="$(bash -c 'source "$1"; parse_args --benchmark-root /dataset \
        --benchmark-target-selection label-semantic' _ "$test_root/run_amodal_slurm.sh" 2>&1)"; then
    printf 'FAIL: semantic label matching was accepted outside the SAM profile\n' >&2
    exit 1
fi
[[ "$invalid_profile" == *'requires released-sam'* ]]
printf 'PASS: semantic label matching requires the SAM profile\n'
