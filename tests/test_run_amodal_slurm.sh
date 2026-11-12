#!/usr/bin/env bash
# Local regression checks; no Conda installation, Slurm job, or GPU required.
set -Eeuo pipefail
test_script="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"
# shellcheck source=../run_amodal_slurm.sh
source "$(dirname -- "$test_script")/../run_amodal_slurm.sh"

# The libblas MKL hook reads this variable without a default, exactly as in
# the supplied log. Intentionally keep that behavior to reproduce the crash.
mkl_deactivate_hook() {
    # shellcheck disable=SC2154
    if [ "${CONDA_MKL_INTERFACE_LAYER_BACKUP}" = "" ]; then
        unset MKL_INTERFACE_LAYER
    else
        export MKL_INTERFACE_LAYER=${CONDA_MKL_INTERFACE_LAYER_BACKUP}
    fi
    unset CONDA_MKL_INTERFACE_LAYER_BACKUP
}

if [[ "${1:-}" == --unsafe-conda-child ]]; then
    unset CONDA_MKL_INTERFACE_LAYER_BACKUP
    mkl_deactivate_hook
    exit 99
fi
if [[ "${1:-}" == --failed-conda-child ]]; then
    conda() { return 23; }
    run_conda install
    exit 99
fi
if [[ "${1:-}" == --print-requirements ]]; then
    printf '%s\n' "${RUNTIME_REQUIREMENTS[@]}" \
        'torch==1.13.1' 'torchvision==0.14.1' 'torchaudio==0.13.1' \
        'setuptools==69.5.1' 'wheel==0.40.0'
    exit 0
fi
if [[ "${1:-}" == --print-pythonpath ]]; then
    REPO_DIR="$2"
    configure_python_imports
    printf '%s\n' "$PYTHONPATH"
    exit 0
fi
if [[ "${1:-}" == --launch-upstream-child ]]; then
    REPO_DIR="$2"
    ENV_DIR="$3"
    SD_MODEL="$4"
    shift 4
    cd -- "$REPO_DIR"
    launch_upstream "$@"
    exit 0
fi
if [[ "${1:-}" == --sd-config-child ]]; then
    ENV_DIR="$2"
    SD_MODEL="$3"
    CACHE_DIR="$4"
    preflight_sd_model
    exit 0
fi
if [[ "${1:-}" == --failed-input-child ]]; then
    # Invoked indirectly by the sourced run_pipeline function.
    # shellcheck disable=SC2329
    prepare_inputs() { die "Expected input preparation failure"; }
    run_pipeline
    exit 99
fi

fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/amodal-runner-tests.XXXXXX")"
cleanup() {
    case "$test_dir" in
        "${TMPDIR:-/tmp}/amodal-runner-tests."*)
            find "$test_dir" -type f -delete
            find "$test_dir" -depth -type d -empty -delete ;;
    esac
}
trap cleanup EXIT
export TEST_CALL_LOG="$test_dir/calls.txt"
ENV_DIR="$test_dir/env"
REPO_DIR="$test_dir/repo"
mkdir -p "$ENV_DIR/bin" "$REPO_DIR"
CUDA_RUNTIME=11.7

gcc() { printf '%s\n' "$MOCK_GCC"; }
g++() { printf '%s\n' "$MOCK_GCC"; }
gcc-10() { [[ -n "${MOCK_GCC10:-}" ]] && printf '%s\n' "$MOCK_GCC10"; }
g++-10() { gcc-10 "$@"; }
for version in 5.5.0 11.5.1 11.6.0 12.2.0; do
    MOCK_GCC="$version"
    if gcc_compatible gcc; then fail "accepted unsupported GCC $version"; fi
done
for version in 6.0.0 9.5.0 10.4.0 11.4.0 11.5.0; do
    MOCK_GCC="$version"
    gcc_compatible gcc || fail "rejected supported GCC $version"
done
MOCK_GCC=12.2.0 MOCK_GCC10=10.4.0
unset CC CXX
select_host_compiler
[[ "$CC" == gcc-10 && "$CXX" == g++-10 && "$CUDAHOSTCXX" == g++-10 ]] \
    || fail "did not select the matching fallback compiler pair"
printf 'PASS: compiler compatibility boundaries and CC/CXX selection\n'

make_toolkit() {
    local prefix="$1" version="$2" header library
    mkdir -p "$prefix/bin" "$prefix/include/thrust" "$prefix/lib"
    printf '#!/usr/bin/env bash\nprintf "release %s, V%s.0\\n"\n' "$version" "$version" > "$prefix/bin/nvcc"
    chmod +x "$prefix/bin/nvcc"
    for header in "${CUDA_BUILD_HEADERS[@]}"; do
        mkdir -p "$(dirname -- "$prefix/include/$header")"
        touch "$prefix/include/$header"
    done
    for library in "${CUDA_BUILD_LIBRARIES[@]}"; do
        touch "$prefix/lib/$library"
    done
}
make_toolkit "$test_dir/system-cuda" 12.1
if cuda_toolkit_usable "$test_dir/system-cuda"; then fail 'accepted CUDA 12 with runtime 11.7'; fi
make_toolkit "$ENV_DIR" 11.7
cuda_toolkit_usable "$ENV_DIR" || fail 'rejected a complete matching CUDA toolkit'
mv "$ENV_DIR/include/thrust/complex.h" "$test_dir/thrust-backup.h"
if cuda_toolkit_usable "$ENV_DIR"; then fail 'accepted missing Thrust headers'; fi
mv "$test_dir/thrust-backup.h" "$ENV_DIR/include/thrust/complex.h"
for header in cuda.h cuda_runtime_api.h cusparse.h cublas_v2.h cublasLt.h cusolverDn.h cusolver_common.h; do
    mv "$ENV_DIR/include/$header" "$test_dir/header-backup.h"
    if cuda_toolkit_usable "$ENV_DIR"; then fail "accepted a toolkit without $header"; fi
    mv "$test_dir/header-backup.h" "$ENV_DIR/include/$header"
done
for library in libcusparse.so libcublas.so libcublasLt.so libcusolver.so; do
    mv "$ENV_DIR/lib/$library" "$test_dir/library-backup.so"
    if cuda_toolkit_usable "$ENV_DIR"; then fail "accepted a toolkit without $library"; fi
    mv "$test_dir/library-backup.so" "$ENV_DIR/lib/$library"
done
printf 'PASS: missing CUDA library development headers and linker libraries are rejected\n'

cat > "$ENV_DIR/bin/python" <<'MOCK'
#!/usr/bin/env bash
printf 'python %s\n' "$*" >> "$TEST_CALL_LOG"
if [[ " $* " == *' - '* ]]; then while IFS= read -r line; do :; done; fi
MOCK
chmod +x "$ENV_DIR/bin/python"
mock_python="$test_dir/mock-python"
cp "$ENV_DIR/bin/python" "$mock_python"
conda() {
    [[ "$-" != *u* ]] || fail 'Conda hooks were called with nounset enabled'
    unset CONDA_MKL_INTERFACE_LAYER_BACKUP
    mkl_deactivate_hook
    printf 'conda %s\n' "$*" >> "$TEST_CALL_LOG"
    case "${1:-}" in
        activate) export CONDA_PREFIX="$2" ;;
        create)
            mkdir -p "$ENV_DIR/bin"
            cp "$mock_python" "$ENV_DIR/bin/python" ;;
        fail) return 23 ;;
    esac
    if [[ "$*" == *gcc_linux-64=10* ]]; then
        local compiler
        for compiler in gcc g++; do
            printf '#!/usr/bin/env bash\nprintf "10.4.0\\n"\n' > "$ENV_DIR/bin/$(uname -m)-conda-linux-gnu-$compiler"
            chmod +x "$ENV_DIR/bin/$(uname -m)-conda-linux-gnu-$compiler"
        done
    elif [[ "$*" == *cuda-nvcc=* ]]; then
        make_toolkit "$ENV_DIR" "$CUDA_RUNTIME"
    fi
    return 0
}

if bash "$test_script" --unsafe-conda-child > "$test_dir/unsafe-conda.log" 2>&1; then
    fail 'the original MKL hook crash was not reproduced'
else
    status=$?
    [[ "$status" == 1 ]] || fail "unexpected original hook failure status: $status"
    grep -q 'CONDA_MKL_INTERFACE_LAYER_BACKUP: unbound variable' "$test_dir/unsafe-conda.log" \
        || fail 'did not reproduce the logged unset-variable error'
fi
run_conda activate "$ENV_DIR"
[[ "$-" == *u* ]] || fail 'nounset was not restored after successful activation'
[[ "$CONDA_PREFIX" == "$ENV_DIR" ]] || fail 'activation changes were lost in a subshell'
if run_conda fail; then fail 'failed Conda call appeared successful'; else
    status=$?
    [[ "$status" == 23 && "$-" == *u* ]] || fail 'failure status or nounset was lost'
fi
set +u
run_conda activate "$ENV_DIR"
[[ "$-" != *u* ]] || fail 'wrapper enabled nounset for a caller that disabled it'
set -u
if bash "$test_script" --failed-conda-child > "$test_dir/failed-conda.log" 2>&1; then
    fail 'runner continued after a failed Conda command'
else
    status=$?
    [[ "$status" == 23 ]] || fail "Conda error propagation returned $status instead of 23"
fi
printf 'PASS: original MKL hook crash reproduced; safe calls preserve activation, shell flags, and errors\n'

MOCK_GCC10=""
unset CC CXX
export CUDA_HOME="$test_dir/system-cuda"
configure_cuda_toolchain
[[ "$CUDA_HOME" == "$ENV_DIR" && "$CXX" == "$ENV_DIR/bin/$(uname -m)-conda-linux-gnu-g++" ]] \
    || fail 'did not use private matching toolkit and GCC 10'
grep -q 'gcc_linux-64=10.*gxx_linux-64=10' "$TEST_CALL_LOG" || fail 'did not install both compiler packages'
grep -q 'gxx_linux-64=10.*sysroot_linux-64=2.17' "$TEST_CALL_LOG" || fail 'compiler could target a newer glibc than the cluster'
[[ "$CC" == "$ENV_DIR/bin/$(uname -m)-conda-linux-gnu-gcc" ]] || fail 'nvcc would use the wrong host compiler'
printf 'PASS: CUDA mismatch recovery and private compiler installation\n'

# Reproduce a cached toolkit that has nvcc but incomplete headers.
mv "$ENV_DIR/include/thrust/complex.h" "$test_dir/thrust-backup.h"
configure_cuda_toolchain
grep -q 'cuda-nvcc=11.7.*cuda-cudart-dev=11.7.*cuda-driver-dev=11.7.*cuda-cccl=11.7' "$TEST_CALL_LOG" \
    || fail 'did not repair the complete toolkit together'
printf 'PASS: incomplete CUDA toolkit repair\n'

# Reproduce the new log: runtime/Thrust exist, but cuSPARSE's dev header does not.
mv "$ENV_DIR/include/cusparse.h" "$test_dir/cusparse-backup.h"
configure_cuda_toolchain
cuda_toolkit_usable "$ENV_DIR" || fail 'missing cuSPARSE header was not repaired'
grep -q 'libcusparse-dev=11.7.4.91.*libcublas-dev=11.10.3.66.*libcusolver-dev=11.4.0.1' "$TEST_CALL_LOG" \
    || fail 'CUDA library development packages are missing or unpinned'
printf 'PASS: the logged cusparse.h failure triggers installation of all PyTorch CUDA development headers\n'

load_conda() { :; }
unzip() { :; }
touch "$ENV_DIR/.amodal_torch_11.7" "$ENV_DIR/.amodal_dependencies_${REPO_COMMIT}"
export PYTHONPATH=/unrelated/python PYTHONHOME=/unrelated/home
create_environment
[[ "${PYTHONNOUSERSITE:-}" == 1 && -z "${PYTHONPATH:-}" && -z "${PYTHONHOME:-}" ]] \
    || fail 'inherited Python paths or user-site packages are still enabled'
marker="$ENV_DIR/.amodal_dependencies_${REPO_COMMIT}_${CUDA_RUNTIME}_${DEPENDENCY_REVISION}"
[[ -f "$marker" ]] || fail 'old cache marker prevented dependency repair'
[[ -f "$ENV_DIR/.amodal_torch_11.7_${TORCH_SETUP_REVISION}" ]] || fail 'PyTorch repair was not recorded'
grep -q '^numpy==1.23.5$' "$ENV_DIR/.amodal-constraints-${DEPENDENCY_REVISION}.txt" || fail 'NumPy constraint missing'
grep -q '^albumentations==0.5.2$' "$ENV_DIR/.amodal-constraints-${DEPENDENCY_REVISION}.txt" || fail 'LaMa compatibility pin missing'
grep -q 'pip install .*--no-build-isolation -c .*GroundingDINO' "$TEST_CALL_LOG" || fail 'GroundingDINO install is unconstrained'
grep -q 'pip install .*--no-deps --force-reinstall opencv-python-headless==4.8.1.78' "$TEST_CALL_LOG" \
    || fail 'headless OpenCV was not restored'
printf 'PASS: cache invalidation, shared constraints, and Python isolation\n'

ENV_DIR="$test_dir/fresh-env"
create_environment
[[ -x "$ENV_DIR/bin/python" && -f "$ENV_DIR/.amodal_torch_11.7_${TORCH_SETUP_REVISION}" \
    && -f "$ENV_DIR/.amodal_dependencies_${REPO_COMMIT}_${CUDA_RUNTIME}_${DEPENDENCY_REVISION}" \
    && "$-" == *u* ]] || fail 'fresh setup failed or left strict shell checks disabled'
printf 'PASS: fresh environment creation survives Conda activation and install hooks\n'

ENV_DIR="$test_dir/prepared-v4-env"
mkdir -p "$ENV_DIR/bin"
cp "$mock_python" "$ENV_DIR/bin/python"
touch "$ENV_DIR/.amodal_torch_11.7_v4" "$ENV_DIR/.amodal_dependencies_${REPO_COMMIT}_${CUDA_RUNTIME}_v4"
migration_start="$(wc -l < "$TEST_CALL_LOG")"
create_environment
[[ -f "$ENV_DIR/.amodal_dependencies_${REPO_COMMIT}_${CUDA_RUNTIME}_${DEPENDENCY_REVISION}" ]] \
    || fail 'prepared v4 environment was not migrated'
grep -q '^kornia==0.5.0$' "$ENV_DIR/.amodal-constraints-${DEPENDENCY_REVISION}.txt" \
    || fail 'LaMa Kornia API pin missing from the migration'
migration_calls="$(tail -n "+$((migration_start + 1))" "$TEST_CALL_LOG")"
if grep -Eq 'conda install|--no-build-isolation|pip install .* -e ' <<< "$migration_calls"; then
    fail 'Python-only migration reinstalled PyTorch or rebuilt native extensions'
fi
grep -q 'pip install .*kornia==0.5.0' <<< "$migration_calls" || fail 'migration did not repair Kornia'
printf 'PASS: prepared v4 environment repairs Kornia without rebuilding PyTorch or GroundingDINO\n'

while IFS= read -r call; do
    case "$call" in
        'conda install '*|'conda create '*)
            [[ "$call" == *'numpy=1.23.5'* && "$call" == *'mkl<2024.1'* \
                && "$call" == *'setuptools=69.5.1'* && "$call" == *'wheel=0.40.0'* \
                && "$call" == *'packaging=23.2'* && "$call" == *'python=3.10'* \
                && "$call" == *'cuda-version=11.7'* \
                && "$call" == *'libcurand=10.2.10.91'* && "$call" == *'libcufile=1.3.1.18'* ]] \
                || fail "unconstrained Conda transaction: $call" ;;
    esac
done < "$TEST_CALL_LOG"
grep -q 'conda install .*pytorch=1.13.1.*pytorch-cuda=11.7' "$TEST_CALL_LOG" \
    || fail 'legacy PyTorch marker prevented native dependency repair'
printf 'PASS: every Conda package transaction constrains MKL, NumPy, Python, CUDA, and build tooling\n'

calls_before="$(wc -l < "$TEST_CALL_LOG")"
RUN_ONLY=1
create_environment
[[ "$(wc -l < "$TEST_CALL_LOG")" -eq $((calls_before + 3)) ]] \
    || fail 'run-only attempted installation instead of activation and version checks'
ENV_DIR="$test_dir/missing-env"
if (create_environment) >/dev/null 2>&1; then fail 'run-only accepted a missing environment'; fi
printf 'PASS: run-only never creates or installs a missing environment\n'

if bash "$test_script" --failed-input-child > "$test_dir/input-failure.log" 2>&1; then
    fail 'input preparation failure was swallowed'
else
    status=$?
    [[ "$status" == 1 ]] || fail "wrong input failure status: $status"
    grep -q 'Expected input preparation failure' "$test_dir/input-failure.log" || fail 'input error was lost'
fi
printf 'PASS: input preparation failures propagate\n'
