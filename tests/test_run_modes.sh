#!/usr/bin/env bash
# Exercise main's dispatch without submitting jobs or initializing a GPU.
set -Eeuo pipefail
runner_test_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"
source "$(dirname -- "$runner_test_path")/../run_amodal_slurm.sh"

if [[ "${1:-}" == --probe ]]; then
    shift
    submit_to_slurm() { printf 'SUBMITTED\n'; }
    # Stop at the first operation after main chooses direct execution. The
    # dispatch remains real; only infrastructure boundaries are substituted.
    load_optional_modules() { printf 'DIRECT\n'; exit 0; }
    main "$@"
    exit 0
fi

expect_mode() {
    local expected="$1" actual
    shift
    actual="$("$@")"
    [[ "$actual" == "$expected" ]] || {
        printf 'FAIL: expected %s, got %s\n' "$expected" "$actual" >&2
        exit 1
    }
}

expect_mode SUBMITTED env -u SLURM_JOB_ID bash "$runner_test_path" --probe
expect_mode DIRECT env SLURM_JOB_ID=123 bash "$runner_test_path" --probe
expect_mode DIRECT env -u SLURM_JOB_ID bash "$runner_test_path" --probe --run-in-allocation
expect_mode DIRECT env -u SLURM_JOB_ID bash "$runner_test_path" --probe --local
expect_mode DIRECT env -u SLURM_JOB_ID bash "$runner_test_path" --probe --local \
    --benchmark-root /dataset --benchmark-limit 2 --partition unused
printf 'PASS: login submission, existing allocation, --local, and compatibility alias dispatch\n'
