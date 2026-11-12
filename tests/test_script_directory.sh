#!/usr/bin/env bash
# Helpers and default paths must follow the project, even in an allocation
# submitted from another directory. Slurm spool copies still need a fallback.
set -Eeuo pipefail
test_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
runner="$test_root/run_amodal_slurm.sh"
test_tmp="$(mktemp -d)"
spool_dir="$test_tmp/spool job"
mkdir -- "$spool_dir"
cp -- "$runner" "$spool_dir/slurm_script"
cleanup() {
    rm -f -- "$spool_dir/slurm_script"
    rmdir -- "$spool_dir" "$test_tmp"
}
trap cleanup EXIT

probe='source "$1"; printf "%s\n" "$SCRIPT_DIR" "$INPUT_DIR" "$OUTPUT_DIR" "$CACHE_DIR"'
expect_project() {
    local project="$1" actual expected
    shift
    expected="$(printf '%s\n' "$project" "$project/input" "$project/output" "$project/.amodal-cache")"
    actual="$(env -u AMODAL_INPUT_DIR -u AMODAL_OUTPUT_DIR -u AMODAL_CACHE_DIR "$@")"
    if [[ "$actual" != "$expected" ]]; then
        printf 'FAIL: project paths should be:\n%s\nbut resolved to:\n%s\n' "$expected" "$actual" >&2
        exit 1
    fi
}

expect_project "$test_root" env -u AMODAL_SCRIPT_DIR SLURM_JOB_ID=123 \
    SLURM_SUBMIT_DIR="$test_tmp" bash -c "$probe" -- "$runner"
printf 'PASS: an existing allocation does not redirect helpers to its submission directory\n'

expect_project "$test_root" env -u AMODAL_SCRIPT_DIR -u SLURM_JOB_ID \
    SLURM_SUBMIT_DIR="$test_tmp" bash -c "$probe" -- "$runner"
printf 'PASS: direct execution ignores an unrelated submission directory\n'

expect_project "$test_root" env -u AMODAL_SCRIPT_DIR SLURM_JOB_ID=123 \
    SLURM_SUBMIT_DIR="$test_root" bash -c "$probe" -- "$spool_dir/slurm_script"
printf 'PASS: a direct sbatch spool copy resolves the submission project\n'

expect_project "$test_tmp" env AMODAL_SCRIPT_DIR="$test_tmp" SLURM_JOB_ID=123 \
    SLURM_SUBMIT_DIR="$test_root" bash -c "$probe" -- "$runner"
printf 'PASS: an explicit project override takes precedence\n'
