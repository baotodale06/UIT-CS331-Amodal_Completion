#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
set -Eeuo pipefail

# Reproducible Slurm runner for:
#   Amodal Completion via Progressive Mixed Context Diffusion (CVPR 2024)
#
# The script submits itself when launched from a login node. Once Slurm starts
# the job, it clones the pinned upstream repository, creates an isolated
# environment, downloads the four non-Hugging Face checkpoints, and runs the
# official main.py with absolute paths.

SCRIPT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"
readonly SCRIPT_PATH
# An interactive allocation can inherit SLURM_SUBMIT_DIR from another folder.
# Prefer the original script's directory. Slurm's spool directory lacks the
# original runner, so a direct sbatch job falls back to its submission directory.
# Self-submitted jobs preserve the project explicitly via AMODAL_SCRIPT_DIR.
SCRIPT_DIR="${AMODAL_SCRIPT_DIR:-$(dirname -- "$SCRIPT_PATH")}"
if [[ -z "${AMODAL_SCRIPT_DIR:-}" && -n "${SLURM_JOB_ID:-}" &&
      ! -f "$SCRIPT_DIR/run_amodal_slurm.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"
fi
readonly SCRIPT_DIR
readonly REPO_URL="https://github.com/k8xu/amodal.git"
readonly REPO_COMMIT="bfbfcdec581484586774d9ca53f5c603c97c725b"
readonly DEPENDENCY_REVISION="v5"
# Python-only compatibility repairs do not require reinstalling PyTorch.
readonly TORCH_SETUP_REVISION="v4"
readonly -a CUDA_BUILD_HEADERS=(
    'cuda.h' 'cuda_runtime.h' 'cuda_runtime_api.h' 'thrust/complex.h'
    'cusparse.h' 'cublas_v2.h' 'cublasLt.h' 'cusolverDn.h' 'cusolver_common.h'
)
readonly -a CUDA_BUILD_LIBRARIES=(
    'libcudart.so' 'libcusparse.so' 'libcublas.so' 'libcublasLt.so' 'libcusolver.so'
)
# These constraints also belong on Conda transactions. Pip constraints do not
# stop Conda from selecting MKL 2024.1+, NumPy 2, or incompatible build tooling.
readonly -a CONDA_RUNTIME_REQUIREMENTS=(
    'python=3.10' 'numpy=1.23.5' 'mkl<2024.1'
    'setuptools=69.5.1' 'wheel=0.40.0' 'packaging=23.2'
)
readonly -a RUNTIME_REQUIREMENTS=(
    'numpy==1.23.5' 'scipy==1.10.1' 'scikit-image==0.20.0'
    'scikit-learn==1.2.2' 'opencv-python==4.8.1.78'
    'opencv-python-headless==4.8.1.78' 'Pillow==9.5.0' 'tqdm==4.66.1'
    'PyYAML==6.0.1' 'omegaconf==2.3.0' 'easydict==1.9.0'
    'matplotlib==3.7.5' 'pandas==2.0.3' 'joblib==1.3.2'
    'albumentations==0.5.2' 'imgaug==0.4.0' 'hydra-core==1.3.2'
    'pytorch-lightning==1.9.5' 'torchmetrics==0.11.4' 'tabulate==0.9.0'
    'kornia==0.5.0' 'webdataset==0.2.86' 'packaging==23.2'
    'gdown==4.7.3' 'pycocotools==2.0.7' 'cvbase==0.5.5'
    'accelerate==0.24.1' 'transformers==4.30.2'
    'huggingface-hub==0.19.4' 'safetensors==0.4.1' 'tokenizers==0.13.3'
    'protobuf<4' 'urllib3==2.0.7' 'importlib_metadata==6.8.0'
    'addict==2.4.0' 'yapf==0.32.0' 'timm==0.6.12' 'supervision==0.6.0'
)

INPUT_DIR="${AMODAL_INPUT_DIR:-$SCRIPT_DIR/input}"
OUTPUT_DIR="${AMODAL_OUTPUT_DIR:-$SCRIPT_DIR/output}"
CACHE_DIR="${AMODAL_CACHE_DIR:-$SCRIPT_DIR/.amodal-cache}"
REPO_DIR="${AMODAL_REPO_DIR:-$CACHE_DIR/amodal}"
ENV_DIR="${AMODAL_ENV_DIR:-$CACHE_DIR/conda-env}"
REPO_DIR_EXPLICIT=0
ENV_DIR_EXPLICIT=0
[[ -n "${AMODAL_REPO_DIR:-}" ]] && REPO_DIR_EXPLICIT=1
[[ -n "${AMODAL_ENV_DIR:-}" ]] && ENV_DIR_EXPLICIT=1
CLASSES_FILE=""
CLASSES=""
FILENAMES_FILE=""
MC_TIMESTEP="${AMODAL_MC_TIMESTEP:-35}"
MAX_ITER="${AMODAL_MAX_ITER:-5}"
SD_MODEL="${AMODAL_SD_MODEL:-sd2-community/stable-diffusion-2-inpainting}"
PARTITION="${AMODAL_PARTITION:-}"
GRES="${AMODAL_GRES:-gpu:1}"
CPUS="${AMODAL_CPUS_PER_TASK:-8}"
MEMORY="${AMODAL_MEM:-48G}"
TIME_LIMIT="${AMODAL_TIME:-24:00:00}"
JOB_NAME="${AMODAL_JOB_NAME:-amodal-completion}"
JOB_DEPENDENCY=""
SBATCH_PARSABLE=0
ACCOUNT="${AMODAL_ACCOUNT:-}"
QOS="${AMODAL_QOS:-}"
CUDA_RUNTIME="${AMODAL_CUDA_RUNTIME:-11.7}"
BUILD_JOBS="${AMODAL_BUILD_JOBS:-4}"
TORCH_ARCH="${AMODAL_TORCH_CUDA_ARCH_LIST:-}"
RUN_IN_ALLOCATION=0
SETUP_ONLY=0
RUN_ONLY=0
BENCHMARK_ROOT=""
BENCHMARK_CONFIG=""
BENCHMARK_PIPELINE_PROFILE=benchmark
BENCHMARK_BACKGROUND_SIZING=""
BENCHMARK_DETECTOR_VOCABULARY=released
BENCHMARK_REUSE_RESULTS_FROM=""
REFERENCE_DIR=""
BENCHMARK_SOURCES="VG,COCOA,free_images,laion"
BENCHMARK_LIMIT=""
BENCHMARK_SEED=0
BENCHMARK_WORKERS=auto
BENCHMARK_INFERENCE_ONLY=0
BENCHMARK_SCORE_ONLY=0
BENCHMARK_ALLOW_INCOMPLETE=0
BENCHMARK_TARGET_SELECTION=auto
BENCHMARK_BOUNDARY_CHECK=paper
BENCHMARK_BOUNDARY_CHECK_EXPLICIT=0
BENCHMARK_METRIC_ALIGNMENT=released
BENCHMARK_GENERATE_REFERENCES=0
BENCHMARK_REFERENCE_ONLY=0
LISA_MODEL="xinlai/LISA-13B-llama2-v1"
LISA_PRECISION=fp16
LISA_DEVICE_MAP=auto
LISA_PROMPT_TEMPLATE='{label}'
LISA_REPO_DIR=""
readonly LISA_REPO_COMMIT="3cb2d4301f1af4691bd4f3938335ef06e76f155a"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

log() {
    printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

warn() {
    printf 'WARNING: %s\n' "$*" >&2
}

on_error() {
    local status=$?
    printf 'ERROR: command failed (exit %s) at %s:%s\n' "$status" "${BASH_SOURCE[1]:-$0}" "${BASH_LINENO[0]:-0}" >&2
    exit "$status"
}
trap on_error ERR

usage() {
    cat <<'EOF'
Usage:
  bash run_amodal_slurm.sh [options]

The command submits a one-GPU Slurm job when run outside an allocation. If it
is already running inside Slurm, it executes directly in that allocation.
Use --local to run on the current compute node even without SLURM_JOB_ID.

Execution:
  --local              Execute here without submitting another Slurm job

Input and output:
  --input-dir DIR       Images (default: ./input)
  --output-dir DIR      Results (default: ./output)
  --filenames FILE      One image filename per line (default: all images)
  --classes-file FILE   Classes to detect, one per line (default: repo classes.txt)
  --benchmark-detector-vocabulary MODE
                        released (default) or annotated (add missing image annotation categories)
  --classes LIST        Comma-separated classes, e.g. "person,bicycle"

Pipeline:
  --mc-timestep N       Mixed Context timestep (default: 35, paper code default)
  --max-iter N          Maximum progressive iterations (default: 5)
  --sd-model ID_OR_DIR  SD2 inpainting repo or local Diffusers directory
                       (default: sd2-community/stable-diffusion-2-inpainting)
  --setup-only          Prepare the repo, environment, and checkpoints only
  --run-only            Do not install or download anything; require a complete setup

Open-world benchmark (see OPEN_WORLD_REPRODUCTION.md):
  --benchmark-root DIR  Evaluate the verified 2379-image/2565-target benchmark
  --benchmark-config FILE  Record a JSON experiment contract (use amodal plan to resolve all runner flags)
  --benchmark-pipeline-profile benchmark|released-sam  Default: benchmark; released-sam uses no LISA
  --benchmark-background-sizing released|auto  Auto enlarges undersized backdrops (released-sam default: released)
  --benchmark-reuse-results-from DIR  Reuse verified released-sam results in a new run with background sizing auto
  --reference-dir DIR   Existing visible references (default: DATASET/visible_references,
                       generated automatically with LISA when omitted)
  --benchmark-sources S Comma-separated sources (default: VG,COCOA,free_images,laion)
  --benchmark-limit N   Smoke-test first N targets; explicitly reports a subset
  --benchmark-seed N    Base seed (default: 0; stable per-target seeds)
  --benchmark-workers auto|N  PD-MC workers, one per visible GPU (default: auto)
  --benchmark-inference-only  Generate completions without reference metrics
  --benchmark-score-only      Score saved completions; skip diffusion inference
  --benchmark-allow-incomplete Keep missing images explicit in a partial report
  --benchmark-target-selection auto|label|label-semantic|reference|reference-best  Auto: best reference IoU
                       label-semantic requires released-sam; dedicated SAM launcher defaults to it
  --benchmark-boundary-check paper|released  Default: paper; repairs reversed edge coordinates
  --benchmark-metric-alignment released|original-frame  Default: released crop scoring
  --benchmark-generate-references Generate missing LISA references in --reference-dir
  --benchmark-reference-only     Generate references and stop before PD-MC
  --lisa-model ID_OR_DIR         Default: xinlai/LISA-13B-llama2-v1 (pinned revision)
  --lisa-precision fp16|bf16      Default: fp16, unquantized
  --lisa-device-map auto|single  Auto balances LISA across visible GPUs (default)
  --lisa-prompt-template TEXT    Default: '{label}', as passed by the authors' code
  --lisa-repo-dir DIR            Default: CACHE/LISA

Locations (persistent across jobs):
  --cache-dir DIR       Cache root (default: ./.amodal-cache)
  --repo-dir DIR        Upstream checkout (default: CACHE/amodal)
  --env-dir DIR         Conda environment prefix (default: CACHE/conda-env)

Slurm resources (used only when submitting from a login node):
  --partition NAME      Slurm partition (or AMODAL_PARTITION)
  --gres SPEC           GPU request (default: gpu:1)
  --cpus N              CPUs per task (default: 8)
  --mem SIZE            Memory request (default: 48G)
  --time LIMIT          Wall time (default: 24:00:00)
  --account NAME        Slurm account
  --qos NAME            Slurm QoS
  --job-name NAME       Slurm job name
  --dependency afterok:JOBID  Start after an earlier job succeeds (shared-cache controls)
  --parsable            Ask sbatch for a machine-readable job ID

Authentication:
  Set HF_TOKEN in the environment before submitting if the Hugging Face model
  asks for access. The token is inherited by the Slurm job and never written
  to the command line.

Examples:
  bash run_amodal_slurm.sh --input-dir ./input
  bash run_amodal_slurm.sh --input-dir /scratch/$USER/images --classes "person,bicycle"
  bash run_amodal_slurm.sh --setup-only --cache-dir /scratch/$USER/amodal-cache
EOF
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

require_download_tool() {
    command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1 \
        || die "A downloader is required; install curl or wget"
}

fetch_url() {
    local url="$1" destination="$2"
    if command -v curl >/dev/null 2>&1; then
        if ! curl -fL --retry 5 --retry-delay 5 -C - -o "$destination" "$url"; then
            curl -fL --retry 5 --retry-delay 5 -o "$destination" "$url"
        fi
    else
        if ! wget --continue --tries=5 --waitretry=5 -O "$destination" "$url"; then
            wget --tries=5 --waitretry=5 -O "$destination" "$url"
        fi
    fi
}

absolute_dir() {
    local path="$1"
    mkdir -p -- "$path"
    cd -- "$path" && pwd -P
}

absolute_file() {
    local path="$1"
    [[ -f "$path" ]] || die "File does not exist: $path"
    cd -- "$(dirname -- "$path")" && printf '%s/%s\n' "$(pwd -P)" "$(basename -- "$path")"
}

parse_args() {
    while (($#)); do
        case "$1" in
            --input-dir)
                [[ $# -ge 2 ]] || die "--input-dir requires a value"
                INPUT_DIR="$2"; shift 2 ;;
            --output-dir)
                [[ $# -ge 2 ]] || die "--output-dir requires a value"
                OUTPUT_DIR="$2"; shift 2 ;;
            --filenames)
                [[ $# -ge 2 ]] || die "--filenames requires a value"
                FILENAMES_FILE="$2"; shift 2 ;;
            --classes-file)
                [[ $# -ge 2 ]] || die "--classes-file requires a value"
                CLASSES_FILE="$2"; shift 2 ;;
            --classes)
                [[ $# -ge 2 ]] || die "--classes requires a value"
                CLASSES="$2"; shift 2 ;;
            --mc-timestep)
                [[ $# -ge 2 ]] || die "--mc-timestep requires a value"
                MC_TIMESTEP="$2"; shift 2 ;;
            --max-iter)
                [[ $# -ge 2 ]] || die "--max-iter requires a value"
                MAX_ITER="$2"; shift 2 ;;
            --sd-model)
                [[ $# -ge 2 && -n "${2:-}" ]] || die "--sd-model requires a repo ID or directory"
                SD_MODEL="$2"; shift 2 ;;
            --cache-dir)
                [[ $# -ge 2 ]] || die "--cache-dir requires a value"
                CACHE_DIR="$2"; shift 2 ;;
            --repo-dir)
                [[ $# -ge 2 ]] || die "--repo-dir requires a value"
                REPO_DIR="$2"; REPO_DIR_EXPLICIT=1; shift 2 ;;
            --env-dir)
                [[ $# -ge 2 ]] || die "--env-dir requires a value"
                ENV_DIR="$2"; ENV_DIR_EXPLICIT=1; shift 2 ;;
            --partition)
                [[ $# -ge 2 ]] || die "--partition requires a value"
                PARTITION="$2"; shift 2 ;;
            --gres)
                [[ $# -ge 2 ]] || die "--gres requires a value"
                GRES="$2"; shift 2 ;;
            --cpus)
                [[ $# -ge 2 ]] || die "--cpus requires a value"
                CPUS="$2"; shift 2 ;;
            --mem)
                [[ $# -ge 2 ]] || die "--mem requires a value"
                MEMORY="$2"; shift 2 ;;
            --time)
                [[ $# -ge 2 ]] || die "--time requires a value"
                TIME_LIMIT="$2"; shift 2 ;;
            --account)
                [[ $# -ge 2 ]] || die "--account requires a value"
                ACCOUNT="$2"; shift 2 ;;
            --qos)
                [[ $# -ge 2 ]] || die "--qos requires a value"
                QOS="$2"; shift 2 ;;
            --job-name)
                [[ $# -ge 2 ]] || die "--job-name requires a value"
                JOB_NAME="$2"; shift 2 ;;
            --dependency)
                [[ $# -ge 2 ]] || die "--dependency requires a value"
                JOB_DEPENDENCY="$2"; shift 2 ;;
            --parsable)
                SBATCH_PARSABLE=1; shift ;;
            --setup-only)
                SETUP_ONLY=1; shift ;;
            --run-only)
                RUN_ONLY=1; shift ;;
            --benchmark-root)
                [[ $# -ge 2 ]] || die "--benchmark-root requires a value"
                BENCHMARK_ROOT="$2"; shift 2 ;;
            --benchmark-config)
                [[ $# -ge 2 ]] || die "--benchmark-config requires a value"
                BENCHMARK_CONFIG="$2"; shift 2 ;;
            --benchmark-pipeline-profile)
                [[ $# -ge 2 ]] || die "--benchmark-pipeline-profile requires a value"
                BENCHMARK_PIPELINE_PROFILE="$2"; shift 2 ;;
            --benchmark-background-sizing)
                [[ $# -ge 2 ]] || die "--benchmark-background-sizing requires a value"
                BENCHMARK_BACKGROUND_SIZING="$2"; shift 2 ;;
            --benchmark-detector-vocabulary)
                [[ $# -ge 2 ]] || die "--benchmark-detector-vocabulary requires a value"
                BENCHMARK_DETECTOR_VOCABULARY="$2"; shift 2 ;;
            --benchmark-reuse-results-from)
                [[ $# -ge 2 ]] || die "--benchmark-reuse-results-from requires a value"
                BENCHMARK_REUSE_RESULTS_FROM="$2"; shift 2 ;;
            --reference-dir)
                [[ $# -ge 2 ]] || die "--reference-dir requires a value"
                REFERENCE_DIR="$2"; shift 2 ;;
            --benchmark-sources)
                [[ $# -ge 2 ]] || die "--benchmark-sources requires a value"
                BENCHMARK_SOURCES="$2"; shift 2 ;;
            --benchmark-limit)
                [[ $# -ge 2 ]] || die "--benchmark-limit requires a value"
                BENCHMARK_LIMIT="$2"; shift 2 ;;
            --benchmark-seed)
                [[ $# -ge 2 ]] || die "--benchmark-seed requires a value"
                BENCHMARK_SEED="$2"; shift 2 ;;
            --benchmark-workers)
                [[ $# -ge 2 ]] || die "--benchmark-workers requires a value"
                BENCHMARK_WORKERS="$2"; shift 2 ;;
            --benchmark-inference-only)
                BENCHMARK_INFERENCE_ONLY=1; shift ;;
            --benchmark-score-only)
                BENCHMARK_SCORE_ONLY=1; shift ;;
            --benchmark-allow-incomplete)
                BENCHMARK_ALLOW_INCOMPLETE=1; shift ;;
            --benchmark-target-selection)
                [[ $# -ge 2 ]] || die "--benchmark-target-selection requires a value"
                BENCHMARK_TARGET_SELECTION="$2"; shift 2 ;;
            --benchmark-boundary-check)
                [[ $# -ge 2 ]] || die "--benchmark-boundary-check requires a value"
                BENCHMARK_BOUNDARY_CHECK="$2"; BENCHMARK_BOUNDARY_CHECK_EXPLICIT=1; shift 2 ;;
            --benchmark-metric-alignment)
                [[ $# -ge 2 ]] || die "--benchmark-metric-alignment requires a value"
                BENCHMARK_METRIC_ALIGNMENT="$2"; shift 2 ;;
            --benchmark-generate-references)
                BENCHMARK_GENERATE_REFERENCES=1; shift ;;
            --benchmark-reference-only)
                BENCHMARK_REFERENCE_ONLY=1; BENCHMARK_GENERATE_REFERENCES=1; shift ;;
            --lisa-model)
                [[ $# -ge 2 ]] || die "--lisa-model requires a value"
                LISA_MODEL="$2"; shift 2 ;;
            --lisa-precision)
                [[ $# -ge 2 ]] || die "--lisa-precision requires a value"
                LISA_PRECISION="$2"; shift 2 ;;
            --lisa-device-map)
                [[ $# -ge 2 ]] || die "--lisa-device-map requires a value"
                LISA_DEVICE_MAP="$2"; shift 2 ;;
            --lisa-prompt-template)
                [[ $# -ge 2 ]] || die "--lisa-prompt-template requires a value"
                LISA_PROMPT_TEMPLATE="$2"; shift 2 ;;
            --lisa-repo-dir)
                [[ $# -ge 2 ]] || die "--lisa-repo-dir requires a value"
                LISA_REPO_DIR="$2"; shift 2 ;;
            --local|--run-in-allocation)
                RUN_IN_ALLOCATION=1; shift ;;
            -h|--help)
                usage; exit 0 ;;
            *)
                die "Unknown option: $1 (use --help for usage)" ;;
        esac
    done

    [[ "$MC_TIMESTEP" =~ ^[0-9]+$ ]] || die "--mc-timestep must be a non-negative integer"
    [[ "$MAX_ITER" =~ ^[1-9][0-9]*$ ]] || die "--max-iter must be a positive integer"
    [[ "$CPUS" =~ ^[1-9][0-9]*$ ]] || die "--cpus must be a positive integer"
    [[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || die "AMODAL_BUILD_JOBS must be a positive integer"
    [[ "$RUN_ONLY" -eq 0 || "$SETUP_ONLY" -eq 0 ]] || die "--run-only and --setup-only cannot be combined"
    [[ "$BENCHMARK_SEED" =~ ^[0-9]+$ ]] || die "--benchmark-seed must be non-negative"
    [[ "$BENCHMARK_WORKERS" == auto || "$BENCHMARK_WORKERS" =~ ^[1-9][0-9]*$ ]] \
        || die "--benchmark-workers must be auto or a positive integer"
    [[ -z "$BENCHMARK_LIMIT" || "$BENCHMARK_LIMIT" =~ ^[1-9][0-9]*$ ]] || die "--benchmark-limit must be positive"
    [[ "$BENCHMARK_TARGET_SELECTION" == auto || "$BENCHMARK_TARGET_SELECTION" == label || "$BENCHMARK_TARGET_SELECTION" == label-semantic || "$BENCHMARK_TARGET_SELECTION" == reference || "$BENCHMARK_TARGET_SELECTION" == reference-best ]] \
        || die "--benchmark-target-selection must be auto, label, label-semantic, reference or reference-best"
    [[ "$BENCHMARK_BOUNDARY_CHECK" == paper || "$BENCHMARK_BOUNDARY_CHECK" == released ]] \
        || die "--benchmark-boundary-check must be paper or released"
    [[ "$BENCHMARK_METRIC_ALIGNMENT" == released || "$BENCHMARK_METRIC_ALIGNMENT" == original-frame ]] \
        || die "--benchmark-metric-alignment must be released or original-frame"
    [[ "$LISA_PRECISION" == fp16 || "$LISA_PRECISION" == bf16 ]] || die "--lisa-precision must be fp16 or bf16"
    [[ "$LISA_DEVICE_MAP" == auto || "$LISA_DEVICE_MAP" == single ]] || die "--lisa-device-map must be auto or single"
    [[ "$BENCHMARK_INFERENCE_ONLY" -eq 0 || "$BENCHMARK_SCORE_ONLY" -eq 0 ]] \
        || die "Benchmark inference-only and score-only cannot be combined"
    [[ "$BENCHMARK_PIPELINE_PROFILE" == benchmark || "$BENCHMARK_PIPELINE_PROFILE" == released-sam ]] \
        || die "--benchmark-pipeline-profile must be benchmark or released-sam"
    [[ "$BENCHMARK_TARGET_SELECTION" != label-semantic || "$BENCHMARK_PIPELINE_PROFILE" == released-sam ]] \
        || die "--benchmark-target-selection label-semantic requires released-sam"
    [[ "$BENCHMARK_DETECTOR_VOCABULARY" == released || "$BENCHMARK_DETECTOR_VOCABULARY" == annotated ]] \
        || die "--benchmark-detector-vocabulary must be released or annotated"
    if [[ "$BENCHMARK_DETECTOR_VOCABULARY" == annotated ]]; then
        [[ "$BENCHMARK_PIPELINE_PROFILE" == released-sam && "$BENCHMARK_TARGET_SELECTION" == label-semantic ]] \
            || die "annotated detector vocabulary requires released-sam and label-semantic matching"
    fi
    [[ -z "$BENCHMARK_BACKGROUND_SIZING" || "$BENCHMARK_BACKGROUND_SIZING" == released || "$BENCHMARK_BACKGROUND_SIZING" == auto ]] \
        || die "--benchmark-background-sizing must be released or auto"
    if [[ -n "$BENCHMARK_BACKGROUND_SIZING" ]]; then
        [[ -n "$BENCHMARK_ROOT" ]] || die "--benchmark-background-sizing requires --benchmark-root"
        [[ "$BENCHMARK_PIPELINE_PROFILE" == released-sam || "$BENCHMARK_BACKGROUND_SIZING" == auto ]] \
            || die "The benchmark profile requires --benchmark-background-sizing auto"
    fi
    if [[ -n "$BENCHMARK_REUSE_RESULTS_FROM" ]]; then
        [[ "$BENCHMARK_PIPELINE_PROFILE" == released-sam && "$BENCHMARK_BACKGROUND_SIZING" == auto &&
           "$BENCHMARK_SCORE_ONLY" -eq 0 && "$BENCHMARK_REFERENCE_ONLY" -eq 0 ]] \
            || die "--benchmark-reuse-results-from requires released-sam inference with --benchmark-background-sizing auto"
    fi
    if [[ "$BENCHMARK_PIPELINE_PROFILE" == released-sam ]]; then
        [[ -n "$BENCHMARK_ROOT" ]] || die "released-sam requires --benchmark-root"
        [[ "$BENCHMARK_GENERATE_REFERENCES" -eq 0 && "$BENCHMARK_REFERENCE_ONLY" -eq 0 ]] \
            || die "released-sam generates SAM references during inference; LISA reference preparation is disabled"
        [[ "$BENCHMARK_TARGET_SELECTION" == auto || "$BENCHMARK_TARGET_SELECTION" == label || "$BENCHMARK_TARGET_SELECTION" == label-semantic ]] \
            || die "released-sam requires --benchmark-target-selection label or label-semantic"
        [[ "$BENCHMARK_BOUNDARY_CHECK_EXPLICIT" -eq 0 || "$BENCHMARK_BOUNDARY_CHECK" == released ]] \
            || die "released-sam requires --benchmark-boundary-check released"
        [[ "$BENCHMARK_TARGET_SELECTION" != auto ]] || BENCHMARK_TARGET_SELECTION=label
        BENCHMARK_BOUNDARY_CHECK=released
        [[ -n "$REFERENCE_DIR" ]] || REFERENCE_DIR="$OUTPUT_DIR/visible_references"
    fi
    if [[ -n "$BENCHMARK_ROOT" ]]; then
        [[ -z "$CLASSES" && -z "$FILENAMES_FILE" ]] \
            || die "Benchmark mode uses annotated targets; use --classes-file or --benchmark-sources/limit"
        if [[ -z "$REFERENCE_DIR" && ( "$BENCHMARK_INFERENCE_ONLY" -eq 0 || "$BENCHMARK_GENERATE_REFERENCES" -eq 1 ) ]]; then
            REFERENCE_DIR="$BENCHMARK_ROOT/visible_references"
            [[ "$BENCHMARK_SCORE_ONLY" -eq 1 ]] || BENCHMARK_GENERATE_REFERENCES=1
        fi
    fi
    if [[ "$BENCHMARK_REFERENCE_ONLY" -eq 1 ]]; then
        [[ -n "$BENCHMARK_ROOT" && "$BENCHMARK_INFERENCE_ONLY" -eq 0 && "$BENCHMARK_SCORE_ONLY" -eq 0 ]] \
            || die "--benchmark-reference-only requires benchmark mode without inference-only or score-only"
    fi
    [[ -n "$LISA_REPO_DIR" ]] || LISA_REPO_DIR="$CACHE_DIR/LISA"
    if [[ -d "$LISA_MODEL" ]]; then
        LISA_MODEL="$(cd -- "$LISA_MODEL" && pwd -P)"
    fi
    [[ -z "$JOB_DEPENDENCY" || "$JOB_DEPENDENCY" =~ ^afterok:[1-9][0-9]*$ ]] \
        || die "--dependency must be afterok:JOBID with a positive numeric job ID"
    case "$CUDA_RUNTIME" in
        11.6|11.7) ;;
        *) die "AMODAL_CUDA_RUNTIME must be 11.6 or 11.7 for PyTorch 1.13.1" ;;
    esac
    if [[ "$REPO_DIR_EXPLICIT" -eq 0 ]]; then
        REPO_DIR="$CACHE_DIR/amodal"
    fi
    if [[ "$ENV_DIR_EXPLICIT" -eq 0 ]]; then
        ENV_DIR="$CACHE_DIR/conda-env"
    fi
    # main.py runs from the upstream checkout, so resolve a supplied local
    # model directory while still in the user's submission directory.
    if [[ -d "$SD_MODEL" ]]; then
        SD_MODEL="$(cd -- "$SD_MODEL" && pwd -P)"
    fi
}

submit_to_slurm() {
    require_command sbatch
    mkdir -p -- "$SCRIPT_DIR/logs"

    local -a sbatch_args
    sbatch_args=(
        --job-name "$JOB_NAME"
        --nodes 1
        --ntasks 1
        --cpus-per-task "$CPUS"
        --mem "$MEMORY"
        --time "$TIME_LIMIT"
        --gres "$GRES"
        --export=ALL
        --output "$SCRIPT_DIR/logs/%x-%j.out"
    )
    [[ -n "$PARTITION" ]] && sbatch_args+=(--partition "$PARTITION")
    [[ -n "$ACCOUNT" ]] && sbatch_args+=(--account "$ACCOUNT")
    [[ -n "$QOS" ]] && sbatch_args+=(--qos "$QOS")
    [[ -n "$JOB_DEPENDENCY" ]] && sbatch_args+=(--dependency "$JOB_DEPENDENCY")
    [[ "$SBATCH_PARSABLE" -eq 1 ]] && sbatch_args+=(--parsable)

    log "Submitting to Slurm"
    export AMODAL_SCRIPT_DIR="$SCRIPT_DIR"
    sbatch "${sbatch_args[@]}" "$SCRIPT_PATH" --run-in-allocation "$@"
}

load_optional_modules() {
    # Clusters commonly expose CUDA through environment modules. Set
    # AMODAL_MODULES="cuda/11.7 gcc/11" when the site requires this.
    if [[ -n "${AMODAL_MODULES:-}" ]]; then
        if ! type module >/dev/null 2>&1; then
            for module_init in \
                /etc/profile.d/modules.sh \
                /etc/profile.d/lmod.sh \
                /usr/share/Modules/init/bash \
                /usr/share/lmod/lmod/init/bash; do
                if [[ -f "$module_init" ]]; then
                    # shellcheck disable=SC1090
                    source "$module_init"
                    break
                fi
            done
        fi
        type module >/dev/null 2>&1 || die "AMODAL_MODULES is set but the module command is unavailable"
        # shellcheck disable=SC2086
        module load $AMODAL_MODULES
    fi
}

check_gpu() {
    require_command nvidia-smi
    nvidia-smi -L || die "Slurm allocated no usable NVIDIA GPU"
    local gpu_memory
    # Read the complete nvidia-smi stream before selecting the first GPU. With
    # pipefail enabled, head can close the pipe early and turn a valid query
    # into a SIGPIPE (exit 141) on multi-GPU nodes.
    gpu_memory="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | tr -d '[:space:]' | sed -n '1p')"
    if [[ "$gpu_memory" =~ ^[0-9]+$ ]] && ((gpu_memory < 20000)); then
        warn "GPU has ${gpu_memory} MiB; the paper used a 24 GiB GPU and the pipeline may run out of memory"
    fi
}

run_conda() {
    local amodal_conda_had_nounset=0 amodal_conda_status
    [[ "$-" == *u* ]] && amodal_conda_had_nounset=1
    # Package activation/deactivation hooks are vendor shell code. In
    # particular, newly installed libblas MKL hooks may read a backup variable
    # that the previously active environment never set. Conda install also
    # runs these hooks automatically when it reactivates the current prefix.
    set +u
    if conda "$@"; then
        amodal_conda_status=0
    else
        amodal_conda_status=$?
    fi
    if [[ "$amodal_conda_had_nounset" -eq 1 ]]; then
        set -u
    fi
    return "$amodal_conda_status"
}

conda_environment_packages() {
    local operation="$1"
    shift
    local -a cuda_library_requirements
    # Old NVIDIA metadata leaves these two libraries without upper bounds or
    # cuda-version constraints. Pin the toolkit's published component versions.
    case "$CUDA_RUNTIME" in
        11.6) cuda_library_requirements=('libcurand=10.2.9.124' 'libcufile=1.2.1.4') ;;
        11.7) cuda_library_requirements=('libcurand=10.2.10.91' 'libcufile=1.3.1.18') ;;
        *) die "Unsupported CUDA runtime for Conda package installation: $CUDA_RUNTIME" ;;
    esac
    run_conda "$operation" -y -p "$ENV_DIR" \
        "${CONDA_RUNTIME_REQUIREMENTS[@]}" "cuda-version=$CUDA_RUNTIME" \
        "${cuda_library_requirements[@]}" "$@" \
        --override-channels -c pytorch -c nvidia -c conda-forge
}

load_conda() {
    local conda_sh=""
    if command -v conda >/dev/null 2>&1; then
        conda_sh="$(run_conda info --base 2>/dev/null || true)/etc/profile.d/conda.sh"
    fi
    if [[ -z "$conda_sh" || ! -f "$conda_sh" ]]; then
        for candidate in \
            "${HOME:-}/miniconda3/etc/profile.d/conda.sh" \
            "${HOME:-}/anaconda3/etc/profile.d/conda.sh" \
            "${HOME:-}/mambaforge/etc/profile.d/conda.sh" \
            "$CACHE_DIR/miniforge3/etc/profile.d/conda.sh"; do
            if [[ -f "$candidate" ]]; then
                conda_sh="$candidate"
                break
            fi
        done
    fi

    if [[ -z "$conda_sh" || ! -f "$conda_sh" ]]; then
        require_download_tool
        local arch installer prefix url
        arch="$(uname -m)"
        case "$arch" in
            x86_64) arch="x86_64" ;;
            aarch64|arm64) arch="aarch64" ;;
            *) die "Unsupported Linux architecture for the automatic Miniforge bootstrap: $arch" ;;
        esac
        prefix="$CACHE_DIR/miniforge3"
        installer="$CACHE_DIR/Miniforge3-Linux-${arch}.sh"
        url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${arch}.sh"
        mkdir -p -- "$CACHE_DIR"
        if [[ ! -x "$prefix/bin/conda" ]]; then
            log "No Conda installation found; bootstrapping Miniforge under $prefix"
            fetch_url "$url" "$installer.part"
            mv -f -- "$installer.part" "$installer"
            bash "$installer" -b -p "$prefix"
        fi
        conda_sh="$prefix/etc/profile.d/conda.sh"
    fi
    [[ -f "$conda_sh" ]] || die "Could not locate conda.sh after Conda bootstrap"
    # shellcheck disable=SC1090
    source "$conda_sh"
    command -v conda >/dev/null 2>&1 || die "Conda initialization failed"
}

cuda_toolkit_usable() {
    local prefix="$1" version header library
    [[ -x "$prefix/bin/nvcc" ]] || return 1
    for header in "${CUDA_BUILD_HEADERS[@]}"; do
        [[ -f "$prefix/include/$header" ]] || return 1
    done
    for library in "${CUDA_BUILD_LIBRARIES[@]}"; do
        [[ -e "$prefix/lib64/$library" || -e "$prefix/lib/$library" ]] || return 1
    done
    version="$("$prefix/bin/nvcc" --version)" || return 1
    [[ "$version" == *"release $CUDA_RUNTIME,"* ]]
}

gcc_compatible() {
    local version major minor patch
    version="$("$1" -dumpfullversion -dumpversion 2>/dev/null)" || return 1
    [[ "$version" =~ ^([0-9]+)\.([0-9]+)(\.([0-9]+))?$ ]] || return 1
    major="${BASH_REMATCH[1]}"; minor="${BASH_REMATCH[2]}"; patch="${BASH_REMATCH[4]:-0}"
    # PyTorch 1.13 allows GCC 6.0 through 11.5.0 with CUDA 11.6/11.7.
    ((major >= 6 && (major < 11 || (major == 11 && (minor < 5 || (minor == 5 && patch == 0))))))
}

select_host_compiler() {
    local cc cxx suffix arch pair
    local -a pairs=("${CC:-gcc}|${CXX:-g++}")
    arch="$(uname -m)"
    pairs+=("$ENV_DIR/bin/${arch}-conda-linux-gnu-gcc|$ENV_DIR/bin/${arch}-conda-linux-gnu-g++")
    for suffix in 10 11 9 8 7 6; do
        pairs+=("gcc-$suffix|g++-$suffix")
    done
    for pair in "${pairs[@]}"; do
        cc="${pair%%|*}"; cxx="${pair#*|}"
        if command -v "$cc" >/dev/null 2>&1 && command -v "$cxx" >/dev/null 2>&1 \
            && gcc_compatible "$cc" && gcc_compatible "$cxx" \
            && [[ "$("$cc" -dumpfullversion -dumpversion)" == "$("$cxx" -dumpfullversion -dumpversion)" ]]; then
            CC="$(command -v "$cc")"; CXX="$(command -v "$cxx")"
            export CC CXX
            # torch passes CC as nvcc's -ccbin; also cover nvcc invoked elsewhere.
            export CUDAHOSTCXX="$CXX"
            log "Using host compilers: $CC / $CXX ($("$CXX" -dumpfullversion -dumpversion))"
            return 0
        fi
    done
    return 1
}

configure_cuda_toolchain() {
    local prefix="${CUDA_HOME:-}" nvcc_path
    local -a cuda_packages=(
        "cuda-nvcc=$CUDA_RUNTIME" "cuda-cudart-dev=$CUDA_RUNTIME"
        "cuda-driver-dev=$CUDA_RUNTIME" "cuda-cccl=$CUDA_RUNTIME"
    )
    # PyTorch's ATen/cuda/CUDAContext.h includes these library headers even
    # from GroundingDINO's CPU source file. Runtime packages do not supply them.
    case "$CUDA_RUNTIME" in
        11.6) cuda_packages+=(
            'libcusparse-dev=11.7.2.124' 'libcublas-dev=11.9.2.110' 'libcusolver-dev=11.3.4.124'
        ) ;;
        11.7) cuda_packages+=(
            'libcusparse-dev=11.7.4.91' 'libcublas-dev=11.10.3.66' 'libcusolver-dev=11.4.0.1'
        ) ;;
        *) die "Unsupported CUDA build toolkit: $CUDA_RUNTIME" ;;
    esac
    if ! cuda_toolkit_usable "$prefix"; then
        nvcc_path="$(command -v nvcc || true)"
        if [[ -n "$nvcc_path" ]]; then
            nvcc_path="$(readlink -f "$nvcc_path" 2>/dev/null || printf '%s' "$nvcc_path")"
            prefix="$(dirname -- "$(dirname -- "$nvcc_path")")"
        fi
    fi
    if ! cuda_toolkit_usable "$prefix"; then
        prefix="$ENV_DIR"
        if ! cuda_toolkit_usable "$prefix"; then
            log "Installing a complete CUDA $CUDA_RUNTIME build toolkit into the private environment"
            conda_environment_packages install "${cuda_packages[@]}" \
                || die "A complete CUDA $CUDA_RUNTIME toolkit is required; alternatively set AMODAL_MODULES"
        fi
    fi
    cuda_toolkit_usable "$prefix" || die "CUDA $CUDA_RUNTIME toolkit is incomplete at $prefix (nvcc, CUDA/Thrust/cuSPARSE/cuBLAS/cuSOLVER development headers or libraries missing)"
    export CUDA_HOME="$prefix" PATH="$prefix/bin:$ENV_DIR/bin:$PATH"
    log "Using CUDA $CUDA_RUNTIME toolkit at $CUDA_HOME"

    if ! select_host_compiler; then
        log "Installing GCC/G++ 10 for CUDA $CUDA_RUNTIME (the node compiler is incompatible or missing)"
        conda_environment_packages install 'gcc_linux-64=10' 'gxx_linux-64=10' 'sysroot_linux-64=2.17' \
            || die "Could not install CUDA-compatible host compilers; set AMODAL_MODULES to a compatible GCC module"
        # Apply the compiler package's sysroot and linker configuration.
        run_conda activate "$ENV_DIR"
        select_host_compiler || die "No matching GCC/G++ 6.0-11.5.0 pair found after installing GCC 10"
    fi

    "$ENV_DIR/bin/python" - <<'PY'
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import torch
from torch.utils.cpp_extension import get_compiler_abi_compatibility_and_version, _check_cuda_version, include_paths
assert torch.cuda.is_available(), "CUDA must be available while building GroundingDINO"
compatible, version = get_compiler_abi_compatibility_and_version(os.environ["CXX"])
assert compatible, "Selected C++ compiler is incompatible with PyTorch"
_check_cuda_version(os.environ["CXX"], version)
# File existence does not verify transitive includes or the active compiler's
# search paths. Compile the header that failed in the log with both compilers.
include_flags = [flag for path in include_paths(cuda=True) + [sysconfig.get_path("include")]
                 for flag in ("-I", path)]
source = """#include <cuda.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <thrust/complex.h>
int main() { return 0; }
"""
with tempfile.TemporaryDirectory(prefix="amodal-cuda-headers-") as directory:
    cpp = Path(directory) / "probe.cpp"
    cu = Path(directory) / "probe.cu"
    cpp.write_text(source)
    cu.write_text(source)
    subprocess.run([os.environ["CXX"], "-std=c++14", "-fsyntax-only", *include_flags, str(cpp)], check=True)
    subprocess.run([str(Path(os.environ["CUDA_HOME"]) / "bin/nvcc"),
                    "-ccbin", os.environ["CC"], "-std=c++14", "--expt-relaxed-constexpr",
                    *include_flags, "-c", str(cu), "-o", str(Path(directory) / "probe.o")], check=True)
print("Compiler probes OK: PyTorch CUDAContext, cuSPARSE, cuBLAS, cuSOLVER, and Thrust")
PY
}

create_environment() {
    load_conda
    if [[ "$RUN_ONLY" -eq 1 ]]; then
        [[ -x "$ENV_DIR/bin/python" ]] || die "Run-only environment is missing; omit --run-only once"
    fi
    mkdir -p -- "$(dirname -- "$ENV_DIR")"
    if [[ ! -x "$ENV_DIR/bin/python" ]]; then
        log "Creating isolated Python 3.10 environment at $ENV_DIR"
        conda_environment_packages create pip
    fi

    local python="$ENV_DIR/bin/python"
    run_conda activate "$ENV_DIR"
    # The log shows packages leaking in from ~/.local. Isolate both pip and
    # inference from packages and Python paths inherited from the login node.
    export PYTHONNOUSERSITE=1
    unset PYTHONPATH PYTHONHOME
    export LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    [[ -x "$python" ]] || die "Conda environment has no Python executable: $ENV_DIR"
    "$python" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version' \
        || die "The environment must use Python 3.10: $($python --version 2>&1)"

    local torch_marker="$ENV_DIR/.amodal_torch_${CUDA_RUNTIME}_${TORCH_SETUP_REVISION}"
    if [[ ! -f "$torch_marker" ]] || ! "$python" -c \
        'import sys, torch; assert torch.__version__.split("+")[0] == "1.13.1"; assert torch.version.cuda == sys.argv[1]' \
        "$CUDA_RUNTIME" >/dev/null 2>&1; then
        [[ "$RUN_ONLY" -eq 0 ]] || die "PyTorch setup is missing or incompatible; omit --run-only once"
        log "Installing PyTorch 1.13.1 with CUDA runtime $CUDA_RUNTIME"
        if ! conda_environment_packages install \
            pytorch=1.13.1 torchvision=0.14.1 torchaudio=0.13.1 \
            "pytorch-cuda=$CUDA_RUNTIME"; then
            [[ "$CUDA_RUNTIME" == "11.7" ]] \
                || die "Conda PyTorch installation failed; the wheel fallback only supports CUDA 11.7 (requested $CUDA_RUNTIME)"
            warn "Conda PyTorch installation failed; trying the official CUDA 11.7 wheels"
            "$python" -m pip install --no-cache-dir \
                'numpy==1.23.5' 'setuptools==69.5.1' 'wheel==0.40.0' 'packaging==23.2'
            "$python" -m pip install --upgrade \
                --extra-index-url https://download.pytorch.org/whl/cu117 \
                torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1
        fi
        "$python" -c 'import torch; assert torch.__version__.startswith("1.13."), torch.__version__'
        touch -- "$torch_marker"
    fi

    if ! command -v unzip >/dev/null 2>&1; then
        [[ "$RUN_ONLY" -eq 0 ]] || die "unzip is missing from the run-only environment"
        log "Installing unzip into the private environment"
        conda_environment_packages install unzip
        export PATH="$ENV_DIR/bin:$PATH"
    fi
    command -v unzip >/dev/null 2>&1 || die "unzip is required to extract model checkpoints"

    local deps_marker="$ENV_DIR/.amodal_dependencies_${REPO_COMMIT}_${CUDA_RUNTIME}_${DEPENDENCY_REVISION}"
    if [[ "$RUN_ONLY" -eq 1 ]]; then
        [[ -f "$deps_marker" ]] || die "Dependencies are not marked complete; omit --run-only once to install them"
        return
    fi
    if [[ -f "$deps_marker" ]]; then
        log "Using cached Python dependencies"
        return
    fi

    local constraints="$ENV_DIR/.amodal-constraints-${DEPENDENCY_REVISION}.txt"
    printf '%s\n' "${RUNTIME_REQUIREMENTS[@]}" \
        'torch==1.13.1' 'torchvision==0.14.1' 'torchaudio==0.13.1' \
        'setuptools==69.5.1' 'wheel==0.40.0' > "$constraints"
    # v4 already installed the custom packages and compiled the CUDA extension.
    # v5 changes the Python import paths and restores LaMa's Kornia API only.
    local previous_deps_marker="$ENV_DIR/.amodal_dependencies_${REPO_COMMIT}_${CUDA_RUNTIME}_v4"
    if [[ -f "$previous_deps_marker" ]]; then
        log "Updating Python dependencies in the prepared environment"
        "$python" -m pip install --no-cache-dir -c "$constraints" "${RUNTIME_REQUIREMENTS[@]}"
        "$python" -m pip check
        touch -- "$deps_marker"
        return
    fi
    # Apply the same constraints to every pip install, including upstream's
    # unpinned requirements. --no-build-isolation keeps the CUDA build on torch 1.13.
    "$python" -m pip install --upgrade -c "$constraints" pip setuptools wheel ninja
    # cpp_extension imports pkg_resources, so install the compatible build
    # tooling before inspecting the CUDA compiler on a freshly created env.
    configure_cuda_toolchain
    log "Installing runtime dependencies"
    "$python" -m pip install --no-cache-dir -c "$constraints" "${RUNTIME_REQUIREMENTS[@]}"
    log "Installing the repository's custom Diffusers fork and CUDA extensions"
    "$python" -m pip install --no-cache-dir --no-build-isolation -c "$constraints" -e "$REPO_DIR/diffusers"
    "$python" -m pip install --no-cache-dir --no-build-isolation -c "$constraints" -e "$REPO_DIR/Grounded-Segment-Anything/segment_anything"
    if [[ -n "$TORCH_ARCH" ]]; then
        export TORCH_CUDA_ARCH_LIST="$TORCH_ARCH"
    fi
    export MAX_JOBS="$BUILD_JOBS"
    "$python" -m pip install --no-cache-dir --no-build-isolation -c "$constraints" \
        -e "$REPO_DIR/Grounded-Segment-Anything/GroundingDINO"

    # Upstream declares opencv-python while Albumentations declares headless.
    # Keep their versions identical for metadata, and install headless last so
    # the shared cv2 files do not require libGL/X11 on a compute node.
    "$python" -m pip install --no-cache-dir --no-deps --force-reinstall 'opencv-python-headless==4.8.1.78'

    "$python" -m pip check
    touch -- "$deps_marker"
}

download_file() {
    local url="$1" destination="$2"
    [[ -s "$destination" ]] && return 0
    mkdir -p -- "$(dirname -- "$destination")"
    require_download_tool
    log "Downloading $(basename -- "$destination")"
    fetch_url "$url" "${destination}.part"
    mv -f -- "${destination}.part" "$destination"
    [[ -s "$destination" ]] || die "Downloaded file is empty: $destination"
}

clone_repository() {
    require_command git
    mkdir -p -- "$(dirname -- "$REPO_DIR")"
    if [[ ! -d "$REPO_DIR/.git" ]]; then
        [[ ! -e "$REPO_DIR" ]] || die "Repository path exists but is not a Git checkout: $REPO_DIR"
        log "Cloning upstream repository"
        git -c http.version=HTTP/1.1 clone --depth 1 --branch main "$REPO_URL" "$REPO_DIR"
    fi
    git -C "$REPO_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
        || die "Repository checkout is incomplete: $REPO_DIR"
    if ! git -C "$REPO_DIR" cat-file -e "${REPO_COMMIT}^{commit}" 2>/dev/null; then
        log "Fetching pinned upstream commit"
        git -C "$REPO_DIR" fetch --depth 1 origin "$REPO_COMMIT"
    fi
    git -C "$REPO_DIR" checkout --detach --quiet "$REPO_COMMIT"
    [[ -f "$REPO_DIR/main.py" ]] || die "Pinned checkout has no main.py"
}

download_checkpoints() {
    local gdino="$REPO_DIR/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
    local sam="$REPO_DIR/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"
    local insta="$REPO_DIR/InstaOrder/InstaOrder_ckpt/InstaOrder_InstaOrderNet_od.pth.tar"
    local lama="$REPO_DIR/lama/big-lama/models/best.ckpt"

    [[ "$RUN_ONLY" -eq 1 ]] && {
        [[ -f "$gdino" && -f "$sam" && -f "$insta" && -f "$lama" ]] \
            || die "One or more checkpoints are missing; omit --run-only once to download them"
        return
    }

    download_file \
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" \
        "$gdino"
    download_file \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" \
        "$sam"

    if [[ ! -f "$insta" ]]; then
        require_command unzip
        log "Downloading InstaOrder checkpoints (this archive is several GB)"
        local insta_zip="$CACHE_DIR/InstaOrder_ckpt.zip"
        if [[ ! -s "$insta_zip" ]]; then
            # Drive IDs are case-sensitive: upstream uses LSkJ, not LSKJ.
            "$ENV_DIR/bin/python" -m gdown.cli 1_GEmCmofLSkJZnidfp4vsQb2Nqq5aqBU -O "$insta_zip.part"
            mv -f -- "$insta_zip.part" "$insta_zip"
        fi
        unzip -q -o "$insta_zip" -d "$REPO_DIR/InstaOrder"
    fi

    if [[ ! -f "$lama" ]]; then
        require_command unzip
        local lama_zip="$CACHE_DIR/big-lama.zip"
        download_file "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip" "$lama_zip"
        unzip -q -o "$lama_zip" -d "$REPO_DIR/lama"
    fi

    [[ -f "$gdino" ]] || die "GroundingDINO checkpoint was not found after download"
    [[ -f "$sam" ]] || die "SAM checkpoint was not found after download"
    [[ -f "$insta" ]] || die "InstaOrder checkpoint was not found after extraction"
    [[ -f "$lama" ]] || die "LaMa checkpoint was not found after extraction"
}

prepare_inputs() {
    [[ -d "$INPUT_DIR" ]] || die "Input directory does not exist: $INPUT_DIR"
    local manifest_dir="$CACHE_DIR/manifests"
    local manifest="$manifest_dir/images.txt"
    mkdir -p -- "$manifest_dir"

    if [[ -n "$FILENAMES_FILE" ]]; then
        FILENAMES_FILE="$(absolute_file "$FILENAMES_FILE")"
        cp -- "$FILENAMES_FILE" "$manifest"
    else
        find "$INPUT_DIR" -maxdepth 1 -type f \
            \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) \
            -printf '%f\n' | LC_ALL=C sort > "$manifest"
    fi
    [[ -s "$manifest" ]] || die "No .jpg, .jpeg, .png, or .webp images were found in $INPUT_DIR"

    local classes="$manifest_dir/classes.txt"
    if [[ -n "$CLASSES_FILE" ]]; then
        CLASSES_FILE="$(absolute_file "$CLASSES_FILE")"
        cp -- "$CLASSES_FILE" "$classes"
    elif [[ -n "$CLASSES" ]]; then
        printf '%s\n' "$CLASSES" | tr ',' '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | sed '/^$/d' > "$classes"
    else
        cp -- "$REPO_DIR/classes.txt" "$classes"
    fi
    [[ -s "$classes" ]] || die "No object classes were supplied"

    printf '%s\n' "$manifest" "$classes"
}

configure_python_imports() {
    # The SAM checkout is named segment_anything, and contains the real package
    # in another segment_anything directory. Its checkout root must precede
    # Grounded-Segment-Anything, whose outer directory otherwise becomes a
    # namespace package that shadows setuptools' editable import finder.
    # Put Diffusers' src directory on the path for the same reason.
    local gsa="$REPO_DIR/Grounded-Segment-Anything"
    export PYTHONPATH="$gsa/segment_anything:$REPO_DIR/diffusers/src:$REPO_DIR/lama:$REPO_DIR/InstaOrder:$gsa/GroundingDINO:$REPO_DIR:$gsa"
}

preflight_python() {
    local python="$ENV_DIR/bin/python"
    [[ -x "$python" ]] || die "Python environment is missing: $ENV_DIR"
    configure_python_imports
    export MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false
    "$python" -m pip check
    pushd "$REPO_DIR" >/dev/null
    "$python" - "$CUDA_RUNTIME" "$REPO_DIR" <<'PY'
import importlib
import os
from pathlib import Path
import re
import runpy
import sys
import numpy as np
import torch

assert torch.__version__.split("+")[0] == "1.13.1", torch.__version__
assert torch.version.cuda == sys.argv[1], (torch.version.cuda, sys.argv[1])
assert torch.cuda.is_available(), "torch.cuda.is_available() is false"
assert np.__version__ == "1.23.5", np.__version__
# Exercise NumPy's ABI as well as CUDA driver initialization.
assert torch.from_numpy(np.ones(1, dtype=np.float32)).cuda().item() == 1
print("Python:", os.sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
for module in ("cv2", "skimage", "sklearn", "diffusers", "groundingdino"):
    importlib.import_module(module)
    print("Import OK:", module)
from diffusers import StableDiffusionInpaintMixedContextPipeline
print("Import OK: StableDiffusionInpaintMixedContextPipeline")
import segment_anything as sam
assert sam.__file__, "segment_anything resolved to an outer namespace, not the SAM package"
expected_sam = Path(sys.argv[2]) / "Grounded-Segment-Anything/segment_anything/segment_anything/__init__.py"
assert Path(sam.__file__).resolve() == expected_sam.resolve(), (sam.__file__, str(expected_sam))
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator
assert callable(build_sam) and isinstance(SamPredictor, type) and isinstance(SamAutomaticMaskGenerator, type)
print("Import OK: Segment Anything APIs from", sam.__file__)
import cv2
assert re.search(r"GUI:\s+NONE", cv2.getBuildInformation()), "cv2 must use the headless build"
from albumentations import DualIAATransform
from kornia import SamplePadding
from groundingdino.util.slconfig import SLConfig
assert SLConfig(dict(device="cuda")).pretty_text
from groundingdino import _C
# Importing groundingdino alone succeeds even when the extension did not build.
# A tiny real operator call also detects a CPU-only or ABI-incompatible build.
value = torch.ones((1, 4, 1, 4), device="cuda")
shapes = torch.tensor([[2, 2]], dtype=torch.long, device="cuda")
starts = torch.zeros(1, dtype=torch.long, device="cuda")
locations = torch.full((1, 2, 1, 1, 2, 2), 0.5, device="cuda")
weights = torch.full((1, 2, 1, 1, 2), 0.5, device="cuda")
result = _C.ms_deform_attn_forward(value, shapes, starts, locations, weights, 64)
assert result.shape == (1, 2, 4) and torch.allclose(result, torch.ones_like(result))
torch.cuda.synchronize()
print("CUDA operator OK: groundingdino._C")
# main.py guards its inference entry point. Load its complete import graph
# without constructing models or downloading Hugging Face weights.
runpy.run_path("main.py", run_name="amodal_preflight")
print("Import OK: main.py (including InstaOrder and LaMa)")
PY
    popd >/dev/null
}

configure_model_caches() {
    mkdir -p -- "$CACHE_DIR/huggingface" "$CACHE_DIR/torch"
    export HF_HOME="$CACHE_DIR/huggingface"
    export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
    export TRANSFORMERS_CACHE="$HF_HOME/transformers"
    export TORCH_HOME="$CACHE_DIR/torch"
}

preflight_sd_model() {
    configure_model_caches
    log "Checking Stable Diffusion inpainting source: $SD_MODEL"
    "$ENV_DIR/bin/python" - "$SD_MODEL" <<'PY'
import json
from pathlib import Path
import sys
from huggingface_hub import hf_hub_download

model = sys.argv[1]
def config(filename):
    if Path(model).is_dir():
        path = Path(model) / filename
    else:
        path = Path(hf_hub_download(model, filename))
    return json.loads(path.read_text())

try:
    index = config("model_index.json")
    unet = config("unet/config.json")
    text = config("text_encoder/config.json")
    vae = config("vae/config.json")
    assert index["unet"][1] == "UNet2DConditionModel", "Expected a Diffusers UNet"
    assert unet["in_channels"] == 9 and unet["out_channels"] == 4, "Expected a 9-channel inpainting UNet"
    assert unet["cross_attention_dim"] == 1024 and text["hidden_size"] == 1024, "Expected the SD2 text encoder"
    assert vae["latent_channels"] == 4, "Expected four VAE latent channels"
except Exception as error:
    raise SystemExit(
        f"Cannot use SD2 inpainting model {model}: {error}\n"
        "Use --sd-model with an accessible SD2 inpainting repository or a local Diffusers model directory. "
        "For a private repository, set HF_TOKEN before submitting."
    ) from error
print("Model configs OK: SD2 inpainting, 9-channel UNet and 1024-dimensional text encoder")
PY
}

launch_upstream() {
    export AMODAL_SD_MODEL="$SD_MODEL"
    # Keep the pinned checkout unchanged. Change only the two hardcoded model
    # arguments in main.py's syntax tree, retaining the custom pipeline class,
    # float16 settings, command-line arguments, and original traceback lines.
    "$ENV_DIR/bin/python" - "$@" <<'PY'
import ast
import os
from pathlib import Path
import sys

_amodal_filename = str(Path("main.py").resolve())
_amodal_tree = ast.parse(Path(_amodal_filename).read_text(), filename=_amodal_filename)

class _AmodalModelSource(ast.NodeTransformer):
    replacements = 0
    def visit_Call(self, node):
        self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "from_pretrained"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "stabilityai/stable-diffusion-2-inpainting"):
            node.args[0] = ast.copy_location(ast.Constant(os.environ["AMODAL_SD_MODEL"]), node.args[0])
            self.replacements += 1
        return node

_amodal_selector = _AmodalModelSource()
_amodal_tree = _amodal_selector.visit(_amodal_tree)
if _amodal_selector.replacements != 2:
    raise SystemExit("Expected two SD2 model-loading calls in the pinned main.py; refusing an unsupported checkout")
ast.fix_missing_locations(_amodal_tree)
sys.argv[0] = _amodal_filename
__file__ = _amodal_filename
__package__ = None
__spec__ = None
exec(compile(_amodal_tree, _amodal_filename, "exec"), globals())
PY
}

run_pipeline() {
    local manifest classes
    local -a input_files
    # A failing process substitution is not propagated by mapfile. Capture the
    # producer's exit status before unpacking its paths.
    local prepared_inputs
    prepared_inputs="$(prepare_inputs)"
    mapfile -t input_files <<< "$prepared_inputs"
    [[ "${#input_files[@]}" -eq 2 ]] || die "Input preparation did not return manifest and class paths"
    manifest="${input_files[0]}"
    classes="${input_files[1]}"

    mkdir -p -- "$OUTPUT_DIR"
    configure_model_caches
    configure_python_imports
    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false
    export MPLBACKEND=Agg
    export OMP_NUM_THREADS="$CPUS"
    export MKL_NUM_THREADS="$CPUS"

    [[ -f "$REPO_DIR/images/gray_wallpaper.jpeg" ]] \
        || die "The clean Mixed Context background is missing from the checkout"

    log "Running progressive Mixed Context diffusion"
    pushd "$REPO_DIR" >/dev/null
    launch_upstream \
        --input_dir "$INPUT_DIR" \
        --img_filenames_txt "$manifest" \
        --classes_txt "$classes" \
        --gdino_config "$REPO_DIR/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
        --gdino_ckpt "$REPO_DIR/Grounded-Segment-Anything/groundingdino_swint_ogc.pth" \
        --sam_ckpt "$REPO_DIR/Grounded-Segment-Anything/sam_vit_h_4b8939.pth" \
        --instaorder_ckpt "$REPO_DIR/InstaOrder/InstaOrder_ckpt/InstaOrder_InstaOrderNet_od.pth.tar" \
        --lama_config_path "$REPO_DIR/lama/big-lama/config.yaml" \
        --lama_ckpt_path "$REPO_DIR/lama/big-lama/models/best.ckpt" \
        --mc_clean_bkgd_img "$REPO_DIR/images/gray_wallpaper.jpeg" \
        --mc_timestep "$MC_TIMESTEP" \
        --max_iter_id "$MAX_ITER" \
        --output_dir "$OUTPUT_DIR"
    popd >/dev/null
}

prepare_benchmark_metrics() {
    [[ -n "$BENCHMARK_ROOT" && "$BENCHMARK_INFERENCE_ONLY" -eq 0 ]] || return 0
    if [[ "$RUN_ONLY" -eq 0 ]]; then
        # Torch/torchvision and all other metric dependencies are already pinned
        # by the shared environment. Never let LPIPS upgrade them transitively.
        "$ENV_DIR/bin/python" -m pip install --no-deps 'lpips==0.1.4'
    fi
    "$ENV_DIR/bin/python" - <<'PY'
import importlib.metadata
import lpips
assert importlib.metadata.version("lpips") == "0.1.4", "Benchmark requires lpips==0.1.4; omit --run-only once"
PY
}

prepare_lisa_runtime() {
    local extras="$CACHE_DIR/lisa-python" marker="$CACHE_DIR/lisa-python/.runtime-v1"
    if [[ ! -d "$LISA_REPO_DIR/.git" ]]; then
        [[ "$RUN_ONLY" -eq 0 ]] || die "LISA checkout is missing; omit --run-only once"
        git clone --filter=blob:none --no-checkout https://github.com/JIA-Lab-research/LISA.git "$LISA_REPO_DIR"
        git -C "$LISA_REPO_DIR" checkout --detach "$LISA_REPO_COMMIT"
    fi
    [[ "$(git -C "$LISA_REPO_DIR" rev-parse HEAD)" == "$LISA_REPO_COMMIT" ]] \
        || die "LISA checkout must use pinned commit $LISA_REPO_COMMIT: $LISA_REPO_DIR"
    if [[ ! -f "$marker" ]]; then
        [[ "$RUN_ONLY" -eq 0 ]] || die "LISA Python overlay is missing; omit --run-only once"
        mkdir -p -- "$extras"
        # The base Torch/CUDA environment is reused. Only this subprocess sees
        # LISA's Transformers version; PD-MC keeps its pinned 4.30.2 installation.
        "$ENV_DIR/bin/python" -m pip install --target "$extras" --upgrade --no-deps \
            'transformers==4.31.0' 'sentencepiece==0.1.99' 'einops==0.4.1'
    fi
    PYTHONPATH="$extras:$LISA_REPO_DIR:$SCRIPT_DIR" "$ENV_DIR/bin/python" - <<'PY'
import importlib.metadata
import torch
import transformers
assert transformers.__version__ == "4.31.0", transformers.__version__
assert importlib.metadata.version("sentencepiece") == "0.1.99"
from model.LISA import LISAForCausalLM
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from accelerate import dispatch_model
print("LISA reference imports OK; shared Torch", torch.__version__, "isolated Transformers", transformers.__version__)
PY
    touch "$marker"
}

run_reference_preparation() {
    configure_model_caches
    local -a reference_args=(
        --dataset-root "$BENCHMARK_ROOT" --reference-dir "$REFERENCE_DIR"
        --lisa-repo "$LISA_REPO_DIR" --lisa-model "$LISA_MODEL" --precision "$LISA_PRECISION"
        --lisa-device-map "$LISA_DEVICE_MAP"
        --sources "$BENCHMARK_SOURCES" --seed "$BENCHMARK_SEED" --prompt-template "$LISA_PROMPT_TEMPLATE"
    )
    [[ -n "$BENCHMARK_LIMIT" ]] && reference_args+=(--limit "$BENCHMARK_LIMIT")
    [[ "$BENCHMARK_ALLOW_INCOMPLETE" -eq 1 ]] && reference_args+=(--allow-incomplete)
    PYTHONPATH="$CACHE_DIR/lisa-python:$LISA_REPO_DIR:$SCRIPT_DIR" \
        "$ENV_DIR/bin/python" "$SCRIPT_DIR/prepare_visible_references.py" "${reference_args[@]}" "$@"
}

run_benchmark() {
    configure_model_caches
    configure_python_imports
    export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
    export OMP_NUM_THREADS="$CPUS" MKL_NUM_THREADS="$CPUS"
    local -a benchmark_args=(
        --dataset-root "$BENCHMARK_ROOT" --output-dir "$OUTPUT_DIR" --repo-dir "$REPO_DIR"
        --sources "$BENCHMARK_SOURCES" --seed "$BENCHMARK_SEED"
        --workers "$BENCHMARK_WORKERS"
        --mc-timestep "$MC_TIMESTEP" --max-iter "$MAX_ITER" --sd-model "$SD_MODEL"
        --target-selection "$BENCHMARK_TARGET_SELECTION"
        --boundary-check "$BENCHMARK_BOUNDARY_CHECK"
        --pipeline-profile "$BENCHMARK_PIPELINE_PROFILE"
        --detector-vocabulary "$BENCHMARK_DETECTOR_VOCABULARY"
        --metric-alignment "$BENCHMARK_METRIC_ALIGNMENT"
    )
    [[ -n "$REFERENCE_DIR" ]] && benchmark_args+=(--reference-dir "$REFERENCE_DIR")
    [[ -n "$BENCHMARK_CONFIG" ]] && benchmark_args+=(--config "$BENCHMARK_CONFIG")
    [[ -n "$BENCHMARK_BACKGROUND_SIZING" ]] && benchmark_args+=(--background-sizing "$BENCHMARK_BACKGROUND_SIZING")
    [[ -n "$BENCHMARK_REUSE_RESULTS_FROM" ]] && benchmark_args+=(--reuse-results-from "$BENCHMARK_REUSE_RESULTS_FROM")
    [[ -n "$CLASSES_FILE" ]] && benchmark_args+=(--classes-file "$CLASSES_FILE")
    [[ -n "$BENCHMARK_LIMIT" ]] && benchmark_args+=(--limit "$BENCHMARK_LIMIT")
    [[ "$BENCHMARK_INFERENCE_ONLY" -eq 1 ]] && benchmark_args+=(--inference-only)
    [[ "$BENCHMARK_SCORE_ONLY" -eq 1 ]] && benchmark_args+=(--score-only)
    [[ "$BENCHMARK_ALLOW_INCOMPLETE" -eq 1 ]] && benchmark_args+=(--allow-incomplete)
    "$ENV_DIR/bin/python" "$SCRIPT_DIR/evaluate_open_world.py" "${benchmark_args[@]}" "$@"
}

main() {
    parse_args "$@"

    if [[ "$RUN_IN_ALLOCATION" -eq 0 && -z "${SLURM_JOB_ID:-}" ]]; then
        submit_to_slurm "$@"
        return 0
    fi

    load_optional_modules
    check_gpu
    require_download_tool
    require_command git

    INPUT_DIR="$(absolute_dir "$INPUT_DIR")"
    OUTPUT_DIR="$(absolute_dir "$OUTPUT_DIR")"
    CACHE_DIR="$(absolute_dir "$CACHE_DIR")"
    if [[ -d "$REPO_DIR" ]]; then
        REPO_DIR="$(cd -- "$REPO_DIR" && pwd -P)"
    else
        REPO_DIR="$(cd -- "$(dirname -- "$REPO_DIR")" && pwd -P)/$(basename -- "$REPO_DIR")"
    fi
    if [[ -d "$ENV_DIR" ]]; then
        ENV_DIR="$(cd -- "$ENV_DIR" && pwd -P)"
    else
        ENV_DIR="$(cd -- "$(dirname -- "$ENV_DIR")" && pwd -P)/$(basename -- "$ENV_DIR")"
    fi

    log "Job ${SLURM_JOB_ID:-local} on $(hostname)"
    log "Project: $SCRIPT_DIR"
    log "Input: $INPUT_DIR"
    log "Output: $OUTPUT_DIR"
    log "Cache: $CACHE_DIR"

    clone_repository
    create_environment
    preflight_python
    if [[ -n "$BENCHMARK_ROOT" && "$BENCHMARK_GENERATE_REFERENCES" -eq 1 ]]; then
        if [[ "$SETUP_ONLY" -eq 0 ]]; then
            run_reference_preparation --check-only
        fi
        prepare_lisa_runtime
        if [[ "$SETUP_ONLY" -eq 0 ]]; then
            log "Generating visible references with LISA-13B before loading PD-MC"
            run_reference_preparation
            if [[ "$BENCHMARK_REFERENCE_ONLY" -eq 1 ]]; then
                log "Visible reference preparation complete: $REFERENCE_DIR"
                return 0
            fi
        fi
    fi
    if [[ -n "$BENCHMARK_ROOT" && "$SETUP_ONLY" -eq 0 ]]; then
        # Validate before downloading large checkpoints or constructing models.
        run_benchmark --check-only
    fi
    preflight_sd_model
    download_checkpoints
    prepare_benchmark_metrics

    if [[ "$SETUP_ONLY" -eq 1 ]]; then
        log "Setup complete. Re-run with --run-only to skip installation and downloads."
        return 0
    fi

    if [[ -n "$BENCHMARK_ROOT" ]]; then
        run_benchmark
    else
        run_pipeline
    fi
    log "Completed. Results are in $OUTPUT_DIR"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
