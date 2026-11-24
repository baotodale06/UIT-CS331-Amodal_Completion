#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -Eeuo pipefail
CODE_DIR="${SLURM_SUBMIT_DIR:?Submit from the isolated evaluation source}"
STUDY="$(dirname -- "$CODE_DIR")"
PROJECT_DIR="${SHAPE_PROJECT_DIR:-$HOME/Vu_Hung/cv_nangcao}"
CACHE_DIR="$PROJECT_DIR/.amodal-cache"
ENV_DIR="$CACHE_DIR/conda-env"
REPO_DIR="$CACHE_DIR/amodal"
PYTHON_EXEC="$CACHE_DIR/surface-metrics-env/bin/python"
CONTROL="$PROJECT_DIR/output/surface-annotated-20261112-freshrefs/baseline"
NATIVE="$PROJECT_DIR/output/shape-calibrated-20261123"
DATASET="$PROJECT_DIR/datasets/open_world_amodal"
HOLDOUT="$PROJECT_DIR/output/shape-v2-slurm-20261123/holdout-data"
cd "$CODE_DIR"
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME="$CACHE_DIR/huggingface" HUGGINGFACE_HUB_CACHE="$CACHE_DIR/huggingface/hub"
export TRANSFORMERS_CACHE="$CACHE_DIR/huggingface/transformers" TORCH_HOME="$CACHE_DIR/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME="$ENV_DIR" PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export AMODAL_CACHE_DIR="$CACHE_DIR" AMODAL_ENV_DIR="$ENV_DIR" AMODAL_REPO_DIR="$REPO_DIR"
unset PYTHONHOME PYTHONPATH
source "$CODE_DIR/run_amodal_slurm.sh"
configure_python_imports
export PYTHONPATH="$CODE_DIR:$PYTHONPATH"
PREPARE=(scripts/prepare_shape_calibrated_evaluation.py --study "$STUDY" --control "$CONTROL" --dataset-root "$DATASET")
if [[ "${1:-}" == preflight ]]; then
    IFS=',' read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES:?Slurm allocation required}"
    [[ "${#DEVICES[@]}" == 1 ]] || { echo 'Native geometry requires one visible Quadro' >&2; exit 2; }
    if [[ ! -d "$STUDY/admitted" ]]; then
        "$PYTHON_EXEC" scripts/package_shape_calibrated.py --study "$NATIVE" --output "$STUDY/admitted" \
            --data "$PROJECT_DIR/output/shape-v2-slurm-20261123/data" \
            --sam-validation "$PROJECT_DIR/output/shape-validation-sam-20261123/result"
    fi
    "$PYTHON_EXEC" "${PREPARE[@]}" --mode freeze --holdout-data "$HOLDOUT"
    bash tests/validate_cuda_extension.sh "$ENV_DIR" "$REPO_DIR" > "$STUDY/cuda-validation.log" 2>&1
    bash tests/validate_pipeline_imports.sh "$ENV_DIR" "$REPO_DIR" > "$STUDY/import-validation.log" 2>&1
    "$PYTHON_EXEC" - "$STUDY" <<'PY'
import sys
from datetime import datetime, timezone
from pathlib import Path
from amodal_benchmark.dataset import sha256, write_json
study = Path(sys.argv[1])
write_json(study / 'pipeline-validation.json', dict(verified=True, at=datetime.now(timezone.utc).isoformat(),
    freeze_sha256=sha256(study / 'policy-freeze.json'),
    source_manifest_sha256=sha256(study / 'evaluation-source-manifest.json'),
    logs={name: sha256(study / name) for name in ('cuda-validation.log', 'import-validation.log')},
    benchmark_success=False))
PY
    if [[ ! -f "$STUDY/holdout-native.json" ]]; then
        "$PYTHON_EXEC" -u scripts/evaluate_shape_v2_geometry.py --shape-calibrated-package "$STUDY/admitted" \
            --data "$HOLDOUT" --split holdout --freeze "$STUDY/policy-freeze.json" --output "$STUDY/holdout-native.json"
    fi
    "$PYTHON_EXEC" "${PREPARE[@]}" --mode prepare
    exit 0
fi
[[ "${1:-}" == --benchmark-workers && "${2:-}" == 4 ]] || { echo 'Require four benchmark workers' >&2; exit 2; }
STAGE="${3:?Expected smoke or full}"
IFS=',' read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES:?Slurm allocation required}"
[[ "${#DEVICES[@]}" == 4 ]] || { echo 'Require four allocated GPUs' >&2; exit 2; }
"$PYTHON_EXEC" "${PREPARE[@]}" --mode prepare
ARGS=(--dataset-root "$DATASET" --repo-dir "$REPO_DIR"
    --pipeline-profile released-sam --target-selection label-semantic --boundary-check released
    --background-sizing auto --detector-vocabulary annotated --workers 4 --seed 0 --mc-timestep 35 --max-iter 5
    --shape-calibrated-package "$STUDY/admitted" --reference-dir "$STUDY/benchmark/visible_references")
case "$STAGE" in
    smoke)
        "$PYTHON_EXEC" -u evaluate_open_world.py "${ARGS[@]}" --output-dir "$STUDY/smoke" \
            --target-manifest "$STUDY/smoke-manifest.json"
        "$PYTHON_EXEC" "${PREPARE[@]}" --mode verify-smoke
        ;;
    full)
        "$PYTHON_EXEC" "${PREPARE[@]}" --mode verify-smoke
        "$PYTHON_EXEC" -u evaluate_open_world.py "${ARGS[@]}" --output-dir "$STUDY/benchmark"
        "$PYTHON_EXEC" -m amodal_benchmark verify --run-dir "$STUDY/benchmark" --dataset-root "$DATASET" \
            --require-full --require-provenance > "$STUDY/full-verification.json"
        "$PYTHON_EXEC" -m amodal_benchmark register --run-dir "$STUDY/benchmark" --dataset-root "$DATASET" \
            --registry "$STUDY/registry.csv" --hypothesis 'Shape-only validation-calibrated amodal prior versus original PMC'
        "$PYTHON_EXEC" scripts/audit_shape_v2.py --control "$CONTROL" --method "$STUDY/benchmark" \
            --dataset-root "$DATASET" --freeze "$STUDY/policy-freeze.json" --selector-method shape-calibrated-probability \
            --unseen-inventory "$STUDY/unseen-inventory.json" \
            --geometry-dir "$PROJECT_DIR/output/shape-20261121/evaluation_only/cocoa" --output "$STUDY/full-audit.json"
        ;;
    *) echo "Unknown calibrated stage $STAGE" >&2; exit 2;;
esac
