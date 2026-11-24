#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -Eeuo pipefail
CODE_DIR="${SLURM_SUBMIT_DIR:?Submit from the frozen evaluation source}"
STUDY="$(dirname -- "$CODE_DIR")"
PROJECT_DIR="${SHAPE_PROJECT_DIR:-$HOME/Vu_Hung/cv_nangcao}"
CACHE_DIR="$PROJECT_DIR/.amodal-cache"
ENV_DIR="$CACHE_DIR/conda-env"
REPO_DIR="$CACHE_DIR/amodal"
CONTROL="$PROJECT_DIR/output/surface-annotated-20261112-freshrefs/baseline"
PYTHON_EXEC="$CACHE_DIR/surface-metrics-env/bin/python"
CHECKPOINT="$STUDY/training/best.pt"
cd "$CODE_DIR"
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
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
if [[ "${1:-}" == geometry ]]; then
    "$PYTHON_EXEC" scripts/evaluate_shape_v2_geometry.py --checkpoint "$CHECKPOINT" \
        --data "$STUDY/data" --split validation --output "$STUDY/validation-native.json"
    "$PYTHON_EXEC" scripts/freeze_shape_v2.py --checkpoint "$CHECKPOINT" \
        --validation "$STUDY/validation-native.json" --holdout-manifest "$STUDY/holdout-data/manifest.json" \
        --control "$CONTROL" --source-manifest "$STUDY/evaluation-source-manifest.json" \
        --output "$STUDY/policy-freeze.json"
    "$PYTHON_EXEC" scripts/evaluate_shape_v2_geometry.py --checkpoint "$CHECKPOINT" \
        --data "$STUDY/holdout-data" --split holdout --freeze "$STUDY/policy-freeze.json" \
        --output "$STUDY/holdout-native.json"
    exit 0
fi
[[ "${1:-}" == --benchmark-workers && "${2:-}" == 4 ]] || { echo 'Require four benchmark workers' >&2; exit 2; }
STAGE="${3:?Expected smoke or full}"
IFS=',' read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES:?Slurm allocation required}"
[[ "${#DEVICES[@]}" == 4 ]] || { echo 'Require four allocated GPUs' >&2; exit 2; }
"$PYTHON_EXEC" scripts/prepare_shape_v2_evaluation.py --study "$STUDY" --control "$CONTROL" \
    --dataset-root "$PROJECT_DIR/datasets/open_world_amodal"
ARGS=(--dataset-root "$PROJECT_DIR/datasets/open_world_amodal" --repo-dir "$REPO_DIR"
    --pipeline-profile released-sam --target-selection label-semantic --boundary-check released
    --background-sizing auto --detector-vocabulary annotated --workers 4
    --shape-v2-checkpoint "$CHECKPOINT" --reference-dir "$STUDY/benchmark/visible_references")
case "$STAGE" in
    smoke)
        "$PYTHON_EXEC" -u evaluate_open_world.py "${ARGS[@]}" --output-dir "$STUDY/smoke" \
            --target-manifest "$STUDY/smoke-manifest.json" --inference-only
        "$PYTHON_EXEC" scripts/prepare_shape_v2_evaluation.py --study "$STUDY" --control "$CONTROL" \
            --dataset-root "$PROJECT_DIR/datasets/open_world_amodal" --verify-smoke
        ;;
    full)
        [[ -f "$STUDY/smoke-validation.json" ]] || { echo 'Real frozen-checkpoint smoke is required' >&2; exit 2; }
        "$PYTHON_EXEC" -u evaluate_open_world.py "${ARGS[@]}" --output-dir "$STUDY/benchmark"
        "$PYTHON_EXEC" -m amodal_benchmark verify --run-dir "$STUDY/benchmark" \
            --dataset-root "$PROJECT_DIR/datasets/open_world_amodal" --require-full --require-provenance \
            > "$STUDY/full-verification.json"
        "$PYTHON_EXEC" -m amodal_benchmark register --run-dir "$STUDY/benchmark" \
            --dataset-root "$PROJECT_DIR/datasets/open_world_amodal" --registry "$STUDY/registry.csv" \
            --hypothesis 'Shape-only ResNet34 amodal prior and expected-IoU selection versus frozen original PMC'
        "$PYTHON_EXEC" scripts/audit_shape_v2.py --control "$CONTROL" --method "$STUDY/benchmark" \
            --dataset-root "$PROJECT_DIR/datasets/open_world_amodal" --freeze "$STUDY/policy-freeze.json" \
            --unseen-inventory "$STUDY/unseen-inventory.json" \
            --geometry-dir "$PROJECT_DIR/output/shape-20261121/evaluation_only/cocoa" \
            --output "$STUDY/full-audit.json"
        ;;
    *) echo "Unknown evaluation stage $STAGE" >&2; exit 2;;
esac
