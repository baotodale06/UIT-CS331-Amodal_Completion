#!/usr/bin/env bash
# Check the real SAM exports and complete inference imports without loading
# models or checkpoints. This check also works in a matching CPU environment.
set -Eeuo pipefail
test_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=../run_amodal_slurm.sh
source "$test_dir/../run_amodal_slurm.sh"
[[ $# -eq 2 ]] || die 'Usage: bash tests/validate_pipeline_imports.sh ENV_PREFIX UPSTREAM_CHECKOUT'
ENV_DIR="$(cd -- "$1" && pwd -P)"
REPO_DIR="$(cd -- "$2" && pwd -P)"
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset PYTHONHOME PYTHONPATH
configure_python_imports
pushd "$REPO_DIR" >/dev/null
"$ENV_DIR/bin/python" - "$REPO_DIR" <<'PY'
from pathlib import Path
import runpy
import sys
import segment_anything as sam
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator
from diffusers import StableDiffusionInpaintMixedContextPipeline
from kornia import SamplePadding

expected = Path(sys.argv[1]) / "Grounded-Segment-Anything/segment_anything/segment_anything/__init__.py"
assert sam.__file__ and Path(sam.__file__).resolve() == expected.resolve()
assert callable(build_sam) and isinstance(SamPredictor, type) and isinstance(SamAutomaticMaskGenerator, type)
print("PASS: real SAM exports from", sam.__file__)
print("PASS: custom Mixed Context Diffusers pipeline import")
runpy.run_path("main.py", run_name="amodal_import_validation")
print("PASS: full main.py import graph, including InstaOrder and LaMa")
PY
popd >/dev/null
