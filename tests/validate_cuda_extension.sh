#!/usr/bin/env bash
# Real compiler/GPU verification against an already provisioned environment and
# pinned upstream checkout. Does not download model checkpoints or run inference.
set -Eeuo pipefail
test_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=../run_amodal_slurm.sh
source "$test_dir/../run_amodal_slurm.sh"
[[ $# -eq 2 ]] || die 'Usage: bash tests/validate_cuda_extension.sh ENV_PREFIX UPSTREAM_CHECKOUT'
ENV_DIR="$(cd -- "$1" && pwd -P)"
REPO_DIR="$(cd -- "$2" && pwd -P)"
[[ -x "$ENV_DIR/bin/python" ]] || die 'Validation environment has no Python'
[[ -f "$REPO_DIR/Grounded-Segment-Anything/GroundingDINO/setup.py" ]] || die 'GroundingDINO checkout is missing'
export CUDA_HOME="$ENV_DIR" PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
configure_cuda_toolchain
# One build worker keeps this optional test usable on smaller development PCs.
export MAX_JOBS="${AMODAL_BUILD_JOBS:-1}"
# Include PTX so the cluster's 7.5 target can also be exercised on newer GPUs.
export TORCH_CUDA_ARCH_LIST="${AMODAL_TORCH_CUDA_ARCH_LIST:-7.5+PTX}"
"$ENV_DIR/bin/python" -m pip install --no-deps --no-build-isolation \
    -e "$REPO_DIR/Grounded-Segment-Anything/GroundingDINO"
"$ENV_DIR/bin/python" - <<'PY'
import torch
from groundingdino import _C

value = torch.ones((1, 4, 1, 4), device="cuda")
shapes = torch.tensor([[2, 2]], dtype=torch.long, device="cuda")
starts = torch.zeros(1, dtype=torch.long, device="cuda")
locations = torch.full((1, 2, 1, 1, 2, 2), 0.5, device="cuda")
weights = torch.full((1, 2, 1, 1, 2), 0.5, device="cuda")
output = _C.ms_deform_attn_forward(value, shapes, starts, locations, weights, 64)
assert output.shape == (1, 2, 4)
assert torch.allclose(output, torch.ones_like(output))
grad_value, grad_locations, grad_weights = _C.ms_deform_attn_backward(
    value, shapes, starts, locations, weights, torch.ones_like(output), 64
)
assert torch.allclose(grad_value, torch.full_like(value, 0.5))
assert torch.allclose(grad_locations, torch.zeros_like(locations))
assert torch.allclose(grad_weights, torch.full_like(weights, 4))
torch.cuda.synchronize()
print("PASS: real GroundingDINO CUDA build and forward/backward operators")
signal = torch.arange(16 * 64 * 64, dtype=torch.float32, device="cuda").reshape(1, 16, 64, 64)
signal = signal / signal.numel()
spectrum = torch.fft.rfftn(signal, dim=(-2, -1))
reconstructed = torch.fft.irfftn(spectrum, s=signal.shape[-2:], dim=(-2, -1))
assert torch.allclose(signal, reconstructed, atol=1e-5, rtol=1e-5)
torch.cuda.synchronize()
print("PASS: real cuFFT forward/inverse round trip for LaMa")
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY
