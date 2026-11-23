#!/usr/bin/env bash
# Tạo hai environment riêng để source C2F không xung đột dependencies PMC/LaMa.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
conda env create -f environments/pmc.yml
conda env create -f environments/c2f.yml
# Cài editable fork; không cài requirements.txt legacy có nhiều pin mâu thuẫn.
conda run -n pmc python -m pip install -e ./diffusers
conda run -n pmc python -m pip install --no-build-isolation -e ./Grounded-Segment-Anything/GroundingDINO --no-deps
conda run -n pmc python -m pip install -e ./Grounded-Segment-Anything/segment_anything --no-deps
conda run -n pmc python -m pip check
conda run -n pmc-c2f python -m pip check
conda run -n pmc python scripts/download_c2f.py
printf '%s\n' 'Cài xong. Truyền --c2f_python /đường/dẫn/envs/pmc-c2f/bin/python khi chạy.'
