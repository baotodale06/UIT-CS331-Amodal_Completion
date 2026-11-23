#!/usr/bin/env bash
# Chạy trong environment pmc; truyền thêm --c2f_python nếu worker dùng env riêng.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python main.py "$@"
