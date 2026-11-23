# Kiểm thử CPU

File: `tests/test_pmc.py`

unittest discover -s tests -v. Bao phủ crop/canvas, invalid masks, support, score selection, IoU hidden, JSON Path, cleanup khi exception, CLI metrics-only và run_iteration với fake backend. Không thay thế kiểm thử model/checkpoint/CUDA thực tế.

Xem [overall.md](overall.md) để biết cách chạy, dữ liệu đầu vào, artifacts và giới hạn kiểm thử.
