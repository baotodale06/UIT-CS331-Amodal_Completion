# Tải C2F

File: `scripts/download_c2f.py`

Clone official repo và checkout commit đã kiểm tra, từ chối source dirty khi cần checkout. Tải bốn file COCOA bằng URL data.dgl.ai, ghi .part rồi rename; file đã có được reuse và hash. manifest ghi nguồn, revision, SHA256, bytes. SHA ghi nhận provenance tải được, không phải checksum xác thực do tác giả ký. --code-only không tải weights.

Xem [overall.md](overall.md) để biết cách chạy, dữ liệu đầu vào, artifacts và giới hạn kiểm thử.
