# Evaluation full hoặc subset

File: `scripts/evaluate.py`

Manifest JSONL một object/dòng. --limit mặc định 5, --full, --start và --metrics-only. Input quan sát tách khỏi GT; chạy process_object trước khi đọc GT. Reproject canvas về F0 bằng placement thật, không tối ưu alignment với GT. Lưu đầy đủ pool/ảnh/CSV/JSON/log; failure không bị âm thầm loại khỏi báo cáo. --lpips chạy CPU, metric masked proxy có mô tả trong overall.

Xem [overall.md](overall.md) để biết cách chạy, dữ liệu đầu vào, artifacts và giới hạn kiểm thử.
