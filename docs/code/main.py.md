# Entry point và vòng lặp PMC

File: `main.py`

parse_args kiểm tra seeds, thresholds, C2F options. preflight kiểm tra assets/worker/CUDA. process_object dự đoán prior một lần, tạo canvas, chạy tối đa max_iter_id. run_iteration sinh K ảnh, lưu pool, segment/scoring CPU, tăng iter_id đúng một lần rồi check_occlusion. Chỉ ảnh được chọn cập nhật canvas. Output mỗi run có config/summary/log; không xóa thư mục cũ.

Xem [overall.md](overall.md) để biết cách chạy, dữ liệu đầu vào, artifacts và giới hạn kiểm thử.
