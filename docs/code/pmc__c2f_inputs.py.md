# Tiền xử lý C2F

File: `pmc/c2f_inputs.py`

visible_crop_bounds tính crop chữ nhật mở rộng từ visible mask, clamp trong ảnh gốc; không nhận full-mask/GT bbox. Expand mặc định 2, tune bằng validation nếu cần.

Xem [overall.md](overall.md) để biết cách chạy, dữ liệu đầu vào, artifacts và giới hạn kiểm thử.
