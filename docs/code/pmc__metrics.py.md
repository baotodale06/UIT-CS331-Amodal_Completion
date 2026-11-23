# Metrics độc lập

File: `pmc/metrics.py`

Chỉ evaluator import module này. Kiểm tra mask alignment và GT chứa visible; tính IoU/Dice/hidden IoU, RGB MAE/PSNR. Giá trị không xác định trả None để JSON không có NaN/Infinity; aggregate luôn kèm count.

Xem [overall.md](overall.md) để biết cách chạy, dữ liệu đầu vào, artifacts và giới hạn kiểm thử.
