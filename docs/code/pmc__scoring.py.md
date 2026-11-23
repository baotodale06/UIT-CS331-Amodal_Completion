# Chấm điểm A.6

File: `pmc/scoring.py`

ScoreContext chỉ có prior/visible/valid, không chứa GT. target_mask chọn max IoU với visible, kiểm tra visible recall. Score=0.70 IoU+0.20 visible recall−0.10 leakage, chỉ trong support ảnh gốc. None là invalid. Hòa điểm giữ seed đầu; tất cả invalid chọn candidate 0 và ghi fallback_reason.

Xem [overall.md](overall.md) để biết cách chạy, dữ liệu đầu vào, artifacts và giới hạn kiểm thử.
