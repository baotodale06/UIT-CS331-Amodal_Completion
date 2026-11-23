# Prior và cache

File: `pmc/shape_predictor.py`

validate_c2f hash source/configs/weights/worker/preprocessing. ShapePredictor cache npz theo nội dung RGB/V/O và identity; không đọc candidate/GT. predict_c2f ghi IPC npz tạm, chạy worker bằng c2f_python và stream stdout/stderr vào JSONL, heartbeat mỗi 30 giây; timeout/exit lỗi được truyền lên. Prior binarize rồi union visible. save_json ghi atomic và serialize Path.

Xem [overall.md](overall.md) để biết cách chạy, dữ liệu đầu vào, artifacts và giới hạn kiểm thử.
