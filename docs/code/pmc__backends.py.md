# Các model PMC

File: `pmc/backends.py`

Factory lazy cho DINO/SAM/InstaOrder/LaMa/SD. Segment cả pool theo DINO trước, SAM sau; SAM chunks boxes. InstaOrder load backbone inference không optimizer; LaMa load generator chuẩn không trainer; SD dùng fork diffusers trong repo, FP16, attention slicing. Mọi tensor output cần cho scorer được chuyển CPU. Lỗi inference được truyền lên, không giấu bằng except trống.

Xem [overall.md](overall.md) để biết cách chạy, dữ liệu đầu vào, artifacts và giới hạn kiểm thử.
