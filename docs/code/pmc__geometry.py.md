# Canvas và crop

File: `pmc/geometry.py`

Kế thừa quy tắc PMC. x trong tên legacy là hàng, y là cột. Canvas giữ dtype, crop bounds không âm, masks không mutate khi lấy biên. Dùng cùng bounds cho RGB/prior/visible/support; compute_offset giữ quy tắc output cũ.

Xem [overall.md](overall.md) để biết cách chạy, dữ liệu đầu vào, artifacts và giới hạn kiểm thử.
