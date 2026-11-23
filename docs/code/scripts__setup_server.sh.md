# Cài server

File: `scripts/setup_server.sh`

Tạo env pmc và pmc-c2f, cài editable local diffusers, GroundingDINO và SAM; build DINO cần toolchain CUDA phù hợp. pip check rồi tải C2F. Không dùng requirements.txt cũ chứa pin mâu thuẫn. Nếu đã có env, tự dùng conda env update thay vì chạy create lại.

Xem [overall.md](overall.md) để biết cách chạy, dữ liệu đầu vào, artifacts và giới hạn kiểm thử.
