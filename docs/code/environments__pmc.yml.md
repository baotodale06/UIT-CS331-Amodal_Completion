# Environment PMC

File: `environments/pmc.yml`

Pin Python/Torch/NumPy cho compatibility với legacy SAM/DINO/InstaOrder/LaMa và fork diffusers. Torchvision phải cùng Torch. Đây là baseline cần chạy preflight và smoke inference trên server; không phải environment lock đã xác minh mọi dependency.

Xem [overall.md](overall.md) để biết cách chạy, dữ liệu đầu vào, artifacts và giới hạn kiểm thử.
