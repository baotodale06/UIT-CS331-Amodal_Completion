# Worker C2F độc lập

File: `pmc/c2f_worker.py`

Process mới tránh namespace utils/models của InstaOrder. Tạo VQModel, ResNet encoder, MaskedTransformer và refine modules trên CPU; strict load g_model/model/img_encoder/refine; chuyển lên GPU cho inference. Bỏ ImageNet download vì checkpoint có encoder. Rescale input 256² theo upstream, MaskGIT top-k 3/T=3, refine sigmoid, resize xác suất về crop rồi paste vào F0. Seed cố định; không loss/evaluator/DDP/optimizer/GT. --check_environment chỉ kiểm tra imports.

Xem [overall.md](overall.md) để biết cách chạy, dữ liệu đầu vào, artifacts và giới hạn kiểm thử.
