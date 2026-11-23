# Logging và model lifecycle

File: `pmc/runtime.py`

JSONL có time UTC, stage, stage_id, object/image/seed context, thời gian và peak allocated/reserved VRAM. ModelManager.use cấm nested inference; finally reset SAM image embeddings, đưa weights về CPU, gc rồi empty_cache. policy unload bỏ cache weights; factory phải khởi tạo trên CPU.

Xem [overall.md](overall.md) để biết cách chạy, dữ liệu đầu vào, artifacts và giới hạn kiểm thử.
