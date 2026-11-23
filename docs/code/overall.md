# Shape-Aware PMC với C2F-Seg

## Phạm vi

Đã thay backend SAMEO đang dở bằng C2F-Seg COCOA. Đây là **shape-aware candidate selection** theo A.4–A.7: không sửa diffusion weights, không dùng oracle/GT để chấm điểm. Code gốc được tách thành các module có comment/docstring tiếng Việt.

Luồng: visible mask ban đầu → C2F prior (cache) → InstaOrder/LaMa chuẩn bị context → SD sinh K ảnh tuần tự → DINO/SAM đo target của từng ảnh → CPU chấm điểm → chọn một ảnh → kiểm tra occlusion và cập nhật canvas một lần. Trên ảnh thật, DINO/SAM và InstaOrder khám phá target/occluder trước bước prior. Evaluation có thể cung cấp observed visible/occluder từ manifest.

## Cài đặt trên server

```bash
bash scripts/setup_server.sh
conda activate pmc
python scripts/download_c2f.py
python main.py --preflight_only --c2f_python /path/to/envs/pmc-c2f/bin/python
python main.py --input_dir /data/images --c2f_python /path/to/envs/pmc-c2f/bin/python
```

Cần cài Conda, driver hỗ trợ CUDA 11.8 và toolchain CUDA/PyTorch phù hợp để build GroundingDINO. Script tạo env mới, không sửa env hiện có. Các checkpoint PMC (DINO, SAM ViT-H, InstaOrder, LaMa) vẫn cần nằm ở đường dẫn CLI; SD tải từ Hugging Face khi chạy đầu tiên hoặc truyền `--sd_model /local/model`.

`environments/pmc.yml` là baseline Python 3.10, Torch 2.1.2/CUDA 11.8, NumPy 1.23.5 cho code cũ. C2F dùng env riêng với cùng Torch, không pip install setup.py upstream vì file đó pin NumPy 1.22.4. `--preflight_only` kiểm tra assets, SHA source/weights, imports worker và CUDA; **không chạy inference thật**. Chưa xác minh toàn bộ môi trường trên server GPU trong workspace này.

## VRAM và RAM

`ModelManager` chỉ cho phép một model PMC trên GPU; model ra khỏi scope được chuyển về CPU. Chính sách mặc định `cpu` giữ weights ở RAM. Dùng `--offload_policy unload` khi RAM hạn chế: tốn thời gian đọc lại checkpoint. C2F chạy subprocess riêng khi manager không có model active; process kết thúc giải phóng weights CUDA. K=4 sinh tuần tự, không batch bốn ảnh. SAM xử lý box theo chunk `--sam_box_batch_size 8`.

LaMa chỉ load generator inference, không load Lightning trainer/evaluator. Đây là adapter cho big-lama chuẩn; noise augmentation bị từ chối rõ ràng. Bộ nhớ peak thực tế vẫn phải đo trên server; không bảo đảm cứng mọi input đều vừa 24 GB.

## Evaluation

Một dòng manifest là một object. Đường dẫn tương đối tính từ vị trí manifest; masks là ảnh grayscale nhị phân (0=background, giá trị >0=foreground). `image` là ảnh đã bị che; `visible_mask`, `occluder_mask` là quan sát đầu vào. `full_mask` và `gt_rgb` chỉ được đọc sau inference. Không dùng annotation amodal để xác định crop predictor.

```json
{"id":"object-1","image":"images/occluded.png","visible_mask":"masks/visible.png","occluder_mask":"masks/occluder.png","category":"cat","full_mask":"gt/full.png","gt_rgb":"gt/unoccluded.png"}
```

`full_mask`/`gt_rgb` không bắt buộc. Không có GT thì vẫn lưu ảnh và runtime, không tự tạo quality metrics. `category` là prompt target.

```bash
# Chạy thử 5 object, mặc định K=4
python scripts/evaluate.py --manifest /data/eval.jsonl --limit 5 \
  --c2f_python /path/to/envs/pmc-c2f/bin/python
# Chạy full, có LPIPS tùy chọn trên CPU
python scripts/evaluate.py --manifest /data/eval.jsonl --full --lpips \
  --c2f_python /path/to/envs/pmc-c2f/bin/python
# Đánh giá prediction có sẵn: thêm prediction_mask và tùy chọn prediction_rgb vào manifest
python scripts/evaluate.py --manifest /data/predictions.jsonl --metrics-only --full
```

Có `--start N`, các cờ pipeline khác được chuyển tiếp. Cờ `--limit` tính theo object, không phải ảnh. Mỗi evaluation có thư mục riêng, không xóa output cũ. Lỗi từng sample được ghi và chạy tiếp; OOM dừng. Có sample thất bại thì exit code 1.

## Metrics và artifacts

Mỗi mẫu có input, visible, prior, prediction mask/RGB, overlay, comparison; từng iteration có K candidates, target masks hợp lệ, prior/visible/valid/editable và selection.json. GT chỉ lưu nếu có. `metrics.csv`, `per_sample.json`, `metrics_summary.json` và `pipeline.jsonl` nằm ở thư mục run.

- Amodal IoU, Dice trên toàn object trong ảnh gốc.
- Hidden IoU: so `prediction & ~visible` với `GT & ~visible`; null nếu GT không có hidden region.
- Visible recall, prior IoU, thời gian/sample.
- Hidden MAE trên RGB [0,1]; PSNR dB null nếu hoàn hảo (có cờ `perfect_hidden_rgb`) hoặc không có hidden.
- `--lpips`: masked LPIPS proxy; ngoài hidden hai ảnh dùng cùng RGB GT. Đây không phải LPIPS được chuẩn hóa theo diện tích hidden.
- Mỗi mean có count. Sample lỗi không bị âm thầm mất: summary có failed/invalid và `amodal_iou_with_failures_zero` tính lỗi có GT là 0. RGB metrics chỉ có khi có output RGB hợp lệ; đọc count khi so sánh.

## Kiểm thử và giới hạn

Đã chạy unittest CPU cho scorer/fallback, valid support, geometry, metrics, model offload khi exception, luồng iteration dùng backend giả và CLI evaluation từ prediction có sẵn. Compile và CLI help chạy được. Chưa chạy weights C2F, DINO, SAM, SD, InstaOrder hoặc LaMa thật vì môi trường local không có Torch/CUDA/checkpoints. Không báo cáo accuracy hoặc VRAM chưa đo.

Nguồn C2F: https://github.com/amazon-science/c2f-seg (Apache-2.0), pin commit `817d656281b92f7f3d457b0937501b5ad162ffcc`. Worker giữ modules/MaskGIT upstream nhưng bỏ DDP, optimizer, loss và GT loader. Crop 2x từ visible là cấu hình deployable, không phải cam kết tái lập đúng benchmark dùng protocol khác.
