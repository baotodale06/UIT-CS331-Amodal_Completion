# Kế hoạch 4 tuần — Reproduce & Improve Progressive Mixed Context Diffusion (PMC)

> **Dành cho:** 2 thành viên, 1 GPU 24 GB VRAM; CS331 Advanced Computer Vision  
> **Paper:** *Amodal Completion via Progressive Mixed Context Diffusion*, Xu, Zhang & Shi, CVPR 2024.  
> **Mục tiêu:** Báo cáo giảng viên sau 4 tuần, có baseline đã chạy, **mỗi người một cải tiến riêng**, ablation công bằng, qualitative results và slide.  
> **Ngày lập kế hoạch:** 21/10/2026. Lịch bên dưới giả định W1 = 21–27/10; W2 = 28/10–03/11; W3 = 04–10/11; W4 = 11–17/11. Dời ngày tương ứng nếu thời hạn báo cáo thực tế khác.  
> **Điểm xuất phát:** Cả hai **đã đọc và hiểu baseline**, nên **không** phân bổ lại một tuần để đọc paper. Chưa giả định baseline đã chạy end-to-end hoặc benchmark đã có số liệu.

## 0. Quy tắc ưu tiên — đọc trước khi bắt đầu

- **P0 / Bắt buộc:** Code baseline chạy; COCO pseudo-occlusion + GT chuẩn; candidate caching; **Shape-PMC**, **Surface-PMC** có output độc lập; evaluation công bằng; ablation; slide và rehearsal.
- **P1 / Nên có:** Predicted-shape (non-oracle) đủ để công bố kết quả chính; kết quả easy/hard; failure cases; runtime/VRAM; ít nhất một ablation riêng cho mỗi người.
- **P2 / Optional:** DINO/CLIP patch feature sâu; fine-tuning U-Net lớn; shape-refinement vòng thứ hai; gradient-guided diffusion; open-world dataset; full 3,000-image benchmark; training một diffusion mới; video completion.
- **Freeze scope:** Sau **ngày 7 của W2**, ngừng đề xuất cơ chế mới; sau **ngày 3 của W3**, ngừng thay scorer trừ sửa lỗi; W4 tập trung chứng minh, viết và trình bày.

**Câu chuyện nghiên cứu:** PMC tạo hình bằng context + mixed-context diffusion. Người 1 nghiên cứu **shape-aware candidate selection**. Người 2 nghiên cứu **surface/appearance-aware candidate selection**. Kết hợp cả hai là thí nghiệm phụ chung, **không thay cho hai đóng góp độc lập**.

## 1. Kết quả tối thiểu phải có khi báo cáo (Definition of Done)

1. **Baseline:** chạy bản PMC gốc có ghi SHA commit, environment, tham số, ảnh input/output và failure rate. Không nhất thiết khớp hoàn toàn mọi con số của paper vì protocol/dataset có thể khác.
2. **Dataset:** tập COCO pseudo-occlusion có `input`, `GT full object mask`, `GT RGB`, `visible target mask`, `occluder mask`, `hidden GT mask`, `occlusion_ratio`, `sample_id`; chia dev/val/test **theo source image ID**, không leak.
3. **Improvement A — Người 1:** shape prior scorer độc lập, có so sánh **no-prior K candidates / oracle shape (chỉ upper bound) / predicted shape (để claim deployable)**. Nếu predicted shape chưa ổn, trình bày thành công/thất bại rõ ràng, không giả vờ oracle là kết quả thực.
4. **Improvement B — Người 2:** surface scorer độc lập dùng boundary/color/texture từ visible object; ablation **boundary-only vs full score** tối thiểu.
5. **Same-candidate comparison:** mỗi sample và mỗi iteration, **cùng K candidates, seed và mọi tham số PMC** cho `first-seed`, `shape`, `surface`, `joint`. Giữ K cố định cho đối chứng. Nếu reranking ở từng vòng lặp tạo nhánh khác nhau, chạy **single-iteration controlled test** trước, rồi full-loop end-to-end riêng.
6. **Metrics:** object/hidden-only MAE + ít nhất LPIPS **hoặc** DreamSim nếu cài kịp; amodal mask IoU; runtime và failure rate; qualitative grids. Cần ghi rõ crop, coordinate frame, output alignment và mask area khi tính metric.
7. **Slides:** ~12–16 slide, có baseline, hai hypothesis, hai method diagram, bảng ablation, success + failure, giới hạn và định hướng tương lai. Hai người đều nói rõ phần mình.
8. **Reproducibility:** `README`, file cấu hình, `results.csv`, 10–20 hình cố định, scripts tái chạy, Git commit/tag; không cần nộp weights nếu học phần không yêu cầu.

## 2. Cách chia vai trò để tránh block nhau

| Thành viên | Chịu trách nhiệm chính (owner) | Hỗ trợ cho người còn lại |
|---|---|---|
| **Người 1 / A** | COCO annotation + mask contract; shape prior; shape scorer; shape IoU; shape ablation | Verify coordinate transforms, code review dataset và selection |
| **Người 2 / B** | PMC baseline run; candidate generation/cache; surface scorer; appearance metrics; profiling | Hỗ trợ visualizations, code review integration |
| **Cả hai** | Dataset protocol, experiment matrix, validation split, end-to-end QA, joint scorer, slides | Review ngang tối thiểu 1 PR/người/tuần |

**Một GPU:** A và B không đồng thời chạy CUDA jobs nặng. Thống nhất khung giờ GPU từ đầu mỗi ngày; ưu tiên **inference + candidate cache** chạy theo batch ban đêm, scorer dùng CPU hoặc lightweight GPU. Mọi run có `run_id`, người chạy, git SHA, seed, dataset manifest hash.

## 3. Tập dữ liệu & ngân sách thực nghiệm trong 4 tuần

| Giai đoạn | Số mẫu đề xuất | Mục đích |
|---|---:|---|
| Smoke test | **10** | Test output, object detection, annotation integrity, mask alignment |
| Development | **30–50** | Debug scorer nhanh, nhiều failure modes |
| Validation | **50–100** | Tune threshold và weights; không đưa vào final metrics |
| Test chính | **100–200** (50% easy, 50% hard) | So sánh chính thức; nâng lên nếu runtime cho phép |
| Expanded test | **300–500 — Optional** | Đánh giá ổn định hơn nếu GPU đủ |
| Full paper scale | **3,000 — Optional** | Không bắt buộc với deadline 4 tuần |

**Hardness:** easy = 20–50% target-mask occlusion, hard = 50–80%; theo định nghĩa `|S_GT ∩ O| / |S_GT|`. Đây là tỉ lệ của **target bị che**, không phải `|O| / |image|`.

**Quan trọng về tính khoa học:** paper báo cáo 3,000 pseudo-occluded objects tổng hợp từ **COCO và Open Images**; chỉ dùng COCO và tự tạo pseudo-occlusion là **reproduction-style evaluation**, **không được ghi là exact reproduction of the paper's dataset/protocol**. Repo `download_dataset.sh` tải ảnh COCO 2014 nhưng **không tự cung cấp** COCO instance annotation JSON hoặc script tái tạo nguyên bộ 3,000 ảnh như paper. Với scope 4 tuần, hãy đóng băng dataset tự tạo, công bố quy tắc lọc + seed và báo cáo chênh lệch với paper.

### Dataset contract tối thiểu (`manifest.jsonl`)

```json
{"sample_id":"coco_000001_occ_000017","split":"test","target_image_id":1,"occluder_image_id":17,"category":"cat","input_path":".../input.png","gt_rgb_path":".../gt_rgb.png","target_full_mask_path":".../full_mask.png","visible_mask_path":".../visible_mask.png","occluder_mask_path":".../occ_mask.png","hidden_gt_mask_path":".../hidden_gt.png","occlusion_ratio":0.47,"rng_seed":2026}
```

- `full_mask` là mask target **trước khi thêm occluder**; COCO vốn chỉ có modal instance mask, nên chỉ chọn target đủ nhìn thấy (không sát biên, không có dấu hiệu bị che sẵn; kiểm tra tay subset).
- `gt_rgb` là ảnh **trước overlay**, và chỉ so sánh trong vùng `hidden_gt = full_mask & occ_mask`, không so sánh RGB occluder với hidden target.
- Input deployable chỉ gồm `input` + những mask **dự đoán/được cấp khi inference**; không để path `gt_rgb` hoặc `full_mask` lọt vào object dùng cho predicted-shape/surface selection.
- Thêm JSON annotations COCO riêng, vì chỉ tải ảnh không thể tái tạo instance masks từ annotations.

## 4. WEEK 1 — Khóa baseline + dữ liệu + candidate cache (21–27/10)

**Mục tiêu:** Cuối tuần có **cùng một tập candidate images** để A và B làm hai hướng cải tiến trên dữ liệu thực. Không dành tuần này để đọc lại paper.

| Ngày | Người 1 — Shape/data | Người 2 — Baseline/infra | Bắt buộc phải giao |
|---|---|---|---|
| **D1** | Đọc/ghi data contract; viết checklist chọn full target COCO | Checkout/pin PMC SHA + environment + smoke test 3 ảnh | `experiment_spec.md`, Git branch policy, GPU time slots |
| **D2** | Code tạo mask `S,V,O,H`, occlusion ratio; unit tests | Trace `run_iteration()` → `run_mixed_context_diffusion()` → `check_occlusion()`; lưu trung gian | 10 samples có đủ 6 artifacts; baseline 5–10 outputs |
| **D3** | Tạo 30–50 dev + 50–100 val + 100–200 test; split by source IDs | Làm `candidate_hook`: K=2 rồi K=4, seed tracking, cache output + masks | Frozen `manifest_v1.jsonl`; 1 directory candidates mẫu |
| **D4** | Kiểm tra mask/coordinate overlays, ratio easy/hard; xuất contact sheet | Chạy baseline `first-seed` trên 30–50 dev; kiểm tra VRAM/runtime | Dataset QA: 20 hình kiểm tra bằng mắt; failure log |
| **D5** | Viết `shape_score` interface + oracle mask transform test | Cache candidates K=2/4 cho dev; dùng segmentation cùng logic baseline | `candidates/<sample>/<iteration>/<seed>` + metadata |
| **D6** | Chạy shape oracle scorer 10–20 mẫu | Implement hidden-region MAE + mask IoU + output alignment | First comparable table trên 10–20 cases |
| **D7** | Review implementation B | Review masks A; merge stable branch | **Gate W1:** baseline + dataset + cached candidates + scoring API chạy thông |

**P0 quality gates W1:** (a) mọi mask cùng shape trong một coordinate frame; (b) `H_GT ⊆ S_GT`; (c) `V ∩ H_GT = ∅`; (d) first-seed từ cache trùng output baseline (modulo run nondeterminism); (e) 10/10 samples tính metric không crash.  
**Optional W1:** phục dựng exact 3,000 dataset; tạo Web demo; thử models khác Stable Diffusion.

**Nếu gate thất bại:** giảm K=2 và test=50, ưu tiên sửa dataset/alignment trước khi viết scorer nâng cao.

## 5. WEEK 2 — Hai independent improvements (28/10–03/11)

**Mục tiêu:** **Cả hai người đều có contribution chạy được độc lập** trước khi thử kết hợp. Không chuyển sang gradient-guided diffusion.

| Ngày | Người 1 — A: Shape | Người 2 — B: Surface | Deliverable |
|---|---|---|---|
| **D8** | Implement shape IoU + visible recall + leakage penalty, dùng GT mask **oracle only** | Implement boundary-band detection (visible-side & generated-side), handle empty boundary | Hai scorers có unit tests |
| **D9** | Rerank K=4 trên 30–50 dev; nhìn success/failure | Boundary color distance Lab (masked pixels), test seam visualization | 20 before/after grid mỗi hướng |
| **D10** | Viết predicted shape **không GT**: ưu tiên pretrained amodal model nếu có sẵn hoặc lightweight shape predictor | Thêm color histogram/gradient texture nhẹ; normalize scores | `shape_predictor.py` chạy inference / `surface_score.py` |
| **D11** | Kiểm tra predicted shape IoU và lỗi đối xứng/thiếu bộ phận | So sánh boundary-only, boundary+color, full surface bằng dev | Failures documented, không tuning trên test |
| **D12** | Tune alpha/beta/gamma trên val; giữ oracle upper bound riêng | Tune surface weights trên val; fallback khi seam không hợp lệ | Config đã chọn (đóng băng) |
| **D13** | Chạy A predicted trên 50 val; chuẩn bị branch merge | Chạy B trên 50 val; review result quality | A vs first-seed; B vs first-seed trên **same candidate pool** |
| **D14** | Review B; export A function API | Review A; export B function API | **Gate W2:** `A` và `B` có bảng riêng, mỗi người có 3+ ví dụ tốt/xấu |

**P0:** A có `predicted` hoặc ít nhất implementation GT oracle + báo cáo trung thực về phần pending; B có boundary scorer hoạt động và controlled comparison.  
**P1:** A pretrained predictor / small U-Net đủ chất lượng; B gradient texture và histogram cải thiện so với boundary.  
**Optional W2:** DINOv2/v3 patch features, LoRA, learned cross-object texture critic, train hình từ đầu.

**Decision rule (D11):**
- Nếu `oracle shape` **không hơn** random/fixed K selection trên dev: KHÔNG huấn luyện shape predictor tốn GPU; thử sửa score và segmentation target selection, hoặc thu hẹp claim vào failure analysis. Oracle không chứng minh success tự động.
- Nếu `surface scorer` cải thiện chỉ **seam metric của chính scorer** nhưng không cải thiện hidden-region perceptual metric/human preference: giảm global color, dùng local boundary features; tránh claim cải thiện reconstruction fidelity.
- Nếu cả hai scorer fail: vẫn có bài báo cáo hợp lệ về controlled reproduction + negative results, nhưng dành D12–D14 sửa bug trước khi giảm scope.

## 6. WEEK 3 — Evaluation, controlled ablation và freeze result (04–10/11)

**Mục tiêu:** Hoàn thành **scientific evidence**, không thêm model mới. Đây là tuần quan trọng nhất với giảng viên.

| Ngày | Người 1 | Người 2 | Output |
|---|---|---|---|
| **D15** | Merge shape scorer | Merge surface scorer; build `joint_selector` | 4 modes chạy cùng cached K candidates |
| **D16** | Test robustness shape theo occlusion ratio; shape oracle gap | Run test 100–200 mẫu, xuất all metrics | Per-sample `results.csv`, errors.csv |
| **D17** | Check predicted-shape vs oracle, visualize 10 failed masks | Run surface ablation: boundary-only vs full | 2 bảng riêng cho từng contribution |
| **D18** | Benchmark amodal IoU + shape failures | Benchmark masked LPIPS/MAE + runtime/VRAM | Main ablation table, confidence intervals nếu kịp |
| **D19** | Joint vs A (có bổ sung lợi ích không?) | Joint vs B; cân bằng score, không tune test | Combined comparison, synergy analysis |
| **D20** | Chọn 5 successes + 5 failures, explain cause | Tạo visual grid `Input / GT / PMC / A / B / A+B` | 10–20 slide-ready qualitative figures |
| **D21** | Audit leakage, reproducibility | Audit scripts + source image split | **Gate W3:** freeze code + configs + results, Git tag `report-freeze` |

### Bảng bắt buộc (giữ cùng evaluation/test set)

| Method | K generated | Shape prior | Surface guidance | Amodal IoU ↑ | Hidden MAE ↓ | LPIPS ↓ | Time/sample ↑ |
|---|---:|---|---|---:|---:|---:|---:|
| Original PMC / fixed seed | 1 | — | — | TBD | TBD | TBD | TBD |
| PMC K-control / fixed first seed | 4 | — | — | TBD | TBD | TBD | TBD |
| A: Shape-PMC (**predicted**) | 4 | Predicted | — | TBD | TBD | TBD | TBD |
| B: Surface-PMC | 4 | — | Boundary+texture | TBD | TBD | TBD | TBD |
| A+B: SS-PMC | 4 | Predicted | Boundary+texture | TBD | TBD | TBD | TBD |
| A-oracle (**diagnostic only**) | 4 | **GT mask** | — | TBD | TBD | TBD | TBD |

> `PMC K-control` ở bảng trên là **fixed first-seed, K=4 candidates được tạo và cache nhưng scorer không được quyền chọn**; chất lượng thường tương đương K=1 nếu cùng first seed; khác ở chi phí `generate K`. Có thể bổ sung `random-K choice` và **offline best-of-K using GT** như upper bound — **Optional**. Mọi so sánh A/B/joint phải giữ candidate set như nhau và trình bày đúng chi phí.

**Vấn đề quan trọng:** Rerank từng vòng lặp làm các nhánh có state khác nhau ở vòng sau; vì vậy có **hai đánh giá**: (1) *fixed candidate pool, single iteration* kiểm soát selection; (2) *full progressive pipeline* đo kết quả hệ thống. Không tuyên bố dùng “cùng candidate pool mọi iteration” cho các trajectories đã phân nhánh.

**P0 W3:** bảng chính + bản xuất CSV + ≥10 grids + errors; W3 không được trôi sang W4.  
**Optional W3:** statistical bootstrap 95% CI, category-wise breakdown 55 classes, K=8, full 3k images, expanded human study.

## 7. WEEK 4 — Report + slides + rehearsal (11–17/11)

**Mục tiêu:** Biến kết quả thành câu chuyện khoa học kiểm chứng được. Dự kiến 12–16 slides / 12–15 phút **là giả định**; chỉnh theo rubric và thời lượng giảng viên công bố.

| Ngày | Người 1 | Người 2 | Milestone |
|---|---|---|---|
| **D22** | Viết Methods A (hình + loss/score + oracle gap) | Viết Methods B (feature regions + scores + ablation) | 2 pages technical appendix |
| **D23** | Tạo charts IoU/error vs occlusion | Tạo charts LPIPS/MAE/runtime + qualitative grids | Figures high resolution + captions |
| **D24** | Soạn slides 1–7 (problem, paper, reproducibility, A) | Soạn slides 8–14 (B, results, discussion, limits) | First full deck |
| **D25** | Review nội dung B + phát biểu giới hạn | Review nội dung A + GT leakage claims | Slide content freeze |
| **D26** | Rehearsal #1, sửa câu trả lời về shape oracle | Rehearsal #1, sửa câu trả lời về surface metric/rerank | Timing log + mock Q&A |
| **D27** | Final QA outputs + backup figures | Final QA CSV, scripts, offline copy | Reproducibility package |
| **D28** | Rehearsal #2, dẫn phần A | Rehearsal #2, dẫn phần B + conclusion | **Gate W4:** submission-ready |

### Slide structure (14 slides gợi ý)

1. Title + 1 hình ví dụ trực quan; 2. Task definition / why occlusion hard; 3. PMC original idea & key contributions; 4. Data and pseudo-occlusion protocol; 5. Baseline reproduction + failure cases; 6. Two limitations → two hypotheses; **7–8. Person A: Shape prior method + results**; **9–10. Person B: Surface method + results**; 11. Joint method / ablation; 12. Qualitative comparisons + failure; 13. Runtime, limitations, data leakage controls; 14. Conclusion + future work.

**Slide ít giá trị nên để Optional:** lịch sử diffusion dài; liệt kê mọi package đã cài; thuật toán SAM/GroundingDINO chi tiết; nhiều bảng gần giống nhau; live demo phụ thuộc mạng; 3D/robotics applications không liên quan kết quả.

### Câu hỏi giảng viên rất có thể hỏi

- Cải tiến **được đưa vào bước nào** của code gốc? Có sửa weights của SD không?
- Nếu tăng K=4 thì tăng chất lượng do better sampling hay do scorer?
- Shape oracle có được dùng trên test của predicted method không? Đảm bảo không leak bằng cách nào?
- Làm sao chắc COCO modal mask đại diện complete visible target trước khi overlay?
- Boundary/color consistency có đồng nghĩa appearance đúng không? Chỗ nào scorer bias?
- Mô hình chạy khi **không có GT amodal mask** thế nào? Có fallback nếu shape predictor kém không?
- Lợi ích vs inference time/VRAM có đáng không?
- Đâu là cải tiến **độc lập** của từng thành viên, đâu là nhóm phối hợp?

## 8. Rủi ro và phương án cắt giảm (cut-scope ladder)

| Rủi ro | Dấu hiệu | Biện pháp bắt buộc | Optional cắt trước |
|---|---|---|---|
| PMC dependency / 24 GB OOM | baseline không end-to-end | K=1–2, FP16 nếu tương thích, tuần tự, giải phóng các model; giảm sample count | DINO features, K=8, 3k dataset |
| COCO data không đủ GT | nhiều targets che sẵn | Filter nặng; kiểm tra 20–30 hình; loại trường hợp ambiguous | Thêm object classes mới |
| Có output nhưng không measure được | bbox/shape misalignment | Unit tests cho crop/pad/resize; 5 ví dụ pixel overlay | Thêm metrics phụ |
| Predicted shape quá kém | oracle tốt nhưng predicted IoU thấp | Dùng oracle làm diagnostic và báo cáo limitation; train small predictor hoặc sử dụng baseline pretrained có sẵn | Fine-tune larger amodal model |
| Surface reranking không cải thiện GT | chỉ giảm self-score | Giảm global histogram; ablate boundary/texture; báo cáo failure trung thực | DINO, learned critic |
| Generation K quá chậm | 200 samples không xong | Cache ngay khi chạy; test 50–100, giữ split hard/easy | 500/3000 samples |
| Integrated method không tốt | A+B < max(A,B) | Báo cáo conflict/complementarity; **không ép joint thành thắng** | Retune trên test (không được làm) |

**Hard stop:** Trong 4 tuần này **không** triển khai gradient-guided diffusion, training diffusion từ đầu, 3D shape prediction, hay video completion như thành phần bắt buộc.

## 9. Git / experiment convention

```text
repo/
  vendor/pmc_original/        # pinned git SHA / untouched reference
  data/manifests/             # only paths + metadata; GT inaccessible to test scorer
  src/data/                   # pseudo-occlusion, transforms
  src/pmc_hooks/              # candidate generation, segmentation wrapper
  src/shape/                  # Person A
  src/surface/                # Person B
  src/selectors/              # selection/joint
  src/eval/                   # metrics, alignment
  configs/
  runs/<run_id>/              # log JSON, outputs CSV, images
  figures/
  slides/
  docs/
```

Mỗi experiment cần lưu `method`, `K`, `seeds`, `mc_timestep`, `max_iter_id`, `image IDs`, `shape_model_version`, `score_weights`, `eval_frame`, `GPU`, `time_ms`, `failure_reason`, `git_sha`. Không xóa cache candidates sau khi hoàn thành từng batch.

### Daily coordination (10–15 phút)

1. Hôm qua xong gì (artifact/PR cụ thể)? 2. Hôm nay chạy gì trên GPU? 3. Có blocker ảnh hưởng gate không? 4. Có đẩy thêm vào **Optional** không? 5. Mỗi ngày trước khi nghỉ phải commit/log và kiểm tra **1 qualitative case + 1 quantitative case**.

## 10. Checklist cuối cùng

- [ ] Baseline PMC chạy và pin version/seed.
- [ ] Pseudo-occlusion manifest và dataset splits frozen.
- [ ] GT RGB/GT shape **không** lọt vào predicted inference.
- [ ] K candidates cache + masks + coordinate mapping kiểm tra.
- [ ] Người 1 có Shape-PMC (ít nhất scorer + oracle diagnostic + giải pháp non-oracle).
- [ ] Người 2 có Surface-PMC (boundary scorer và ablation).
- [ ] Bảng A, B, A+B vs same-K controls.
- [ ] Test easy/hard, metrics và failure rate.
- [ ] At least 10 comparative grids, không chỉ cherry-pick.
- [ ] Runtime/VRAM và limitation.
- [ ] Slides + speaker notes + rehearsal 2 lần.
- [ ] Git tag / reproducibility package.

---

## Tài liệu gốc đã kiểm tra (21/10/2026)

1. Paper CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Amodal_Completion_via_Progressive_Mixed_Context_Diffusion_CVPR_2024_paper.html
2. PDF: https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Amodal_Completion_via_Progressive_Mixed_Context_Diffusion_CVPR_2024_paper.pdf
3. Official code: https://github.com/k8xu/amodal
4. `main.py` — `run_iteration`, `run_mixed_context_diffusion`, `check_occlusion`: https://github.com/k8xu/amodal/blob/main/main.py
5. `main.sh` — COCO val2014 input, MC timestep 35: https://github.com/k8xu/amodal/blob/main/main.sh
6. `download_dataset.sh` — COCO train2014/val2014 image ZIP (annotations cần lấy riêng): https://github.com/k8xu/amodal/blob/main/download_dataset.sh

**Phân biệt nguồn:** Các mô tả PMC, cấu trúc code và benchmark 3,000/easy/hard là từ paper/repo; số mẫu, lịch, cải tiến shape/surface, scorer, threshold, các ưu tiên và protocol trong tài liệu này là **đề xuất cho nhóm**, không phải số liệu hoặc thiết kế được paper gốc chứng minh.
