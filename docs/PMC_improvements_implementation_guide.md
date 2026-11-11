# Implementation Guide — Hai cải tiến độc lập cho Progressive Mixed Context Diffusion

> **Tài liệu kỹ thuật cho 2 thành viên** — đọc độc lập và có thể triển khai theo từng checklist.  
> **Scope:** COCO pseudo-occlusion, input single RGB image; 1 GPU 24 GB; 4 tuần còn lại đến report.  
> **Hai proposal của nhóm, KHÔNG thuộc paper gốc:**
> - **A / Người 1:** *Shape-Aware PMC* — dùng amodal shape prior để **chọn** candidate có silhouette hợp lý.
> - **B / Người 2:** *Surface-Aware PMC* — dùng appearance của vùng visible để **chọn** candidate có boundary/color/texture nhất quán.
> 
> **Mức bắt buộc:** training-free *candidate reranking* quanh original diffusion; **optional:** finetune model, deep appearance encoders, gradient-guided diffusion, shape refinement.

## 0. Đọc nhanh: thứ tự thực hiện

1. **Cả hai:** duy trì một bản PMC gốc chạy được; tạo COCO pseudo-occlusion kèm ground truth và crop/canvas transform chính xác; tạo K candidate images (K=2 trước, sau đó 4), lưu để dùng chung.
2. **A:** implement `shape_score(candidate_mask, predicted_shape, visible_mask)` và kiểm tra với GT shape dưới dạng **oracle diagnostic**; đưa vào inference một shape predictor **không dùng GT**.
3. **B:** implement `surface_score(candidate_rgb, candidate_mask, observed_rgb, visible_mask, editable_mask)` bằng boundary, color, texture **không dùng GT RGB**.
4. **Cả hai:** cùng candidate pool → first-seed / shape-only / surface-only / joint; chọn score weights trên validation; chạy test duy nhất một lần; báo cáo metrics, failures, runtime.

### Minimal completion criteria

- A phải xuất `scores_A.csv` + ảnh `input / S_pred / candidate masks / selected / GT` + bảng predicted-vs-oracle (oracle đánh dấu rõ **not deployable**).
- B phải xuất `scores_B.csv` + ảnh `input / visible/hidden boundary / candidate samples / selected / GT` + bảng boundary-vs-full surface.
- Cả hai có `predict(...)->idx` hoạt động **không nhận bất kỳ GT path/mask/ảnh nào** khi đánh giá test.

---

# PHẦN I — Ground truth, coordinate frame và điểm can thiệp PMC

## 1. COCO pseudo-occlusion: định nghĩa toán học

Chọn ảnh/instance target gốc có RGB `I_gt ∈ uint8[H,W,3]` và binary instance mask `S_gt ∈ {0,1}[H,W]`. Chỉ giữ targets đủ visible trước overlay; COCO masks **không phải amodal GT mặc định**. Chọn một occluder instance được cắt từ một ảnh khác, hình/texture `I_o`, mask alpha `O ∈ {0,1}[H,W]` sau resize/placement. Composite:

\[
I_{occ}(x,y)=O(x,y)I_o(x,y)+(1-O(x,y))I_{gt}(x,y).
\]

Các mask:

\[
V=S_{gt}\cap\neg O,\quad H_{gt}=S_{gt}\cap O,
\]

\[
r_{occ}=\frac{|H_{gt}|}{|S_{gt}|}.
\]

- `V` = original target pixels **còn nhìn thấy**.
- `H_gt` = vùng target pixels **bị occluder che** (chỉ dùng evaluation).
- `S_gt` = complete target mask **trước pseudo-occlusion** (chỉ dùng train shape predictor/validation/evaluation/oracle).
- `O` = occluder **toàn phần**; thường lớn hơn `H_gt` do một phần che background.
- **Không được dùng `H_gt`, `S_gt`, `I_gt` trong inference của phương pháp non-oracle.**

**Quan trọng:** script `download_dataset.sh` của official PMC chỉ tải các ảnh `train2014`, `val2014`. Dataset builder cần lấy `instances_train2014.json`, `instances_val2014.json` riêng từ bộ COCO annotations. Kết quả COCO-only custom pseudo-occlusion **không tự động** giống bộ benchmark 3,000 mẫu COCO+Open Images của paper (1,500 easy 20–50%, 1,500 hard 50–80%).

### Cách lọc target và dựng split

1. Loại `iscrowd=1`, mask diện tích quá nhỏ, instance chạm biên ảnh, instance hình mảnh không đủ surface visible.
2. Loại target bị che sẵn rõ rệt bằng kiểm tra overlap với instance khác + xem 20–30 mẫu bằng mắt (annotation 2D không đủ chứng minh không bị che trong đời thực).
3. Ưu tiên target diện tích ≥ 5% ảnh, visible area sau overlay ≥ 15% object và hidden fraction 20–80%.
4. Occluder lấy từ ảnh khác; mask overlay alpha phải đúng. Sample seed deterministic. Để đảm bảo không leak, split theo **cả source target image ID và source occluder ID** hoặc dùng pool occluders riêng cho từng split.
5. `train/dev/val/test` không dùng chung original targets; **tune scoring weights trên val**; report test không dùng để điều chỉnh phương pháp.
6. Nên lấy 50/50 easy/hard; **optional** thêm nhiều bins 20–30%, ... 70–80%.

### Kiểm tra unit tests data contract

```python
import numpy as np

def verify_gt_masks(full_mask, visible_mask, occ_mask, hidden_gt):
    s = np.asarray(full_mask, dtype=bool)
    v = np.asarray(visible_mask, dtype=bool)
    o = np.asarray(occ_mask, dtype=bool)
    h = np.asarray(hidden_gt, dtype=bool)
    assert s.ndim == 2
    assert s.shape == v.shape == o.shape == h.shape
    assert np.array_equal(v, s & ~o)
    assert np.array_equal(h, s & o)
    assert not np.any(v & h)
    assert s.sum() > 0
    return float(h.sum() / s.sum())
```

## 2. Những gì có THẬT trong code gốc

Original `main.py` có đường đi:

```text
run_pipeline()
  ├── segment()  [GroundingDINO + SAM]
  └── for each query object:
      run_iteration()
        ├── analyze_masks() + aggregate_occluders()  [InstaOrder]
        ├── create_canvas()  [ở iteration đầu]
        ├── crop()  [canvas -> cropped patch]
        ├── sd_modal_mask = crop_query_mask
        ├── sd_occ_mask = crop_occ_mask + crop_outpaint_mask
        ├── dilate(sd_occ_mask)
        ├── run_mixed_context_diffusion()
        │   ├── swap background
        │   ├── LaMa object-removed background
        │   └── StableDiffusionInpaintMixedContextPipeline
        │       .images[0]
        ├── resize(output, original crop size)
        └── check_occlusion()
            ├── segment() [GroundingDINO + SAM]
            ├── filter_out_amodal_segmentation()
            ├── analyze_masks() [InstaOrder]
            └── update canvas / should run next iteration?
```

- PMC custom diffusion được load bằng `StableDiffusionInpaintMixedContextPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16)` khi `mc_timestep` được bật.
- `main.sh` dùng `--mc_timestep 35`; `run_mixed_context_diffusion()` có `mixed_context_up_ft_indices=[2]`.
- `run_iteration()` sau inpaint resize candidate về `(input_width,input_height)` trước `check_occlusion()`.
- `filter_out_amodal_segmentation(crop_query_mask, amodal_masks)` chọn mask theo overlap với **visible query mask**. Nên **tận dụng cùng quy tắc** khi chọn target candidate, tránh nhầm với occluder.
- `check_occlusion()` làm segmentation + occlusion analysis và cập nhật canvas. Nếu đã segment ứng viên để rerank, có thể cache masks; tuy nhiên **refactor check_occlusion dùng cache là Optional**, dễ gây lệch baseline. Ban đầu chấp nhận segmentation chạy lại.

**Điểm cần lưu ý:** `sd_occ_mask` bao gồm **outpaint region**, không chỉ occluder pixel. Trong score phải phân biệt `editable_mask` và `hidden_gt_mask`. Trên ảnh thật không có hidden GT.

## 3. Hệ tọa độ — phần dễ sai nhất

Có ít nhất bốn frame:

1. `F0`: original image `(H0,W0)`.
2. `Fcanvas`: canvas mở rộng (trong PMC `size_multiplier=6` ở vòng đầu).
3. `Fcrop(t)`: crop vuông ở iteration t, với `row0=crop_x_min`, `row1=crop_x_max`, `col0=crop_y_min`, `col1=crop_y_max`. **Tên `x` trong code PMC được dùng cho chỉ số hàng** (do `np.where`), không phải tọa độ x theo convention ảnh.
4. `Fsd`: image/masks dùng cho Stable Diffusion (thường 512×512; pipeline có thể tự transform). Kết quả `.images[0]` phải được resize về `Fcrop(t)` trước segmentation/scoring nếu scorer dùng crop.

**Quy tắc bắt buộc:** cùng một score thì `I_candidate`, `M_candidate`, `M_visible`, `M_editable`, `M_shape_prior` **phải cùng (h,w), cùng alignment**. Resize masks bằng `NEAREST`, RGB bằng `BILINEAR/BICUBIC`.

### Hàm đặt prior từ original image lên canvas/crop

```python
import numpy as np

def to_centered_canvas(mask_f0, canvas_h, canvas_w):
    """Áp dụng nếu verified create_canvas() đặt ảnh gốc chính giữa.
    KHÔNG dùng nếu code gốc có convention padding khác.
    """
    h, w = mask_f0.shape
    top = (canvas_h - h) // 2
    left = (canvas_w - w) // 2
    out = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    out[top:top+h, left:left+w] = mask_f0.astype(np.uint8)
    return out


def extract_crop(mask_canvas, crop_x_min, crop_x_max,
                 crop_y_min, crop_y_max):
    crop = mask_canvas[crop_x_min:crop_x_max,
                       crop_y_min:crop_y_max]
    return crop.astype(bool)
```

**Tốt nhất** không tự suy đoán padding: so `to_centered_canvas(mask_f0,...)` với `query_obj.init_query_mask_canvas` trên một case, hoặc trực tiếp dùng `create_canvas()` của repo; code ở trên chỉ là unit-testable adapter minh họa.

**Giới hạn bắt buộc của shape prior dự đoán trên ảnh F0:** nếu object mở rộng ngoài original image, shape predictor không biết điều gì nằm ngoài F0. Với COCO pseudo-occlusion project, lọc targets không chạm biên rồi **chỉ score in-frame relevant region**; không phạt phần nằm ngoài F0 như thể có GT shape bằng 0. Full boundary outpainting = **Optional**.

## 4. API chung cho người 1 và người 2

Mỗi candidate phải có thông tin sau (cùng coordinate crop):

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class Candidate:
    image_rgb: np.ndarray      # uint8 [h,w,3]
    object_mask: np.ndarray    # bool  [h,w]; target mask (not all masks)
    seed: int
    iter_id: int
    valid: bool = True

@dataclass
class ScoreContext:
    observed_rgb: np.ndarray   # uint8 [h,w,3], original observed pixels
    visible_mask: np.ndarray   # bool [h,w], original visible target
    editable_mask: np.ndarray  # bool [h,w], occluder + outpaint candidate region
    shape_prior: np.ndarray | None   # bool/float [h,w] predicted only
    prior_valid: np.ndarray | None = None  # bool [h,w] in-frame prior support
```

Hàm scorer:

```python
# Shape: def score_shape(candidate: Candidate, ctx: ScoreContext) -> float|None
# Surface: def score_surface(candidate: Candidate, ctx: ScoreContext) -> float|None
# Higher score = better. Return None if feature not identifiable.
```

**Cấm leakage qua object:** `ScoreContext` không có `gt_rgb`, `full_gt_mask`, `hidden_gt_mask` hay bất cứ GT path nào. Evaluator dùng một `EvalRecord` khác, chỉ chạy **sau** khi `selected_idx` được ghi nhận.

---

# PHẦN II — Người 1: Shape-Aware PMC

## A.1. Research question & hypothesis

> **RQ-A:** Liệu amodal shape prior độc lập có giúp PMC chọn hình hoàn chỉnh hợp lý hơn về hình học, nhất là dưới occlusion lớn, so với việc chọn seed cố định trong cùng tập candidates?

**Hypothesis A:** predicted shape prior tăng **amodal IoU** và giảm geometric failures trong best-of-K PMC. Dự kiến có thể giúp mạnh hơn ở hard occlusion, nhưng đó là điều **cần kiểm chứng**.

**Phân biệt contribution:** Người 1 **không** sửa diffusion weights; người 1 đưa prior vào **candidate selection sau khi sinh**. Đây là *shape-aware selection*, chưa phải *shape-conditioned diffusion*.

## A.2. Pipeline A

```text
I_occ + initial visible target mask V + occluder mask O
                  │
                  ├── shape predictor f_shape ──> S_pred (F0)
                  │                                 │
                  └── original PMC, seeds s1..sK    │
                                      │             │
                                RGB candidates      │
                                      │             │
                          target segmentation       │
                                      │             │
                             candidate masks        │
                                      └──────┬──────┘
                                             ↓
                                      shape scoring
                                             ↓
                                     selected RGB
                                             ↓
                               original check_occlusion()
```

## A.3. Bước 1 — Oracle diagnostic (PHẢI tách khỏi deployable result)

Trên val, dùng `S_gt` làm shape prior để test **upper-bound usefulness**: nếu với shape hoàn hảo mà reranking không chọn ra candidate tốt hơn, dự đoán shape sẽ không giải quyết được.

- Run original PMC K=2/4 candidates, **cache nguyên candidate pool**.
- Segment từng candidate dùng chính detector/SAM PMC.
- Score mask vs `S_gt` đã transform sang Fcrop.
- So với `first-seed` + `random-seed` trong cùng pool; dùng `hidden RGB GT` **chỉ evaluate**, không dùng để chọn candidate.
- Ghi `oracle_shape=true` vào config/results; không cho vào result deployable chính.

**Giả thuyết có thể bị bác bỏ:** shape đúng nhưng texture sai; oracle IoU cao mà masked LPIPS tệ; hoặc candidates đều thiếu chân/thân nên không gì để chọn. Ghi lại các tình huống này.

## A.4. Bước 2 — Shape predictor KHÔNG dùng GT khi test

### Lựa chọn ưu tiên (theo deadline)

1. **Ưu tiên A:** model amodal segmentation pretrained có checkpoint/inference chạy trong 24 GB đã xác minh. API `predict(I_occ, V, O) -> S_pred`. Kiểm tra input expectation: prompt, category, crop, mask; không mặc định mô hình nào cũng nhận 5-channel mask.
2. **Fallback A:** train **lightweight 5-channel U-Net** trên *training split* COCO pseudo-occlusions. Inputs `RGB(3)+V(1)+O(1)` ở 256², output `S_pred(1)` sigmoid. Train 2–10k generated training samples, 5–15 epochs **là điểm bắt đầu để đo**, không bảo đảm accuracy.
3. **Fallback tối thiểu:** mask geometry heuristic (convex hull/ellipse extrapolation từ V) **chỉ được gọi là heuristic non-oracle**, cần chứng minh có ích trên val; giới hạn cho nhóm categories có hình dạng đơn giản. Không tự nhận là amodal segmentation model.

### Tiny U-Net contract (nếu fallback)

\[
X=\operatorname{concat}(I_{occ}/255,V,O)\in\mathbb R^{5\times256\times256}
\]

\[
S_{pred}=\sigma(f_\theta(X))\in[0,1]^{256\times256}
\]

\[
\mathcal L=\mathcal L_{BCE}+\mathcal L_{Dice}+\lambda_v\mathcal L_{visible}
\]

\[
\mathcal L_{visible}=\frac{\sum V(1-S_{pred})}{\sum V+\epsilon}.
\]

- Dùng threshold 0.5 khởi tạo; tune 0.4–0.6 trên validation, **không test**.
- Bắt buộc `S_pred[V]=1` sau binarization hoặc thêm visible-coverage term. Shape prior phải **chứa** visible object.
- Nên cache một `S_pred` **mỗi input object**, reuse cho mọi seed/candidate và không dự đoán lại từ candidate output (tránh feedback circular).
- Tránh overfitting synthetic overlays; đánh giá `IoU(S_pred,S_gt)` riêng theo easy/hard.

### Nguy cơ dự đoán shape nhìn 'quá đúng'

Nếu predicted model đọc các file annotation/GT từ data loader của **test**, hoặc `S_pred` được tạo từ chính `S_gt` trước khi gọi scorer, đó là data leakage. Viết assert API **không nhận GT** và lưu hash/version predictor.

## A.5. Bước 3 — Segment target cho K candidates

Original `check_occlusion()` dùng `segment(gdino_model, run_sam, sam_ckpt,...)`, rồi `filter_out_amodal_segmentation(crop_query_mask, amodal_segmentations)` (chọn theo IoU với vùng modal query).

- Tách một `segment_target_candidate(image_pil, original_crop_query_mask, ...)` wrapper.
- Output `M_k [h,w]` bool; **không** dùng tất cả instance masks chồng nhau.
- Nếu không detect được target / mask rất nhỏ / overlap original visible dưới threshold (ví dụ 0.3 trên val), score `None`, không phải score tốt.
- Nếu candidate segmentation khác với final `check_occlusion` segmentation, chỉ log sai biệt; **Optional** refactor để reuse kết quả.

**Compute note:** segment K candidates khá tốn GPU; cache masks ra PNG, và scorer A/B chỉ đọc cache. Đừng chạy DINO/SAM mới trong từng scorer riêng.

## A.6. Bước 4 — Shape score có kiểm soát

Với candidate target mask `M_k`, prior predicted `S_pred`, và `V`, trên một **valid prior support** `R` (chỉ in-frame nếu shape predictor không hỗ trợ outpaint):

\[
M'_k=M_k\cap R,\quad S'=S_{pred}\cap R,\quad V'=V\cap R.
\]

\[
Q_{IoU}=\frac{|M'_k\cap S'|}{|M'_k\cup S'|+\epsilon},\quad
Q_{vis}=\frac{|M'_k\cap V'|}{|V'|+\epsilon}
\]

\[
L_{leak}=\frac{|M'_k\setminus S'|}{|M'_k|+\epsilon}.
\]

\[
\boxed{Q_A(k)=w_iQ_{IoU}+w_vQ_{vis}-w_lL_{leak}}
\]

Gợi ý bắt đầu `wi=.70,wv=.20,wl=.10`; các giá trị **đề xuất**, chỉ tune trên val. `R` nên là vùng ảnh gốc được đặt vào crop canvas, không phải toàn bộ padding trống. Cần lưu cả component scores.

### Code shape scorer (có thể sử dụng trực tiếp khi cùng coordinates)

```python
import numpy as np


def mask_iou(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    assert a.shape == b.shape and a.ndim == 2
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / max(int(union), 1))


def shape_score(candidate_mask, shape_prior, visible_mask,
                prior_valid=None, wi=.70, wv=.20, wl=.10):
    m = np.asarray(candidate_mask, bool)
    s = np.asarray(shape_prior, bool)
    v = np.asarray(visible_mask, bool)
    assert m.shape == s.shape == v.shape
    r = np.ones_like(m, dtype=bool) if prior_valid is None else np.asarray(prior_valid, bool)
    assert r.shape == m.shape

    m, s, v = m & r, s & r, v & r
    if not m.any() or not v.any():
        return None

    iou = mask_iou(m, s)
    visible_recall = float((m & v).sum() / v.sum())
    leak = float((m & ~s).sum() / m.sum())
    return wi * iou + wv * visible_recall - wl * leak
```

**Cần sửa scorer nếu**: predicted prior rất không chắc chắn, object thin structures, mask target sai, hoặc candidate có geometry tốt nhưng lệch vùng pose hợp lý. Lúc đó có thể thêm shape-boundary Chamfer distance (**Optional**, sau khi IoU baseline chạy).

## A.7. Bước 5 — Chọn candidate và fallback

```python

def choose_best_by_score(candidates, score_fn):
    """score_fn(i, candidate) -> float|None; higher = better."""
    scored = []
    for i, candidate in enumerate(candidates):
        score = score_fn(i, candidate)
        if score is not None and np.isfinite(score):
            scored.append((float(score), i))
    if not scored:
        return 0  # baseline seed; log fallback_reason
    return max(scored)[1]
```

`fallback_idx=0` chỉ an toàn khi candidate 0 là baseline hợp lệ; nếu segmentation candidate 0 hỏng, log `invalid_output` để đánh giá failure rate. Không âm thầm loại sample khỏi denominator.

## A.8. Thí nghiệm A bắt buộc

| Run | K | Prior | Điều cần chứng minh |
|---|---:|---|---|
| Original fixed-seed | 1 | — | reproduction baseline |
| First-of-K control | 4 | — | cùng candidate pool, không score |
| Shape A (predicted) | 4 | Predicted `S_pred` | **deployable contribution** |
| Shape A (oracle) | 4 | **GT `S_gt`** | upper bound, **không phải claim chính** |
| Shape A IoU-only | 4 | Predicted | **Optional:** role visible/leakage terms |

Report: `Amodal mask IoU`, hidden-region `MAE/LPIPS` (khi đủ hạ tầng), `invalid segmentation %`, time/sample, good/bad cases easy/hard. Nếu shape improves mask IoU nhưng hidden RGB không tốt lên, nói đúng như vậy.

---

# PHẦN III — Người 2: Surface/Appearance-Aware PMC

## B.1. Research question & hypothesis

> **RQ-B:** Có thể dùng visible-object surface cues (continuity tại seam, màu sắc và local texture) để ưu tiên kết quả PMC có appearance hidden hợp lý hơn khi giữ nguyên pretrained diffusion không?

**Hypothesis B:** appearance-aware reranking trên cùng K candidate pool giảm seam mismatch và tăng perceptual fidelity/human preference so với fixed-seed/K-control; còn semantic/perceptual gains phải đo chứ không mặc định.

**Lưu ý khoa học:** Visible patch vs hidden patch đôi khi **không cùng màu/texture** (đầu mèo khác thân, mặt áo khác tay áo). Vì vậy surface scorer phải có cơ chế **abstain/fallback** và không được coi toàn bộ object là vật liệu đồng nhất.

## B.2. Pipeline B

```text
Input observed RGB I_occ, visible target V, editable E
                 │
                 ├── extract observed visible RGB patches / features
                 │
                 └── original PMC -> K RGB candidates I_k
                                         │
                                    target mask M_k
                                         │
                                  hidden proposal H_k
                                         │
                    boundary / local color / texture consistency
                                         │
                                       Q_B(k)
                                         │
                             best candidate -> check_occlusion()
```

**Important:** Surface score không dùng GT RGB, không trực tiếp thay noise predict hoặc cross-attention: đây là **post-generation reranking**, hoàn toàn thống nhất triết lý training-free.

## B.3. Bước 1 — Chuẩn bị immutable appearance anchor

`I_obs` phải chứa **ảnh đầu vào gốc** ở các pixels visible, **không phải** bản đã bị synthetic clean-background swap của mixed context và **không phải** RGB visible đã bị sửa qua các vòng diffusion. Nếu t>0, lấy immutable `query_obj.init_img_canvas` + original visible `init_query_mask_canvas` crop về Fcrop(t).

Định nghĩa:

- `V_obs`: vùng target **đã được quan sát thật** ban đầu (trong current crop frame).
- `M_k`: mask target được segment từ candidate RGB.
- `E`: edit mask PMC hiện tại (`sd_occ_mask`, occluder + outpaint, sau dilation).
- `H_k = (M_k ∩ E) \ V_obs`: phần *được sinh và được cho là target*; **không lấy cả E làm hidden object**.

Nếu `H_k` trống, candidate không có nội dung hidden → scorer `None`/invalid; evaluator cần ghi failure.

## B.4. Bước 2 — Hai dải biên lân cận (seam)

Giả sử người đang nhìn thấy bề mặt target V và model sinh hidden H. Tạo hai tập pixels:

\[
B_v=V\cap\operatorname{Dilate}(H,r),\quad
B_h=H\cap\operatorname{Dilate}(V,r).
\]

`r=3–8` trên crop 512² là điểm bắt đầu. Dilation xác định **gần vùng tiếp xúc**, không dùng toàn bộ object. Để giảm nhiễu anti-alias/boundary, có thể bỏ đi pixel sát mép không tin cậy; **Optional**.

### Code tạo band (chạy được với OpenCV)

```python
import cv2
import numpy as np


def seam_bands(visible, hidden, radius=5):
    v = np.asarray(visible, bool)
    h = np.asarray(hidden, bool)
    assert v.shape == h.shape and v.ndim == 2
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1)
    )
    dv = cv2.dilate(v.astype(np.uint8), k).astype(bool)
    dh = cv2.dilate(h.astype(np.uint8), k).astype(bool)
    bv = v & dh
    bh = h & dv
    return bv, bh
```

Nếu visible và hidden không có seam (ví dụ hai bộ phận disconnected, hoặc target detection lỗi), `bv`/`bh` có diện tích quá nhỏ thì **không đo boundary**. Fallback score neutral hoặc dùng color/texture với low confidence; không trả 0 và hiểu nhầm là hoàn hảo.

## B.5. Bước 3 — Boundary color distance

Vì RGB Euclidean distance nhạy ánh sáng, đổi sang Lab (OpenCV 8-bit Lab 0–255) và so **các dải gần seam**:

\[
L_{boundary}=\frac{\|\mu_{Lab}(I_{obs}[B_v])-\mu_{Lab}(I_k[B_h])\|_2}{255\sqrt{3}}.
\]

Dùng `observed_rgb` cho **visible** và `candidate_rgb` cho **hidden**; rất quan trọng vì model có thể thay đổi cả phần visible.

```python

def boundary_color_loss(candidate_rgb, observed_rgb, visible, hidden,
                        radius=5, min_pixels=8):
    assert candidate_rgb.shape == observed_rgb.shape
    bv, bh = seam_bands(visible, hidden, radius)
    if bv.sum() < min_pixels or bh.sum() < min_pixels:
        return None
    a = cv2.cvtColor(observed_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    b = cv2.cvtColor(candidate_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    return float(np.linalg.norm(a[bv].mean(0)-b[bh].mean(0)) / (255.0*np.sqrt(3)))
```

**Cảnh báo:** mean Lab của hai patch có thể không khớp vì shadows/lighting, texture stripes hoặc hai bộ phận khác nhau. Nên xem đây là tín hiệu rẻ nhất, không đủ để kết luận appearance fidelity.

## B.6. Bước 4 — Color-distribution consistency (P1)

Dùng histogram Lab hoặc HSV (ví dụ `H:16 bins`, `S:8`, hoặc per-channel Lab 16 bins) ở `B_v/B_h`; normalize cả hai về xác suất rồi tính Bhattacharyya distance:

\[
L_{color}=D_{Bhat}(\operatorname{hist}(I_{obs}[B_v]),\operatorname{hist}(I_k[B_h])).
\]

Dùng cùng bin edges cho hai vùng. Quan trọng: sample size đủ để histogram không nhiễu. Không dùng histogram toàn object nếu input gồm áo hai màu, động vật với mặt/thân khác màu. *Nếu diện tích band nhỏ, tắt thành phần này*.

```python

def gray_hist(image_rgb, mask, bins=32):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    vals = gray[np.asarray(mask, bool)]
    if len(vals) < 16:
        return None
    hist, _ = np.histogram(vals, bins=bins, range=(0,256))
    hist = hist.astype(np.float64)
    return hist / (hist.sum()+1e-12)


def bhattacharyya_prob_distance(p, q):
    if p is None or q is None:
        return None
    return float(np.sqrt(max(0.0, 1.0 - np.sqrt(p*q).sum())))
```

Code là **grayscale minimal viable reference**; chuyển sang Lab/HSV nếu grayscale không phân biệt màu sắc tốt. Không khẳng định metric này là perceptual color fidelity chuẩn.

## B.7. Bước 5 — Texture-consistency (P1)

### Phương án nhẹ, dễ xong trong 4 tuần: Gradient magnitude + orientation histogram

Trích xuất Sobel gradient của `observed_rgb` trong visible seam và `candidate_rgb` trong hidden seam. Texture ở đây là **thống kê local**, không phải so pixel vị trí đối xứng vì thiếu correspondence thật.

\[
F_{texture}(I,B)=[\mu(|\nabla I|),\ \sigma(|\nabla I|),\ h_{ori}(I,B)].
\]

\[
L_{texture}=\|F_{texture}(I_{obs},B_v)-F_{texture}(I_k,B_h)\|_1.
\]

Implementation gợi ý (simple scalar gradient statistics để MVP):

```python

def grad_stats(image_rgb, mask, min_pixels=16):
    m = np.asarray(mask, bool)
    if m.sum() < min_pixels:
        return None
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)[m]
    return np.array([mag.mean(), mag.std()], dtype=np.float32)


def gradient_texture_loss(candidate_rgb, observed_rgb, visible, hidden, radius=5):
    bv, bh = seam_bands(visible, hidden, radius)
    f1 = grad_stats(observed_rgb, bv)
    f2 = grad_stats(candidate_rgb, bh)
    if f1 is None or f2 is None:
        return None
    return float(np.abs(f1-f2).mean())
```

**Lưu ý:** Sobel trên crop sẽ thu gradient từ background/occluder quanh biên nên có thể sinh artifacts trong thống kê. P1 cải thiện bằng mask erosion 1–2 px, masked-gradient support, hoặc patch-only features. Gradient magnitude mean/std không bắt được hướng sọc nên hãy trung thực gọi đây là **simple texture-statistics proxy**.

### Optional nâng cấp

- HOG/orientation histograms, LBP, Gabor filters.
- DINOv2/v3 patch embeddings: lấy masked local patches cùng semantic part, nearest-neighbor patch features / feature-statistic distance (không chỉ global crop cosine); có thể cần GPU thứ hai hoặc inference theo batch nhỏ.
- Learned scorer hoặc adapter training — **Optional cao rủi ro**, không triển khai trong 4 tuần trừ đã có code.

## B.8. Bước 6 — Ghép loss và chọn candidate

\[
L_{surface}(I_k)=\lambda_bL_{boundary}+\lambda_cL_{color}+\lambda_tL_{texture};\quad Q_B(k)=-L_{surface}(I_k).
\]

**Cực kỳ quan trọng:** ba loss có thang đo khác nhau, cần fit normalizer (median/MAD hoặc mean/std) trên **validation** rồi cố định trên test; đừng cộng score chưa chuẩn hóa và cho weight 'vừa mắt'.

```python

def robust_z(v, median, mad, clip=5.0):
    return float(np.clip((v-median)/(1.4826*max(mad,1e-6)), -clip, clip))


def combine_surface_losses(losses, normalizers, weights):
    """losses: name->float|None; normalizers: name->(median,MAD)."""
    present = [n for n in weights if losses.get(n) is not None]
    if not present:
        return None
    wsum = sum(weights[n] for n in present)
    # Average over available terms; absence is not zero-loss 'perfect score'
    z = sum(weights[n]*robust_z(losses[n], *normalizers[n]) for n in present)
    return float(-z / max(wsum, 1e-12))
```

Dùng `normalizers` fit từ dev/validation **không dùng test labels**. `weights` khởi đầu có thể `boundary=0.6, color=0.1, texture=0.3` (gợi ý tuning, không phải paper hyperparams). Nếu gradient proxy không giúp masked LPIPS/human preference trên val, bỏ nó; ablation cần phản ánh kết quả thật.

**Quan trọng về dữ liệu:** hình ảnh `candidate_rgb` sau mixed-context thường chứa clean synthetic background, nên appearance score **chỉ** áp dụng trên `M_k` và những phần visible/hidden đúng object; so global whole RGB sai ý nghĩa.

## B.9. Experiment B bắt buộc

| Method | K | Selection | Mục tiêu |
|---|---:|---|---|
| Original PMC | 1 | first seed | baseline |
| First-of-K control | 4 | first seed | same K cost control |
| B-boundary | 4 | min boundary loss | test đơn giản nhất |
| B-full surface | 4 | weighted boundary/color/texture | đề xuất chính |
| B-feature ablation — **Optional** | 4 | color-only, texture-only | sensitivity |

Evaluate **masked hidden-region MAE / LPIPS** và human preference hoặc một metric độc lập khác; ghi seam mismatch như **diagnostic only** bởi tối ưu bằng seam score sẽ tự làm nó tốt hơn, không tự chứng minh hidden RGB đúng. Với object surface không đồng nhất, B có thể thua baseline: phải trích failure modes thật.

---

# PHẦN IV — Tích hợp vào original PMC (cùng làm)

## 5. Điểm hook ít rủi ro nhất

Trong original `main.py`, `run_iteration()`:

```python
# Original logic (simplified):
...
amodal_completion = run_mixed_context_diffusion(...)
query_obj.iter_id += 1
amodal_completion = amodal_completion.resize((input_width, input_height))
query_obj.amodal_completion = amodal_completion
query_obj = check_occlusion(amodal_completion, sd_modal_mask, ...)
```

**Thay phần generate 1 sample bằng K samples + select, nhưng giữ `query_obj.iter_id += 1` CHỈ một lần mỗi PMC iteration**. Nếu tăng `iter_id` trong vòng for-seeds, pipeline sẽ thay đổi stopping logic và invalidates comparison. Không chạy `check_occlusion()` trên mỗi candidate nếu function có side effects cập nhật canvas; hãy chỉ chạy `segment_target_candidate()` không side effects để score, rồi gọi `check_occlusion()` một lần sau selection.

### Pseudocode integration (đọc kỹ warning)

```python
# PSEUDOCODE: adjust function signatures to your pinned PMC checkout.
# Candidate generation might require extending custom diffusion pipeline
# with an explicit torch.Generator argument; verify it is accepted.

seed_list = [101, 203, 307, 409][:K]

candidates = []
for seed in seed_list:
    pil_img = generate_with_original_mixed_context(
        sd_img=sd_img,
        sd_modal_mask=sd_modal_mask,
        sd_occ_mask=sd_occ_mask,
        sd_prompt=sd_prompt,
        mc_timestep=mc_timestep,
        seed=seed,
    )
    # After generation, align to crop canvas (not arbitrary 512 pixels)
    candidate_rgb = np.asarray(
        pil_img.resize((input_width, input_height)), dtype=np.uint8
    )
    candidates.append(candidate_rgb)

candidate_masks = []
for candidate_rgb in candidates:
    mask = segment_target_candidate(
        candidate_rgb, crop_query_mask=sd_modal_mask,
        # original GroundingDINO/SAM refs...
    )
    candidate_masks.append(mask)

idx = select_candidate(
    mode=mode, candidates=candidates,
    candidate_masks=candidate_masks,
    visible_mask=visible_mask_in_crop,
    editable_mask=sd_occ_mask,
    observed_rgb=original_observed_rgb_in_crop,
    shape_prior=predicted_shape_in_crop,
    prior_valid=original_image_support_in_crop,
)

amodal_completion = Image.fromarray(candidates[idx])
query_obj.iter_id += 1
query_obj.amodal_completion = amodal_completion
query_obj = check_occlusion(amodal_completion, sd_modal_mask, ...)
```

### Cảnh báo về `generator` và side effects

- Code `run_mixed_context_diffusion()` gốc **không có tham số seed/generator trong signature**. Nhóm cần thêm đối số vào wrapper rồi thử `torch.Generator(device='cuda').manual_seed(seed)` vào custom pipeline call **nếu `__call__` chấp nhận `generator`**. Không được giả định signature Diffusers upstream giống fork của PMC. Nếu không hỗ trợ, dùng `torch.manual_seed(seed)` ngay trước từng call như fallback, log version và kiểm tra deterministic repeat.
- Để tránh K lần LaMa/background prep không cần thiết, **Optional** refactor thành `prepare_mixed_context_inputs()` một lần → diffusion K lần. Làm sau khi correctness pass, vì refactor dễ lệch baseline.
- Mọi `candidate_rgb`/mask đã resize phải cùng crop shape. Nếu scorer compute trên 512 frame thay vì crop thì cần resize *toàn bộ* masks prior/observed tương ứng; không mix hai frame.
- `query_obj.init_query_mask_canvas` là baseline's original visible target canvas; `query_obj.query_mask_canvas` có thể đã chứa content hallucinated ở vòng sau. Surface anchor phải lấy từ **init**, không từ vùng đã hallucinate nếu claim visible fidelity.
- Nếu segmentation no-mask: log invalid; không gọi GT để tìm target mask cho scorer. Fallback first valid candidate hoặc first-seed theo protocol đã đóng băng.

## 6. Cache và fair comparison

**Định dạng cache khuyên dùng:**

```text
cache/
  sample_001/
    iter_00/
      context.json       # prompt, crop coords, original visible/editable masks, SHA
      observed.png
      visible_mask.png
      editable_mask.png
      shape_pred.png      # prediction only; oracle saved in evaluator location
      seed_101.png
      seed_101_mask.png
      seed_203.png
      seed_203_mask.png
      seed_307.png
      seed_307_mask.png
      seed_409.png
      seed_409_mask.png
```

- **Single-iteration controlled comparison:** candidate pool/cache y hệt, offline chọn `first-seed/A/B/joint`; cách này kiểm soát selection improvement rõ nhất.
- **Full progressive integration:** sau mỗi selection, canvas và next iterations thay đổi. Không thể đảm bảo pool của bước tiếp theo giống nhau giữa các trajectories. Hãy báo cáo experiment end-to-end **tách** khỏi fixed-pool study, dùng cùng K, seed schedule, max iterations và preprocessing config.
- **Compute fairness:** K=1 vs K=4 là so sánh baseline deployment speed vs cost; A vs B vs joint nên same K=4. `first-of-K` không thật sự dùng K để cải thiện, chỉ là quality-control budget; report time K separate.

## 7. Joint method A+B

Tất cả component scores tính trước rồi normalize trên val:

\[
Q_{joint}(k)=\lambda_A\widetilde{Q}_A(k)+\lambda_B\widetilde{Q}_B(k).
\]

- Start `λA=λB=.5`; tune trên validation, **không test**.
- Chỉ combine khi cả A và B độc lập hoạt động; nếu joint kém, có thể báo cáo conflict và giữ độc lập là hai contributions chính.
- **Optional:** Pareto filtering thay vì linear weighted sum, confidence gating khi prior xấu, or two-stage shape-then-surface reranking.

```python

def select_joint(score_a, score_b, valid_indices, wa=.5, wb=.5):
    # score_a/score_b: already normalized; higher is better
    scored = [(wa*score_a[i] + wb*score_b[i], i)
              for i in valid_indices
              if np.isfinite(score_a[i]) and np.isfinite(score_b[i])]
    return max(scored)[1] if scored else 0
```

---

# PHẦN V — Evaluation đúng và presentation-ready

## 8. Metric implementation & protocol

### 8.1. Shape

\[
\mathrm{IoU}=\frac{|M_{output}\cap S_{gt}|}{|M_{output}\cup S_{gt}|}
\]

Chỉ sau khi đưa `M_output` từ crop/canvas về cùng pixel coordinates với `S_gt`. Nếu output object extends beyond original frame, phải nói rõ evaluation region là **in-frame only** để không phạt vùng không có GT.

### 8.2. Hidden RGB

\[
MAE_{hidden}=\frac{\sum_{x,y,c}H_{gt}(x,y)|I_{output}(x,y,c)-I_{gt}(x,y,c)|}{3|H_{gt}|+\epsilon}.
\]

Lấy `H_gt = S_gt & O` từ test GT trong **evaluator**, không từ candidate scorer. Nếu output có hallucinated pixels ngoài S_gt, dùng shape IoU/false-positive rate riêng thay vì coi hidden-RGB-only metric đã đo đủ.

```python

def masked_mae(output_rgb, gt_rgb, hidden_gt):
    out = np.asarray(output_rgb, np.float32)
    gt = np.asarray(gt_rgb, np.float32)
    m = np.asarray(hidden_gt, bool)
    assert out.shape == gt.shape and out.ndim == 3
    assert out.shape[:2] == m.shape
    if not m.any():
        return None
    return float(np.abs(out[m]-gt[m]).mean())
```

### 8.3. LPIPS / DreamSim / CLIP

- **LPIPS/DreamSim:** cung cấp toàn crop đã align hoặc masked/composited crop theo một protocol thống nhất; không thể chỉ lấy trung bình scalar feature map mà gọi là masked LPIPS nếu library không hỗ trợ.
- **CLIP:** paper đánh giá high-level bằng image–text category similarity, không phải pixel reconstruction. Nếu dùng CLIP theo paper, ghi rõ `CLIP(image, category prompt)`; không gọi nhầm là CLIP similarity giữa candidate và GT.
- **Boundary mismatch:** là diagnostic metric, không thay thế quality metric độc lập.
- **Failure rate:** đếm cả object not detected / no valid generation, không âm thầm drop.
- **Runtime:** total wall clock cả generation/segmentation/scoring (mean/median) + per-phase; seed K và GPU memory.

### 8.4. Minimum tables

**Ablation chính (test):**

| Method | K | Mask IoU ↑ | Hidden MAE ↓ | LPIPS ↓ | Fail% ↓ | Sec/sample ↓ |
|---|---:|---:|---:|---:|---:|---:|
| PMC K=1 | 1 |  |  |  |  |  |
| PMC first-of-K | 4 |  |  |  |  |  |
| A predicted shape | 4 |  |  |  |  |  |
| B surface score | 4 |  |  |  |  |  |
| A+B joint | 4 |  |  |  |  |  |
| A oracle (**upper bound only**) | 4 |  |  |  |  |  |

**Các bảng riêng:** A `oracle vs predicted` / B `boundary only vs boundary+color+texture`; easy 20–50% vs hard 50–80% tối thiểu. Đừng báo nhiều chữ số thập phân nếu N nhỏ.

### 8.5. Qualitative grid chuẩn

```
Input (occluded) | Complete GT | Original PMC | Shape-PMC | Surface-PMC | Joint
                 | GT mask      | predicted mask and visible/hidden boundary when useful
```

- Chọn cả success và failure (≥5 mỗi loại nếu có), ít nhất một case shape thắng surface, một case surface thắng shape, một case all fail.
- Dùng **đúng sample_id/test** để không chỉ chọn ví dụ từ validation.
- Captions giải thích *what changed* và có bằng chứng thị giác, không chỉ “our method better”.

## 9. Unit tests/integration tests bắt buộc

1. `mask_iou`: identical masks = 1; disjoint = 0.
2. `verify_gt_masks`: decomposition S = V ∪ H không overlap.
3. Crop/canvas test: coordinate path `F0 -> Fcanvas -> Fcrop` trùng với PMC output cho 3 mẫu; overlay pixel-level.
4. `select_candidate(first_seed)` trả idx=0.
5. Same cached K candidate output hash across A/B/joint (fixed-pool eval).
6. Shape prior original input, không được rebuild từ candidate/GT trên test.
7. Surface observable RGB anchor bằng `I_occ` và `init_img_canvas`, không từ `gt_rgb` hoặc `sd_img_syn`.
8. Missing segmentation → fallback + failure log, không throw/drop test case.
9. Fixed seeds / `iter_id` tăng **một lần** mỗi PMC iteration, không tăng K lần.
10. Visual 10 images: boundary bands nhìn đúng, không chứa occluder background, GT alignment pixel đúng.

## 10. Debug checklist nhanh

| Hiện tượng | Nguyên nhân thường gặp | Check trước |
|---|---|---|
| Shape score cao nhưng GT IoU rất thấp | F0/Fcrop offset sai; object mask thuộc occluder | overlay masks, target selection |
| A oracle tăng mạnh, A predicted không tăng | shape predictor yếu / `S_pred` threshold sai | `IoU(S_pred,S_gt)` theo occlusion ratio |
| Mọi candidates có shape_score tương tự | sample diversity nhỏ; K seeds không thực sự khác | hash outputs, generator support |
| Boundary loss = None nhiều | no seam, mask lỗi, editable mask không đúng | visualize V, H, bands |
| Surface chỉ sinh đồng màu/trơn | global histogram/texture score over-regularized | weight giảm, patch-specific analysis |
| PMC output bị đổi khi thêm hook | ngẫu nhiên bị tiêu thụ bởi segmentation/LaMa; `iter_id` bị tăng; model state | seed at SD call, `no_grad` and replay |
| LPIPS rất cao nhưng nhìn tốt | crop/GT misaligned hoặc output target relocated | offset transform, image support |
| Vượt VRAM 24 GB | SD + GDINO/SAM + other nets cùng resident | serial inference, unload/cpu-offload, cache |
| Runtime tăng 4–8x | K=4 generations + K segmentations | cache, K=2 ablation, optional refactor segmentation reuse |

---

# PHẦN VI — Kế hoạch hành động cá nhân (chỉ cần làm theo)

## Người 1 — checklist 4 tuần

### Tuần 1
- [ ] Build dataset mask contract; verify GT samples và coordinate transforms.
- [ ] Implement `shape_score()` và unit tests.
- [ ] Nhận cached candidate masks từ Người 2; thử **GT oracle** trên 10–20 samples.

### Tuần 2
- [ ] Run oracle on dev; ghi failures.
- [ ] Implement/select predicted shape strategy **không GT**; chạy `shape_prior_predict()`.
- [ ] Validate predicted shape IoU; chọn threshold/weights trên validation.
- [ ] Xuất shape branch API `select_shape_idx(candidates, ctx)`.

### Tuần 3
- [ ] Chạy shape method test cùng candidate pool.
- [ ] Xuất A predicted vs first-of-K vs A oracle diagnostic.
- [ ] Xuất 5 success/5 failure + hard-vs-easy table.
- [ ] Review code B và kiểm tra label leakage.

### Tuần 4
- [ ] Soạn slides method A, shape prior visualization, oracle gap, geometry failures.
- [ ] Thuyết trình hypothesis, giới hạn của predicted shape, và kết quả thực (kể cả negative).

**Optional A:** mask-boundary Chamfer score; real-world pretrained model swap nếu chưa chắc khả năng cài; shape-constrained second-pass inpainting; 3D/depth priors.

## Người 2 — checklist 4 tuần

### Tuần 1
- [ ] Pin và chạy original PMC; profile GPU/time.
- [ ] Viết `generate_candidates(K,seeds)` + save image/mask metadata.
- [ ] Xây dựng common evaluator (masked MAE + IoU; LPIPS nếu cài kịp).

### Tuần 2
- [ ] Implement `seam_bands()`, boundary Lab scorer, `min_pixels` fallback.
- [ ] Thêm color and simple gradient texture (chỉ nếu boundary chạy tốt).
- [ ] Normalize scores trên val; xuất surface branch API `select_surface_idx()`.
- [ ] Chạy B boundary-only vs B-full validation.

### Tuần 3
- [ ] Chạy B test + joint scoring; runtime/memory.
- [ ] Xuất appearance quantitative + qualitative grids.
- [ ] Review A predicted-shape/GT separation; freeze output CSV/config.

### Tuần 4
- [ ] Soạn slides method B, boundary/texture diagram và ablation.
- [ ] Chuẩn bị câu trả lời: vì sao low seam loss không tương đương true hidden reconstruction?

**Optional B:** DINO patch feature scorer, LBP/Gabor, gradient-guided sampling, learned appearance reward model.

## 11. Một số thí nghiệm không nên làm trong deadline này

- **Không:** train Stable Diffusion hoặc Wan/world model mới.
- **Không:** sửa mixed-context UNet latent sampling trước khi K-candidate scorer có measured results.
- **Không:** lấy GT full mask làm hidden inference prior rồi ghi đó là deployable enhancement.
- **Không:** tune weights bằng test results.
- **Không:** chỉ so improvement K=4 với baseline K=1 mà bỏ same-K controls.
- **Không:** claim reranking là “shape-conditioned diffusion” hoặc “texture-guided denoising”; đó là một cơ chế khác.
- **Optional:** larger-scale 3k dataset, open-world generalization, advanced statistics, latent-space gradient loss, full image boundary outpainting.

## 12. Deliverables trước ngày report

```text
submission/
  README.md
  dataset_protocol.md             # how built, split IDs, samples, limitations
  baseline_config.yaml
  shape_config.yaml
  surface_config.yaml
  joint_config.yaml
  results.csv                     # every sample/method, all metrics, error code
  summary_tables.md
  figures/
    pipeline_baseline.png
    pipeline_shape.png
    pipeline_surface.png
    qualitative_grid_*.png
    ablation_easy_hard.png
  slides/                          # pptx or pdf if assignment requires
  code_ref.txt                    # Git SHA + env versions
```

## 13. Nguồn gốc đã xác minh và mức độ tin cậy

- **PMC paper (CVPR 2024):** https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Amodal_Completion_via_Progressive_Mixed_Context_Diffusion_CVPR_2024_paper.html
- **PMC arXiv:** https://arxiv.org/abs/2312.15540
- **Original official code:** https://github.com/k8xu/amodal
- **Original `main.py`:** https://github.com/k8xu/amodal/blob/main/main.py — xác minh `run_iteration`, `check_occlusion`, `filter_out_amodal_segmentation`, mixed-context call và sử dụng Stable Diffusion 2 Inpainting.
- **Original `main.sh`:** https://github.com/k8xu/amodal/blob/main/main.sh — `--mc_timestep 35`, input `dataset/val2014`.
- **Original `download_dataset.sh`:** https://github.com/k8xu/amodal/blob/main/download_dataset.sh — tải ảnh COCO, không phải annotations hoặc full 3,000 pseudo-occlusion dataset.

**Tuyên bố ranh giới nguồn:** Tất cả shape/surface scores, masks conventions, việc dùng K candidates, train shape U-Net, thresholds, pseudocode selector, ablations và kế hoạch 4 tuần là **đề xuất kỹ thuật của nhóm**, chưa có benchmark thực nghiệm được xác nhận. Các đoạn code ghi “pseudocode” chỉ minh họa integration; những hàm Python riêng lẻ ghi là runnable cần dependencies thường dùng (NumPy/OpenCV) và input đúng shape.
