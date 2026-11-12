# Surface-Consistent PMC: selected source text

Captured on 2026-11-05 from the local `ChatGPT - Assess Reproduction Feasibility.html`.
Original HTML SHA-256: `a398b6d068bd7abb2e7aee9453c6796f2c486d9b5143624b2c0112f624eeed3f`.

Read [the structured research reference](../SURFACE_CONSISTENT_PMC.md) first.
This file preserves the Vietnamese source wording, formulas and code text relevant
to Surface-Consistent PMC, plus shared integration and evaluation context. The
standalone Shape-Guided implementation section and suggested repository tree are
omitted. Tables are flattened text; diagrams and source-site citation widgets are
not reproduced visually. KaTeX equations are extracted from their TeX annotations.
The text below is source material, not repository instructions or executable code.
The original is an AI-generated proposal, not evidence of measured improvement.

Each line range refers to the temporary plain-text extraction, not HTML line numbers.

## Proposal status and project context

Extracted lines 8-66.

````text
Thiết kế chi tiết hai hướng cải tiến cho Progressive Mixed Context Diffusion (PMC)

Với project 8 tuần, 2 người và GPU 24 GB, mình đề xuất phát triển hai phương pháp độc lập, đều giữ nguyên pretrained diffusion model của paper gốc:

 |  | 
Người 1
 | 
Người 2

 | 
Tên phương pháp
 | 
Shape-Guided PMC
 | 
Surface-Consistent PMC

 | 
Vấn đề
 | 
Hình dạng đối tượng được sinh có thể không hợp lý
 | 
Texture, màu sắc, hoa văn của phần sinh có thể không khớp phần nhìn thấy

 | 
Thông tin bổ sung
 | 
Amodal shape prior
 | 
Visible-object appearance features

 | 
Điểm can thiệp
 | 
Chọn candidate và điều khiển quá trình mở rộng object
 | 
Chấm điểm và lựa chọn diffusion candidates

 | 
Phương án khả thi nhất
 | 
Shape-constrained candidate selection
 | 
Appearance-aware candidate reranking

 | 
Hướng nâng cao
 | 
Shape-guided inpainting mask
 | 
Appearance guidance trong diffusion

 | 
Có cần huấn luyện lại diffusion?
 | 
Không
 | 
Không

Đây là hai research proposals của nhóm, không phải hai phương pháp được tác giả PMC công bố.
````

## Original PMC integration context

Extracted lines 68-98.

````text
1. Trước hết, cần hiểu đúng điểm can thiệp trong PMC gốc

Paper Amodal Completion via Progressive Mixed Context Diffusion không đơn giản thực hiện:
 $\text{Predict amodal mask}\rightarrow\text{Inpaint RGB}.$ 
Ngược lại, một đóng góp của paper chính là tránh phụ thuộc vào việc phải dự đoán amodal mask chính xác trước khi tạo sinh. Nó mở rộng vùng xử lý ra ngoài object bounding box, thực hiện Mixed Context Diffusion, phân đoạn kết quả, kiểm tra xem object còn bị che hay chạm biên ảnh không, rồi tiếp tục vòng lặp khi cần. Open Access CVF+1
k8xu.github.io
Từ official implementation, có ba vị trí đáng chú ý:

Input image + visible mask + occluder mask

1. run_iteration()

Chuẩn bị crop, mask, padding và gọi diffusion

2. run_mixed_context_diffusion()

Sinh ảnh hoàn chỉnh bằng pretrained Stable Diffusion 2 Inpainting

Người 1 và 2 đều có thể tạo nhiều candidates tại đây

3. Candidate scoring + check_occlusion()

Đánh giá kết quả, chọn candidate, phân đoạn và quyết định lặp tiếp

Điểm tích hợp hai cải tiến ít rủi ro nhất

Completed RGB object + amodal mask

Trong code gốc, run_iteration() gọi run_mixed_context_diffusion() để lấy một completion, sau đó check_occlusion() dùng segmentation và occlusion analysis để cập nhật trạng thái vòng lặp. Mixed Context được thực hiện bằng một custom Stable Diffusion inpainting pipeline, với mặc định trong main.sh là --mc_timestep 35. GitHub+2

Ý tưởng quan trọng nhất: Trước tiên giữ nguyên phần tạo sinh của PMC, chỉ thay đổi cách lựa chọn kết quả. Sau khi chứng minh có cải thiện mới can thiệp sâu vào diffusion.
````

## Surface-Consistent PMC: complete source section 3

Extracted lines 388-698.

````text
3. Người 2 — Surface-Consistent PMC

3.1. Research question và limitation

Research question:

Can appearance features extracted from the visible target object guide diffusion candidate selection toward more visually consistent hidden-region reconstruction?

Đây là hướng đặc biệt thú vị vì nó tập trung vào phần mà shape prior không giải quyết được.

Ví dụ đối tượng là một chiếc áo sọc đỏ-trắng. Diffusion có thể sinh đúng hình dạng chiếc áo, nhưng phần bị che lại trở thành áo trơn màu đỏ, hoặc các sọc bị lệch.

Như vậy:
 $\text{Correct shape}
\not\Rightarrow
\text{Correct appearance}$ 
PMC sử dụng visible object trong quá trình inpainting, nhưng điều đó không có nghĩa phương pháp đã có một cơ chế chấm điểm riêng cho tính nhất quán texture/surface giữa visible và hidden regions.

3.2. Ý tưởng cốt lõi

Người 2 sẽ xây dựng một appearance consistency scorer:
 $Q_{\text{surface}}(I_k,I_{vis})$ 
Nó có nhiệm vụ xác định candidate nào có phần hidden content phù hợp nhất với những thông tin appearance mà ta đã nhìn thấy.

Phân biệt với Người 1:

- 
Người 1 nhìn vào object silhouette / geometry.

- 
Người 2 nhìn vào color, texture, pattern và transition giữa hai vùng.

Một thiết kế khả thi:
 $Q_{\text{surface}}
=
-\lambda_1L_{\text{boundary}}
-\lambda_2L_{\text{color}}
-\lambda_3L_{\text{texture}}$ 
Candidate có score lớn nhất sẽ được chọn.

3.3. Implementation chi tiết

Bước B1 — Xác định visible và generated hidden regions

Giả sử sau khi segment một candidate ta có predicted object mask  $M_k$ .

Phần object được quan sát ban đầu:
 $V=M_{visible}$ 
Phần object mới được khôi phục trong occlusion region:
 $H_k=(M_k\cap O)\setminus V$ 
Trong trường hợp có outpainting, vùng mới sinh ngoài ảnh cũng cần được đưa vào  $H_k$  sau khi căn chỉnh canvas.

Lưu ý: không lấy toàn bộ occluder mask làm hidden object, vì một phần occluder có thể che background chứ không che target.

Bước B2 — Boundary consistency

Đây là thành phần mình khuyên implement đầu tiên.

Ta tập trung vào ranh giới giữa visible object và hidden region được sinh.

Định nghĩa:
 $B_v=\operatorname{Dilate}(H_k,r)\cap V$  $B_h=\operatorname{Dilate}(V,r)\cap H_k$ 
Trong đó  $r$  là độ rộng của dải biên, ví dụ 3–8 pixels trong crop 512×512.

Minh họa các vùng cần so sánh
Hai dải sát ranh giới được dùng để kiểm tra tính liên tục về appearance.
Đổi ảnh RGB sang CIELAB để so sánh màu sắc gần hơn với cảm nhận thị giác.

Với hai dải  $B_v,B_h$ , ta có thể tính:
 $L_{\text{boundary}}
=
\frac{
\|\mu_{\text{Lab}}(B_v)-\mu_{\text{Lab}}(B_h)\|_2
}{C}$ 
Trong đó  $C$  là hệ số chuẩn hóa.

Code tối thiểu:
Python

import cv2
import numpy as np

def boundary_color_loss(
    candidate_rgb,
    visible_mask,
    hidden_mask,
    radius=5,
):
    v = visible_mask.astype(np.uint8)
    h = hidden_mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )

    visible_band = (cv2.dilate(h, kernel) > 0) & (v > 0)
    hidden_band = (cv2.dilate(v, kernel) > 0) & (h > 0)

    if not visible_band.any() or not hidden_band.any():
        return None

    lab = cv2.cvtColor(
        candidate_rgb.astype(np.uint8),
        cv2.COLOR_RGB2LAB,
    ).astype(np.float32)

    mean_visible = lab[visible_band].mean(axis=0)
    mean_hidden = lab[hidden_band].mean(axis=0)

    # Normalized Lab color distance
    return float(
        np.linalg.norm(mean_visible - mean_hidden)
        / (np.sqrt(3) * 255.0)
    )

Đây là baseline scorer, chưa phải một loss hoàn chỉnh cho surface consistency.

Hạn chế của cách so sánh trung bình màu là nó có thể phạt nhầm những thay đổi appearance tự nhiên, ví dụ áo và quần có màu khác nhau, hoặc vùng biên nằm ở chỗ có họa tiết.

Vì vậy nên cải tiến thêm bằng local patch correspondence: so sánh các patch ở hai phía ranh giới, thay vì toàn bộ hai vùng, hoặc chỉ áp dụng boundary loss khi có bằng chứng hai vùng thuộc cùng một bề mặt.

Bước B3 — Color distribution consistency

Thay vì chỉ so sánh màu trung bình, ta lấy histogram màu trong vùng visible và vùng được sinh:
 $h_v=\operatorname{Hist}(I,V)$  $h_h=\operatorname{Hist}(I_k,H_k)$ 
Sau đó:
 $L_{\text{color}}=D_{\text{hist}}(h_v,h_h)$ 
Có thể dùng Bhattacharyya distance, Jensen–Shannon divergence hoặc Wasserstein distance.

Ví dụ:
Python
loss_color = cv2.compareHist(
    hist_visible,
    hist_hidden,
    cv2.HISTCMP_BHATTACHARYYA,
)

Nên dùng histogram trong Lab hoặc HSV, normalize histogram trước khi so sánh.

Điểm yếu: màu của những bộ phận khác nhau trên cùng object không nhất thiết giống nhau. Vì vậy histogram nên là một tín hiệu phụ, không phải hard constraint.

Bước B4 — Texture / pattern consistency

Đây là phần có tiềm năng tạo contribution rõ hơn.

Màu sắc chưa đủ để mô tả texture. Hai bề mặt có thể cùng màu đỏ nhưng một bên có sọc, bên còn lại trơn.

Có ba mức độ implementation:

Mức 1 — Handcrafted texture features

Dùng Sobel gradients, Local Binary Patterns (LBP), Gabor filters hoặc gradient-orientation histograms.
 $L_{\text{texture}}=
D(T(I_{vis}),T(I_{hidden}))$ 
Rất nhẹ, dễ debug, phù hợp để xây dựng baseline.

Mức 2 — Deep visual features

Dùng pretrained DINOv2/DINOv3 hoặc VGG để trích xuất patch-level features rồi so sánh bằng cosine distance, nearest-neighbor patch similarity hoặc feature statistics.
 $F_v=E(I_{vis}),\quad F_h=E(I_{hidden})$ 
Chú ý cần masked patch features và spatial alignment phù hợp. Không nên chỉ so sánh cosine giữa embedding của hai crop toàn cục, vì các bộ phận khác nhau có thể khác nhau về ngữ nghĩa.

Mức 3 — Learned appearance compatibility scorer

Train một network nhỏ nhận các patch visible/hidden để dự đoán mức độ nhất quán appearance.

Hướng này có thể mạnh hơn nhưng cần thêm dữ liệu, training và negative-pair design. Không phải ưu tiên cho project 8 tuần.

Mình khuyên Người 2 chọn Mức 1 + một phần Mức 2. Điều này tạo được hệ thống tương đối mạnh mà không phụ thuộc vào training pipeline mới.

Bước B5 — Tổng hợp scores

Có thể định nghĩa loss:
 $\mathcal L_{\text{surface}}(I_k)
=
\lambda_b L_{\text{boundary}}(I_k)
+
\lambda_c L_{\text{color}}(I_k)
+
\lambda_t L_{\text{texture}}(I_k)$ 
Chọn candidate có loss thấp nhất:
 $\boxed{k^*=\arg\min_k\mathcal L_{\text{surface}}(I_k)}$ 
Skeleton:
Python

def select_by_surface(
    candidates,
    candidate_masks,
    visible_mask,
    occlusion_mask,
):
    scores = []

    for image, object_mask in zip(
        candidates, candidate_masks
    ):
        hidden_mask = (
            object_mask.astype(bool)
            & occlusion_mask.astype(bool)
            & ~visible_mask.astype(bool)
        )

        boundary = boundary_color_loss(
            image,
            visible_mask,
            hidden_mask,
        )

        if boundary is None:
            scores.append(float("inf"))
            continue

        color = color_histogram_loss(
            image, visible_mask, hidden_mask
        )

        texture = texture_feature_loss(
            image, visible_mask, hidden_mask
        )

        loss = (
            0.40 * boundary
            + 0.25 * color
            + 0.35 * texture
        )

        scores.append(loss)

    return int(np.argmin(scores))

Các hàm color_histogram_loss() và texture_feature_loss() cần được nhóm implement; code trên thể hiện giao diện tích hợp. Trước khi cộng, ba loss phải được normalize trên validation vì chúng có scale khác nhau. Các trọng số chỉ là giá trị khởi tạo để thử nghiệm.

Mình cũng khuyên lưu từng thành phần score riêng trong CSV để sau này phân tích:
Plain textsample_id, seed, boundary_loss, color_loss,
texture_loss, total_loss, selected
3.4. Phiên bản nâng cao: Surface-Guided Diffusion

Trong phiên bản reranking:
 $\text{Diffusion}\rightarrow K\text{ outputs}\rightarrow\text{Select}$ 
Surface features chỉ được dùng sau khi diffusion hoàn thành.

Một cải tiến sâu hơn là đưa appearance information vào chính quá trình denoising:
 $\epsilon_{\text{guided}}
=
\epsilon_\theta(z_t,t,c)
-
\eta_t\nabla_{z_t}\mathcal L_{\text{surface}}$ 
Đây là dạng conceptual gradient guidance, không phải một thay đổi có thể áp dụng trực tiếp cho custom PMC pipeline mà không sửa scheduler/denoising logic.

Với hướng này, ta cần:

- 
Tạo appearance loss differentiable trong latent hoặc decoded image space.

- 
Chỉ áp dụng gradient lên vùng có thể sinh.

- 
Giữ nguyên visible region.

- 
Đảm bảo mixed-context compositing và scheduler vẫn hoạt động.

- 
Kiểm tra chi phí VRAM và gradient stability.

Ngoài ra có thể nghiên cứu reference-image conditioning qua image adapter, nhưng việc tích hợp adapter vào custom inpainting pipeline của PMC có thể cần sửa UNet attention processors.

Đây là extension có độ rủi ro cao hơn nhiều so với reranking. Với thời lượng 8 tuần, không nên bắt đầu từ hướng này.

3.5. Experiments riêng cho Người 2

 | 
Experiment
 | 
Mục đích

 | 
PMC baseline, K=1
 | 
Mốc ban đầu

 | 
PMC random candidate, K=4
 | 
Đối chứng cùng ngân sách sampling

 | 
PMC + boundary score
 | 
Kiểm tra đóng góp của seam consistency

 | 
PMC + color + boundary
 | 
Kiểm tra consistency của màu

 | 
PMC + texture + boundary
 | 
Kiểm tra texture guidance

 | 
PMC + full surface scorer
 | 
Phương pháp chính

Metrics: hidden-region LPIPS/DreamSim, masked color/texture discrepancy, tỷ lệ seam artifacts và một tập human preference evaluation.

Tiêu chí thành công: phương pháp nâng chất lượng appearance của hidden region so với control cùng số candidates, đồng thời không làm giảm shape plausibility.
````

## Joint selection and integration context

Extracted lines 700-799.

````text
4. Kết hợp hai cải tiến thành một phương pháp hoàn chỉnh

Sau khi mỗi người đã có phương pháp riêng, nhóm có thể xây dựng:

Shape-and-Surface-Aware Progressive Mixed Context Diffusion (SS-PMC) — tên đề xuất của nhóm.

Ý tưởng là tạo  $K$  candidates, rồi đánh giá mỗi candidate trên cả hai khía cạnh hình học và appearance.

Occluded image + visible mask + occluder mask

Shape predictor

Người 1

 $\hat S$ 

Appearance feature extractor

Người 2

 $F_{\text{vis}}$ 

Original PMC — K candidates

Giữ pretrained Stable Diffusion + Mixed Context sampling

Shape score
 $Q_s(I_k,\hat S)$ 
Surface score
 $Q_a(I_k,F_{\text{vis}})$ 
Joint candidate selection
 $k^*=\arg\max_k[\lambda_sQ_s+\lambda_aQ_a]$ 
Progressive occlusion check → Final completion

Với  $Q_a=-\mathcal L_{\text{surface}}$ , score tổng hợp có thể viết:
 $Q_{\text{joint}}(I_k)
=
\lambda_s\widetilde Q_{\text{shape}}(I_k)
-
\lambda_a\widetilde{\mathcal L}_{\text{surface}}(I_k)$ 
Dấu ngã thể hiện score đã được chuẩn hóa trên validation set.

Tích hợp vào code gốc như thế nào?

Trong file main.py của PMC, run_iteration() hiện tạo một amodal_completion rồi gọi check_occlusion().

Thay đổi tối thiểu là thêm candidate selection giữa hai bước đó:
Python

# Pseudocode for modified run_iteration()

# 1. Prepare inputs using original PMC logic
# sd_img, sd_modal_mask, sd_occ_mask, sd_prompt

# 2. Generate K candidates using the original
#    Mixed Context Diffusion implementation
candidates = generate_pmc_candidates(
    sd_img=sd_img,
    sd_modal_mask=sd_modal_mask,
    sd_occ_mask=sd_occ_mask,
    sd_prompt=sd_prompt,
    num_candidates=4,
)

# 3. Obtain object masks for candidate scoring
candidate_masks = [
    segment_target(image, query_class)
    for image in candidates
]

# 4. Choose one of four experiment modes
if mode == "baseline":
    selected_idx = 0

elif mode == "shape":
    selected_idx = select_by_shape(
        candidate_masks, shape_prior, visible_mask
    )

elif mode == "surface":
    selected_idx = select_by_surface(
        candidates, candidate_masks,
        visible_mask, occlusion_mask
    )

elif mode == "combined":
    selected_idx = select_by_joint_score(
        candidates, candidate_masks,
        shape_prior, visible_mask, occlusion_mask
    )

amodal_completion = candidates[selected_idx]

# 5. Continue the original PMC iteration logic
query_obj = check_occlusion(
    amodal_completion,
    # ... original arguments
)

Đoạn trên là pseudocode kiến trúc, chưa phải patch chạy trực tiếp. Nhóm cần triển khai các wrapper và đảm bảo generator/seed được truyền đến custom diffusion pipeline. Trong bản gốc, nơi gọi model là run_mixed_context_diffusion(); đây là chỗ nên sửa để tạo nhiều samples, thay vì viết lại toàn bộ sampling algorithm. GitHub+1
````

## Experimental design and metrics

Extracted lines 829-976.

````text
5. Thiết kế experiments để chứng minh từng contribution

Điều mình đặc biệt muốn nhóm chú ý là không được nhầm lợi ích từ việc sinh nhiều candidates với lợi ích từ improvement.

Ví dụ PMC gốc chỉ sinh một hình, nhưng Shape-PMC sinh bốn hình rồi chọn hình tốt nhất. Nếu kết quả tốt lên, nguyên nhân có thể chỉ là đã dùng gấp bốn lần số samples.

Vì vậy, cần hai nhóm so sánh.

Main ablation

 | 
Method
 | 
K
 | 
Shape prior
 | 
Surface score

 | 
Original PMC
 | 
1
 | 
—
 | 
—

 | 
PMC single-candidate control
 | 
4 generated, chọn seed cố định
 | 
—
 | 
—

 | 
Shape-PMC
 | 
4
 | 
✓
 | 
—

 | 
Surface-PMC
 | 
4
 | 
—
 | 
✓

 | 
SS-PMC
 | 
4
 | 
✓
 | 
✓

Với cùng từng image, phải sử dụng cùng bốn seeds ở ba phương pháp cải tiến, để candidate pool giống nhau.

Shape oracle ablation

 | 
Method
 | 
Shape source

 | 
Original PMC
 | 
None

 | 
Shape-PMC (predicted)
 | 
Predicted amodal mask

 | 
Shape-PMC (oracle)
 | 
Ground-truth full mask

Oracle result chỉ dùng để phân tích upper bound; không đưa vào headline result của hệ thống deployable.

Metrics

 | 
Metric
 | 
Ý nghĩa

 | 
Amodal IoU ↑
 | 
Predicted object geometry có khớp full mask không?

 | 
Hidden-region MAE ↓
 | 
Sai khác RGB trên vùng target bị che

 | 
Hidden-region LPIPS ↓
 | 
Sai khác perceptual appearance

 | 
DreamSim ↓
 | 
Perceptual/semantic similarity

 | 
CLIP similarity ↑
 | 
Mức độ nhất quán ngữ nghĩa

 | 
Boundary discrepancy ↓
 | 
Mức độ khác biệt appearance tại seam

 | 
Inference time ↓
 | 
Chi phí tạo và chọn candidates

LPIPS và DreamSim cần tính trên vùng crop đã căn chỉnh và có quy tắc masked evaluation nhất quán. Không nên chỉ lấy metric trên toàn ảnh vì background có thể lấn át chất lượng khôi phục.

Với pseudo-occlusion, hidden-region GT là những pixel của target trước khi bị occluder che:
 $H_{GT}=S_{GT}\cap O$ 
Ví dụ masked MAE:
 $\operatorname{MAE}_{hidden}
=
\frac{
\sum_{x,y,c}H_{GT}(x,y)
|\hat I(x,y,c)-I_{GT}(x,y,c)|
}{
3|H_{GT}|+\epsilon
}$ 
Để tránh data leakage, ảnh và mask hoàn chỉnh chỉ được sử dụng cho training/validation của shape predictor và đánh giá; không được cung cấp chúng cho predicted-shape hay surface scorer khi chạy trên test set.

Cần nhớ rằng có nhiều appearance completion hợp lý dù không giống ground truth. Vì thế, kết quả định lượng nên đi cùng đánh giá trực quan hoặc human preference, nhất là với real occlusions.
````

## Hardware feasibility, schedule and completion criteria

Extracted lines 978-1118.

````text
6. Khả năng triển khai với GPU 24 GB

Mức độ rủi ro và mức độ ưu tiên

Shape reranking + oracle experiment

Thấp–TB

Predicted shape via lightweight U-Net

Trung bình

Boundary/color/texture reranking

Thấp–TB

Joint shape + surface selection

Trung bình

Shape-constrained refinement

Cao

Gradient-guided diffusion

Rất cao
Đánh giá tương đối về độ khó/rủi ro triển khai, không phải benchmark đo lường.
GPU 24 GB không nhất thiết là vấn đề đối với một lần SD 2 inpainting ở 512×512 FP16, nhưng toàn bộ PMC còn có Grounding DINO, SAM, InstaOrder và LaMa, nên không thể mặc định rằng tất cả model đều vừa VRAM cùng lúc.

Một số nguyên tắc triển khai:

- 
Generate  $K=4$  tuần tự, không dùng batch 4 ngay từ đầu.

- 
Cache image, masks, feature embeddings và segmentation outputs khi có thể.

- 
Chạy shape predictor và feature extractor theo từng giai đoạn, giải phóng GPU trước khi chạy diffusion nếu cần.

- 
Dùng torch.inference_mode() cho tất cả pretrained models không fine-tune.

- 
Bắt đầu với  $K=2$ ; tăng lên  $K=4$  sau khi đo peak VRAM/runtime.

- 
Kiểm tra kết quả của official pipeline với môi trường nguyên bản trước khi nâng cấp Diffusers hoặc thay đổi scheduler.

Điểm đánh đổi quan trọng: best-of-K reranking có thể tiêu tốn gần K lần số diffusion generations, nên chất lượng phải được báo cáo cùng inference cost. Nếu chất lượng tăng rất ít nhưng runtime tăng mạnh, đó là một limitation đáng thảo luận.

7. Cách phân công công việc thực tế trong 8 tuần

 | 
Tuần
 | 
Người 1 — Shape
 | 
Người 2 — Surface
 | 
Milestone

 | 
1–2
 | 
Hiểu PMC, dataset masks và baseline
 | 
Environment, PMC inference và evaluation
 | 
Baseline reproducible

 | 
3
 | 
Oracle shape experiments, khảo sát predictor
 | 
Kiểm tra surface failures, boundary/color features
 | 
Có quantitative failure analysis

 | 
4
 | 
Implement shape scorer + oracle result
 | 
Implement boundary, color, texture scorers
 | 
Hai prototype riêng

 | 
5
 | 
Predicted-shape model + evaluation
 | 
Deep texture feature hoặc refined scorer
 | 
Hai improvements independently working

 | 
6
 | 
Shape ablations
 | 
Surface ablations, joint integration
 | 
Main comparison table

 | 
7
 | 
Error analysis theo occlusion severity
 | 
Qualitative, human evaluation, runtime
 | 
Freeze results

 | 
8
 | 
Shape contribution slides
 | 
Surface contribution slides; cả hai rehearsal
 | 
Final presentation

Mình sẽ đặt các mốc quyết định như sau: đến hết tuần 3, nếu oracle shape không giúp cải thiện thì Người 1 phải kiểm tra lại scoring strategy trước khi đầu tư huấn luyện predictor. Đến hết tuần 4, nếu surface scorer ưu tiên các hình mượt nhưng sai nội dung, Người 2 nên giảm global color matching và chuyển sang patch-level boundary/texture matching.

8. Hướng nào nên làm trước, và khi nào coi là hoàn thành?

Mình khuyên Người 1 làm shape reranking với GT mask trước, vì có thể đánh giá rất nhanh xem hypothesis có tiềm năng hay không. Sau đó mới chuyển sang predicted shape để tạo phương pháp sử dụng được trên unseen inputs.

Người 2 nên làm boundary-aware reranking trước, vì không cần pretrained model mới và rất dễ kiểm tra bằng qualitative images. Tiếp theo thêm color histogram, rồi texture features; chỉ giữ lại thành phần nào thực sự cải thiện validation metrics.

Về mặt nghiên cứu, một kết luận như:

Predicted shape priors improve geometric consistency under heavy occlusion, while appearance-aware selection improves boundary continuity; jointly using both provides complementary gains at the cost of additional sampling.

sẽ rất có giá trị nếu được experiments chứng minh. Ngược lại, nếu hai scores xung đột hoặc combined method kém hơn một trong hai phương pháp riêng, đó cũng là một kết quả nghiên cứu có ý nghĩa và cần phân tích.

Kết luận: Mình đề xuất cả nhóm cam kết hoàn thành shape-guided selection, surface-guided selection và joint selection làm ba đóng góp thực nghiệm chính. Shape-constrained refinement và gradient-guided diffusion nên được xem là mục tiêu mở rộng, không phải điều kiện để project thành công trong 8 tuần.
````

