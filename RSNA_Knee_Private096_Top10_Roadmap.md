# RSNA 2026 Knee Abnormality Detection — Top-10 Roadmap

**Mục tiêu:** xây pipeline đủ mạnh và đủ robust để hướng tới **Private LB ~0.96 / Top 10**.  
**Lưu ý:** đây là mục tiêu tham vọng, không phải mức điểm được bảo đảm. Mọi quyết định phải ưu tiên **Private robustness**, không overfit Public LB.

## 1. Điểm xuất phát

### Baseline 04 — DONE
- Frozen DINOv2-Small
- 5 MRI slots
- 8 vị trí/slot
- 2.5D triplet
- crop vật lý 140 mm
- 256×256
- CLS-only features
- 12 pathology queries
- L1 soft labels từ report
- 58 gold chỉ audit
- **Public LB = 0.748**
- **Gold-58 macro AUC ≈ 0.734**

Kết luận: supervision pipeline dùng được, nhưng **vision representation đang là bottleneck lớn**.

---

# 2. Nguyên tắc cố định

1. **58 gold không dùng để train / early stop / tune ensemble.**
2. Train chính trên **4,349 weak studies** với soft labels.
3. Report chỉ là **train-time supervision**; inference cuối phải là:
   `MRI → 12 probabilities`.
4. DICOM phải sort bằng **IOP + IPP**, không filename.
5. Crop theo **physical mm**, không resize thô toàn ảnh.
6. Không dùng horizontal flip.
7. Group duplicate reports cùng fold.
8. Audit thêm **scanner/site leakage**.
9. Không tune trọng số ensemble lặp đi lặp lại theo Public LB.
10. Final submit:
   - Internet OFF
   - inference < 9h
   - mục tiêu an toàn <= 6–7h
   - checkpoint/preprocessing hoàn toàn offline.

---

# 3. Roadmap thực hiện

## Stage 05A — Dense-Slice MIL hiện tại

**Mục tiêu:** kiểm tra dense slice coverage + trainable encoder có vượt sparse frozen DINO hay không.

Hiện tại:
- 16 slices/slot
- 224×224
- 140 mm crop
- ResNet18 `weights=None`
- end-to-end trainable
- MIL pathology queries
- soft BCE
- slice dropout
- submit adaptive 16→12→8

### Bổ sung trước full run
- augmentation nhẹ:
  - rotation khoảng ±7–10°
  - brightness/gamma jitter nhẹ
  - scale/translation nhẹ nếu ổn định
  - **không horizontal flip**
- audit fold theo:
  - manufacturer
  - field strength
  - pixel spacing
  - slice thickness
- benchmark:
  - final full-data model
  - 2–3 fold ensemble
  - 5-fold ensemble
  theo tiêu chí **AUC gain / inference hour**.

### Kết quả 05A — 2026-10-01

- Final full-data schedule: 14 epochs, selected once from fold best epochs 13 và 15.
- Public LB: **0.773** (so với 04: **+0.025**).
- Fold-0 weak diagnostic AUC: **0.7230**; Gold-58 macro AUC: **0.7184**.
- Fold-1 weak diagnostic AUC: **0.7483**; Gold-58 macro AUC: **0.7304**.
- Mean Gold-58 macro AUC: **0.7244**, chưa vượt 04 khoảng **0.734**.
- Kết luận: dense coverage + trainable scratch encoder có tín hiệu Public dương nhưng **chưa đạt Milestone 1**;
  representation/pretraining vẫn là bottleneck. Giữ 05A làm candidate diversity, không tiếp tục tune theo Public LB.
- Quyết định tiếp theo: chạy **05B-B1 ImageNet-pretrained ResNet18** trên đúng folds/pipeline 05A.

---

## Stage 05B — Backbone / Pretraining Ladder

Đây là một trong những experiment quan trọng nhất.

**Giữ cố định input, sampling, folds, MIL head. Chỉ thay backbone / initialization.**

So sánh:

1. ResNet18 scratch — B0 — **DONE (05A)**
2. ResNet18 ImageNet pretrained — B1 — **OPTIONAL one-fold calibration; không phải nhánh chính**
3. RadImageNet-pretrained CNN — B2 — **NEXT primary owned-model experiment**
4. CoAtNet-like backbone — B3 — **strong architecture track sau B2**

Mục tiêu:
- cô lập tác dụng của pretraining;
- kiểm chứng evidence public rằng **CoAtNet / backbone family** có thể tạo gain rất lớn;
- chọn một **strong visual backbone** trước khi làm multimodal.

### Fast strong-anchor policy — 2026-10-01

- Notebook public `rsna-knee-speedy-raptors-coatnet-d4-0943.ipynb` được dùng làm **strong benchmark anchor**,
  không dùng để tune lặp weights theo Public LB.
- Pipeline này không phải một backbone đơn: nó gồm DINO ensemble, A5, RadImageNet và nhiều Raptor/CoAtNet readers.
- Mục tiêu ngắn hạn: xác nhận anchor khoảng 0.943 có thể reproduce nguyên trạng với pinned public artifacts.
- Mục tiêu model tự phát triển: đi thẳng **B2 RadImageNet → B3 CoAtNet-like**, thay vì kỳ vọng B1 ResNet18
  tạo bước nhảy tới vùng 0.9+.
- B1 chỉ chạy một fold nếu cần đo riêng tác dụng ImageNet initialization; không bắt buộc chạy full hai fold.

### Mốc đã xác nhận — 2026-10-02

- Người dùng đã submit public ensemble `rsna-knee-speedy-raptors-coatnet-d4-0943.ipynb` trên tài khoản chính: **Public LB = 0.943**.
- Bước kế tiếp là audit trong `RSNA_0943_Audit_and_Next_Experiment.md` trước khi chọn một thay đổi model/blend.
- Kaggle Code Competition không công bố toàn bộ dự đoán test ẩn của lần chấm. Output version nhìn thấy được có thể chỉ có 3 test studies mẫu; không dùng nó để ước lượng tương quan trên test ẩn hoặc tune weights.
- Ưu tiên tạo holdout/OOF có provenance rõ cho các family mạnh, xem per-target AUC và lỗi theo scanner/site. Chỉ thêm model mới nếu OOF cho thấy giá trị bổ sung.
- `results_0.943.zip` xác nhận run mẫu 3 studies `COMPLETE` sau 257 giây, đủ bốn CoAt members và không có fallback được ghi. Thời gian/diagnostics của hidden rerun không thể suy ra từ ZIP này.
- Child receipts cho thấy Gold58 được dùng chọn checkpoint ở nhiều family CoAt. Không dùng Gold58 AUC từ các artifact đó làm validation độc lập cho blend mới.

---

## Stage 05C — MRI Input Representation

Sau khi chọn backbone đủ mạnh, ablate từng yếu tố:

### Slice strategy
- uniform sparse
- dense uniform
- adjacent windows
- center-focused / local windows
- coarse-to-fine nếu cần

### Input type
- 2D single slice
- 2.5D adjacent 3-slice

### Resolution
- 224
- 336
- 384 nếu runtime cho phép

### Physical crop
- 130 mm
- 140 mm
- 150 mm

Mục tiêu:
> tìm **strong visual baseline** đủ mạnh trước khi thêm report contrastive hay CoPAS-lite.

---

# 4. Milestone 1 — Strong Visual Baseline

Chỉ đi tiếp sang multimodal khi đã có một model visual đủ mạnh.

Desired behavior:
- Gold-58 cải thiện rõ so với 04;
- Public LB tăng mạnh mà không cần LB tuning;
- runtime submit an toàn;
- per-target không chỉ mạnh ở diffuse findings mà còn cải thiện MCL / meniscus / fracture.

Nếu visual model vẫn thấp, phải quay lại:
- sampling;
- backbone;
- labels;
- preprocessing;
chứ chưa đổ thêm multimodal complexity.

---

# 5. Stage 06A — MRI ↔ Multilingual Report Contrastive Probe

**Không generate report.**

Pipeline:

```text
MRI → strong image encoder → z_image
                              ↕
                         contrastive
                              ↕
Report → multilingual text encoder → z_text
```

Đồng thời image branch vẫn:

```text
MRI tokens → 12 disease queries → soft BCE
```

### Quan trọng
Current study batch quá nhỏ cho CLIP-style contrastive trực tiếp.

Do đó 06A nên:
1. freeze strong visual encoder;
2. cache MRI study embeddings;
3. cache report embeddings;
4. train projection/alignment head với batch lớn 128–512.

Text encoder phải hỗ trợ **multilingual reports**.

Candidates:
- BGE-M3
- multilingual sentence encoder
- XLM-R derived encoder
- translation-to-English + medical encoder

Không mặc định BGE-M3 là tốt nhất; phải benchmark.

### z_image
Dùng **global study pooling độc lập** trước.
Không gộp 12 disease queries thành z_image ở version đầu.

---

# 6. Stage 06B — Online MRI–Report Training

Chỉ làm nếu 06A cho tín hiệu dương.

Giải quyết batch nhỏ bằng:
- MoCo-style memory queue / memory bank;
- hoặc loss ít phụ thuộc giant in-batch negatives;
- hoặc teacher embedding distillation.

Mục tiêu:
> cho image encoder thực sự học từ report semantics mà test vẫn chỉ cần MRI.

---

# 7. Stage 07 — True CoPAS-lite

Không gọi “CoPAS-lite” nếu chỉ cộng slot embedding rồi pool chung.

Phải có **3 plane branches thật**:

```text
Sag tokens → Sag branch ─┐
Cor tokens → Cor branch ─┼→ structured cross-plane attention
Ax tokens  → Ax branch ──┘
                          ↓
                 fused MRI tokens
                          ↓
                 12 disease queries
```

Ví dụ:

```text
Sag' = Sag
     + Attn(Q=Sag, K/V=Cor)
     + Attn(Q=Sag, K/V=Ax)
```

Tương tự cho Cor và Ax.

### Sequence handling
Không giả định RSNA có clean PD/T1/T2 labels như CoPAS.

Trước tiên audit:
- Fluid_Sensitive
- possible DICOM sequence metadata
- TE/TR nếu đủ tin cậy

Nếu không recover sequence type đáng tin:
- chỉ gọi `Fluid` / `Non-fluid`
- không giả vờ chúng tương đương T1/T2/PD.

---

# 8. Stage 08 — Global + Local Anatomy

Sau khi global visual model mạnh:

```text
global knee view
+
local anatomical evidence
```

Ưu tiên ít branch có ý nghĩa:
- meniscus
- ACL/intercondylar
- MCL/medial side
- OA / patellofemoral compartment
- bone/fracture

Không build 12 full backbones.

Có thể dùng:
- attention-derived ROI
- weak anatomical localization
- external localization dataset sau này nếu thật sự cần.

---

# 9. Stage 09+ — External / Self-Supervised Families

Chỉ dùng external data khi giải quyết gap cụ thể.

Candidates:
- MRNet → ACL/meniscus/knee anatomy
- fastMRI+ → localization
- SKM-TEA → anatomy / segmentation
- MRI self-supervised pretraining

Không concat external MRI chỉ vì “nhiều data hơn”.

---

# 10. Ensemble để hướng tới Private 0.96

Chỉ ensemble **strong + complementary** models.

Target families:

1. Strong dense 2.5D CoAtNet-like visual model
2. DINO / ViT partially fine-tuned model
3. MRI-report aligned model
4. CoPAS-lite plane-aware model
5. Global+local anatomical model

Đánh giá diversity bằng:
- OOF prediction correlation
- per-target residual correlation
- target-wise strengths
- Gold-58 behavior

Không ensemble nhiều bản gần giống nhau chỉ để tăng số lượng.

Có thể thử:
- probability mean
- rank mean
- target-specific blend

Nhưng weights phải dựa vào **internal evidence**, không Public LB probing.

---

# 11. Những thứ chủ động KHÔNG làm

- MRI → generated report → disease
- full CoPAS ngay từ đầu
- fine-tune backbone trên 58 gold
- DINO Small→Base/Large chỉ vì lớn hơn
- YOLO ROI quá sớm
- external MRI data quá sớm
- Optuna ensemble weights trên Public LB
- ensemble hàng loạt near-duplicate models
- horizontal flip
- repeated Public-LB weight tuning

**Lưu ý:** “không tăng model size mù quáng” **không đồng nghĩa** với “không nghiên cứu backbone family”.

---

# 12. Thứ tự thực thi ngắn gọn

```text
04 DONE — Frozen DINO baseline = 0.748
        ↓
05A — Dense Slice ResNet18 MIL
        ↓
05B — Backbone / Pretraining Ladder
        ↓
05C — Slice / 2.5D / Resolution / Crop ablations
        ↓
=========== STRONG VISUAL BASELINE ===========
        ↓
06A — Cached MRI↔Multilingual-report contrastive probe
        ↓
06B — Online contrastive if positive
        ↓
07 — True CoPAS-lite cross-plane model
        ↓
08 — Global + Local anatomy
        ↓
09+ — SSL / external-data model families
        ↓
DIVERSE ENSEMBLE
        ↓
FINAL PRIVATE-LB-ROBUST SUBMISSION
```

---

# 13. Decision rule sau mỗi experiment

Giữ model nếu:
- internal metrics tăng ổn định;
- Gold-58 không collapse;
- Public LB xác nhận xu hướng nhưng không cần LB tuning;
- runtime submit an toàn;
- hoặc model tạo diversity hữu ích cho ensemble.

Bỏ model nếu:
- gain chỉ xuất hiện sau Public-LB probing;
- runtime quá cao;
- complexity tăng mạnh nhưng AUC không tăng;
- prediction gần như giống model mạnh hơn;
- chỉ cải thiện weak-CV nhưng gold / Public không xác nhận.

---

# 14. Score target

Mục tiêu vận hành:

```text
Stage 1:
đưa single visual model từ 0.748 lên vùng cạnh tranh mạnh

Stage 2:
tạo 3–5 family mạnh và khác nhau

Stage 3:
ensemble có kiểm soát để hướng tới
Private LB ≈ 0.96 / Top 10
```

Không giả định 0.96 chắc chắn đạt được.

**Ưu tiên cuối cùng là Private robustness, không phải Public peak.**
