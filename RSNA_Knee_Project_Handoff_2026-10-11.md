# RSNA Knee Abnormality Detection — Handoff toàn bộ dự án

**Cập nhật:** 11/10/2026, múi giờ Asia/Bangkok (UTC+7).  
**Workspace:** `C:\Users\ADMIN\Documents\RSNA`  
**Mục tiêu cuối:** Private Leaderboard Top 10; ~0.959–0.960 là mục tiêu tham vọng, chưa phải kết quả hay mức điểm được bảo đảm.  
**Mốc Public tốt nhất đã được người dùng xác nhận:** **0.943**.

Tài liệu này tổng hợp lịch sử từ EDA/labels đến notebook 21, phân biệt kết quả thật, kết quả chẩn đoán và hướng chưa triển khai. Các file và ZIP gốc được giữ nguyên.

## 1. Tiếp tục từ đâu

1. **Notebook 21 đã hoàn tất**: `results 21.zip` có receipt `COMPLETE`; người dùng xác nhận Public **0.913**, so với B2 standalone notebook 10 **0.914**. Protected champion vẫn **0.943**.
2. Audit 21 xác nhận đúng schema và fixed `0.8*p80 + 0.2*p32`; nhánh 80 reproduce notebook 10 trên 3 visible studies trong sai số ~7e-8 từ CSV. Receipt ghi 192 decoded images, 12 geometry-sorted series, decode-failure dictionary rỗng, visible elapsed 18,832 giây. Không suy ra full hidden parity/runtime từ output này.
3. **Đóng nhánh frozen B2 density blend**; không tiếp tục quét weights/target groups trên Public. Weak OOF gain exploratory của 20 không chuyển thành Public gain của 21.
4. **Bước tiếp theo là triển khai notebook 22 local**: thêm public 2.5D ConvNeXt reader như một family ngoài base 0.943 giữ nguyên, không thay CoAt incumbent. Exact public contract/provenance/eligibility còn phải audit; chưa có run Kaggle hay Public Score của 22. Đọc `RSNA_21_Audit_and_22_Direction_2026-10-11.md` trước khi tiếp tục.
5. Giữ notebook đạt 0.943 và các artifact/version của nó làm benchmark/backup. Không ghi đè notebook gốc khi tạo thử nghiệm mới.

**Hai ZIP dễ nhầm:**

| Tên file local | Nội dung thực tế |
| --- | --- |
| `results_10.zip` — có dấu gạch dưới | Output notebook **10**, B2 standalone, receipt + 3 dòng test hiển thị |
| `results 10.zip` — có dấu cách | Output notebook **20**, ablation 80/32/16 lát, 15 checkpoints và OOF đủ 4.349 UID |

**Tài liệu cũ không đại diện trạng thái hiện tại:**

- `RSNA_KNEE_CODEX_CONTEXT.md` ghi giai đoạn chuẩn bị notebook 04. Dùng cho dữ kiện EDA và nguyên tắc ban đầu.
- `RSNA_Knee_Full_Handoff_2026-10-10.md` có ngày mới nhưng nội dung vẫn yêu cầu tiếp tục 05A RAMFIX. Đây là handoff cũ về mặt trạng thái; không quay lại bước đó chỉ vì đọc file này.
- Các tên “Stage 06 contrastive / Stage 07 CoPAS” trong roadmap là **giai đoạn ý tưởng**, không phải notebook 06/07 thực tế. Notebook 06/07 hiện có là inventory và evidence export.
- Tài liệu hiện tại thay thế phần “current stage / next action” trong các tài liệu cũ; giữ chúng làm lịch sử nghiên cứu.

## 2. Bằng chứng, mục tiêu và giới hạn đánh giá

Người dùng muốn cải thiện dần nhưng đích cuối là Private Top 10. Người dùng sẵn sàng dùng các lượt submit còn lại để kiểm chứng giả thuyết, đồng thời rất quan tâm GPU quota và đã mất hơn 10 giờ qua những run lỗi. Cần chủ động chuẩn bị notebook cụ thể, kiểm tra trước khi giao chạy, nói rõ nó train, chẩn đoán hay tạo submission.

| Loại số liệu | Cách hiểu đúng |
| --- | --- |
| Public LB do người dùng/screenshot xác nhận | Điểm Kaggle thực tế của bài đã nộp; không suy ra Private hoặc AUC từng nhãn |
| Weak OOF macro-AUC proxy | AUC trên L1 silver threshold ở 0.5 để chẩn đoán; training vẫn dùng soft probabilities |
| Weak validation BCE | Loss trên silver mềm, dùng chọn checkpoint/early stop của model tự phát triển |
| Gold58 audit | Audit trên 58 official binary studies, sample nhỏ; không dùng optimizer/early stop/weights của model tự phát triển |
| Preflight/smoke | Xác nhận input, shape, tài nguyên, throughput; không chứng minh model tốt |
| `COMPLETE` | Run hoàn tất theo contract của notebook; không đồng nghĩa tăng score hay mọi fallback đều bằng 0 |

Weak OOF đo mức phù hợp với report-derived teachers, không đo trực tiếp ground truth ảnh trên Private. Các Public CoAt checkpoints đã dùng Gold58 chọn checkpoint, nên Gold58 không là holdout độc lập cho các family đó. Gold58 cũng đã được xem nhiều lần trong nghiên cứu; không coi đó là một bộ test mới chưa từng được quan sát.

Không có full OOF của ensemble 0.943. Việc ghép B2 vào một proxy chỉ gồm vài public family từng cho gain nội bộ nhưng không tăng Public khi ghép vào toàn ensemble; đây là bằng chứng cụ thể cho hạn chế đó.

## 3. Competition và schema

Task: một MRI study gồm nhiều series/plane/slice → 12 xác suất bất thường. Metric là macro ROC-AUC, mỗi nhãn đóng góp bằng nhau.

Thứ tự output phải là:

```text
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
```

Theo [competition overview/evaluation](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation), kiểm tra ngày 11/10/2026:

- Notebook-only submission; CPU/GPU notebook runtime tối đa 9 giờ.
- Internet phải tắt khi submit; file phải tên `submission.csv`.
- External data/pretrained models phải public và phù hợp điều kiện nguồn/competition.
- Entry/merger deadline: 15/10/2026, 23:59 UTC = 16/10/2026, 06:59 UTC+7.
- Final submission deadline: 22/10/2026, 23:59 UTC = 23/10/2026, 06:59 UTC+7.
- Winners' materials deadline: 05/11/2026, 23:59 UTC.

Context trước đây và người dùng ghi nhận 5 submissions/ngày, tối đa 2 final submissions. Kiểm tra hạn mức và lựa chọn final đang hiển thị trên tài khoản trước vận hành cuối; không coi ngày/limit trong handoff là thay thế thông báo của host nếu có cập nhật.

Ngân sách nội bộ 2/3/6 giờ trong notebook là **guard do chúng ta đặt**, không phải ETA hay giới hạn GPU quota của tài khoản. Quota T4×2 thay đổi theo tài khoản/thời điểm; cần hỏi số còn lại nếu sắp giao run dài.

## 4. EDA và supervision đã xây

### 4.1 Dữ liệu ảnh

- 4.407 train studies, 24.371 series; trung bình khoảng 5,53 series/study.
- 58 studies có đầy đủ 12 official binary labels; 4.349 không có structured targets đầy đủ.
- Mọi train study có sagittal, coronal, axial; acquisition type trong từng plane có thể thiếu.
- Typical series khoảng 30 slices nhưng long tail rất dài; không đọc mọi lát mọi series một cách không giới hạn.
- Scanner/vendor, field strength, pixel spacing, slice thickness và FOV khác nhau đáng kể.
- `Fluid_Sensitive` và `Fat_Suppression` trùng nhau trong train audit, không có nghĩa chúng luôn trùng trên mọi test tương lai.
- `SeriesDescription`, `ProtocolName`, laterality có nhiều missing/dummy values; không dùng free-text metadata làm routing chính khi chưa audit.

EDA snapshot: 6.000 header samples không lỗi, 50/50 pixel decodes thành công; không đủ để bỏ logging lỗi ở inference. Pixel-spacing median ~0,3125 mm; slice-thickness median ~3 mm. Đây là thống kê mẫu, không phải toàn bộ ảnh.

Gold58 positive counts theo label order ở mục 3: 24/9/26/23/15/11/21/35/27/12/19/18. Sample nhỏ, nhiều nhãn chỉ có vài positives/negatives; không coi delta AUC nhỏ trên Gold là ổn định. Vendor snapshot khoảng Siemens 40,5%, Philips 31,3%, GE 22,0%, Canon 5,7%; đây là yếu tố domain shift nên kiểm tra, không phải cơ sở tune vendor-specific weights.

### 4.2 Geometry và preprocessing

Filename/SOP UID sorting không đại diện thứ tự giải phẫu. Audit filename order ban đầu không monotonic trên 150 sampled series; selected-series geometry audit có 250/250 usable geometry.

```python
normal = cross(ImageOrientationPatient[:3], ImageOrientationPatient[3:])
position = dot(ImagePositionPatient, normal)
# sort theo position; fallback InstanceNumber nếu cần và ghi count
```

Contract chủ đạo:

- Geometry sorting; không dùng filename lexical làm primary order.
- Apply slope/intercept; xử lý MONOCHROME1; intensity clipping/normalization deterministic.
- Crop vật lý 140 mm dựa trên row/column spacing, chặn theo image bounds khi FOV nhỏ.
- Resize theo encoder config; normalization đúng pretrained model.
- Slot thiếu → mask, không tạo ảnh giả.
- Không horizontal flip mặc định vì cần giữ thông tin anatomy/medial-lateral; mọi augmentation phải nhất quán trong study.
- Inference MRI-only; report chỉ cho supervision và grouping lúc train.

5 slots thường dùng:

```text
SAG_FLUID, SAG_NONFLUID, COR_FLUID, COR_NONFLUID, AX_FLUID
```

`AX_NONFLUID` chưa là nhánh baseline do coverage thấp. Slot order/id phải lấy đúng checkpoint config của từng notebook, không đoán từ danh sách trong một tài liệu khác.

### 4.3 Audit nguồn labels

8 public label files được audit. S07 (`report_labels_v5.csv`) trùng chính xác toàn bộ 58×12 = 696 gold cells, được đánh dấu `REFERENCE_LEAKY` và loại khỏi việc đánh giá độc lập/teacher family bình thường.

Các file không phải 8 voters độc lập:

- S02/S03 là duplicates.
- S01/S06 rất gần nhau.
- S04/S05 cùng family liên quan.

4 families sau loại nguồn leaky:

```text
GPT56SOL, HYBRID_BLEND, STEVEN_CORE, YUNUS
```

Locked supervision cho các model tự phát triển:

```text
L1_family_mean_RECOMMENDED.csv
4.349 unique weak UID × 12 soft probabilities
58 gold UID excluded
```

L1 family mean audit macro-AUC trên Gold58 khoảng 0.895760; L3 gold-guided khoảng 0.895920 nhưng weights được xây từ cùng Gold58 nên không dùng làm locked supervision. Không hard-threshold L1 cho training; threshold chỉ dùng khi tính diagnostic AUC.

Family disagreement lớn ở Synovitis (mean agreement ~0,655), Effusion (~0,728), Baker's (~0,760). Agreement weights đã có trong `L1_agreement_weights.csv`, chưa được chứng minh hữu ích trong một controlled training experiment.

### 4.4 Folds và leakage

- Owned-model folds tái sử dụng `fold_assignments.csv` của 05A: 870/870/870/870/869 weak studies.
- Cùng `report_group` không được xuất hiện cả train và validation.
- Report-group counts giữa các audit có thể khác do normalization method; không thay fold file để ép khớp một con số trong tài liệu.
- Public OOF có assignment khác 05A. Chỉ khoảng 20,72% weak studies giữ cùng index fold giữa hai bộ; không join bằng index fold hoặc thứ tự dòng.
- Join theo string `StudyInstanceUID`, và kiểm tra fold khi so controlled owned experiments.
- Exact-report group audit không loại trừ near-duplicate ảnh, scanner/site leakage hoặc những dạng chồng lấn khác.

## 5. Lịch sử kỹ thuật 05A và cách tránh lặp run lỗi

### 5.0 Baseline ảnh trước B2

04 dùng frozen DINOv2-Small, ảnh 256, sparse 2.5D khoảng tám positions/slot, CLS features 384 chiều và năm slots; physical crop 140 mm. Public 0.748, Gold audit khoảng 0.7338; run lịch sử khoảng 3 giờ 35 phút. Đó là baseline tự phát triển ban đầu, không phải DINO bundle 20 members trong anchor public 0.943.

05A chuyển sang ResNet18 train từ đầu, ảnh 224, 16 slices/slot (tối đa 80), encoder trainable và query-based MIL với soft BCE. Đây là bước tăng dense coverage nhưng không có medical pretraining. Sau đó B2 thay bằng frozen RadImageNet ResNet50, không phải chỉ sửa learning rate của 05A.

### 5.1 Các lỗi đã gặp

1. Không tìm thấy L1/manifest dù file có trong upstream notebook Output: file phải được **attach vào current notebook**, không chỉ hiện trên trang output của notebook khác. Kaggle có cả path kiểu `/kaggle/input/<slug>` và `/kaggle/input/notebooks/...` hoặc `/kaggle/input/datasets/...`.
2. AMP/DataParallel indexed assignment: destination Float, source Half tại `pooled[nonempty] = po`. SAFE code cast nguồn về dtype đích: `po.to(pooled.dtype)`.
3. Attach output đã save nhưng không có ảnh cache: `fold_0.pt` và `weak_cache_manifest.csv` không thay thế `.npz` thực tế. Absolute path `/kaggle/working` của run cũ không tồn tại trong kernel mới.
4. Host RAM tăng đều theo epoch và sang fold tiếp theo. Run nhiều lần chết hoặc bị RAM guard chủ động dừng. Chỉ nới guard/đổi ngưỡng không xử lý memory retention.

### 5.2 Bằng chứng OOM

Logs người dùng cho thấy process RSS từ ~0,55 GiB lên ~23,4 GiB sau 15 epochs, rồi ~28 GiB ở fold sau; mức tăng thường ~1,42 GiB/epoch. Trong khi CUDA allocated ~0,2 GiB và peak ~1,14 GiB. Đây là host RAM accumulation, không có bằng chứng những logs đó là VRAM OOM.

Những error `HOST-RAM LEAK GUARD` là intentional early stops để bảo vệ kernel. Chúng không được ghi nhận như full training thành công. Comments trong code liên hệ shared-memory mapping/CPU conversion, nhưng chưa có profiling đủ để khẳng định một nguyên nhân allocator duy nhất cho mọi run.

### 5.3 SAFE implementation đã chạy thành công

- Worker trả plain NumPy `uint8 [T,H,W]`; CPU batch còn `uint8 [B,T,H,W]`.
- Tạo tensor trong main process; float/RGB conversion và augmentation thực hiện sau chuyển GPU.
- Study batch 2, worker 1, `persistent_workers=False`, `prefetch_factor=1`, `pin_memory=False` trong SAFE files hiện có.
- Một fold mỗi notebook/kernel mới: `FOLD0_SAFE`, `FOLD1_SAFE`.
- Loader shutdown, `gc`, `malloc_trim` khi hỗ trợ, GPU cleanup, memory-history logs.
- SAFE guard hiện tại theo available system RAM; RSS slope vẫn diagnostic.
- `require_existing_cache=False` trong final SAFE files có thể cho rebuild khi thiếu cache. Không mô tả chúng là tuyệt đối cấm rebuild; ưu tiên attach cache đầy đủ để tránh decode lại tốn giờ.

Các fold0/fold1 SAFE đã chạy thành công, có ZIP artifacts. Final full-data 14 epochs được chọn một lần từ best epochs 13/15. Public 05A = **0.773**, tăng +0.025 so với 04 = 0.748.

| SAFE run | Weak validation AUC proxy | Gold audit | Runtime |
| --- | ---: | ---: | ---: |
| Fold 0 | ~0.7230 | ~0.7184 | ~89,6 phút |
| Fold 1 | ~0.7483 | ~0.7304 | ~95,6 phút |

Hai fold artifacts không chứa toàn bộ ảnh cache theo mặc định và không tự là một full-data final checkpoint. Final14 là notebook train riêng rồi notebook Submit SAFE làm inference.

Final14 ghi `final_progress.pt` mỗi epoch: model/config/labels/slots/completed epochs/training scope. Nó **không có** optimizer/scheduler/scaler/RNG state và không có auto-resume load path trong notebook, nên chỉ là recoverable model snapshot, không phải exact training resume. Artifact cuối để submit là `05A_dense_mil_final.pt`; ready check yêu cầu đủ 14 history epochs và Gold audit. Không tự chuyển một guard checkpoint/progress snapshot thành final checkpoint đã hoàn tất.

### 5.4 Cache đã tái sử dụng

Người dùng tải saved old 05A Output có cache thực, upload thành private Dataset **`Study_case 05_RSNA`**. Path được xác nhận trong notebook 20:

```text
/kaggle/input/datasets/anhtunnguynchu/study-case-05-rsna/
  rsna_05a_dense_mil/study_cache/
    weak_<StudyInstanceUID>.npz
    gold_<StudyInstanceUID>.npz
  rsna_05a_dense_mil_ramfix/fold_assignments.csv
```

Cache đầy đủ: 4.349 weak + 58 gold = 4.407 studies, up to 80 cached slices/study, 224×224 uint8, 140 mm crop. Measured total images = 327.497; average weak valid slices ~74,30. Disk cache khoảng 11,9 GiB, không đồng nghĩa 11,9 GiB resident RAM.

Ảnh sidebar có `study_cache` chỉ xác nhận thư mục hiện diện; asset gates còn phải kiểm tra count và `.npz` theo từng UID. Nếu saved latest Output không chứa cache cũ, attach đúng saved version hoặc Dataset chứa cache. Preserve ảnh cache ngoài ZIP diagnostics nhỏ nếu cần.

Cache attach được tham chiếu tại chỗ, không tự copy vào output mới. Final artifact ZIP không gồm cache hoặc progress; full working Output chỉ chứa cache được build tại run đó, không tự re-export toàn bộ attached cache. Vì vậy việc save version thành công không bảo đảm version mới vẫn là nguồn study cache đầy đủ.

## 6. Registry notebook: từ 00 đến 21

Các tên local 00–03 không nhất thiết có prefix số; giữ numbering logic dưới đây, không rename hay thay lịch sử.

| Số | Notebook local / purpose | Trạng thái và kết luận |
| --- | --- | --- |
| 00 | `RSNA_Knee_EDA_v2.ipynb` | DONE: study/series/report/DICOM/metadata EDA |
| 01 | `RSNA_Knee_Label_Audit_v2.ipynb` | DONE: source leakage/dependency/Gold58 label audit; bản v1 là lịch sử |
| 02 | `RSNA_Knee_Silver_Label_Construction_v1.ipynb` | DONE: L1 soft silver, 4 families, 4.349 UID |
| 03 | `RSNA_Knee_Baseline_V1_Prep.ipynb` | DONE: deterministic series manifest và report groups |
| 04 | `04_RSNA_Knee_Baseline_V1_Train.ipynb` | DONE: frozen DINOv2-Small, sparse 2.5D, Public 0.748 |
| 05A | `05A_RSNA_Knee_DenseSlice_MIL_Train_FOLD0_SAFE.ipynb` và `...FOLD1_SAFE.ipynb` | DONE sau nhiều engineering failures; ResNet18 scratch dense MIL |
| 05A final | `05A_RSNA_Knee_DenseSlice_MIL_Train_FINAL14_SAFE.ipynb` + `...Submit_SAFE.ipynb` | DONE: final14 schedule, Public 0.773 |
| 05B-B1 | `05B_B1_RSNA_Knee_DenseSlice_MIL_ImageNet_FOLD0_SAFE.ipynb` và `...FOLD1_SAFE.ipynb` | Đã tạo; không có kết quả B1 riêng được xác nhận trong handoff này; optional calibration |
| Anchor | `rsna-knee-speedy-raptors-coatnet-d4-0943.ipynb` | Public **0.943**, imported public ensemble, current benchmark |
| 06 | `06_RSNA_0943_Public_Artifacts_Inventory_CPU.ipynb` | DONE CPU: kiểm kê public inputs của anchor |
| 07 | `07_RSNA_0943_OOF_Evidence_Export_CPU.ipynb` | DONE CPU: export OOF/manifest/competition tables; đã sửa size guard |
| 08 | `08_B2_RadImageNet_Encoder_Preflight_T4.ipynb` | DONE: frozen ResNet50 encoder throughput gate |
| 09 | `09_B2_RadImageNet_FrozenMIL_Fold01.ipynb` | DONE **5 folds** dù filename/header cũ có Fold01; owned B2 control |
| 10 | `10_B2_RadImageNet_FrozenMIL_Submit.ipynb` | DONE: B2 standalone Public **0.914** |
| 11 | `11_B3_CoAtNet_Encoder_Preflight.ipynb` | DONE: timm CoAtNet pretrained encoder resource gate |
| 12 | `12_B3_CoAtNet_FrozenMIL_FiveFold.ipynb` | DONE: frozen CoAtNet candidate yếu hơn B2; không submit |
| 13 | `13_RSNA_0943_Visible_Output_Integrity_CPU.ipynb` | DONE: integrity trên 3 visible test studies của anchor |
| 14 | `14_B2_RadImageNet_PlaneAwareMIL_FiveFold.ipynb` | DONE: frozen hierarchical plane-aware head thua B2; không submit |
| 15 | `15_B2_RadImageNet_SampledNeighbor25D_FiveFold.ipynb` | DONE: sampled-cache neighbors + frozen encoder thua lớn; không submit |
| 16 | `16_RSNA_0943_Anchor_Plus_B2_Fixed20_Probe.ipynb` | DONE: all-target 20% B2 rank probe, Public **0.942** |
| 17 G1 | `17_G1_RSNA_0943_B2_Group_Probe.ipynb` | DONE: Public **0.942** |
| 17 G2 | `17_G2_RSNA_0943_B2_Group_Probe.ipynb` | DONE: Public **0.942** |
| 17 G3 | `17_G3_RSNA_0943_B2_Group_Probe.ipynb` | DONE: Public **0.943**, displayed tie, chưa vượt anchor |
| 18 | `18_B2_RadImageNet_Layer4_Finetune_Preflight.ipynb` | DONE: resource smoke, 64 train + 16 val, 2 epochs, 16 tokens |
| 19 | `19_B2_RadImageNet_Layer4_Fold0_Paired.ipynb` | DONE: frozen vs layer4 fine-tune ở 16 tokens, fold0; gain không rõ |
| 20 | `20_B2_RadImageNet_Slice_Density_Ablation.ipynb` | DONE: same encoder/head/folds, 80/32/16 tokens; dense80 tốt nhất |
| 21 | `21_B2_Density80_32_FixedBlend_Submit.ipynb` | DONE: fixed 80/20 probability pool của density80/32, Public **0.913**; nhánh density blend đóng |

### Public Score registry đã xác nhận

| Candidate | Public score | Tình trạng |
| --- | ---: | --- |
| 04 frozen DINO baseline | 0.748 | Historical baseline |
| 05A scratch ResNet18 dense MIL | 0.773 | Historical owned model |
| Public Speedy Raptors/CoAt anchor | **0.943** | Best confirmed benchmark |
| 10 B2 frozen RadImageNet standalone | 0.914 | Valid owned family, chưa bằng anchor |
| 16 anchor + B2 all targets | 0.942 | Chưa cải thiện |
| 17 G1 / G2 / G3 | 0.942 / 0.942 / 0.943 | Không có displayed improvement |
| 21 fixed density blend | **0.913** | Người dùng xác nhận; không cải thiện B2 0.914 |

Kaggle hiển thị ba chữ số thập phân trong các screenshot. Không lấy chênh lệch rounded macro score để suy ngược exact AUC của một group/nhãn.

## 7. Anchor 0.943: thành phần, công thức và những gì chưa biết

Anchor là inference pipeline từ nhiều public artifacts, không phải mô hình đã được chúng ta tự train end-to-end. Giữ attribution, pinned versions, source licenses/provenance khi tiếp tục dùng.

Các nhánh chính:

- DINOv2 ensemble: 20 members.
- A5 attention-pooling: 5 fold checkpoints, applicability gate.
- RadImageNet ResNet50 + nhiều heads + metadata calibrator.
- Raptor: 4 views/arms.
- CoAt family: Residual, Global96, D4, Repair-v1.

Tầng cuối:

```text
H = 0.60 × rank(Raptor) + 0.40 × rank(mean_probability(CoAt4))
Final = rank((1-w) × rank(BTKD) + w × H)
BTKD = DINO → A5 → Rad + calibrator
```

| Target group | w cho cả Raptor+CoAt |
| --- | ---: |
| Mặc định | 0.60 |
| ACL, Lateral OA, Fracture | 0.75 |
| Medial Meniscus | 0.80 |
| Lateral Meniscus | 1.00 |

Tên `_coatnet_weight` dễ gây nhầm: đây là weight của cả nhánh Raptor+CoAt. Lateral Meniscus bỏ BTKD ở tầng cuối; điều đó không chứng minh đây là target yếu của final ensemble. CoAt được mean probability rồi rank; repeated rank và acquisition gates không cho phép diễn giải mọi trọng số như global coefficients trên xác suất gốc.

Ở tầng BTKD, A5 dùng weight 0.52 khi acquisition gate phù hợp; study không phù hợp giữ parent. Rad reference dùng 0.55 trừ Baker's/Fracture, có second-layout branch 0.20 và metadata calibrator. Đây là mô tả công thức đã có, không phải các weights mới được chúng ta tối ưu. Không nhân weights qua nhiều tầng rank để diễn giải contribution toàn cục.

`results_0.943.zip` xác nhận run hiển thị 3 studies, `COMPLETE`, 257,02 giây, đủ DINO 20 / A5 5 / Raptor 4 / CoAt 4 members, không fallback được ghi. Người dùng submit account chính và xác nhận Public 0.943. **ZIP 3 studies không chứa full hidden-scoring predictions, labels, runtime hay per-target AUC.** Không thể tính correlation hữu ích trên hidden cohort từ nó.

Code có fallback/except ở các child, nên cần kiểm tra receipt actual members chứ không chỉ có `submission.csv`. Một run visible đủ nhánh không chứng minh mọi hidden study đều xử lý giống vậy.

## 8. Audit 06/07/13 và evidence public OOF

06 kiểm kê 43 relevant paths và 208 manifest clues trong các input của anchor. 07 export 4 OOF CSV, Pilkwang `oof.npz`, manifests và competition `train.csv`/`train_series.csv`.

07 từng bị size guard tại `train.csv` 5.690.007 bytes khi tổng file trước đó ~3.915.584 bytes. Đây là guard export evidence, không phải GPU/model failure. Bản output cuối `results_07.zip` đủ dữ liệu để audit; không cần chạy lại inventory nếu không có input mới.

Các public OOF candidates:

- E9 RadImageNet `v52_oof.csv`.
- E11 dataset `v52_e11_oof.csv`.
- E11 notebook và E13 notebook cũng có filename `v52_e11_oof.csv`; prediction/hash khác nhau, không phải ba bản copy identical.
- Pilkwang DINO `oof.npz` thuộc 20-member bundle.

Các public CSV/NPZ có predictions cho 4.407 UID và 12 targets với finite probabilities; official binary labels chỉ có ở 58 studies. NPZ `y_derived` không tương đương Gold labels. Exact-report group audit chưa phát hiện crossing ở public folds đã kiểm tra, nhưng chưa xác minh độc lập toàn bộ checkpoint training provenance hoặc near-duplicate leakage.

Gold58 descriptive AUC của các public OOF candidates: E11 dataset ~0.8256, E9 ~0.8543, E11 notebook ~0.8375, E13 ~0.8576, DINO ~0.8400. Đây không phải full-ensemble scores và không dùng để chọn target weights. Public fold sizes là 882/882/881/881/881, Gold phân bố 10/15/16/9/8; khác owned-model weak folds.

**Còn thiếu:** OOF/independent holdout predictions của A5, Raptor, đủ 4 CoAt và final ensemble. Inference full-train checkpoint trên train không hồi tố tạo OOF sạch.

13 `VISIBLE_OUTPUT_PASS` chỉ kiểm tra file schema/provenance/formula của visible run anchor; không xác minh hidden scoring rerun.

## 9. Các model tự phát triển: kết quả và quyết định

### 9.1 Frozen B2 và B3

08 đo trên 64 studies / 4.717 ảnh: encoder mất 12,72 giây (~370,7 ảnh/s), GPU peak allocated ~0,233 GiB. Ngoại suy riêng encoder là ~14,6 phút cho toàn cache, chưa gồm I/O và head training.

09 dùng frozen RadImageNet ResNet50:

- Encode một lần 327.497 ảnh từ 4.407 cache files.
- Feature RAM float16 ~1,345 GiB, mỗi token 2048 chiều, tối đa 80 tokens/study.
- GPU 0, image microbatch 24; grayscale lặp thành RGB; normalization `gray / 127.5 - 1` theo implementation đã pin.
- Head 256 chiều, slot/position embeddings, 12 learnable queries, masked attention.
- Head batch 32; AdamW lr 3e-4, weight decay 1e-4; tối đa 20 epochs, patience 4; chọn checkpoint bằng weak validation BCE.
- Seed 111; best epochs của folds 0–4: 11/10/9/10/7.
- Pooled weak OOF ~0.858970; Gold58 audit bằng mean probability của năm heads ~0.888217; tổng 12,71 phút.
- Notebook 10 giữ preprocessing/feature quantization đó, Public **0.914**.

Pinned RadImageNet encoder:

```text
Dataset: marwanmath/resnet-50-radimagenet-marwan
File: ResNet50.pt
SHA256: 08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734
```

11/12 B3 dùng `timm/coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k`, pretrained generic, **khác** các public CoAt checkpoints đã train cho RSNA. Preflight 11 đo 4.717 ảnh / 64 studies trong 40,49 giây (~116,49 ảnh/s); GPU peak ~0,227 GiB. Ngoại suy riêng encoder ~46,5 phút. File weight 167.032.870 bytes:

```text
model.safetensors SHA256:
12e9f7a9707028b2125f03c389f80d8d3202dd830b381f309a9f28c7a3eeb59f
```

12 hoàn tất năm folds trong 45,71 phút, weak OOF **0.778793**, Gold audit **0.740407**. Frozen generic CoAtNet yếu hơn B2 trong pipeline này, nên không submit/blend cấu hình đó. Điều này không phủ định CoAtNet trainable hoặc RSNA-adapted; nó phủ định giả định “backbone lớn hơn tự động tốt hơn” với frozen features.

### 9.2 Plane-aware head và sampled-neighbor 2.5D

| Cấu hình | Pooled weak OOF | Gold58 audit | Runtime |
| --- | ---: | ---: | ---: |
| 09 frozen RadImageNet control | 0.858970 | 0.888217 | 12,71 phút |
| 12 frozen generic CoAtNet | 0.778793 | 0.740407 | 45,71 phút |
| 14 plane-aware frozen RadImageNet head | 0.855159 | 0.877670 | 25,46 phút |
| 15 sampled-neighbor 2.5D, frozen RadImageNet | 0.816646 | 0.842161 | 12,46 phút |

14 có weak delta −0.003811; validation BCE xấu hơn ở cả năm folds. MCL/PF OA tăng riêng lẻ nhưng 10/12 targets giảm. Hierarchical plane pooling trên cùng frozen 80-token features **đã được thử**; không đề xuất lại thay đổi y hệt như một thí nghiệm mới.

15 dùng previous/current/next **sampled cache slices trong cùng slot**, không nhất thiết là các lát DICOM native liền kề. Frozen encoder với channel layout đó giảm −0.042324 weak macro-AUC; cả 12 targets và BCE ở cả năm folds đều xấu hơn. Chưa thể kết luận về native contiguous 2.5D với encoder được adaptation. Không có Public submission của 12/14/15 được xác nhận.

### 9.3 Blend 16 và groups 17

Một proxy chỉ gồm DINO+E13 rank trên weak OOF tăng từ 0.863469 lên 0.870346 khi thêm cố định 20% B2. Proxy đó thiếu A5/Raptor/CoAt và chưa xác minh độc lập toàn bộ provenance public checkpoints; **không phải OOF của ensemble 0.943**.

16 giữ inference của protected anchor và B2 đã thành công, sau đó áp dụng:

```text
rank(0.80 × rank(anchor_0943) + 0.20 × rank(B2_09))
```

Public **0.942**, thấp hơn anchor ở độ chính xác hiển thị. 17 tách cùng thay đổi đó thành ba groups:

| Group | Targets | Public |
| --- | --- | ---: |
| G1 | ACL, MCL, Medial Meniscus, Lateral Meniscus | 0.942 |
| G2 | Medial OA, Lateral OA, PF OA, Effusion | 0.942 |
| G3 | Synovitis, Baker's, Contusion, Fracture | 0.943 |

Ba notebooks độc lập, đã chạy và submit riêng. Receipts đều `COMPLETE`; tám columns ngoài group giữ nguyên anchor. Trên ba visible studies, rank resolution làm CSV cuối có cùng SHA256 với anchor dù B2 đã chạy. Điều này **không chứng minh** predictions của hidden scoring cohort cũng giống nhau. G3 là displayed tie, chưa chứng minh cải thiện. Không suy ngược AUC từng nhãn từ rounded group scores.

**Bài học:** bổ sung tốt cho một partial proxy không đồng nghĩa bổ sung tốt cho toàn ensemble 0.943.

### 9.4 Layer4 fine-tuning 18/19

18 chỉ là resource gate: 64 train + 16 validation studies, hai epochs, 16 uniformly sampled cached tokens, layer4+head có gradients. Tổng ~16,2 giây, GPU peak allocated ~0,454 GiB, process RSS ~1,757 GiB. Ngoại suy 2,11 giờ cho năm folds × bốn epochs chưa tính đủ validation/I/O/checkpoint overhead; không phải ETA full run.

19 so paired frozen với layer4 fine-tune trên **fold 0, cùng 16 tokens**, 3.479 train / 870 validation / 58 Gold studies. Run `COMPLETE`, ~77,82 phút; GPU peak allocated ~0,462 GiB, host RSS ~1,930 GiB. Run này không OOM.

| Nhánh notebook 19 | Weak fold 0 AUC | Gold58 audit |
| --- | ---: | ---: |
| Frozen 16 | 0.826864 | 0.841696 |
| Layer4 fine-tuned 16 | 0.827381 | 0.831870 |

Weak delta +0.000517; report-group bootstrap 95% CI khoảng [−0.0053, +0.0052], chưa thuyết phục. Train/validation gap tăng; best epochs frozen/layer4 = 6/4, dừng ở epochs 10/8. Gold giảm nhưng vẫn chỉ là audit. Không chi GPU cho năm folds với nguyên cấu hình 16-token này.

So sánh trực tiếp với notebook 09 fold 0 dùng 80 tokens (0.864812) bị confounded bởi cả token count và training/head-batch details. Notebook 20 sau đó mới cô lập density với original 09 head training.

## 10. Notebook 20: kết quả controlled density ablation

Một lần encoder extraction dùng chung; ba nhánh density 80/32/16 train trên cùng năm folds, head của 09, batch 32, seeds, optimizer và BCE checkpoint selection. Nhánh sparse lấy uniformly từ valid cached token order; nhánh 80 giữ original arrays/padding.

Report trong **`results 10.zip`**: `COMPLETE`, 4.349 weak / 58 Gold, 327.497 encoded images, không empty studies; 15 checkpoints cho ba densities × năm folds. Tổng 14,914 phút, encoding 629,10 giây, feature RAM 1,345 GiB.

Cả năm density-80 fold AUC deviations so với 09 đều bằng 0. Predictions gần như nhau nhưng không byte-identical: weak OOF probability drift tối đa ~5,9e-7; checkpoint tensor drift tối đa ~3,7e-5. Chênh lệch cực nhỏ giữa pooled OOF của 09 và report 20 là numerical/serialization, không phải gain mới.

| Density | Weak pooled OOF | Gold58 audit | Weak delta so với 80 |
| --- | ---: | ---: | ---: |
| 80 | 0.858970 | 0.888217 | 0 |
| 32 | 0.846273 | 0.863451 | −0.012697 |
| 16 | 0.821821 | 0.836134 | −0.037148 |

Report-group bootstrap 95% CI, khoảng 250 draws:

- 32 − 80: [−0.01522, −0.01035].
- 16 − 80: [−0.04023, −0.03359].

Cả 12 weak targets giảm ở density 32. Các mức giảm lớn nhất: Lateral Meniscus −0.04572, ACL −0.02469, Baker's −0.02237. **Dense coverage có lợi rõ trong pipeline hiện tại**; giảm còn 16 tokens chưa phải cách giữ chất lượng B2 có bằng chứng.

### Tín hiệu exploratory probability pooling

Hai mixtures được đánh giá sau khi đã xem density results:

| Probability pool | Weak pooled OOF | Delta so với 80 |
| --- | ---: | ---: |
| 80% density 80 + 20% density 32 | 0.860584 | +0.001614 |
| 50% density 80 + 50% density 32 | 0.859658 | +0.000688 |

80/20 có bootstrap CI khoảng [+0.00115, +0.00209]; 50/50 có CI cắt 0. Đây là **exploratory analysis trên cùng weak labels**, chưa phải independent prospective test. Chưa xác lập gain so với Public 0.914, anchor 0.943 hay Private.

Không gọi weight 20% là đã preregister trước notebook 20; không dò nhiều weights bằng Public rồi coi đó là validation độc lập.

## 11. Notebook 21: contract chạy và kiểm tra

Notebook: `21_B2_Density80_32_FixedBlend_Submit.ipynb`  
Builder: `build_21_density_blend_submit.py`  
Local test: `test_21_density_blend.py`

### Inputs cần attach

1. Competition RSNA Knee data: `test.csv`, `test_series.csv`, `test_series/`, `sample_submission.csv`.
2. **Exact successful notebook 20 saved Output**, chứa:
   - `density_ablation_report.json`.
   - `density_80_fold_0_head.pt` đến `density_80_fold_4_head.pt`.
   - `density_32_fold_0_head.pt` đến `density_32_fold_4_head.pt`.
3. Public Dataset `resnet-50-radimagenet-marwan`, có pinned `ResNet50.pt`.

Metadata tự liệt kê competition; notebook 20 Output và RadImageNet Dataset cần attach thủ công nếu import không phục hồi. Ảnh mới nhất của người dùng có đủ ba inputs và Internet Off; cần kiểm tra GPU trong Accelerator submenu. **Không cần** 05A study cache, notebook 02 silver hoặc notebook 09 heads cho inference này.

### Model và guards

```text
Decode/sort/crop một lần → tối đa 80 ảnh → frozen RadImageNet features một lần
Năm density-80 heads → mean probability p80
Uniform subset từ cùng valid token sequence → năm density-32 heads → mean p32
submission = 0.8 * p80 + 0.2 * p32
```

- T4×2 nhưng chỉ dùng GPU 0. Inference vẫn cần GPU dù đã train xong.
- Float16 feature quantization khớp 09/20; head prediction giữ normalization và token conventions.
- Phải tìm được một head root rõ ràng có đủ mười checkpoints.
- Report phải `COMPLETE`, densities [80,32,16], weak 4.349 / Gold 58, đúng encoder SHA; cả năm control fold AUC deviations ≤1e-8.
- Kiểm tra density/fold/label order/config/state dict provenance của mỗi checkpoint.
- Trước khi ghi submission mới, nhánh 80 không thay đổi phải khớp notebook 10 trên ba visible reference UIDs trong tolerance probability 1e-4.
- Hidden UIDs không có visible reference **không được coi là đã independently validated** bằng gate đó. Với cohort ba visible studies, yêu cầu đủ ba UID matches.
- Kiểm tra UID join, sample columns/order/coverage, finite probabilities trong [0,1] và CSV roundtrip.
- Empty study trả 0.5 và được đếm; `COMPLETE` không tự đồng nghĩa zero empty/decode failures.

Outputs:

```text
submission.csv
b2_density80_control.csv
b2_density32_control.csv
b2_density_blend_receipt.json
```

Local verification đã pass syntax, control checkpoint/OOF provenance, token selection và mocked CPU inference với 0/1/10/32/80 tokens. **Cập nhật 11/10: Kaggle visible run đã COMPLETE; người dùng xác nhận Public 0.913.** `results 21.zip` xác nhận visible control parity/schema/fixed blend, không chứa hidden predictions/runtime hoặc checkpoints để re-audit từng head. Xem audit riêng `RSNA_21_Audit_and_22_Direction_2026-10-11.md`. Nếu reproduction gate fail trong run khác, chẩn đoán assets/preprocessing/runtime trước; không nới tolerance rồi gọi đó là model improvement.

## 12. Attach matrix và cách tiết kiệm GPU

| Workflow | Inputs cần thiết | GPU / Internet | Output mục tiêu |
| --- | --- | --- | --- |
| 05A SAFE fold/final training | Competition + 02 silver + 03 manifest + cache/fold assets theo gate; ưu tiên complete cache Dataset | GPU / Off nếu dependencies đã đủ | Fold checkpoints hoặc completed final14 |
| 05A Submit SAFE | Competition + completed Final14 Output với `05A_dense_mil_final.pt` | GPU / Off | Submission; không train lại |
| 06/07 public evidence | 18 sources gốc của anchor: competition + 17 public artifacts | CPU None / Off | Inventory, OOF evidence; không submission |
| 08/09/14/15/18/19/20 owned B2 | Competition + 02 silver + complete 05A cache/folds + RadImageNet weights; upstream control Output nếu code yêu cầu | T4×2 / Off; đa số dùng GPU 0 | Preflight/OOF/checkpoints tùy notebook |
| 11 B3 preflight | Training assets + pinned timm pretrained weight | GPU; bước download có thể cần Internet | `b3_preflight.json` và saved weight |
| 12 B3 train | Competition + 02 silver + 05A cache/folds + successful 11 Output | T4×2 / Off | B3 OOF/checkpoints |
| 10 B2 standalone | Competition + complete five-fold 09 heads + RadImageNet weights | T4×2 / Off | Submission + receipt |
| 16/17 anchor+B2 | Inputs gốc của anchor + complete 09 heads + RadImageNet weights | T4×2 / Off | Controlled rank probes |
| 21 density pool | Competition + 20 Output với 80/32 heads + RadImageNet weights | T4×2 / Off | Fixed probability probe |

Bảng là hướng dẫn theo workflow, không thay asset gate của từng version. 18 inputs của anchor không tự động là minimal set cho owned B2. Notebook Output, Dataset và Model Instance là các attachment types khác nhau; kiểm tra filenames thực mount và gate ở cell đầu.

**Nguồn để phục hồi anchor:** metadata của `rsna-knee-speedy-raptors-coatnet-d4-0943.ipynb` có 18 data sources: competition ID 154281; dataset version IDs 19120128, 19122845, 19134209, 18956429, 19395055, 18229736, 19395706, 19862359, 19459763, 18842180, 18839182, 18673450, 18879001, 18716507; kernel version IDs 342671664, 342849430; model instance version ID 4533. Đây là IDs của imported source metadata, không phải danh sách “latest version” để thay tùy ý. Inventory 06 giúp map các mounted paths; không copy GPU flags cũ mà bỏ kiểm tra Kaggle settings hiện tại.

Trước run tốn GPU:

1. Kiểm tra input gate; phân biệt preprocessing cache, feature cache và checkpoints.
2. Benchmark đúng representation, density, microbatch và trainability dự định. Preflight 16 tokens không chứng nhận chi phí train 80 tokens.
3. Tính cả cache I/O, train, validation, Gold audit và checkpoint overhead.
4. Lưu checkpoints/metrics sau mỗi fold; run bị guard dừng phải có trạng thái PARTIAL rõ ràng.
5. Tránh decode DICOM lại hàng giờ khi đã có cache được xác minh.
6. Theo dõi host RAM và CUDA allocated/reserved riêng; `empty_cache()` không chữa retained host RAM.
7. Không hứa “chắc chắn không OOM” hoặc runtime/score bảo đảm khi chưa chạy đúng Kaggle hardware/cohort.

## 13. Định hướng cải tiến và tiêu chí quyết định

### 13.1 Ngắn hạn: đóng 21 và triển khai 22

Fixed standalone 80/20 probe 21 đã cho Public **0.913**, không tăng so với **0.914**. Bước mới là public 2.5D ConvNeXt reader thêm bên ngoài unchanged anchor 0.943, sau audit exact contract/provenance/eligibility và baseline reproduction; không thay CoAt incumbent. Notebook 22 đang được triển khai local, chưa có run/score.

Nhánh frozen B2 density pool đã dừng. Không dò hàng loạt 32/16 weights hoặc target groups trên Public chỉ để tìm một rounded gain. Kế hoạch chi tiết hiện tại nằm trong `RSNA_21_Audit_and_22_Direction_2026-10-11.md`; không bảo đảm vượt champion hay Private gain.

### 13.2 EDA có thể làm ngay từ outputs hiện có

- Audit per-target errors, counts và paired metrics trên đúng owned UID/fold cohort.
- Phân tầng theo vendor, field strength, spacing, số series, slot availability và teacher agreement có provenance đáng tin. Kiểm tra mỗi subgroup có đủ mẫu và cả hai classes.
- Kiểm tra near-duplicates của ảnh/report, site/scanner leakage, train-validation discrepancy.
- Phân biệt lỗi ảnh với bất đồng teacher: weak CV có thể penalize dự đoán đúng ảnh nhưng khác silver. Gold58 chỉ audit hạn chế, không dùng tối ưu target weights.
- Xem ảnh model thật sự nhận: crop, geometry, plane selection, acquisition thiếu và anatomy bị mất khi sparse sampling. Attention maps là chẩn đoán, không tự chứng minh localization đúng.
- Giữ OOF hiện có; không chạy lại inventory 06/07 nếu inputs không đổi.

Để phân tích định lượng toàn anchor 0.943, cần independent predictions/provenance cho mọi family. Nếu chưa có, ghi rõ khoảng trống; inference train bằng full-data checkpoint không được gọi là OOF.

### 13.3 Model tự phát triển mạnh hơn: giữ dense coverage

Notebook 20 ủng hộ tối đa 80 cached tokens làm reference tiếp theo. Hướng mới phải khác rõ cấu hình 14/15 đã thất bại:

1. **Full-density adaptation:** so frozen 80 với adaptation ở last encoder block, giữ input/folds/head selection. Làm resource gate đúng 80 tokens. Dùng microbatch/gradient accumulation khi cần, kiểm tra effective batch và optimization budget thực tế. Notebook 19 chưa ủng hộ scale nguyên cấu hình sparse 16, cũng chưa chứng minh adaptation 80 sẽ tăng.
2. **Native 2.5D / depth context:** dùng DICOM thực sự liền kề hoặc depth structure trong từng series, với encoder phù hợp/được adaptation. Cache 05A là sampled slices; notebook 15 không chứng nhận native adjacency. Cần cache/config provenance mới và I/O/throughput preflight.
3. **Anatomy / sequence-aware representation:** plane/sequence branches có ý nghĩa, target-conditioned fusion hoặc global+local features, giữ thông tin depth/spatial. Lặp frozen hierarchical pooling của 14 không phải hướng mới. Chỉ routing theo metadata đã audit; `Fluid/Non-fluid` không tự tương đương T1/T2/PD.
4. **Resolution / physical crop / selection:** mỗi lần thay một factor: 224→resolution cao hơn, 140 mm→crop khác, hoặc anatomical/quality selection không dùng labels. Upsample cache 224 lên 336/384 không khôi phục native detail. Preprocessing mới có thể tốn nhiều giờ; đo trước full train.

Bắt đầu bằng matched one-fold/pilot và cost estimate, rồi scale nếu gain có tính lặp lại. Không gọi model mạnh chỉ vì backbone có nhiều parameters hơn.

### 13.4 Supervision và label uncertainty

L1 là teacher usable, chưa được chứng minh là trần chất lượng. Controlled tests có thể thay teacher-family composition, confidence weighting hoặc finding definitions. Giữ model/folds không đổi để cô lập supervision; ghi nguồn nào bị Gold contamination; tránh reliability weights được tối ưu trên 58 Gold.

Synovitis/Effusion/Baker's có disagreement đáng nghiên cứu. Không mặc định mọi lỗi đều do backbone, cũng không đổi silver nhiều lần rồi chọn bằng Gold/Public.

### 13.5 MRI ↔ report alignment: chưa triển khai

`Hướng khác.md` và roadmap có ý tưởng ConVIRT/CLIP-style study-level alignment:

```text
MRI → image/study encoder → z_image
Report → multilingual text encoder → z_text
Train: soft BCE + alignment/contrastive/distillation
Test: MRI-only image branch → 12 probabilities
```

4.349 weak MRI studies **đã được dùng** cho classification qua L1. Report alignment là thêm semantic supervision, không phải “bắt đầu sử dụng các ảnh đang bị bỏ qua vì thiếu official labels”.

Pilot có thể cache frozen study/text embeddings, train projection heads với effective batch 128–512; đây là thiết kế đề xuất, chưa phải cấu hình thành công. Reports multilingual; BGE-M3, XLM-R hoặc translation+medical encoder là candidates, chưa chọn winner. Queue/distillation có thể hỗ trợ online batch nhỏ.

Group duplicate reports và không cho validation report/image tiếp xúc pretraining khi yêu cầu clean cross-fitting. Gold58 giữ ngoài optimization theo owned-model contract. Retrieval Recall@K chỉ là auxiliary; cần classification OOF gain. Chưa triển khai MRI→generated report→classifier pipeline. Ý tưởng train Gold/all pairs trong tài liệu cũ không tự thay locked audit-only policy.

### 13.6 External pretraining / localization: chưa triển khai

MRNet, fastMRI+, SKM-TEA, OAI/MOST, KneeMRI, ROCO/BiomedCLIP được nhắc như nguồn năng lực. **Chưa xác nhận external raw MRI dataset nào đã được dùng train owned experiments.**

Trước triển khai phải kiểm tra access/license/competition eligibility, overlap và protocol/pathology fit. Publicly discoverable không đồng nghĩa unrestricted derivative use. Chọn nguồn để giải quyết một gap cụ thể như anatomy, ACL, meniscus hay localization, không chỉ thêm volume. Public pretrained weights đã dùng trong B2/B3 không tương đương tự train raw external datasets.

### 13.7 Ensemble và final selection

Mục tiêu là ít families nhưng mạnh và bổ sung thực sự, có supervision/OOF provenance rõ; không phải nhiều near-duplicates. Đánh giá per-target strengths, prediction/residual correlation, paired OOF blend gain và tổng hidden-inference budget. Gold/Public nên kiểm chứng hoặc cảnh báo, không đóng vai weight optimizer.

Trước deadline: giữ version 0.943 cùng inputs/checkpoints; kiểm tra final selection trên Kaggle. Các screenshot trước từng chọn 0.748 hoặc 0.942 ở thời điểm khác; **current final selection chưa được xác minh**. Ghi seed/version/hash/weights để offline reproduce. Đừng để candidate yếu hơn được chọn chỉ vì nó mới nhất.

## 14. Những hướng không nên lặp nguyên trạng

| Hướng/cấu hình | Bằng chứng | Quyết định hiện tại |
| --- | --- | --- |
| Nới RAM growth guard trong 05A | RSS vẫn tăng đều, guards fail lặp lại | Không coi là fix; dùng SAFE dataflow/cache đã chạy |
| Frozen generic CoAtNet 12 | Weak OOF 0.7788 so với B2 0.8590 | Không submit/blend cấu hình này |
| Frozen hierarchical plane head 14 | Tổng giảm; 10/12 labels giảm | Không đổi tên thành “CoPAS mới” rồi chạy lại y hệt |
| Sampled-cache 2.5D 15 + frozen encoder | Cả 12 labels giảm | Không submit; phân biệt native adjacency |
| B2 rank blends 16/17 | Public 0.942/0.942/0.942/0.943 | Chưa có gain so với anchor |
| Layer4 fine-tune 19 ở 16 tokens | Delta rất nhỏ, CI cắt 0; ~77,8 phút | Không scale nguyên cấu hình lên năm folds |
| Sparse 32/16 một mình ở 20 | 32 giảm cả 12 targets | Không thay dense 80 nếu chưa có lý do mới |
| “Full anchor OOF EDA” từ ba visible UIDs | Không có hidden labels/full OOF | Không suy AUC từng nhãn hoặc Private score |

## 15. File và ZIP map để tiếp quản

### Output archives

```text
RSNA_Knee_EDA_Artifacts.zip                           → 00 EDA
RSNA_Knee_Label_Audit_v2_Artifacts.zip                → 01 label audit
RSNA_Knee_Silver_Label_v1_Artifacts.zip               → 02 L1 labels
RSNA_Knee_Baseline_V1_Prep_Artifacts.zip              → 03 preparation
RSNA_05A_Fold0_DenseMIL_Artifacts.zip                 → 05A fold 0
RSNA_05A_Fold1_DenseMIL_Artifacts.zip                 → 05A fold 1
results_0.943.zip                                   → protected anchor visible output
results_06 phân tích các input của notebook 0.943.zip → 06 inventory
results_07.zip                                      → 07 public OOF evidence
results_08.zip                                      → 08 encoder preflight
results_09.zip                                      → 09 five-fold B2
results_10.zip                                      → 10 standalone visible output
results_11.zip                                      → 11 B3 preflight + weight
results_12.zip                                      → 12 B3 five-fold
results_13.zip                                      → 13 visible integrity
results_14.zip                                      → 14 plane-aware head
results_15.zip                                      → 15 sampled-neighbor 2.5D
results_16.zip                                      → 16 anchor + B2
results_17 G1.zip                                   → 17 G1
results 17 G2.zip                                   → 17 G2
results_17 G3.zip                                   → 17 G3
results 18.zip                                      → 18 layer4 preflight
results 19.zip                                      → 19 paired fold 0
results 10.zip                                      → 20 density ablation, KHÔNG PHẢI 10
results 21.zip                                      → 21 visible receipt + 80/32 controls + submission; Public user-confirmed 0.913
```

`results.zip`, old notebooks và filenames chung chung phải được nhận diện bằng manifest/receipt trước khi dùng. Filename không đủ làm provenance. Output 21 đã được audit; notebook 22 đang triển khai local và chưa có Output ZIP/Kaggle score.

### Analysis, builders và tests

| File | Vai trò |
| --- | --- |
| `audit_raptor_0943.py` | Audit anchor receipts và visible branch integrity |
| `analyze_results_13_14.py` | Paired 09/14 metrics |
| `analyze_16_public_oof_b2_diagnostic.py` | Partial public OOF proxy + B2 diagnostic |
| `audit_20_slice_density.py` | UID/fold reconciliation, density metrics, exploratory pools, group bootstrap |
| `build_13_14_notebooks.py` | Generate 13/14 |
| `build_15_notebook.py` | Generate 15 |
| `build_16_anchor_b2_probe.py` | Generate 16 |
| `build_17_group_ablation_probes.py` | Generate 17 G1/G2/G3 |
| `build_18_layer4_preflight.py` | Generate 18 |
| `build_19_layer4_fold0.py` | Generate 19 |
| `build_20_slice_density.py` | Generate 20 |
| `build_21_density_blend_submit.py` | Generate 21; đưa visible 10 references từ old ZIP vào gate |
| `test_20_slice_density.py` | CPU token selection/padding/MIL forward-backward |
| `test_21_density_blend.py` | CPU syntax/control provenance/selection/mocked inference |

Builders tạo notebooks; khi sửa cần giữ builder và generated artifact nhất quán. Không chạy tất cả builders mù quáng vì có thể ghi đè user-edited notebooks. Workspace không phải Git repository tại thời điểm kiểm tra; giữ originals bằng tên/version mới.

Các lệnh local hỗ trợ tiếp quản, **chưa chạy lại toàn bộ trong turn viết handoff này**:

```powershell
python audit_20_slice_density.py
python test_20_slice_density.py
python test_21_density_blend.py
```

Local Python trước đây có thể in `Failed to find real location ...python.exe` rồi vẫn chạy tiếp thành công. Kiểm tra exit code và final output; không kết luận failure từ riêng dòng đó. Local torch là CPU; tests không xác minh full Kaggle GPU training.

## 16. Hồ sơ cần lưu cho mỗi thí nghiệm mới

**Trước run:** parent notebook/version; hypothesis; một factor thay đổi; training cohort; group-fold hash; seed; preprocessing/weight SHA; resource budget; checkpoint-selection rule; comparison xác định trước.

**Sau run:** COMPLETE/PARTIAL; OOF UID/fold coverage; weak BCE; per-target AUC và counts; paired deltas/uncertainty; Gold audit với caveats; runtime/host RAM/GPU peak; decode failures/empty studies/fallbacks; checkpoints; Public score nếu submit; quyết định giữ, bỏ hay pilot tiếp.

Nếu thiếu input hoặc chưa biết Public score, ghi “unknown”, không thế bằng metric trên cohort khác. Guard stop, successful import và inference ba visible studies không được báo như full training hoặc Private performance.

## 17. Nguồn và thứ tự đọc

Đọc handoff này trước, sau đó code/receipts thực tế của notebook hiện tại. Thứ tự ưu tiên bằng chứng:

1. User-confirmed Kaggle score/screenshot cho Public.
2. ZIP receipts, OOF, configs và checkpoints cho run.
3. Dated decision notes cho diễn giải.
4. Roadmaps cũ chỉ cho hypotheses, không phải current next action.

### Tài liệu local chi tiết

Mọi filename dưới đây nằm trong workspace `C:\Users\ADMIN\Documents\RSNA`:

- `RSNA_0943_Audit_and_Next_Experiment.md`: lịch sử 06/07/08/09; các đoạn “next step” cũ không còn current.
- `RSNA_0943_Direct_Decomposition_2026-10-06.md`: exact anchor formula và OOF gap.
- `RSNA_B2_0914_to_B3_2026-10-05.md`: B2 Public result, B3 preflight/weight.
- `RSNA_13_14_Audit_and_15_Decision_2026-10-06.md`: kết quả âm của 14.
- `RSNA_15_16_Decision_2026-10-06.md`: kết quả âm của 15, rationale cho proxy 16.
- `RSNA_17_Audit_and_18_Decision_2026-10-08.md`: Public 17 scores/provenance.
- `RSNA_19_Public_Code_and_Domain_Research_2026-10-08.md`: research hypotheses và 18/19.
- `RSNA_20_Audit_and_21_Decision_2026-10-09.md`: 20 audit và kế hoạch probe 21 tại thời điểm đó.
- `RSNA_21_Audit_and_22_Direction_2026-10-11.md`: 21 COMPLETE/Public 0.913, đóng density branch và current ConvNeXt-addition direction 22.
- `RSNA_Knee_Private096_Top10_Roadmap.md`: roadmap lớn, nhiều hướng chưa triển khai.
- `Hướng khác.md`: MRI-report alignment ideas, chưa phải executed protocol.

### Nguồn chính thức/nghiên cứu đã tham khảo

- [Competition overview/evaluation](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation) và [rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules): kiểm tra quy định trước final submission.
- [RadImageNet paper](https://pubs.rsna.org/radiology/doi/full/10.1148/ryai.210315): động cơ medical transfer learning, không bảo đảm LB của chúng ta.
- [PyTorch transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html): implementation reference cho frozen/fine-tuning.
- [CoAtNet model card](https://huggingface.co/timm/coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k): nhận diện generic pretrained B3.
- [Learning co-plane attention across MRI sequences for diagnosing twelve types of knee abnormalities](https://www.nature.com/articles/s41467-024-51888-4): động cơ structured plane/sequence fusion; dataset/evaluation khác cuộc thi.
- [Clinically Prioritized Attention-Based Fusion of Multi-Plane Knee MRI](https://waseda.elsevierpure.com/en/publications/clinically-prioritized-attention-based-fusion-of-multi-plane-knee/): động cơ learned slice/plane fusion; MRNet results không phải ước lượng RSNA LB.
- Các [single-model](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304), [pipeline/slice choices](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743148), [ensemble suggestions](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742926) là participant self-reports để phát sinh hypotheses, không phải controlled proof.

## 18. Prompt để chuyển sang chat/agent khác

```text
Read RSNA_Knee_Project_Handoff_2026-10-11.md first in
C:\Users\ADMIN\Documents\RSNA.

We are improving RSNA Knee toward Private Top 10, aspirational ~0.96.
Best confirmed Public is the protected imported ensemble at 0.943.
Owned B2 standalone notebook 10 is 0.914.
Anchor+B2 probes 16/17 did not beat the displayed 0.943.

Notebook 20 completed five-fold density 80/32/16 and replicated 09 control.
Its Output is 'results 10.zip' (space), NOT 'results_10.zip' (underscore).
Notebook 21 is a fixed 80/20 probability blend of density-80/32 B2 heads.
It completed: 'results 21.zip' contains its COMPLETE visible receipt,
80/32 controls and submission. User-confirmed Public is 0.913 versus
B2 standalone 0.914. Visible 80-token parity passed; hidden parity is unknown.

Close the frozen B2 density-blend branch. Notebook 22 is being implemented
locally: add a public 2.5D ConvNeXt reader outside the unchanged 0.943
anchor, rather than replacing its CoAt incumbent. Exact public contract,
provenance and eligibility are pending audit; no Kaggle 22 run/score yet.
Read RSNA_21_Audit_and_22_Direction_2026-10-11.md. Continue from 22,
not 21 rerun/weight sweeps or 05A RAMFIX in the old handoff.
Keep L1 soft labels, original report-group folds, Gold58 audit-only,
MRI-only inference, geometry sorting, physical crop and offline weights.
Preserve the 0.943 baseline; distinguish weak OOF from Public/Private.
Frozen plane head 14 and sampled-cache 2.5D 15 already failed.
New anatomical/plane/depth proposals must differ meaningfully and use
matched controls. Density 20 supports retaining dense coverage.

Prepare concrete, reviewable notebooks with proportionate checks.
Do not promise no OOM, guaranteed runtime, or 0.96 from static tests.
All proposed strong directions in section 13 are hypotheses, not
completed experiments or guaranteed improvements.
```
