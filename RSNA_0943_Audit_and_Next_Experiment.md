# RSNA Knee 0.943: audit trước khi cải tiến

Ngày 2026-10-02. Người dùng xác nhận đã submit notebook Speedy Raptors trên tài khoản chính và nhận Public LB **0.943**. Đây là mốc thực tế; file `.ipynb` lưu trên máy chỉ ghi log chạy 3 test studies nên không thay thế bằng chứng của lần submit đó.

## Kết quả kiểm tra ZIP do người dùng gửi

`results_0.943.zip` là output của run hiển thị trên **3 study mẫu**. Receipt ghi `COMPLETE`, 257.02 giây (~4 phút 17 giây), DINO 20 members, A5 5 folds, Raptor 4 views, CoAt đủ 4 members (Residual, Global96, D4, Repair-v1). CoAt dùng `rank_of_member_probability_mean`; không có event fallback/failure/neutral; cả bốn child phase đều `ok: true`. `diagnostics/a5_applicability.json` ghi cả 3 studies khả dụng. Đây là bằng chứng tốt cho run mẫu, **không cho biết thời gian hay trạng thái chi tiết của lần Kaggle chấm trên khoảng 1.300 study ẩn**.

ZIP không chứa file OOF/train/validation/gold prediction nào; 33 nguồn prediction được lưu đều chỉ bao phủ 3 study mẫu. Không có cơ sở trong ZIP để tính AUC từng nhánh, tương quan family trên cohort đủ lớn hay độ cải thiện của một blend mới.

Các child receipts ghi rõ checkpoint của Residual, Global96 và Repair-v1 được chọn bằng **Gold58 macro-AUC**. D4 cũng chứa Gold58 AUC và checkpoint selection riêng. Vì vậy AUC Gold58 trong các receipts là số đã qua quá trình chọn model, không dùng làm holdout độc lập để quyết định blend tiếp theo.

## Notebook đang làm gì

`rsna-knee-speedy-raptors-coatnet-d4-0943.ipynb` là pipeline inference từ nhiều artifact public, không phải một mô hình được train trong notebook:

1. DINOv2: 20 members, gộp rank theo fold.
2. A5 attention pooling: 5 fold checkpoints.
3. RadImageNet: ResNet50 và nhiều bộ heads, có calibrator dùng metadata số lượng acquisition.
4. Raptor: 4 views/arms.
5. CoAtNet: Residual, Global96, D4, Repair-v1; mặc định chia đều trong family nếu đủ 4 nhánh.
6. Blend cuối: trọng số CoAt/Raptor khác nhau theo target: 0.60 mặc định, ACL 0.75, Medial Meniscus 0.80, Lateral Meniscus 1.00, Lateral OA 0.75, Fracture 0.75.

Điểm cuối là macro ROC-AUC nên từng target đóng góp bằng nhau. Không thể suy ra chất lượng từng target từ một con số 0.943 tổng.

## Vấn đề đã thấy từ code

| Mức ưu tiên | Quan sát | Vì sao quan trọng | Cách kiểm tra |
|---|---|---|---|
| Cao | Mỗi nhánh CoAtNet có `except` để tiếp tục khi child lỗi; nếu thiếu một nhánh, code vẫn có thể xuất `submission.csv`. | Bài nộp có thể khác công thức 4 nhánh mô tả trên đầu notebook. | Đọc `_coat_raptor_blend_receipt.json`: `members` phải đủ 4 tên và `family_reduction` phải là `rank_of_member_probability_mean`; đọc thêm `btkd_v559_complete.json/events`. |
| Cao | Không có per-target AUC/OOF của ensemble 0.943 trong output inference. | Chưa biết target nào kéo điểm xuống và nhánh nào bổ sung thông tin có ích. | Lấy OOF/holdout predictions và nhãn được giữ ngoài training của **từng** family; đánh giá macro/per-target AUC, bootstrap theo study/report group. |
| Cao | Blend cuối dùng trọng số theo target đã cố định trong code, nhưng notebook này không cung cấp bằng chứng validation độc lập cho từng trọng số. | Thử thêm trọng số theo Public LB dễ overfit; 0.943 không đủ chứng minh Private LB. | Chỉ thay trọng số khi có OOF hợp lệ; giữ nguyên 0.943 làm control. |
| Trung bình | Nhiều tầng rank và hai lần blend; Rad branch có calibrator theo acquisition metadata. | Khó biết gain đến từ mô hình ảnh hay phân bố scanner/protocol; có thể lệch domain trên Private. | Audit phân bố score và bất đồng giữa family theo vendor, field strength, plane/slot availability, study source. |
| Cao | Child receipts xác nhận Gold58 được dùng chọn checkpoint cho ít nhất Residual, Global96 và Repair-v1. | Gold58 AUC 0.90–0.93 trong receipts là kết quả sau chọn checkpoint, không phải holdout độc lập cho blend. | Không dùng Gold58 để tune tiếp; lập validation theo report group và xem provenance của toàn bộ public family. |
| Trung bình | Run mẫu 3 studies mất 257 giây; với cohort >48 code chuyển 4 CoAt readers sang chạy tuần tự. | Không thể ngoại suy 4 phút 17 giây thành thời gian chấm ẩn. | Chỉ dùng runtime lần chấm nếu Kaggle thực sự công bố; giữ biên an toàn 9 giờ. |

## Thí nghiệm tiếp theo

**Bước 1 — kiểm tra output hiển thị, không dùng GPU.** Tải Output của đúng version đạt 0.943 và chạy:

```powershell
python audit_raptor_0943.py C:\duong-dan\output.zip
```

Script đọc trực tiếp ZIP hoặc thư mục, in lỗi/fallback, xác nhận 4 CoAt members và kiểm tra số study. Kaggle Code Competition chạy lại notebook trên test ẩn để chấm và **không công bố output đầy đủ của lần chấm đó**. Nếu ZIP chỉ có 3 study mẫu, script sẽ dừng trước phần tương quan. Receipt 3 study chỉ xác minh lần chạy hiển thị, không chứng minh rằng 4 nhánh đều hoạt động trong lần chấm ẩn. Nếu có một output holdout độc lập đủ lớn, script mới xuất `raptor_0943_branch_diagnostics.csv`; đây là độ khác biệt, **không phải AUC**.

**Bước 2 — xác minh holdout.** Muốn tăng score một cách có cơ sở, cần predictions OOF và nhãn hợp lệ cho các family chính. Kiểm tra study overlap, report-group overlap và provenance của từng checkpoint trước. Nếu không có OOF tin cậy, không chỉnh blend theo Public LB; chuyển sang train một family mới có OOF riêng.

Notebook `06_RSNA_0943_Public_Artifacts_Inventory_CPU.ipynb` chỉ kiểm kê các public input đã dùng bởi notebook 0.943. Chạy trên Kaggle với Accelerator `None`, Internet `Off`, giữ 18 nguồn input trong metadata (competition + 17 public). Kết quả cần gửi lại là Output ZIP chứa `artifact_inventory.csv` và `manifest_evidence.csv`. Notebook không chạy inference và không tiêu thụ GPU quota. Nếu Kaggle không tự phục hồi các Input khi import, dừng và kiểm tra sidebar trước khi Save & Run All.

### Kết quả kiểm kê 06 (ZIP người dùng gửi ngày 2026-10-02)

`results_06 phân tích các input của notebook 0.943.zip` chứa 43 đường dẫn liên quan và 208 dòng manh mối manifest. Có **4 CSV được đặt tên OOF**: `v52_oof.csv` của RadImageNet E9 và ba bản `v52_e11_oof.csv` từ dataset E11 / notebook E11 / notebook E13; thêm `oof.npz` của Pilkwang. Đây là ứng viên OOF, **chưa xác minh** schema, ID, nhãn, fold independence hay việc ba E11 CSV có trùng nội dung. JSON `public_oof_diagnostic.json` tự khai 4.407 OOF rows, 5 outer folds và 58 gold rows; không được coi đây là AUC độc lập cho ensemble 0.943. Manifest các CoAt nhắc rõ `selection_metric = official_gold58_macro_auc`, củng cố cảnh báo Gold58 đã dùng chọn checkpoint. Chưa thấy OOF của toàn bộ DINO/Raptor/CoAt family trong danh sách 43 file.

Notebook `07_RSNA_0943_OOF_Evidence_Export_CPU.ipynb` là bước kế tiếp, vẫn **CPU only**, Internet Off, 18 Input giống 06. Nó xuất đúng các CSV/NPZ/JSON OOF nhỏ và `train.csv`/`train_series.csv` từ competition (nếu tìm thấy), kèm SHA256 và số dòng. Không chạy model, không nộp bài, không đổi blend. Khi có Output ZIP của 07 mới kiểm được duplicate, cohort overlap, group leakage và per-target AUC hợp lệ. Nếu 07 thiếu competition `train.csv`, không suy diễn metrics; xử lý attachment/schema trước.

### Kết quả kiểm tra 07 (ZIP người dùng gửi ngày 2026-10-02)

`results_07.zip` chứa đủ 4 OOF CSV, `oof.npz` của Pilkwang, hai JSON manifest và cả `train.csv`/`train_series.csv`. Bốn CSV và NPZ đều có đúng **4.407 StudyInstanceUID**, thứ tự ID khớp `train.csv`, 12 target khớp, xác suất hữu hạn trong [0,1], không thiếu ID/giá trị. Bốn CSV có cùng fold assignment (882, 882, 881, 881, 881) và `is_gold`; 58 gold studies chia 10/15/16/9/8 theo fold. Ba CSV mang tên E11 **không trùng dự đoán** (SHA256 khác, chênh lệch số học); không được gộp như ba bản copy của một OOF. `oof.npz` thuộc bundle Pilkwang DINOv2 20 members theo `manifest.json`, nên sửa kết luận của kiểm kê 06: có **một ứng viên DINO OOF**, ngoài các ứng viên Rad/E11/E13. Vẫn **không thấy OOF đầy đủ cho A5/Raptor/4 CoAt**, tức không có OOF của toàn bộ công thức 0.943.

Nhãn official trong `train.csv` chỉ có ở **58/4.407** studies, cả 12 target đều đủ trên cùng 58; 4.349 studies còn lại không có official labels. NPZ có `y_derived` trên 4.407 study, nhưng đó là nhãn suy ra, không tương đương official gold. Trên 58 gold, mức trùng `y_derived` với official label chỉ 60,3% ở Contusion và 70,7% ở Effusion/Synovitis (77,6% ở Lateral Meniscus/PF OA). Vì vậy AUC trên 4.349 weak-label studies chỉ là kiểm tra proxy, không được báo như Private-LB estimate. Exact normalized Report tạo 4.257 groups, 54 groups lặp lại và **không group lặp nào đi qua hai fold**; đây là một kiểm tra chống exact-report leakage, không chứng minh mọi near-duplicate/image-level leakage đều vắng mặt.

Macro AUC mô tả trên cùng 58 gold: E11 dataset 0,8256; E9 Rad 0,8543; E11 notebook 0,8375; E13 notebook 0,8576; Pilkwang DINO OOF 0,8400. Những số này không so trực tiếp với Public LB 0,943 (cohort khác) và Gold58 đã được một số public model sử dụng chọn checkpoint. AUC theo nhãn cho các OOF này thường yếu ở Lateral Meniscus/Lateral OA, nhưng **không suy ra** target yếu của ensemble 0.943 vì thiếu OOF của Raptor/CoAt và Public LB không có breakdown. Không dùng các macro-AUC này để tune lại trọng số hay chọn model.

Kết luận vận hành: không cần chạy thêm notebook kiểm kê và chưa có căn cứ để sửa submission 0.943. Muốn thử một model mới, phải thiết kế group-CV theo report với prediction OOF của model mới, kiểm tra weak-label noise và dùng Gold58 chỉ như audit có ghi rõ selection bias. Để đánh giá blend với 0.943, còn cần OOF/validation predictions của A5, Raptor và cả 4 CoAt hoặc một holdout independent có đủ prediction từ toàn bộ pipeline. Không tạo một trọng số mới bằng Public LB hoặc Gold58.

**B2 execution gate:** `08_B2_RadImageNet_Encoder_Preflight_T4.ipynb` kiểm tra RadImageNet ResNet50 checkpoint pinned bằng SHA256 và 05A uint8 cache đầy đủ trước khi dùng GPU; sau đó encode tối đa 64 weak studies trên một T4, giới hạn benchmark 15 phút, ghi throughput/RAM GPU vào `b2_preflight.json`. Đây là preflight, không train toàn bộ 4.349 study và không tạo submission. Chỉ thiết kế/chạy B2 group-CV dài sau khi số đo này xác nhận thời gian và tài nguyên có biên an toàn. Cần attach đúng saved 05A Output chứa `rsna_05a_dense_mil/study_cache` và public RadImageNet weight; 18 public inputs được giữ trong notebook metadata. Internet Off.

**Kết quả 08:** `results_08.zip` ghi `COMPLETE`, cache private Dataset có đúng 4.349 weak/58 gold, RadImageNet ResNet50 đúng SHA256 pinned. Một T4 encode 4.717 ảnh/64 study trong 12,72 giây (370,7 ảnh/giây), VRAM peak allocated 0,233 GiB, reserved 0,305 GiB, không study mẫu nào rỗng. Ngoại suy tuyến tính riêng encoder trên 4.407 study là 0,243 giờ (~14,6 phút), **chưa gồm** đọc/ghi feature đầy đủ hoặc train head; đây không phải cam kết runtime cuối.

**Notebook 09:** `09_B2_RadImageNet_FrozenMIL_Fold01.ipynb` encode 4.407 study vào feature RAM float16 một lần, sau đó train MIL head trên hai fold 0/1 cũ với soft L1 labels; gold chỉ audit sau khi checkpoint được chọn bằng weak validation BCE. Nó xuất weak OOF của hai fold, gold predictions, metrics và checkpoint head, **không xuất submission**. Đầu vào bổ sung cần attach notebook 02 Output chứa L1 silver; private `Study_case 05_RSNA` Dataset đã chứa cache và `fold_assignments.csv`. B2 này thay cả encoder family lẫn freeze policy so với 05A/B1, do đó chỉ là feasibility/model-family probe, không được báo như hiệu ứng nhân quả riêng của pretraining.

### Audit CPU trước khi chạy 09 (2026-10-02)

- Kiểm tra ZIP 02, 05A và 07 trên máy: L1 có 4.349 ID duy nhất, fold file 4.349 ID duy nhất, `train.csv` có 4.407 study gồm đúng 58 official gold. L1 + gold phủ đúng 4.407 ID, 12 nhãn silver đều hữu hạn trong [0,1]. Năm fold 05A có 870/870/870/870/869 study, không có `report_group` trùng giữa train và validation. Cả 12 target đều có hai lớp trên fold 0 và 1 khi threshold silver ở 0,5, nên macro AUC của phép thử có thể tính đủ target.
- OOF public Rad/E11/E13/Pilkwang có số fold khác 05A: chỉ 901/4.349 study weak giữ cùng chỉ số fold (20,72%). Vì vậy không được ghép bằng số fold hay thứ tự dòng; ghép theo `StudyInstanceUID`. Trên đúng 1.740 study ở fold 0+1 của 05A, AUC proxy với nhãn L1 nhị phân là E11 dataset 0,8287; E9 Rad 0,8341; E11 notebook 0,8307; E13 notebook 0,8406; Pilkwang DINO 0,8514. Các model public không có provenance độc lập theo fold 05A, và nhãn L1 là weak; đây chỉ là ngữ cảnh, **không phải holdout độc lập, không phải ước lượng 0,943 hay Private LB**.
- Control 05A trên chính fold/nhãn này: fold 0 weak macro AUC 0,7230, Gold58 audit 0,7184, runtime 89,6 phút; fold 1 tương ứng 0,7483, 0,7304, runtime 95,6 phút. So sánh 09 với 05A theo từng fold trên weak proxy, đồng thời xem BCE và per-target; không dùng Gold58 để chọn cấu hình.
- Đã thử forward/backward và một lượt train/eval `StudyMIL` bằng dữ liệu giả trên CPU; JSON notebook và AST code hợp lệ. Chưa thể xác minh toàn bộ 4.407 cache hay thời gian train head trên Kaggle bằng CPU local. Preflight 08 chỉ đo encoder khoảng 14,6 phút ngoại suy, không đo head hoặc I/O đầy đủ.
- `max_total_hours` của 09 từng là **3 giờ chốt an toàn nội bộ**, không phải giới hạn Kaggle hay ETA. Sau audit, nâng lên **6 giờ** để tránh dừng sớm theo thời gian chạy control 05A; giữ RAM guard 5 GiB, checkpoint tốt nhất theo weak BCE, patience 4 và báo trạng thái `PARTIAL_GUARD` nếu phải dừng. 6 giờ vẫn là ngân sách tối đa cho thử nghiệm, không bảo đảm thành công hay thời gian thực tế.

Khuyến nghị hiện tại: **không chạy 09 khi quota T4×2 chỉ còn khoảng 1 giờ**. Đợi quota hồi, chạy cell kiểm tra input bằng CPU trước; chỉ khi hiện `ASSET CHECK PASSED` mới bật T4×2 và `Save & Run All`. Kiểm tra `run_report.json` là `COMPLETE` và có đủ fold 0/1 trước khi diễn giải OOF. Không submit 09 trực tiếp.

**Bước 3 — một cải tiến có kiểm soát.** Ưu tiên family ảnh bổ sung cho ensemble hiện tại, không train lại 20 DINO + 4 CoAtNet. Ứng viên đầu tiên là một model plane-aware/focal cho target có OOF yếu và bất đồng cao. Train trên fold group theo report, tạo OOF trên weak studies, audit scanner/site; sau đó kiểm tra gain khi thêm model bằng OOF. Chỉ submit một candidate nếu OOF macro tăng và không có target lớn bị giảm đáng kể. Giữ nguyên notebook 0.943 làm control/backup.

## Dữ liệu còn thiếu  diễn giải như output test ẩn.
- Runtime và per-target breakdown **nếu Kaggle thực sự hiển thị**; mặc định chỉ có score tổng.
- OOF/holdout predictions và provenance của public family checkpoints nếu muốn đánh giá blend/model gain. Không có nhãn test, nên output inference chỉ cho phép EDA về phân bố và bất đồng.

Không có thí nghiệm nào ở đây bảo đảm đạt Private 0.959. Tiêu chí chọn là tiến bộ trên validation đáng tin và khả năng tái hiện trong giới hạn iđể kết luận định lượng

- Output ZIP của version được 0.943 để xác minh run hiển thị và artifact contract. ZIP này có thể chỉ chứa 3 test studies mẫu; không đượcnference.
