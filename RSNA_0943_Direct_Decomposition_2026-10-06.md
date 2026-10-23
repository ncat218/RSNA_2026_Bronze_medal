# Phân rã trực tiếp mốc Public LB 0.943 — 2026-10-06

## Kết luận

`rsna-knee-speedy-raptors-coatnet-d4-0943.ipynb` là **baseline được bảo vệ**. Notebook 09/14/15 thử một family RadImageNet mới bên cạnh nó; chúng **chưa** đo được mức tăng của toàn ensemble 0.943. Notebook 14 đã thua control 09 trong phép so OOF cùng UID/fold/nhãn weak, nên không có căn cứ đưa nó vào baseline. Không thay công thức 0.943 hoặc tiêu lượt submit dựa trên kết quả đó.

## Công thức thực tế ở tầng cuối

Code ở cell 27 lấy `BTKD` (DINOv2 → A5 → Rad + calibrator), `Raptor` và `CoAt family` (4 child khi đủ). Với mỗi nhãn, nó tính:

`H = 0.60 × rank(Raptor) + 0.40 × rank(mean_probability(4 CoAt))`

`Final = rank((1-w) × rank(BTKD) + w × H)`

`w` là biến `_coatnet_weight` trong code, nhưng tên này gây hiểu nhầm: đó là trọng số của **cả nhánh Raptor+CoAt**, không phải CoAt đơn lẻ. Bảng dưới là hệ số đại số của các thành phần rank **trước lần rank cuối**; không phải tỷ lệ đóng góp vào AUC hay hệ số trên xác suất gốc.

| Nhãn | BTKD | Raptor | CoAt family |
|---|---:|---:|---:|
| Mặc định (7 nhãn còn lại) | 0.40 | 0.36 | 0.24 |
| ACL, Lateral OA, Fracture | 0.25 | 0.45 | 0.30 |
| Medial Meniscus | 0.20 | 0.48 | 0.32 |
| Lateral Meniscus | 0.00 | 0.60 | 0.40 |

Riêng Lateral Meniscus, **BTKD bị loại hoàn toàn ở tầng cuối**. Đó là một giả thuyết cần kiểm chứng, không phải bằng chứng nhãn này yếu: Kaggle chỉ công bố Public macro score 0.943, không có per-target score của ensemble. 4 CoAt child được trung bình **xác suất** đều nhau rồi mới rank; vì phép rank phi tuyến, không được diễn giải mỗi child là đúng 1/4 của đóng góp CoAt trong final.

Ở tầng BTKD, A5 dùng `A5_W=0.52` trên study có acquisition phù hợp; study không phù hợp giữ parent. Rad trộn nhánh reference với hệ số 0.55 trừ Baker's/Fracture, sau đó có nhánh second-layout 0.20 và calibrator phụ thuộc số series/plane/fat-suppression/fluid-sensitive. Nhiều lần rank và availability gate khiến không thể nhân hệ số các tầng để ra một tỷ lệ ảnh hưởng toàn cục.

## EDA nào đã có chứng cứ, EDA nào chưa thể kết luận

| Dữ liệu | Có thể kiểm tra | Không thể kết luận |
|---|---|---|
| `results_0.943.zip`: 3 study mẫu, đủ DINO 20/A5 5/Raptor 4/CoAt 4, không fallback trong run hiển thị | Công thức, contract, sự hiện diện nhánh, score/fallback per-study trên **3 study** | AUC/correlation có ý nghĩa, lỗi của hidden scoring rerun, nhãn nào kéo 0.943 xuống |
| `results_07.zip`: 4.407 UID và OOF ứng viên DINO/E9/E11/E13 | Schema, alignment UID, fold/report-group, proxy AUC từng family hiện có | Full-ensemble 0.943 OOF; chưa có A5/Raptor/4 CoAt OOF |
| Gold58 official labels | Audit mô tả nhỏ, theo từng nhãn nếu ghi rõ độ bất định | Holdout độc lập để tune: vài CoAt checkpoint đã được chọn bằng Gold58 |
| 4.349 weak labels | So sánh thử nghiệm **cùng UID/fold** như 09 vs 14; phân tầng protocol như chẩn đoán | Ước lượng Private LB; nhãn weak có sai lệch đáng kể so với Gold58 |

## Hướng đánh giá kế tiếp bám sát 0.943

1. **Giữ nguyên baseline.** Không submit notebook 14; tạm hoãn chạy 15 nếu mục tiêu trước mắt là phân rã 0.943. 15 chỉ trả lời câu hỏi riêng về cách đưa lát 2.5D vào B2.
2. **Thu predictions cùng UID của tất cả family** trên một cohort có nhãn hợp lệ và provenance rõ: BTKD stage, Raptor, từng CoAt child, final. Trước khi tính AUC, lập bảng study/report-group nào từng được dùng train, chọn checkpoint, calibrate hoặc tune blend cho từng family. Những dòng có nguồn gốc chồng lấn chỉ dùng để phân tích disagreement/coverage, không gọi là OOF độc lập.
3. **Phân rã theo nhãn và subgroup**: per-target AUC/PR-AUC (kèm số positive/negative, CI bootstrap theo report group); chênh lệch `final − BTKD`, `final − Raptor`, ablation CoAt; stratify theo plane/series/field strength/vendor/source nếu metadata tin cậy. Quyết định dựa trên paired differences cùng cohort, không so macro score của cohort khác nhau.
4. **Chỉ đề xuất một thay đổi xác định trước** khi có gain lặp lại qua folds, không làm xấu nhiều nhãn và không dùng Gold58/Public LB để tối ưu trọng số. Nếu public checkpoints đã train trên toàn bộ 4.407 study và không có OOF thật cho A5/Raptor/CoAt, không thể hồi tố tạo OOF sạch bằng cách chỉ chạy inference; khi đó cần holdout độc lập hoặc tái huấn luyện có cross-fitting.

Nguồn kiểm tra: notebook baseline cell 22–27, 29; receipts trong `results_0.943.zip`; `results_07.zip`; `RSNA_0943_Audit_and_Next_Experiment.md`; `RSNA_13_14_Audit_and_15_Decision_2026-10-06.md`.
