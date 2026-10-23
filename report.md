# RSNA Knee — Tóm tắt tiến độ và thông tin cần tìm

Cập nhật: 17/10/2026. Mục tiêu cuối: Private Top 10; mốc Public tốt nhất đã xác nhận: **0.957**. Điểm Public không chứng minh điểm Private hoặc thứ hạng cuối.

## 1. Từ 0.943 đến hiện tại

| Giai đoạn | Đã làm | Kết quả Public / kết luận |
|---|---|---|
| Base 0.943 | Tái lập ensemble public; kiểm kê nguồn model, checkpoint và output | **0.943** — mốc ban đầu |
| 08–21: nhánh riêng B2 | Frozen RadImageNet ResNet50 + MIL, 5 fold; phân tích OOF nhãn yếu, thử blend theo nhóm bệnh và mật độ lát | Standalone **0.914**; density blend **0.913**; ghép vào base **0.942–0.943**. Chưa cải thiện base |
| 22: ConvNeXt reader | Thử thêm family public; gặp khác version/hash checkpoint và lỗi khi rerun hidden | Chưa có mốc điểm cải thiện được xác nhận từ nhánh này |
| 23A/B | Áp dụng các công thức ensemble public mạnh hơn, gồm Raptor | Cả hai **0.954** |
| 24: competition-only | Chuẩn bị thử training chỉ dùng MRI cuộc thi, cho phép ImageNet init; benchmark và chia cache 96 lát, 384×384 | Đã xác nhận shard 0: 1.088 study, ~10,22 GiB. Chưa có bằng chứng trong báo cáo này về model full-data hoàn tất hoặc điểm submit |
| 25A/B | Tái lập Raptor + R7; bản B thêm Raptor224 | Cả hai **0.956** |
| 26–28 | Chẩn đoán cửa sổ qua biên series, thử tỷ trọng nhóm OA, triển khai diagnostic mirror R7 | Lọc cửa sổ không giúp gold; notebook 27 vẫn **0.956**. Chưa có kết quả 28 được đánh giá |
| 29A/B/C | Tái lập nguồn public mới dùng R6; tách ảnh hưởng Raptor224 và mirror TTA | Cả ba **0.957**; chưa có cải tiến riêng vượt nguồn public |
| 30 | Phân tích 58 gold với R6 TTA; đo trước/sau stacker và thêm R7 plain cố định | Chạy đủ, không fallback; chưa tìm thấy tín hiệu cải thiện thuyết phục |

**Phân biệt đóng góp:** bước tăng 0.943 → 0.954 → 0.956 → 0.957 chủ yếu đến từ học hỏi và tái lập notebook/checkpoint public mạnh hơn. Các ablation riêng giúp kiểm chứng, loại hướng yếu và ổn định triển khai, nhưng chưa tạo bước tăng Public vượt recipe nguồn.

## 2. Base 0.957 đang có gì?

- **Raptor:** CoAtNet + multi-instance learning, cửa sổ nhiều lát, attention theo bệnh; checkpoint 384 và tùy chọn 224, đã có crop/mirror inference.
- **R6:** bốn fold ConvNeXt-small; định vị giải phẫu, kiểm tra bên gối, head không gian/OA extent/Synovitis; stacker Effusion và Baker’s.
- **29A:** 50% rank R6 TTA + 32,5% rank Raptor384 + 17,5% rank Raptor224.
- **29B:** 50% rank R6 TTA + 50% rank Raptor384.
- **29C:** giống A nhưng tắt mirror TTA của R6.
- Cả ba cùng hiển thị 0.957: chưa đủ kết luận TTA hoặc 224 vô ích vì điểm được làm tròn. R6 phải dùng dataset **Version 1**, không dùng Latest đã thay nội dung bằng R7.

## 3. Điều đã biết từ notebook 30

58/58 gold được score, zero fallback; tái lập R6 visible của 29B với sai số 0. Tổng runtime ~22,5 phút.

| Thay đổi so với blend 29B trên gold | Delta macro AUC |
|---|---:|
| Bỏ stacker Effusion | +0.000104 |
| Bỏ stacker Baker’s | +0.000075 |
| 50% Raptor384 + 40% R6 TTA + 10% R7 plain | −0.000294 |

Các khoảng bootstrap đều chứa 0. Thiếu đặc trưng thể tích Effusion chỉ ở 2/58 study, không phải lỗi phổ biến. R6–R7 plain tương quan cao (~0.967–0.986 trên gold), nên thêm model gần giống chưa tạo đa dạng đủ rõ.

Lateral OA (~0.816) và Synovitis (~0.858) là hai AUC blend thấp nhất trên gold: **manh mối để phân tích, không phải bằng chứng chúng cũng kéo điểm hidden xuống**.

## 4. Nút thắt còn nan giải

1. **Thiếu validation độc lập:** ZIP submit chỉ chứa 3 study visible, không có nhãn hidden. 58 gold đã được dùng trong phát triển/chọn model upstream; AUC và bootstrap trên đó không phải independent CV.
2. **Thiếu recipe training/nhãn đầy đủ:** chưa tái lập được toàn bộ cách tạo labels, chia fold, train các head và fit stacker của R6/Raptor. Hiện hiểu inference rõ hơn training.
3. **Thiếu mô hình thực sự bổ sung:** thay resolution, TTA hoặc trộn R6/R7 vẫn quanh cùng họ mô hình. Chưa có bằng chứng về specialist hoặc backbone mới sửa được lỗi khác biệt.
4. **Nguy cơ overfit Public:** nhiều lượt submit hỗ trợ kiểm chứng giả thuyết, nhưng dò trọng số liên tục không đảm bảo Private tốt hơn.
5. **Chi phí kỹ thuật:** từng gặp RAM leak, dtype, mount/version mismatch; nhánh training mới phải qua kiểm tra bộ nhớ, tốc độ và resume trước khi chạy dài.

## 5. Nhờ bạn tìm và cung cấp — theo thứ tự ưu tiên

| Ưu tiên | Cần tìm gì? | Gửi gì hữu ích nhất? |
|---|---|---|
| **1** | Training recipe của R6/Raptor hoặc một nguồn public mạnh hơn, có khác biệt kỹ thuật thật | Notebook/code train, config, link nguồn và checkpoint/version; score gắn đúng phiên bản, không chỉ tiêu đề |
| **2** | OOF và validation đáng tin cậy | CSV/NPZ prediction có UID + fold, nguồn nhãn, cách chia patient/report/site; xác nhận ngoài fold, kể cả stacker |
| **3** | Nhãn OA và Synovitis tốt hơn | Code tạo nhãn + label table + provenance; quy ước mức độ OA, phân biệt Synovitis/Effusion; ablation nếu có |
| **4** | Specialist giải phẫu thực sự bổ sung | Model/code cho compartment OA, synovium, ligament/meniscus ROI hoặc segmentation; preprocessing, weights, điều kiện sử dụng và kết quả kiểm chứng |
| **5** | Kết quả kiểm chứng stacker/TTA và lỗi theo nhóm | Tác giả fit stacker trên plain hay mirror logits? Có cross-fit không? AUC theo scanner/site, thiếu series, thiếu fluid volume? |

**Câu hỏi có thể mang đi hỏi:**

> Với base R6 + Raptor đang khoảng 0.957, nút thắt nằm ở report labels, chọn vùng giải phẫu hay pooling? Có ablation/OOF ngoài fold chứng minh không? Có thể chia sẻ recipe train, fold assignment và prediction đối chiếu không?

## 6. Đề xuất hướng tiếp theo

- Giữ nguyên 29A/B/C làm anchor, không sửa chồng lên bản đã score.
- Nếu có nguồn public mới: phân tách khác biệt so với base trước, tái lập một bản, rồi ablation từng thành phần mới.
- Nếu có OOF/labels đáng tin: ưu tiên specialist OA/Synovitis hoặc representation khác, đánh giá tính bổ sung trước khi ensemble.
- Chưa nên bỏ stacker, tăng trọng số từng bệnh hoặc train full-data chỉ vì 58 gold tăng rất nhỏ. Không có cơ sở hứa chắc 0.96 hay Private Top 10.
