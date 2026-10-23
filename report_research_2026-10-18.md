# RSNA Knee — Brief nghiên cứu tiếp theo

Cập nhật 18/10/2026. Dùng file này làm trạng thái hiện tại; `report.md` ngày 17/10 chỉ dùng làm lịch sử. Mục tiêu cuối: Private Top 10; không có kết quả Private để đối chiếu.

## 1. Base và bằng chứng hiện tại

Public được người dùng xác nhận: **0.943 → 0.954 → 0.956 → 0.957 → 0.958**. Các bước tăng chủ yếu do tái lập nguồn public; chưa chứng minh cải tiến riêng vượt recipe nguồn.

Anchor được bảo vệ: **33_RSNA_0958_R6_Raptor_Submit.ipynb = 0.958**.

- Raptor384: Nartaa SWA, native320, 96 lát/94 cửa sổ, anatomical mirror probability averaging.
- Raptor224: checkpoint riêng cùng release, một view.
- R6: bốn fold ConvNeXt-small, spatial/extent/Synovitis heads, anatomy anchors, laterality classifier, mirror TTA, Effusion/Baker's stackers.
- Trộn rank: trong họ Raptor, 65% 384 + 35% 224. Tỷ trọng Raptor/R6: Synovitis 80/20, Lateral OA 20/80, PF OA 60/40, Baker's 35/65, các bệnh khác 50/50.
- R6 bundle `fe055406fad6a12f`, original Heliosli v1; 33 dùng w1 copy. Original dataset Latest/v2 đã thay thành R7, không được đánh đồng.

## 2. Những hướng đã thử hoặc đang chạy

| Hướng | Kết quả/trạng thái |
|---|---|
| Frozen RadImageNet/MIL + density/group blends | Không vượt 0.943; standalone 0.914 và density blend 0.913 |
| Loại cửa sổ Raptor qua biên series | Diagnostic gold xấu đi; không chuyển vào base |
| Raptor224 hoặc mirror R6, ablation 29A/B/C | Cả ba hiển thị Public 0.957; không chứng minh tương đương dưới độ làm tròn |
| Bỏ stacker Effusion/Baker's, thêm R7 plain | Diagnostic30: delta macro +0.000104/+0.000075/−0.000294; chưa thuyết phục |
| 32: 18 ca Synovitis/Effusion bất đồng | Đã lập hồ sơ; chưa có manual report/image review. AUC nhóm chọn lọc này yếu, không phải CV độc lập |
| 34: physical sampling phase0 | Đã triển khai header gate + diagnostic gold; chưa nhận kết quả |
| 35: trung bình hai phase R6 | Lượt đầu lỗi thiếu import numpy sau khi R6 original score đủ 3/3. Đã sửa và kiểm thử cell đầy đủ; chưa xác nhận rerun GPU |
| 36: Raptor series-count-normalized attention | Người dùng báo chạy hoàn thành; chưa có Output ZIP/Public score |
| 37: crop R6 cùng kích thước, dịch offset16→24, kết hợp hai pass | Chưa chạy theo thông tin mới nhất; đã sửa cùng lỗi import của 35 |

35–37 là ba phép thử độc lập, giữ nguyên trọng số family của 33. Chưa có điểm mới nào được xác nhận cho chúng. Không tự gán score từ tiêu đề file hoặc trạng thái notebook successful.

## 3. Giới hạn và điều không nên đề xuất lại

- Chỉ 58 gold; gold đã được dùng upstream cho phát triển/chọn mô hình. Gradient exclusion không chứng minh validation độc lập, đặc biệt khi chưa biết tập fit stacker.
- Saved test outputs chỉ có 3 study, không có nhãn hay prediction hidden. Không dùng chúng chọn trọng số/bệnh.
- R6 đã có eval_stride=1, phys_mm=3.5 và slab averaging cho lát rất mỏng. “Giảm stride để lấy dày hơn” không phải thay đổi mới.
- Các yếu tố đã có: multi-resolution, anatomy localization, side correction, spatial heads, attention theo bệnh, mirror TTA, stackers. Không đề xuất lại như phát minh mới.
- Nhãn mềm 0.3–0.5 của Raptor v10 chưa có nghĩa chính thức; v10 không phải checkpoint hiện tại. Hoãn frozen-head uncertainty masking và không tự relabel bằng prediction.
- OAI được tác giả công bố trong cả hai Raptor checkpoints. R6 có auxiliary weights cần truy provenance. Chưa có host ruling trực tiếp được kiểm chứng cho các dependency cụ thể. Không đồng nhất tải công khai/hash khớp với eligibility.

## 4. Nhờ research bên ngoài: tối đa ba ưu tiên

### A. Bằng chứng trực tiếp cho phase/crop/pooling đang thử

Tìm ablation của đúng R6/Raptor hoặc checkpoint gần tương đương: random phase lúc train, multi-phase inference, crop translation, inverse-count/series-aware MIL. Cần source/config, checkpoint revision, baseline và metric. Nêu rõ nếu nghiên cứu chỉ là pipeline khác. Không tìm thêm các trọng số blend lân cận.

### B. Một nhánh thực sự bổ sung, có thể triển khai

Ưu tiên model mạnh khác pipeline hoặc specialist với weights, preprocessing, licence và prediction OOF có UID/fold/label provenance. Có thể là compartment OA hoặc Synovitis, nhưng cần ablation và lỗi bổ sung, không chỉ correlation thấp. Nếu chỉ có paper/no weights, ghi rõ chi phí tái tạo. Không đưa checkpoint yếu vào stack chỉ vì khác backbone.

### C. Recipe supervision và điều kiện dùng final

- Heliosli: training/labels của R6, side_cls.pt, anatomy.pt; gold dùng ở đâu; stacker cross-fit và plain/mirror inputs; khả dụng OOF.
- Nartaa: công thức blendjev, định nghĩa Synovitis/OAI, giá trị khi báo cáo không nhắc, provenance JEV.
- Host/Rules: câu trả lời trực tiếp cho public OAI-trained weights và SKM-TEA/KneeXNet auxiliaries, nghĩa vụ giao nộp/licence của winner. Không suy luận từ im lặng hoặc từ tiêu đề “Private”. Không cần gửi báo cáo y khoa ra dịch vụ bên ngoài để research.

## 5. Kết quả research mong muốn

Mỗi phát hiện: link nguồn gốc + commit/version/date + đã đọc code hay chỉ mô tả + bằng chứng + điểm chưa biết + khác biệt với base + phép thử nhỏ nhất + dữ liệu còn thiếu + điều kiện dừng. Chỉ đề xuất tối đa ba việc có thể hành động. Nếu không tìm thấy, báo rõ; không bù bằng lời khuyên chung hoặc hứa 0.96.

Không chờ research mới để đánh giá output 34–37. Không thay selection final hay gửi câu hỏi công khai tự động; quyết định eligibility khi host im lặng còn cần người dùng chọn mức rủi ro.
