# Chốt đánh giá và năm probe cuối — 22/10/2026

## Kết quả đã xác minh

Anchor 33 và các bản 39, 40B, 41, 44A đều được người dùng xác nhận Public **0.958**. Hai bản S6 standalone 43A/43B đều **0.950**. Các điểm này được hiển thị làm tròn; không chứng minh chúng bằng nhau hoàn toàn hoặc xếp được chất lượng Private.

ZIP 44A chứa bằng chứng ablation chạy thật:

- Audit `COMPLETE`, đủ 3 study nhìn thấy, tổng 628,24 giây. Không có cảnh báo hoặc fallback R6/Raptor/S6 trong lượt lưu này.
- `44_postprocessor_audit.json` và log hai worker khớp: chỉ bỏ `--fluid-stacker` của Effusion; vẫn giữ `--state-stacker`, mirror TTA, model và tiền xử lý.
- So với ZIP 40B, CSV cuối chỉ đổi **Effusion**: độ lệch lớn nhất 0,15000006 và đảo thứ tự một trong ba cặp study. Mười một cột còn lại giữ nguyên. Xác suất S6 giống hệt.
- 39 và 40B dùng cùng đầu ra anchor/S6. Điểm số cuối khác do rank so với xác suất, nhưng thứ tự trên ba study nhìn thấy giống nhau ở cả 12 bệnh.

Ba study này chỉ kiểm tra triển khai, không đủ để đánh giá AUC hay tính tổng quát. Thời gian trên chúng không phải dự báo thời gian chấm ẩn. 44A không cho bằng chứng tăng Public ở độ chính xác đang hiển thị; 44B lỗi khi chấm nên chưa đánh giá được ý tưởng.

## Những gì đang và chưa được kiểm tra về overfit

Các notebook submit gần đây có **kiểm tra tính đúng của pipeline**, không có cơ chế chứng minh không overfit:

| Có kiểm tra | Không thể suy ra từ kiểm tra đó |
|---|---|
| Hash checkpoint/code, đúng phiên bản input | Dữ liệu huấn luyện không rò rỉ, hoặc model tổng quát tốt |
| UID/nhãn cột, số dòng, NaN, miền giá trị, fallback | AUC hay độ hiệu chỉnh trên Private |
| Nguồn model được giữ nguyên, chỉ đổi một yếu tố | Thay đổi đó có hiệu quả ngoài mẫu |
| Không train mới, không fit trọng số trong notebook submit | Không overfit do chọn cấu hình sau nhiều lần xem gold/Public |
| Bootstrap cặp trên 58 gold | Xóa được thiên lệch do tái sử dụng gold và chọn mô hình nhiều lần |

Chưa có tập holdout chưa từng dùng hoặc OOF độc lập đủ tin cậy cho việc chọn ensemble cuối. Gold58 đã được xem nhiều lần; checkpoint công khai cũng có lịch sử lựa chọn chưa được kiểm chứng đầy đủ. Rủi ro **overfit do lựa chọn theo Public/gold là có thật**, nhưng không thể đo mức giảm Private từ bằng chứng hiện tại. Không có cơ sở dự báo Top 10 Private.

Nhánh B2 cũ từng có weak-label OOF và chọn epoch trên fold validation. Đó là đánh giá trong thí nghiệm huấn luyện ấy, không phải OOF độc lập của ensemble R6/Raptor/S6 hiện tại. Không chuyển kết luận từ nhánh cũ sang recipe mới.

Năm probe 45 được chốt trước khi xem kết quả của chúng, không fit nhãn hay quét trọng số từng bệnh. Điều này hạn chế thêm các bậc tự do, **không biến chúng thành validation độc lập**. Chọn bản tốt nhất trong năm vẫn là thích nghi với Public.

## Đề xuất selected final

**Chưa đổi selected 39 + 40B trong lúc các probe đang chạy.** Không tự động thay selected bằng notebook mới hoặc bản có cùng 0.958.

Nếu phải chốt mà không có bằng chứng mới đáng tin cậy, đề xuất ưu tiên:

1. **40B** làm ứng viên chính: đã chấm thành công, giữ recipe gốc và S6 center-only, không thêm nhánh TTA mới.
2. **33** làm phương án phòng ngừa rủi ro phần S6: không có phần bổ sung S6 của các bản 39/40B.

Đây là **lựa chọn phân tán một loại rủi ro**, không phải dự báo 40B/33 sẽ thắng Private. Chúng vẫn chia sẻ phần lớn backbone/trọng số. Hai lựa chọn hiện tại **39 + 40B cũng hợp lý** nếu muốn giữ hai cách pha S6, nhưng mức độc lập rất thấp: cùng model, chỉ khác biểu diễn điểm S6. Không có dữ liệu để khẳng định cặp nào tối ưu.

Chưa có bằng chứng để ưu tiên 41 hoặc 44A hơn 40B: 41 thêm chi phí TTA trong khi 43A/B chưa chứng minh lợi ích; 44A chỉ thay hậu xử lý Effusion và không tăng điểm hiển thị. Không đổi selected vì thứ hạng tăng vài bậc ở hai thời điểm khác nhau. Người dùng cần xác nhận hai ô selected cuối; tài liệu này không thực hiện thay đổi đó.

## Năm notebook 45 — chưa có score

Đây là năm **thử nghiệm fusion dùng lại các đầu ra model đã có**. Không train mới, không thêm lượt suy luận model, không đổi crop/chuỗi/lát cắt/geometry và không dùng gold/test labels. Mỗi bản độc lập với các bản khác. Cấu hình chính xác và mọi thay đổi phải được audit riêng, không âm thầm thay checkpoint.

| Probe | Giả thuyết cố định | Mức ưu tiên |
|---|---|---|
| **45A — chuẩn hóa percentile của anchor** | Chuẩn hóa thang điểm ensemble anchor trước khi pha S6, kiểm tra tác động của thang điểm tổng hợp | **Chạy trước** |
| **45B — probability pooling** | Thử trộn các xác suất thành phần thay vì chỉ xếp hạng chúng | Sau A/D nếu còn thời gian |
| **45C — logit pooling** | Thử trộn trên miền logit, chặn xác suất bằng `eps = 1e-5` để tránh logit vô hạn | Sau A/D; nhạy với calibration |
| **45D — trọng số family đồng đều** | Raptor/R6 cùng 0,5 cho tất cả bệnh, thay các ưu tiên từng bệnh; giảm số giả định target-specific | **Chạy trước** |
| **45E — leave-224-out** | Giữ nguồn Raptor384, bỏ đóng góp Raptor224 khỏi fusion để kiểm tra giá trị của nhánh độ phân giải thấp | Sau A/D nếu còn thời gian |

Trọng số cố định không được chọn lại sau khi xem điểm từng bệnh trên gold. Không chọn mask bệnh cho mỗi probe dựa trên kết quả của probe khác. Probability/logit pooling có rủi ro vì calibration giữa các model có thể khác; không xem chúng mặc nhiên tốt hơn rank. Leave-224-out là phép bỏ một nguồn đã có, không phải kiến trúc mới. `p6` là xác suất R6 sau các bộ hậu xử lý gốc, không phải điểm thô từ head.

Ký hiệu: `p384`, `p224`, `p6`, `pS` là xác suất Raptor384, Raptor224, R6 và S6; `w_j` là trọng số Raptor theo bệnh của 33. Các trọng số này giữ nguyên ở mọi probe trừ 45D. Công thức hai probe về confidence:

- **45B:** `0.9 × [w_j × (0.65 × p384 + 0.35 × p224) + (1 − w_j) × p6] + 0.1 × pS`.
- **45C:** `sigmoid(0.9 × [w_j × (0.65 × logit(p384) + 0.35 × logit(p224)) + (1 − w_j) × logit(p6)] + 0.1 × logit(pS))`. Chặn từng xác suất trong `[1e-5, 1 − 1e-5]` trước khi lấy logit.

Rank được tính trên đúng tập dòng đã chấm của nguồn tương ứng. Nếu một thành phần cần cho probe thiếu ở một study, dòng đó trở về **đúng baseline 40B**, có ghi audit; không tự tạo xác suất “an toàn” hoặc âm thầm coi probe đã chạy đầy đủ. Việc giữ các lượt suy luận gốc vẫn cần thiết cho baseline/fallback, ngay cả khi 45E không dùng 224 trong công thức candidate.

**Cung cấp năm notebook theo yêu cầu không có nghĩa phải chạy đủ năm.** Ưu tiên 45A và 45D; cân nhắc hàng đợi/thời gian thực tế rồi mới chạy phần còn lại. Mỗi notebook vẫn chạy lại toàn bộ pipeline suy luận 40B; không phải chạy CPU trên CSV đã lưu. Mọi bản đều chưa có điểm Public/Private, không hứa 0.96+. Các bản 0.958 đã thành công vẫn phải được bảo toàn.

## Input và kiểm tra trước khi nộp

Dùng đúng **năm input/version của 40B đã thành công**, không phải bộ ba input của S6 standalone:

1. Competition RSNA Knee Abnormality Detection, có test DICOMs.
2. R6 Frozen **w1**, bundle `fe055406fad6a12f` giống 33/40B; không đổi sang R7/latest chỉ vì cùng tên checkpoint.
3. Nartaa publication/selected SWA weights **Version 1**, có cả Raptor384 và Raptor224.
4. Heliosli OrthoFoundation code/anatomy **Version 1**, có `anatomy.pt` và `cloud_code`.
5. Nartaa S6 all-data weights **Version 5**.

**GPU T4×2, Internet Off, Save & Run All.** Không cần Output cũ hoặc study cache. Tên dataset xuất hiện trong ảnh không xác nhận version; metadata import không bảo đảm thay được attachment trong draft cũ.

Trước khi submit: audit phải `COMPLETE`, đúng experiment, các stage đủ UID, số dòng/nhãn khớp và công thức đúng probe. Xem warnings/fallback, không coi CSV từ một lượt lỗi là bản hợp lệ. CPU test chỉ xác nhận logic đã kiểm thử; không bảo đảm GPU, OOM hoặc hidden rerun thành công. Không dùng CSV ba study nhìn thấy để suy diễn metric chấm ẩn.

## Hạn chót và cách chốt

Lịch chính thức nêu hạn nộp cuối **23:59 UTC ngày 22/10/2026**, tương đương **06:59 sáng 23/10/2026 tại Bangkok/Việt Nam (UTC+7)**. Nguồn: [Kaggle — Overview, Timeline và Code Requirements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview). Kaggle ghi runtime tối đa 9 giờ; hàng đợi và lượt rerun làm thời gian thực tế khác thời gian notebook nhìn thấy.

Không diễn giải “mai ngày cuối” theo đồng hồ máy chạy lệnh. Kiểm tra countdown trực tiếp trước khi launch. Nếu sát hạn hoặc GPU queue không ổn định, ưu tiên **chốt hai submission đã chấm thành công**, không thay bằng bản chưa có score. Giữ một bản sao notebook/input version/receipt của các lựa chọn cuối. Không dùng hết quota chỉ để thay đổi vài chữ số mà chưa có giả thuyết rõ.

## Research còn thiếu nhưng không chặn năm probe

- OOF thật sự độc lập, cùng UID/fold/label source cho R6, Raptor và S6; không thể tái tạo validation sạch chỉ từ ZIP ba study.
- Lịch sử dùng gold để train/chọn epoch/chọn stacker, provenance và điều kiện tái phân phối từng weight.
- Một checkpoint độc lập về giám sát hoặc nguồn dữ liệu với bằng chứng tái lập, thay vì thêm biến thể của cùng ensemble.

Không gửi báo cáo MRI thô cho dịch vụ ngoài, không giả định quyền sử dụng chỉ từ hash khớp và không thêm dependency mới sát hạn nếu chưa xác minh. Các điểm thiếu này cần ghi nhận trung thực; không có nghiên cứu nhanh nào bảo đảm Private Top 10.

## Kiểm chứng bàn giao

`44A_Output_Audit.json` lưu audit ZIP thực tế. `45_Visible_Regression.json` dựng lại baseline40B và anchor33 từ các CSV thực tế: sai số lớn nhất khi dựng anchor khoảng 2,39e-8. Đây là hồi quy triển khai, không phải metric.

`45_Local_Validation.json` ghi kiểm thử CPU độc lập: công thức cả năm bản trên 1/2/3/1.513 study; UID đảo thứ tự, ties, xác suất 0/1, coverage lỗi một phần và fallback đúng40B; 25 ca chạy toàn bộ parent với model/mount giả lập, gồm lỗi stage và lỗi sát lúc xuất file. Các nguồn S6 và prefix anchor được so sánh byte, notebook33 gốc không đổi. Chưa kiểm thử GPU/DICOM/chấm ẩn của năm bản mới.

Lớp ghi audit mới đổi correlation chẩn đoán không xác định thành JSON null, nhưng vẫn từ chối NaN trong dự đoán được chấm. Đây là sửa một đường lỗi có thể tái hiện trên fixture cột hằng; **chưa chứng minh nó là nguyên nhân các lỗi chấm ẩn 40A/44B trước đây**.
