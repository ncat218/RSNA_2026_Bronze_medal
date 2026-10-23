# Đối chứng hai notebook Public 0.954 — 2026-10-12

## Bản nào đã được xác minh?

| Bản nguồn đã ghi điểm | File đối chứng nguyên bản | Public LB trên trang nguồn | Blend |
| --- | --- | ---: | ---: |
| [Helios954 V1](https://www.kaggle.com/code/sameerk2004/rsna-knee-sil-d5m-helios954) | `23A_Exact_Public0954_Helios_70_30.ipynb` | 0.954 | Raptor 70%, Compact S6 30% |
| [E030 Compact35 V1](https://www.kaggle.com/code/wocenmma/rsna-e030-compact35-native?scriptVersionId=357282371) | `23B_Exact_Public0954_E030_65_35.ipynb` | 0.954 | Raptor 65%, Compact S6 35% |

Hai file trên là bản `.ipynb` tải từ chính các trang phiên bản đã ghi điểm và được sao chép nguyên byte vào workspace. Chưa sửa code, metadata hay output. Bốn code cell đầu giống nhau từng ký tự; code cell blend chỉ đổi hằng số 0.70/0.30 thành 0.65/0.35. E030 có thêm cell cuối để lưu hash/receipt, không thay đổi `submission.csv`.

Hai file mà người dùng đưa trước đó trong `Kaggle public/` **không phải cả hai bản exact này**. `rsna-knee-lb-0-954-dual-sota-ensemble.ipynb` thêm preflight chưa chạy và khác một điều kiện tìm mount; `kaggle-rsna-knee 0.954 public.ipynb` dùng trọng số theo từng nhãn. Không suy ra Public 0.954 cho bản trọng số theo nhãn khi chưa submit bản đó.

## Chạy đối chứng trên tài khoản thi

1. Import `23A_Exact_Public0954_Helios_70_30.ipynb` vào Kaggle notebook mới. Cài GPU T4×2, Internet Off. Metadata tải về có `isGpuEnabled=false` dù trang phiên bản nguồn hiển thị T4×2, nên **kiểm tra accelerator trên giao diện** trước Save & Run All.
2. Attach competition `RSNA Knee Abnormality Detection` và đúng ba dataset hiển thị ở tab Input của notebook nguồn: `RSNA Knee Compact v1`, `RSNA Knee selected SWA weights 0949 and 0945`, `RSNA OrthoFoundation Fold0 Epoch7 Private Weights`. Khi thêm Input, chọn đúng phiên bản/checkpoint theo trang nguồn; đừng thay checkpoint chỉ vì tên file gần giống. Các code cell tự kiểm tra manifest/hashes và sẽ dừng nếu khác.
3. Save & Run All, rồi chỉ nộp khi Output có `submission.csv` và log có `NOTEBOOK_COMPLETE` (bản B còn có `E030_REPLICA_COMPLETE`). Kiểm tra `blend_receipt.json`: số study, `raptor_failed`, `ours_failed`, thời gian.
4. Ghi Public Score của chính tài khoản mình. Chỉ sau khi A hoàn tất mới chạy `23B_Exact_Public0954_E030_65_35.ipynb` để đo khác biệt 65/35. Hai bản có chung ảnh và mô hình nên kết quả rất tương quan; chênh lệch 0.001–0.002 Public có thể chỉ là nhiễu trên tập Public nhỏ.

Hai notebook nguồn báo thời gian chạy **visible test 3–4 phút**, không phải cam kết thời gian hidden rerun. Đây là bản đối chứng nguyên gốc, **chưa được thử chạy lại trong tài khoản của người dùng**; không thể hứa chắc không gặp lỗi môi trường/input/hidden test.

## Giới hạn về nguồn dữ liệu và bước tiếp theo

Raptor dùng checkpoint có tên chứa `oai`, và mã nguồn notebook dẫn tới dữ liệu OAI. Điểm Public 0.954 **không xác nhận** mọi nguồn dữ liệu/weight đều hợp lệ cho Private/final standings. Người dùng đã ưu tiên tính hợp lệ và tái lập: trước khi chọn làm lời giải cuối, cần ghi rõ nguồn, giấy phép và điều kiện truy cập của OAI cùng các weights liên quan; so với [quy định cuộc thi](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules) và [giải thích của host về OAI](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819). Không coi một lần Public probe thành xác nhận pháp lý.

Sau khi đối chứng A/B, dùng outputs/receipts để phân tích: tỷ lệ fail, mối tương quan theo nhãn giữa Raptor và Compact, và liệu thêm thành phần 0.943 hiện có cung cấp lỗi *khác* hay không. Chỉ thử blend mới khi có căn cứ OOF/validation và giữ một cấu hình cố định trước khi nhìn Public LB; không tune nhiều vòng bằng Public Score để tránh overfit leaderboard.
