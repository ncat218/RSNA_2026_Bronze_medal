# RSNA: audit hai notebook được giới thiệu là Public 0.954

Cập nhật 11/10/2026. Đây là phân tích **file `.ipynb` local**, không phải xác nhận độc lập leaderboard. Best Public đã được người dùng submit/xác nhận cho pipeline hiện tại vẫn là **0.943**. Notebook 22 (anchor 0.943 + reader ConvNeXt 2.5D) chưa có điểm Kaggle.

## Bằng chứng và độ tin cậy

| File | SHA-256 | Công thức blend trong source | Bằng chứng có trong file |
| --- | --- | --- | --- |
| `Kaggle public/kaggle-rsna-knee 0.954 public.ipynb` | `450d8fb5e3c7c185d25a1b25bff1335a85cdb69d94fa103227b04c1c5bd99cb2` | Raptor/S6 theo nhãn: 0.65/0.35 cho mô mềm; 0.60/0.40 PF OA & synovitis; 0.58/0.42 cho OA khác, contusion, fracture | Saved run 3 visible studies, `OPTIMIZED_VELCIRAPTOR_COMPLETE`; **không có Public Score trong output** |
| `Kaggle public/rsna-knee-lb-0-954-dual-sota-ensemble.ipynb` | `fc2055b92e6ca5aadad346d19bd92946d4e2c8581d5ef5674f06e137503880f4` | Cố định 0.70 Raptor / 0.30 S6 cho cả 12 nhãn | Saved run 3 visible studies, `NOTEBOOK_COMPLETE`; **không có Public Score trong output**. Cell preflight mới thêm chưa chạy (`execution_count=null`). |

Hai file dùng cùng source Raptor; S6 chỉ khác một kiểm tra tìm mount (bản thứ hai cho phép nhiều candidate và lấy candidate đầu); không phải hai model/score độc lập. Metadata cả hai ghi `isGpuEnabled:false` dù saved output T4×2; khi import cần kiểm tra Settings. Saved `papermill` của file đầu là 173 giây, file thứ hai 208 giây cho **3 visible studies**, không chứng minh thời gian hidden ~1.300 studies. Lời hứa “zero OOM risk” trong markdown không thể suy ra từ việc tách subprocess. Chỉ có thể xác thực 0.954 từ URL phiên bản Kaggle/ảnh Public Score hoặc lần submit của người dùng.

## Cái mới thực chất so với anchor 0.943

1. **Raptor/OAI CoAtNet-384**: một checkpoint `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` với SHA-256 pin `7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef`. Nó dùng 96 lát/5 slot, crop 140 mm, 384 px, tối đa 94 cửa sổ tại inference; không phải ba checkpoint Raptor v5/v8/v10 trong anchor 0.943. [Tác giả recipe CoAtNet báo Public 0.949](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792) cho mô hình một nhánh, nhưng chưa xác minh checkpoint trong hai file này có đúng lineage/điểm đó.
2. **OrthoFoundation ConvNeXt S6**: 4 fold `E_lrprior_s0_f0.pt`–`f3.pt`, geometry-aware DICOM, physical slice position, anatomical routing/laterality, 2.5D patches. Source so khớp SHA-256 cho modules và các weights theo `manifest.json`/`bundle_id=209a32c8b50a6b1c`. Đây là nhánh có thể bổ sung tính đa dạng so với anchor, nhưng local file không có OOF sạch hoặc AUC riêng của nhánh.
3. **Rank fusion**: Raptor và S6 được percentile-rank rồi cộng trọng số. Đây là hai model families mới, khác hẳn việc điều chỉnh trọng số B2 frozen 0.914 đã thất bại. Bản 0.70/0.30 là một hypothesis cố định; bản trọng số theo nhãn được thêm sau và **không có bằng chứng ablation** trong hai file để gọi là “optimal”.

## Chỗ cần chặn trước khi dùng GPU/submission

- Trong source Raptor, tên checkpoint chứa `oai`; [tác giả công khai dùng 2.399 OAI knees](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792). [Host chỉ cho phép OAI nếu tất cả thí sinh có thể truy cập miễn phí và không cần ký duyệt cấp tổ chức](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819); nguồn gốc/phương thức truy cập chính xác của checkpoint này và quyền tái phân phối cần xác minh trước khi xem là candidate Private Top 10 hợp lệ. Tương tự `anatomy.pt`/OrthoFoundation nhắc đến SKM-TEA; cần kiểm tra provenance và license của weights này. Public Score không thay thế được eligibility.
- File B preflight chỉ tìm **tên file**, không kiểm tra hashes tại bước đó; S6 có kiểm tra sâu hơn sau khi import. Raptor hash được kiểm tra ở cell tiếp theo. Nếu attach nhầm phiên bản, kiểm tra sớm sẽ tiết kiệm GPU.
- Hai run visible có 3 studies và không có per-label AUC/correlation đáng tin cậy. Không suy ra đóng góp S6 hay độ ổn định Private từ rank correlation 3 điểm.
- Trong blend, `rap_fail` suy từ 12 raw outputs đều đúng 0.5; đó là heuristic, không phải receipt lỗi đầy đủ. Giới hạn thất bại Raptor 3% vẫn cho phép fallback; với mục tiêu high-stakes cần audit từng study và báo cáo số lượng.

## Thứ tự thực nghiệm đề xuất

1. **Xác minh nguồn**: nhận URL Kaggle/version hoặc screenshot score của đúng hai file; ghi rõ phiên bản nào thật sự 0.954. Cùng lúc kiểm tra ba dataset mounts/checkpoint hashes và provenance OAI/SKM-TEA. Không đổi hoặc xóa bản anchor 0.943.
2. **Tái lập đúng recipe 0.70/0.30 một lần**, chỉ khi nguồn/eligibility chấp nhận được. Preflight CPU/GPU nhanh, verify tất cả artifacts và visible 3-study parity trước hidden run. Lưu Raptor raw, S6 raw, receipt, SHA-256 và full-cohort failure counts. Dùng T4×2, Internet Off theo source; *không cam kết runtime/OOM*.
3. **Ablation có giá trị**: lấy standalone Raptor, standalone S6, và fixed blend từ cùng run; nếu đủ quyền submit, mỗi candidate một lượt. So với 0.943 và 0.954 được xác minh, không fit per-label weights trên Public. Tiếp theo mới thử một global blend pre-registered giữa champion và family S6, vì Public 0.954 có thể không chuyển sang Private.
4. **Hướng tự chủ/hợp lệ** nếu OAI/SKM-TEA chưa rõ: chuyển các ý tưởng vật lý (slice sorting, 3.5 mm normalization, 100–140 mm local crops, laterality, finding-specific attention, missing-label masking) vào model train bằng competition data/labels có provenance rõ. Chạy OOF theo report-group, audit 58 gold chỉ để chẩn đoán, giữ Private Top 10 là mục tiêu cuối chứ không lấy leaderboard làm tập train.

**Quyết định hiện tại:** Chưa thay champion 0.943. Hai file là gợi ý một *family change* mạnh, nhưng chưa đủ bằng chứng để khẳng định bất kỳ file local nào tái lập 0.954 hoặc hợp lệ cho final. Khi có URL/ảnh leaderboard và provenance dataset, chọn đúng một scored recipe để đóng gói notebook thử nghiệm có fail-fast receipts.
