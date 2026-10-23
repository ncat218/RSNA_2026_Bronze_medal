# Rebuild notebook 0.958 — 2026-10-17

## Kết quả

Notebook mới: `31_RSNA_0958_Rebuild_Strict.ipynb`. Đây là bản build từ `Kaggle public/0.958.ipynb`; công thức chấm mặc định `v6a`, checkpoint và danh sách 19 inputs được giữ nguyên. Không gán điểm Public/Private mới cho bản build này vì chưa chạy Kaggle.

## Những gì đã sửa

1. Metadata Kaggle của bản gốc ghi `isGpuEnabled: false`, mâu thuẫn với code yêu cầu đúng hai T4. Bản mới đặt GPU bật, T4, Internet tắt.
2. Bốn manifest CoAt với SHA256 gốc được kiểm tra trước các bước MRI tốn thời gian.
3. Chỉ cho xuất kết quả khi đủ đúng bốn reader CoAt và bốn artifact xác suất tương ứng. Bản gốc có thể tiếp tục bằng 1–3 reader hoặc chỉ Raptor khi child reader lỗi.
4. Công thức `flat3` phải có cả Raptor public và CoAt family; không tự đổi sang công thức hai nhánh.
5. Tất cả arm Raptor phải chuẩn bị đủ mọi test UID trước khi publish. Bản gốc chỉ ghi sự kiện nếu thiếu.
6. Xóa output cũ và execution count để người chạy không nhầm log ba study với run mới; giới hạn override cấu hình cơ bản.

## Quyết định từ research cũ

- `29A/29B` có Public 0.957 theo kết quả người dùng, nhưng chỉ ba study hiện trong output ZIP; không có prediction thẳng hàng với pipeline 0.958 trên một validation độc lập.
- Experiment 30 tái hiện R6 TTA và thử bỏ từng stacker hoặc pha R7 plain trên 58 gold. Chênh lệch macro AUC rất nhỏ, và 58 gold đã được dùng trong phát triển/chọn model. Chưa có bằng chứng đủ để đưa R6/R7 hay weight mới vào công thức 0.958.
- Thay đổi serving-only bỏ các cửa sổ Raptor qua ranh giới series ở experiment 26 làm kết quả blend gold xấu đi; không mang nó vào bản rebuild.
- Vì vậy cải tiến hiện tại tập trung vào tính đúng đắn và khả năng tái hiện. Muốn cải thiện AUC tiếp theo cần dự đoán OOF thẳng hàng, fold/provenance rõ, hoặc một model bổ sung được đánh giá độc lập.

## Kiểm tra tại máy

- 55 cells; mọi code cell compile.
- So với nguồn, chỉ code cells 6, 48, 52 thay đổi; một preflight cell và một mô tả mới được thêm.
- Kiểm tra metadata, không có saved outputs, ID cell duy nhất, SHA256 manifest khớp notebook.
- Preflight giả lập nhận đủ bốn manifest và từ chối khi thiếu một manifest.
- Chưa chạy DICOM/GPU/Kaggle hidden test. Run thật cần kiểm tra `rsna_v6_blend_report.json`, `btkd_v559_complete.json`, child receipts và `submission.csv`.

`31_RSNA_0958_Rebuild_Manifest.json` ghi hash nguồn và notebook mới. `build_31_0958_rebuild.py` tái tạo notebook một cách xác định, và sẽ dừng nếu các đoạn code được sửa không còn khớp nguồn đã nhận diện.
