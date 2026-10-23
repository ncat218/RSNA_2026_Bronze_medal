# Notebook thay thế cho lần submit 31

`results 31.zip` là output của pipeline v6/CoAt 55 cell: `status=COMPLETE`, 3/3 study mẫu, 4/4 CoAt reader, không thấy lỗi tại lần chạy này. Ảnh Kaggle báo exception ở lần chấm lại tập ẩn, nhưng ZIP không chứa traceback của lần đó; chưa thể khẳng định lỗi do thời gian, dữ liệu ẩn hay một reader cụ thể.

File hiện được gửi dưới tên `31_RSNA_0958_Rebuild_Strict.ipynb` là notebook khác, gồm 9 cell: Raptor 384 + Raptor 224 + R6 mirror TTA. User báo Public 0.958 cho nhánh này. Hai code cell Raptor giống hệt notebook 29A đã submit thành công; R6 chỉ đổi slug sang bản copy checkpoint R6 được pin, và blend cuối dùng trọng số theo nhãn.

`33_RSNA_0958_R6_Raptor_Submit.ipynb` đóng gói nhánh 9 cell đó thành file mới. Bảy code cell, checkpoint hash, công thức blend và bốn data sources giữ nguyên từng byte. Metadata Kaggle được sửa thành GPU T4×2 bật, Internet tắt; output 3-study cũ được xóa. Notebook nguồn không bị ghi đè.

Đã kiểm tra JSON, biên dịch code (sau khi bỏ dòng Jupyter `%%writefile`), hash của cả bảy code cell, metadata và danh sách input. Chưa chạy lại DICOM/GPU hoặc xác nhận Public 0.958 từ tài khoản Kaggle trong môi trường này. Khi Save & Run All, dùng đúng bốn input đã lưu trong notebook; xác nhận `NOTEBOOK_COMPLETE`, `rows` và `fallback` trước khi submit. Nếu Kaggle tiếp tục báo exception, cần log traceback của hidden rerun để sửa đúng nguyên nhân.
