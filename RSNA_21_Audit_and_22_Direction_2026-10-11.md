# RSNA — Audit notebook 21 và hướng notebook 22

Cập nhật 11/10/2026 (UTC+7). Workspace: `C:\Users\ADMIN\Documents\RSNA`.

## Kết luận hiện tại

Người dùng xác nhận notebook 21 Public **0.913**, thấp hơn B2 standalone notebook 10 **0.914**; protected ensemble vẫn **0.943**. Nhánh frozen B2 density blend đóng lại: không dò thêm density weights hoặc target groups để tìm rounded Public gain. Điều này không chứng minh mọi mô hình RadImageNet đều hết tiềm năng; nó kết thúc phép thử frozen encoder/head/density hiện tại.

Hướng 22 đã được triển khai thành notebook local: thêm **public 2.5D ConvNeXt reader** như một family mới bên ngoài công thức ensemble 0.943 được giữ nguyên. Công thức thử trước công bố là global rank blend 70/30, cố định một lần; không dò target groups hoặc weights trên Public. Chưa có run Kaggle, submission hay Public Score của 22.

## Bằng chứng notebook 21

Nguồn: `results 21.zip` (2.917 bytes), gồm receipt, hai CSV controls và `submission.csv`. So sánh visible control với `results_10.zip` (dấu gạch dưới, notebook 10), không nhầm với `results 10.zip` (dấu cách, notebook 20).

| Kiểm tra | Kết quả đã xác minh |
| --- | --- |
| Receipt | `COMPLETE`; `B2_density_80_32_fixed_80_20_standalone` |
| Heads được notebook yêu cầu | Density 80/32 × folds 0–4 = 10 heads; root thuộc saved Output notebook 20 |
| Visible submission | 3 unique studies × 13 columns; đúng UID/12-label order; không NaN/nonfinite hoặc xác suất ngoài [0,1] |
| Probability range | 0.03870462–0.84295356 |
| 80-token parity với 10 | Receipt: max absolute diff 5,96e-8; tính lại từ CSV decimal: 7,00e-8 trên cùng 3 visible studies |
| Fixed pool | Tính lại `0.8*p80 + 0.2*p32`, max CSV difference 4,80e-8 |
| DICOM | 12 geometry-sorted series, 192 decoded images; `decode_failures_by_transfer_syntax` rỗng |
| Visible elapsed | 18,832 giây; không phải full hidden runtime |
| Encoder provenance | SHA256 `08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734`, khớp receipt notebook 10 |

Source notebook 21 kiểm tra density, fold, label order, encoder SHA, config và nonempty state của cả 10 heads trước inference. ZIP 21 không chứa checkpoint files, per-head hashes hay logs, nên không độc lập re-audit được từng head từ ZIP này. Cũng không có dedicated empty-study count trong receipt để suy ra nó bằng 0.

Không thấy lỗi export/schema/phép blend trên cohort hiển thị. Receipt ba studies không chứng minh hidden-run parity, AUC từng nhãn hay Private performance. Public 0.913 là bằng chứng người dùng xác nhận riêng, không suy ra từ receipt hoặc weak OOF. Gain exploratory của notebook 20 trên weak labels đã không chuyển thành gain Public của probe 21.

## Notebook 22 — implementation và evidence

Artifact: `22_RSNA_0943_Plus_ConvNeXt25D_Fixed30_Probe.ipynb`; builder: `build_22_convnext_probe.py`. Nó sao chép nguyên trạng 30 cell anchor 0.943 rồi nối phần reader, không đổi ensemble cũ.

- Data dependency mới duy nhất: [GoodPJW RSNA Knee 2.5D ConvNeXt reader v1](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader), gồm 3 EMA checkpoints. SHA-256 được pin trong notebook. License ghi Apache-2.0; attribution giữ trong markdown.
- Contract lấy từ public `knee.py` / `preprocess.py`: ConvNeXt-Tiny, sáu plane/fat-suppression slots, 12 native 3-slice windows, canonical 0.4-mm/384 volume, 256-pixel 0.92 centre crop, ImageNet normalization, study transformer + per-target attention; average probability của 3 folds. Inference chạy one-study, microbatch 8, không DataLoader workers / DataParallel.
- Blend cố định: `rank(0.70 * rank(anchor_0.943) + 0.30 * rank(reader))` trên 12 nhãn. Mức 30% là mốc tác giả đã thử trong stack khác: reader riêng báo Public 0.929, stack 0.943 → 0.944 ở weight .30; weight .20 của mô hình yếu hơn trong cùng repo lại kéo 0.943 xuống 0.937. Đây là chứng cứ chọn probe, không phải dự báo điểm cho ta. Nguồn độc lập: [repo/ablation log](https://github.com/aditbytes/kaggle-rsna-knee-abnormality), [notebook GoodPJW được fork làm baseline](https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944).
- Bảo vệ trước thời gian GPU: preflight pin/hash ba weights, schema/cohort, DICOM decode mẫu và `timm` trước khi anchor chạy; nếu codec không đọc được thì dừng sớm. Sau anchor bắt buộc xác minh 20/5/4/4 receipts + output hashes. Reader failure hoặc decode/empty failure không phát hành submission. Safety budget 8.5 giờ, nhưng runtime T4 thực tế chưa benchmark.
- Verification tại local: builder tạo 38 cell; AST parse toàn bộ code cells; xác nhận 30 cell anchor nguyên trạng; PyTorch model-batching tests chưa chạy được vì Python local không có `torch`. CPU validation suite 9/9 pass. Đây không phải Kaggle smoke run.
- **Asset correction, 11/10/2026:** bản build đầu đã pin sai ba `.pt` local tải dở (86,9/96,3/85,9 MB; đều mở đầu `PK` nhưng không có ZIP central directory). CPU diagnostic 22A trên dataset Kaggle `goodpjw2008/rsna-knee-2-5d-convnext-reader` cho ba checkpoint thật, mỗi file 123.035.271 bytes, với SHA-256 lần lượt `0b64baa9d1c29c8d5c4dffe06d9ab2c941119b37738e5a2346e882ab694e7b80`, `61696b6e67474695242fc8049e96e2ea13987db6e20d04410dc2961a689585cb`, `f5cf19cf591600c1c1a04f1aa5ff467751ef01513426a4ea3baa789443656cb6`. `preprocess.py`, `knee.py`, `infer.py` trên mount khớp source review byte-for-byte. Builder/notebook 22 đã được sửa để pin ba hash Kaggle này; kiểm tra fold/backbone/tensor shapes bằng CPU vẫn bắt buộc trước inference. Chưa có Kaggle run thành công của bản sửa, vì vậy không tuyên bố đã hết mọi lỗi.
- **Artifact để chạy sau correction:** `22B_RSNA_0943_Plus_ConvNeXt25D_Fixed30_AssetCorrected.ipynb`. Không dùng bản notebook 22 đã upload/saved trên Kaggle trước đó. Có thể chạy riêng hai cell đầu bằng CPU/Accelerator None để chứng minh preflight asset trước khi bật T4×2 và Save & Run All; sau khi đổi Accelerator, Kaggle sẽ khởi động kernel mới và chạy lại từ đầu.
- Provenance limit: source code cho biết labels từ 4 public report-label tables và nhóm fold theo report text; 58 gold bị loại khỏi optimizer nhưng dùng cho checkpoint selection. Upstream ancestry của weak labels/cohort membership không độc lập chứng thực bởi ta. Notebook ghi rõ giới hạn và không tuyên bố eligibility beyond reviewed public source. [Host OAI ruling](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819) áp dụng riêng cho OAI; source reader v1 được kiểm tra ở đây không dùng OAI raw MRI.

## Cách chạy tiếp

GPU T4×2, Internet Off. Giữ nguyên toàn bộ 18 dataSources của notebook 0.943; attach thêm đúng Dataset `goodpjw2008/rsna-knee-2-5d-convnext-reader` version 1. Không cần attach notebook outputs 09/20/21, 05A cache, hay silver labels. Save Version → Save & Run All; trước khi submit kiểm tra `convnext_probe_receipt.json` là COMPLETE, 3 parity checks, zero selected-series decode/empty failures, exact anchor member counts, đúng 1,300-ish hidden study count/schema. Sau đó mới nộp một lượt và gửi Public score + Output ZIP.

Đích dài hạn vẫn Private Top 10 (~0.96 aspirational), nhưng gain Public một tick chưa chứng minh Private gain. Giữ 0.943 làm benchmark/backup; chỉ nâng champion sau kết quả Kaggle thực tế.

Đích dài hạn vẫn Private Top 10. Giữ 0.943 làm benchmark/backup; chỉ nâng champion sau kết quả Kaggle thực tế, đồng thời ghi rõ Public improvement chưa chứng minh Private improvement.
