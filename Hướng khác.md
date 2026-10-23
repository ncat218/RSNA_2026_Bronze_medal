# Hướng khác

## ConVIRT-style image–text pretraining cho RSNA Knee Abnormality Detection

## 1. Ý tưởng tổng quát

Thay vì chỉ train một mô hình thị giác trực tiếp từ phần nhỏ dữ liệu MRI có nhãn 12 bệnh, có thể tận dụng **radiology report đi kèm MRI** làm nguồn supervision bổ sung theo tinh thần **ConVIRT / CLIP**.

Ý tưởng cốt lõi:

```text
Knee MRI study  ─────→ Image Encoder ─────→ z_image
                                               ↘
                                             similarity
                                               ↗
Radiology report ───→ Text Encoder ──────→ z_text
```

Trong quá trình pre-training:

- MRI của bệnh nhân A phải có embedding gần report của bệnh nhân A.
- MRI của bệnh nhân A phải xa report của các bệnh nhân khác.
- Model học representation của MRI dựa trên **natural-language supervision**, thay vì chỉ phụ thuộc vào class label 0/1.

Sau khi pre-train xong, chỉ giữ hoặc ưu tiên sử dụng **image encoder** để fine-tune cho bài toán cuối:

```text
Knee MRI
   ↓
Image Encoder đã pre-train bằng MRI–report
   ↓
Multi-label classification head
   ↓
12 xác suất bệnh / abnormality
```

Ở test time không cần report.

---

## 2. Vì sao hướng này phù hợp với RSNA?

Bài toán có ba đặc điểm rất phù hợp với ConVIRT-style learning:

1. **Chỉ một phần training studies có nhãn 12 findings.**
2. **Nhiều study vẫn có radiology report**, tức là phần dữ liệu không có label vẫn chứa supervision ở dạng text.
3. **MRI là dữ liệu y tế**, đúng loại domain mà ConVIRT được xây dựng để tận dụng: ít expert labels nhưng thường có report đi kèm.

Nếu chỉ train supervised:

```text
MRI có label
    ↓
classifier
```

thì phần lớn dữ liệu không có nhãn gần như bị bỏ phí.

Nếu dùng image–text contrastive pretraining:

```text
MRI không có 12 labels
        +
Radiology report
        ↓
Contrastive pretraining
        ↓
Image representation tốt hơn
```

sau đó mới fine-tune trên phần có nhãn.

---

## 3. Không nên xử lý MRI như một ảnh 2D đơn lẻ

Một knee MRI study thường gồm:

- nhiều series,
- mỗi series có nhiều slice,
- có thể khác anatomical plane hoặc sequence.

Vì vậy không nên đơn giản lấy:

```text
1 slice ↔ toàn bộ report
```

vì report mô tả toàn study, trong khi một slice đơn lẻ chỉ chứa một phần thông tin.

### Pipeline hợp lý hơn: study-level representation

```text
Series 1
n slices → Slice Encoder → pooling ──┐
                                     │
Series 2                             │
n slices → Slice Encoder → pooling ──┼→ Study Aggregator
                                     │        ↓
Series 3                             │     z_study
n slices → Slice Encoder → pooling ──┘
                                              ↕ contrastive learning
                                           z_report
                                              ↑
                                         Text Encoder
                                              ↑
                                      Radiology Report
```

Mục tiêu là tạo một embedding đại diện cho **toàn MRI study**, sau đó align embedding đó với report của cùng study.

---

## 4. Hướng mạnh hơn: structured report / finding-aware caption

Không nhất thiết dùng nguyên report dài làm text input.

Radiology report có thể chứa:

- nhiều câu không liên quan trực tiếp đến 12 nhãn cần dự đoán,
- cách diễn đạt không thống nhất,
- thông tin âm tính và dương tính lẫn nhau,
- nhiều ngôn ngữ hoặc thuật ngữ khác nhau.

Có thể chuẩn hóa report thành một **structured medical caption** tập trung vào đúng findings quan tâm.

Ví dụ:

```text
MRI of the knee demonstrates:
- ACL injury
- medial meniscus tear
- joint effusion
- no fracture
```

Pipeline:

```text
Radiology report gốc
        ↓
normalize / translate nếu cần
        ↓
extract các findings quan trọng
        ↓
structured caption
        ↓
Text Encoder
```

Sau đó contrastive với MRI study embedding.

Có thể gọi hướng này là:

> **Finding-aware image–text pretraining**

Mục tiêu là khiến text supervision tập trung hơn vào đúng semantic signal liên quan tới 12 nhãn dự đoán.

---

## 5. Có thể dùng text prompt cho từng bệnh

Ngoài report-level contrastive learning, có thể tạo prompt mô tả từng finding.

Ví dụ:

```text
"Knee MRI showing ACL injury."
"Knee MRI without ACL injury."
```

hoặc:

```text
"Knee MRI showing joint effusion."
"Knee MRI without joint effusion."
```

Khi đó image embedding được khuyến khích:

- gần prompt đúng,
- xa prompt sai.

Ví dụ:

```text
MRI ACL positive
      ↓
closer to: "Knee MRI showing ACL injury"
farther from: "Knee MRI without ACL injury"
```

Điểm hay của cách này là supervision không còn chỉ là:

```text
ACL = 1
```

mà trở thành semantic supervision bằng natural language.

---

## 6. Có nên từ MRI tự sinh caption rồi dùng caption để train không?

Có thể thử về sau, nhưng **không nên là hướng đầu tiên**.

### Vì sao?

Nếu dùng một model khác để đọc MRI và tự sinh caption:

```text
MRI
 ↓
medical VLM
 ↓
caption dự đoán
 ↓
dùng làm supervision
```

thì nếu caption sai, model phía sau sẽ học từ lỗi đó.

Rủi ro:

```text
MRI
 ↓
caption hallucination / sai finding
 ↓
supervision sai
 ↓
model học sai tiếp
```

Trong khi dữ liệu đã có **radiologist report thật**, nên tốt hơn là:

```text
Radiologist report
       ↓
NLP / LLM normalize
       ↓
structured caption
```

Tức là dùng AI để **chuẩn hóa report thật**, thay vì để AI tự bịa report mới từ MRI.

---

## 7. External data có thể tận dụng

### 7.1 MRNet

Rất gần domain vì cũng là knee MRI.

Có thể dùng để:

- pretrain knee MRI encoder,
- học abnormality / ACL / meniscus representation,
- sau đó transfer sang RSNA.

### 7.2 fastMRI+

Có knee MRI và pathology annotations.

Có thể dùng để học:

- localization,
- pathology-aware features,
- vùng giải phẫu quan trọng.

Hữu ích nếu muốn thêm auxiliary task như region/pathology prediction.

### 7.3 SKM-TEA

Có knee MRI kèm các annotation về:

- anatomy,
- cartilage / meniscus,
- pathology localization.

Có thể dùng cho pretraining hoặc auxiliary supervision.

### 7.4 ROCOv2

Có lượng lớn radiology image–text pairs.

Ưu điểm:

- phù hợp cho general medical image–text pretraining.

Nhược điểm:

- domain rộng,
- không chuyên knee MRI.

Vì vậy phù hợp làm **general medical pretraining**, sau đó mới domain-adapt sang knee MRI.

### 7.5 BiomedCLIP

Thay vì train image–text model hoàn toàn từ đầu, có thể dùng một biomedical vision-language model đã được pretrain trước rồi fine-tune/domain-adapt cho knee MRI.

Pipeline có thể là:

```text
BiomedCLIP pretrained
      ↓
RSNA MRI + report contrastive adaptation
      ↓
12-label fine-tuning
```

---

## 8. Lộ trình experiment đề xuất

Nên làm theo từng mức để biết phần nào thật sự đem lại gain.

### A. Visual-only baseline

```text
MRI → encoder/aggregator → 12 sigmoid outputs
```

Đây là baseline bắt buộc.

### B. BiomedCLIP / medical pretrained encoder

Dùng pretrained medical vision encoder, sau đó fine-tune classifier.

Mục tiêu: kiểm tra general medical pretraining có giúp knee MRI hay không.

### C. RSNA ConVIRT-style pretraining

```text
MRI study ↔ original radiology report
```

Contrastive pretrain trên tất cả training pairs có thể dùng được, sau đó fine-tune classifier trên subset có 12 labels.

### D. Structured-report contrastive pretraining

Thay original report bằng report đã được normalize thành findings/caption ngắn gọn.

So sánh trực tiếp:

```text
Original report
vs
Structured finding-aware caption
```

### E. Report-derived silver labels

Từ report, trích xuất các finding có confidence cao để tạo pseudo/silver labels cho phần study chưa có official labels.

Sau đó train với cả:

- gold labels,
- silver labels,
- image–text contrastive supervision.

### F. External knee MRI pretraining

Kết hợp MRNet / fastMRI+ / SKM-TEA trước khi adapt sang RSNA.

---

## 9. Loss function có thể dùng

Một phiên bản joint training đơn giản:

\[
L = L_{classification} + \lambda L_{image-text} + \mu L_{pseudo-label}
\]

Trong đó:

### Classification loss

Multi-label BCE hoặc biến thể phù hợp imbalance:

```text
Image Encoder
     ↓
12 sigmoid outputs
     ↓
classification loss
```

### Image–text contrastive loss

```text
MRI study embedding ↔ report embedding
```

Positive pair phải gần nhau, negative pair phải xa nhau.

### Pseudo-label / silver-label loss

Dùng cho những study không có official 12 labels nhưng có thể suy ra finding từ report với confidence cao.

Tổng thể:

```text
                  MRI Study
                     │
                     ↓
                Image Encoder
                     │
           ┌─────────┴─────────┐
           ↓                   ↓
    12 disease logits       z_image
           ↓                   ↕
 classification loss       z_report
                               ↑
                           Text Encoder
                               ↑
                             Report
```

---

## 10. Experiment quan trọng để chứng minh giá trị của text supervision

Nên so sánh:

```text
A. Generic pretrained visual encoder
B. Biomedical pretrained encoder
C. RSNA MRI–report contrastive pretraining
D. C + structured report / silver labels
E. D + external knee MRI pretraining
```

Ngoài việc train trên 100% labeled subset, nên giảm dần lượng label:

```text
100%
50%
25%
10%
```

Nếu model có image–text pretraining giảm performance ít hơn khi label bị giảm, đây là bằng chứng mạnh rằng:

> natural-language supervision giúp model học useful representation trong low-label regime.

---

## 11. Có thể dùng retrieval metric như auxiliary evaluation

Sau contrastive pretraining có thể kiểm tra:

```text
MRI study A
    ↓
image embedding
    ↓
retrieve among many reports
    ↓
report đúng của A đứng top mấy?
```

Metric có thể dùng:

- Recall@1
- Recall@5
- Recall@10
- Median Rank

Tuy nhiên đây chỉ nên là **auxiliary evaluation**.

Mục tiêu cuối của competition vẫn là chất lượng dự đoán 12 findings.

Do đó câu hỏi quan trọng không phải:

> Model retrieve report tốt tới đâu?

mà là:

> Image–text pretraining có giúp dự đoán 12 bệnh tốt hơn hay không?

---

## 12. Những hướng nên ưu tiên

### Ưu tiên cao

1. Visual-only baseline.
2. BiomedCLIP hoặc medical pretrained backbone.
3. RSNA MRI–report contrastive pretraining.
4. Structured report / finding-aware caption.
5. Silver labels từ report.
6. External knee MRI pretraining.

### Chưa nên ưu tiên ban đầu

- Tự generate report/caption trực tiếp từ MRI rồi dùng làm ground truth.
- Train một CLIP/ConVIRT cực lớn từ scratch.
- Dùng retrieval metric làm objective cuối thay cho classification metric.
- Dùng clustering như hướng chính khi đã có report supervision giàu thông tin hơn.

---

## 13. Kết luận

Hướng đáng thử nhất là:

> **MRI + radiology report contrastive pretraining → fine-tune cho 12-finding multi-label classification.**

Điểm mạnh của hướng này là tận dụng được cả những MRI chưa có official 12 labels.

Một phiên bản mạnh hơn:

> **Finding-aware image–text pretraining**, trong đó report được chuẩn hóa thành structured caption tập trung vào các bệnh/finding cần dự đoán.

Pipeline tổng thể đề xuất:

```text
                         TRAINING

          ┌──────────── MRI Study ─────────────┐
          │                                     │
          ↓                                     ↓
    Image Encoder                         Multi-label Head
          │                                     ↓
          │                               12 probabilities
          │                                     ↓
          │                           Classification Loss
          ↓
      z_image
          ↕ Contrastive Loss
      z_report
          ↑
    Text Encoder
          ↑
Radiology report
          ↓
optional normalize / finding extraction
          ↓
structured medical caption
```

Sau khi train:

```text
                           TEST

Knee MRI Study
     ↓
Image Encoder
     ↓
Study Aggregator
     ↓
12-label Head
     ↓
12 disease probabilities
```

Report chỉ đóng vai trò **training supervision**; inference không cần report.

Đây là một hướng phù hợp để khai thác semi-supervised / weakly supervised setting của bài toán hơn việc chỉ dựa vào phần nhỏ dữ liệu có label.
