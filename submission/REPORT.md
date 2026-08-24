# Lab 21 — Evaluation Report

**Họ tên**: Dương Hải Long  **MSSV**: 01607  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `T4 16GB (Colab)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | `Ticket CSKH tiếng Việt → JSON triage (250 samples)` — mặc định theo lab |
| Train / val | `225 / 25` (seed 42) |
| `max_length` | `1024` — p95 đo được là `98` tokens *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | `2` |

**Template có giữ khối `<think>` không?** `Có` — *(results/template_check.json)*
Qwen3.5 template bảo tồn khối `<think>` nguyên vẹn; reasoning traces được truyền tới loss.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.4149` |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

**Nhận xét**: Mask proof xác nhận — sau khối suy luận, câu trả lời JSON nằm trong phần được tính loss (`supervised_fraction = 41.49%`), trong khi câu hỏi đã bị che phủ (mặc định `assistant-only` mode). Template Qwen3.5 giữ nguyên khối `<think>` nên không có vấn đề mất dữ liệu reasoning.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

**Chạy NB2 để đo ba baseline trước khi train:**
- **(a) base + naive prompt**: Chỉ dùng system message mặc định
- **(b) base + optimized prompt**: Prompt được tối ưu cho tác vụ triage
- **(c) LoRA fine-tune**: Model sau khi fine-tune 2 epochs

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | _FILL_FROM_NB2_ | _FILL_FROM_NB2_ | _FILL_FROM_NB2_ | _FILL_FROM_NB2_ |
| (b) base + optimized prompt | _FILL_FROM_NB2_ | _FILL_FROM_NB2_ | _FILL_FROM_NB2_ | _FILL_FROM_NB2_ |
| (c) LoRA fine-tune | _FILL_FROM_NB5_ | _FILL_FROM_NB5_ | _FILL_FROM_NB5_ | _FILL_FROM_NB5_ |

**Hướng dẫn**: 
- Chạy NB2 trên GPU để đo baseline (a) và (b) trước khi train
- Sau khi NB3 train xong, NB5 sẽ đo kết quả (c)
- Sao chép số liệu từ `results/evals.json` hoặc `results/evals.csv` vào bảng trên

**(b) có thật sự mạnh hơn (a) không?** Điền sau chạy NB2

---

## 4. Giải phẫu cấu hình sai (NB4)

**Chạy NB4 để so sánh ba run đối chứng với run chính xác:**

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | _FILL_ | _FILL_ | _FILL_ | _FILL_ | _FILL_ | _FILL_ |
| `attn_only` | q,v | *(matched)* | _FILL_ | _FILL_ | _FILL_ | _FILL_ | _FILL_ | _FILL_ |
| `wrong_lr` | text-linear | 16 | _FILL_ | _FILL_ | _FILL_ | _FILL_ | _FILL_ | _FILL_ |
| `qlora` | text-linear | 16 | _FILL_ | _FILL_ | _FILL_ | _FILL_ | _FILL_ | _FILL_ |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1.

**Hướng dẫn**: 
- Chạy NB4 sau NB3 (nó sẽ train 3 run đối chứng song song)
- Sao chép số liệu từ `results/runs.csv` vào bảng trên
- Trả lời ba câu hỏi dưới đây sau khi thấy kết quả

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**

_ANSWER_: Điền sau chạy NB4


**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

_ANSWER_: Điền sau chạy NB4


**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**

_ANSWER_: Điền sau chạy NB4

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `_FILL_FROM_NB5_VERDICT_`
`target Δ = _FILL_` · `regression Δ = _FILL_` · `valid_trace_rate = _FILL_`

**Hướng dẫn**: 
- Chạy NB5 sau khi NB3 và NB4 xong
- NB5 sẽ in kết quả phán quyết: `PASSED` hoặc `FAILED`
- Sao chép `target Δ`, `regression Δ`, và `valid_trace_rate` từ output NB5

**Diễn giải (≥100 từ)**:

Điền nội dung diễn giải sau khi chạy NB5. Nếu FAILED, hãy phân tích:
- Tại sao fine-tune không vượt qua cổng (regression hoặc target)?
- Dataset có đủ lớn/chất lượng không?
- Learning rate hoặc epoch có phù hợp không?
- Mask mode có lựa chọn đúng không?

_ANSWER_: Điền sau chạy NB5

---

## 6. Định tính — bắt buộc có cả ca THUA

**Hướng dẫn**:
- Chạy NB5 để lấy kết quả chi tiết từng mẫu
- Chọn 5 ticket: ít nhất 3 ca fine-tune thắng, và ít nhất 2 ca fine-tune thua
- Điền vào bảng dưới với output từ model base (b) và fine-tune (c)
- Giải thích pattern nếu có (ví dụ: FT thua khi ticket ngắn, khi intent hiếm, etc.)

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | _FILL_ | _FILL_ | _FILL_ | _FILL_ | ✅ FT thắng |
| 2 | _FILL_ | _FILL_ | _FILL_ | _FILL_ | ✅ FT thắng |
| 3 | _FILL_ | _FILL_ | _FILL_ | _FILL_ | ✅ FT thắng |
| 4 | _FILL_ | _FILL_ | _FILL_ | _FILL_ | ❌ **FT thua** |
| 5 | _FILL_ | _FILL_ | _FILL_ | _FILL_ | ❌ **FT thua** |

**Có mẫu chung nào ở các ca FT thua không?**

_ANSWER_: Điền sau chạy NB5

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ)**: Bạn có nên deploy bản fine-tune này không, và vì sao? Đâu là đòn
bẩy thật sự trong lab này — vị trí adapter, learning rate, chất lượng dữ liệu, hay mask?

_ANSWER_: Điền sau chạy NB5. Hãy cân nhắc:
- Tỷ lệ cải thiện của fine-tune so với baseline (b)?
- Có vượt qua cổng hồi quy không?
- Có trường hợp nào fine-tune làm hỏng kết quả không (regression)?
- Dataset đủ lớn hay chỉ bé → overfit?


**Ba điều tôi học được** (cụ thể, không generic):
1. _FILL_
2. _FILL_
3. _FILL_

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**

_FILL_: Mô tả cụ thể phương pháp hoặc thí nghiệm bạn muốn thử

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
