# RAG Evaluation Results

## Framework sử dụng

> **RAGAS 0.1.21** — LLM judge: `openai/gpt-4o-mini` qua openrouter.

- Golden dataset: **18 câu hỏi** (song ngữ Anh–Việt, có câu ngoài domain để test fallback)
- top_k = 5 | Thời điểm chạy: 2026-08-04 16:21:27

---

## Overall Scores

| Metric | Config A — hybrid | Config B — dense_only | Δ (A − B) |
|--------|--------------------|--------------------|-----------|
| Faithfulness | 0.777 | 0.764 | +0.013 |
| Answer Relevance | 0.650 | 0.539 | +0.111 |
| Context Recall | 0.630 | 0.769 | -0.139 |
| Context Precision | 0.809 | 0.844 | -0.035 |
| **Average** | **0.716** | **0.729** | **-0.012** |


---

## A/B Comparison Analysis

**Config A — hybrid:**

> Hybrid — semantic (bge-m3) + BM25 → hợp nhất bằng RRF → rerank → fallback PageIndex khi điểm cosine gốc thấp (`retrieve()` của Task 9)

**Config B — dense_only:**

> Dense-only — chỉ semantic search bge-m3, KHÔNG BM25, KHÔNG rerank, KHÔNG fallback (`retrieve_dense_only()` của Task 9)

**Kết luận:**

> Hai config gần như ngang nhau (chênh -0.012). Với corpus nhỏ, dense search đã đủ tìm đúng tài liệu nên phần fusion + rerank chưa tạo khác biệt rõ; nên ưu tiên `dense_only` nếu cần độ trễ thấp.

---

## Worst Performers (Bottom 3)

_Xếp hạng theo điểm trung bình 4 metric của config `dense_only`._

| # | Question | Faithfulness | Relevance | Recall | Precision | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|-----------|---------------|------------|
| 1 | How do I renew my Vietnamese passport at the immigration office? | 0.000 | 0.000 | 0.000 | 0.000 | Ngoài domain (kỳ vọng) | Câu hỏi không có evidence trong corpus — điểm thấp là ĐÚNG, chỉ cần kiểm tra hệ thống có từ chối trả lời thay vì bịa. |
| 2 | Công thức nấu phở bò truyền thống gồm những nguyên liệu gì? | 0.000 | 0.000 | 0.000 | 0.000 | Ngoài domain (kỳ vọng) | Câu hỏi không có evidence trong corpus — điểm thấp là ĐÚNG, chỉ cần kiểm tra hệ thống có từ chối trả lời thay vì bịa. |
| 3 | When does the buyer get the refund money back after returning an item? | 0.500 | 0.000 | 0.667 | 1.000 | Generation | Câu trả lời lạc đề hoặc trả lời chung chung, không bám câu hỏi. |

**Phân tích chi tiết:**

1. **q17 — How do I renew my Vietnamese passport at the immigration office?** (mean = 0.000, 5 contexts, category: `out_of_domain`)
   - Khâu hỏng: **Ngoài domain (kỳ vọng)**
   - Nguyên nhân: Câu hỏi không có evidence trong corpus — điểm thấp là ĐÚNG, chỉ cần kiểm tra hệ thống có từ chối trả lời thay vì bịa.
2. **q18 — Công thức nấu phở bò truyền thống gồm những nguyên liệu gì?** (mean = 0.000, 5 contexts, category: `out_of_domain`)
   - Khâu hỏng: **Ngoài domain (kỳ vọng)**
   - Nguyên nhân: Câu hỏi không có evidence trong corpus — điểm thấp là ĐÚNG, chỉ cần kiểm tra hệ thống có từ chối trả lời thay vì bịa.
3. **q08 — When does the buyer get the refund money back after returning an item?** (mean = 0.542, 5 contexts, category: `returns_refunds`)
   - Khâu hỏng: **Generation**
   - Nguyên nhân: Câu trả lời lạc đề hoặc trả lời chung chung, không bám câu hỏi.

---

## Recommendations

### Cải tiến 1 — Tăng Answer Relevance
**Action:** Thêm few-shot mẫu trả lời ngắn gọn đúng trọng tâm vào prompt và yêu cầu trả lời trực tiếp câu hỏi ở câu đầu tiên trước khi giải thích.

**Expected impact:** Câu trả lời bám câu hỏi hơn, answer_relevancy tăng rõ nhất ở các câu hỏi 'how/what'.

### Cải tiến 2 — Chuẩn hoá chunking cho tài liệu song ngữ
**Action:** Chunk theo heading (markdown-aware) thay vì cắt cứng 800 ký tự, và gắn thêm tiêu đề mục vào đầu mỗi chunk để chunk tự mang ngữ cảnh.

**Expected impact:** Giảm trường hợp evidence bị cắt đôi giữa 2 chunk — cải thiện đồng thời recall và faithfulness.

### Cải tiến 3 — Calibrate lại SCORE_THRESHOLD cho fallback
**Action:** Đo phân bố điểm cosine gốc của semantic_search trên nhóm câu in-domain và nhóm ngoài domain trong golden dataset, chọn ngưỡng nằm giữa 2 phân bố (KHÔNG dùng điểm RRF).

**Expected impact:** Câu ngoài domain đi đúng nhánh PageIndex/từ chối trả lời, giảm hallucination mà không làm hỏng các câu in-domain.

---

## Ghi chú / hạn chế của lần chạy này

- Dùng embedding local bge-m3 cho RAGAS (không có OPENAI_API_KEY)
