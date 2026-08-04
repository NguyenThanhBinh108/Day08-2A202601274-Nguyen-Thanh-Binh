# RAG Evaluation Results

## Framework sử dụng

> **RAGAS 0.1.21** — LLM judge: `openai/gpt-4o-mini` qua openrouter.

- Golden dataset: **18 câu hỏi** (song ngữ Anh–Việt, có câu ngoài domain để test fallback)
- top_k = 5 | Thời điểm chạy: 2026-08-04 16:56:13

---

## Overall Scores

| Metric | Config A — hybrid | Config B — dense_only | Δ (A − B) |
|--------|--------------------|--------------------|-----------|
| Faithfulness | 0.727 | n/a | n/a |
| Answer Relevance | 0.755 | n/a | n/a |
| Context Recall | 0.759 | n/a | n/a |
| Context Precision | 0.730 | n/a | n/a |
| **Average** | **0.743** | **n/a** | **n/a** |


---

## A/B Comparison Analysis

**Config A — hybrid:**

> Hybrid — semantic (bge-m3) + BM25 → hợp nhất bằng RRF → rerank → fallback PageIndex khi điểm cosine gốc thấp (`retrieve()` của Task 9)

**Config B — dense_only:**

> Dense-only — chỉ semantic search bge-m3, KHÔNG BM25, KHÔNG rerank, KHÔNG fallback (`retrieve_dense_only()` của Task 9)

**Kết luận:**

> Chưa đủ điểm ở cả 2 config để kết luận — xem phần Ghi chú bên dưới.

---

## Worst Performers (Bottom 3)

_Xếp hạng theo điểm trung bình 4 metric của config `hybrid`._

| # | Question | Faithfulness | Relevance | Recall | Precision | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|-----------|---------------|------------|
| 1 | How do I renew my Vietnamese passport at the immigration office? | 0.000 | 0.000 | 0.000 | 0.000 | Ngoài domain (kỳ vọng) | Câu hỏi không có evidence trong corpus — điểm thấp là ĐÚNG, chỉ cần kiểm tra hệ thống có từ chối trả lời thay vì bịa. |
| 2 | Công thức nấu phở bò truyền thống gồm những nguyên liệu gì? | 0.000 | 0.000 | 0.000 | 0.000 | Ngoài domain (kỳ vọng) | Câu hỏi không có evidence trong corpus — điểm thấp là ĐÚNG, chỉ cần kiểm tra hệ thống có từ chối trả lời thay vì bịa. |
| 3 | What evidence must a buyer provide when submitting a refund request? | 0.400 | 0.000 | 0.667 | 0.750 | Generation | Câu trả lời chứa khẳng định không có trong context (hallucination) — cần siết system prompt và bắt buộc citation. |

**Phân tích chi tiết:**

1. **q17 — How do I renew my Vietnamese passport at the immigration office?** (mean = 0.000, 2 contexts, category: `out_of_domain`)
   - Khâu hỏng: **Ngoài domain (kỳ vọng)**
   - Nguyên nhân: Câu hỏi không có evidence trong corpus — điểm thấp là ĐÚNG, chỉ cần kiểm tra hệ thống có từ chối trả lời thay vì bịa.
2. **q18 — Công thức nấu phở bò truyền thống gồm những nguyên liệu gì?** (mean = 0.000, 3 contexts, category: `out_of_domain`)
   - Khâu hỏng: **Ngoài domain (kỳ vọng)**
   - Nguyên nhân: Câu hỏi không có evidence trong corpus — điểm thấp là ĐÚNG, chỉ cần kiểm tra hệ thống có từ chối trả lời thay vì bịa.
3. **q03 — What evidence must a buyer provide when submitting a refund request?** (mean = 0.454, 5 contexts, category: `returns_refunds`)
   - Khâu hỏng: **Generation**
   - Nguyên nhân: Câu trả lời chứa khẳng định không có trong context (hallucination) — cần siết system prompt và bắt buộc citation.

---

## Recommendations

### Cải tiến 1 — Tăng Faithfulness
**Action:** Siết system prompt: bắt buộc mỗi câu phải kèm [Source] và trả lời 'Tôi không thể xác minh thông tin này từ nguồn hiện có' khi thiếu evidence; hạ temperature xuống 0.1.

**Expected impact:** Giảm hallucination, các câu ngoài domain trả lời đúng kiểu từ chối thay vì bịa.

### Cải tiến 2 — Chuẩn hoá chunking cho tài liệu song ngữ
**Action:** Chunk theo heading (markdown-aware) thay vì cắt cứng 800 ký tự, và gắn thêm tiêu đề mục vào đầu mỗi chunk để chunk tự mang ngữ cảnh.

**Expected impact:** Giảm trường hợp evidence bị cắt đôi giữa 2 chunk — cải thiện đồng thời recall và faithfulness.

### Cải tiến 3 — Calibrate lại SCORE_THRESHOLD cho fallback
**Action:** Đo phân bố điểm cosine gốc của semantic_search trên nhóm câu in-domain và nhóm ngoài domain trong golden dataset, chọn ngưỡng nằm giữa 2 phân bố (KHÔNG dùng điểm RRF).

**Expected impact:** Câu ngoài domain đi đúng nhánh PageIndex/từ chối trả lời, giảm hallucination mà không làm hỏng các câu in-domain.

---

## Ghi chú / hạn chế của lần chạy này

- Dùng embedding local bge-m3 cho RAGAS (không có OPENAI_API_KEY)
