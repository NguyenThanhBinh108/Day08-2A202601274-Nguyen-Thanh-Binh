# Bài Tập Nhóm — Trợ lý RAG Hỗ trợ Thương mại điện tử

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1: Sản phẩm nhóm Chatbot RAG

Xây dựng chatbot trả lời câu hỏi về chính sách thương mại điện tử và hỗ trợ khách hàng liên quan.

**Yêu cầu:**

- Giao diện chat HTML/CSS/JS trong `index.html`
- Trả lời có trích dẫn nguồn (dựa trên Task 10)
- Hỗ trợ câu hỏi tiếp nối (bộ nhớ hội thoại)
- Hiển thị tài liệu nguồn đã dùng

**Luồng gợi ý:**

```
Giao diện HTML/CSS/JS → Truy xuất (Task 9) → Sinh câu trả lời (Task 10) → Hiển thị
```

---

## Yêu cầu 2: Pipeline Đánh giá RAG

Sử dụng **1 trong 3 framework** sau để đánh giá pipeline RAG của nhóm:

### Framework lựa chọn

| Framework                                           | Cài đặt               | Đặc điểm                                      |
| --------------------------------------------------- | ------------------------ | ------------------------------------------------- |
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas`    | Chuẩn industry cho RAG eval, 3 trục chính      |
| [TruLens](https://github.com/truera/trulens)         | `pip install trulens`  | Dashboard UI, feedback functions mạnh            |

### Yêu cầu đánh giá

1. **Tạo Golden Dataset** — tối thiểu 15 cặp hỏi đáp (`question`, `expected_answer`, `expected_context`)
2. **Chạy đánh giá** trên toàn bộ golden dataset với các chỉ số sau:
   - **Faithfulness** — câu trả lời có bám đúng ngữ cảnh không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ bằng chứng không?
   - **Context Precision** — trong ngữ cảnh lấy về, bao nhiêu phần thực sự hữu ích?
3. **So sánh A/B** — chạy đánh giá trên ít nhất 2 cấu hình khác nhau, ví dụ có reranking và không reranking, hoặc hybrid và dense-only
4. **Báo cáo** — bảng điểm, phân tích các trường hợp kém nhất và đề xuất cải tiến

Xem code mẫu (DeepEval/RAGAS/TruLens) chi tiết trong `README.md` gốc mục "Yêu cầu 2".

### Sản phẩm cần nộp cho phần đánh giá

- [ ] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [ ] File `group_project/evaluation/eval_pipeline.py` — script chạy đánh giá
- [ ] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [ ] So sánh A/B ít nhất 2 cấu hình

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày, chạy local hoặc deploy
3. **Pipeline đánh giá** chạy được và có báo cáo kết quả
4. **Đẩy code lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

```
[Vẽ diagram kiến trúc ở đây]
```

---

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
| ------------ | ---- | ---------- | ------------ |
|              |      |            |              |
|              |      |            |              |
|              |      |            |              |
|              |      |            |              |

---

## Hướng Dẫn Chạy

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy giao diện demo có nối backend Task 10
py server.py
# hoặc
py app.py

# Sau đó mở http://localhost:5500
```

---

## Lưu ý

Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
