# Phân Công Nhóm 5 Người — Lab 08 RAG Pipeline

| Role | Tên | Trách nhiệm chính |
|---|---|---|
| Role 1 — Team Leader & RAG Architect | **Bình** | Điều phối, Task 3, Task 9 (ghép pipeline), review |
| Role 2 — Data & Dense Search Dev | **Đăng** | Task 1, Task 4 (ChromaDB), Task 5 (Semantic + HyDE) |
| Role 3 — Sparse Search & Reranking Dev | **Vũ** | Task 6 (BM25), Task 7 (RRF), Task 8 (PageIndex) |
| Role 4 — Frontend & Chatbot Dev | **Linh** | Task 2 (crawl), Task 10 (Generation), `app.py` |
| Role 5 — Evaluation & QA Engineer | **Liễu** | `golden_dataset.json`, RAGAS, báo cáo A/B |

---

## 1. Quy tắc chống chồng chéo: mỗi file chỉ một người chạm

Đây là nguyên tắc quan trọng nhất. Nếu tôn trọng bảng này thì gần như **không bao giờ có git conflict**.

| File | Chủ sở hữu duy nhất |
|---|---|
| `src/task1_collect_legal_docs.py` | Đăng |
| `src/task2_crawl_news.py` | **Linh** |
| `src/task3_convert_markdown.py` | **Bình** |
| `src/task4_chunking_indexing.py` | Đăng |
| `src/task5_semantic_search.py` | Đăng |
| `src/task6_lexical_search.py` | Vũ |
| `src/task7_reranking.py` | Vũ |
| `src/task8_pageindex_vectorless.py` | Vũ |
| `src/task9_retrieval_pipeline.py` | Bình |
| `src/task10_generation.py` | Linh |
| `app.py` | Linh |
| `group_project/evaluation/*` | Liễu |
| `requirements.txt`, `.env`, `README.md` | Bình |

**Hai thay đổi so với bản Phương Án B gốc, có lý do:**

1. **Task 2 (crawl) giao Linh, không giao Đăng.** Bản gốc dồn Task 1–3 + 4 + 5 cho Role 2, tức Đăng một mình ôm **toàn bộ đường găng** — 4 người còn lại ngồi chờ. Linh nhận Task 2 được vì `app.py` **đã nối sẵn** `generate_with_citation` ([app.py dòng 120–123](app.py#L120-L123)), Linh không có việc gấp ở 30 phút đầu.
2. **Task 3 giao Bình.** Task 3 là điểm hợp lưu của Task 1 (Đăng) và Task 2 (Linh); để người điều phối cầm thì chạy được ngay khi file đầu tiên về. Code Task 3 chỉ là bỏ comment ~10 dòng ([task3 dòng 39–43](src/task3_convert_markdown.py#L39-L43)), không tốn thời gian của Bình.

> Bản gốc còn giao Role 2 "kết nối `generate_with_citation()` vào `app.py`" ở CP5 — **việc này đã có sẵn trong starter**, đừng giao lại cho ai.

**Branch:** mỗi người một nhánh `feature/<tên>-task<N>`, PR vào `main`, Bình merge.

---

## 2. Đồ thị phụ thuộc thật (đã kiểm chứng từ `tests/test_individual.py`)

```
Task 1 (Đăng: PDF)  ─┐
                     ├→ Task 3 (Bình: .md) ─┬→ Task 4 (Đăng: chroma_db) → Task 5 (Đăng)
Task 2 (Linh: crawl)─┘                      │                                    │
                                            └→ Task 6 (Vũ: BM25) ────────────────┤
                                            └→ Task 8 (Vũ: PageIndex)            │
                                                                                 ↓
Task 7 (Vũ: RRF) ── KHÔNG phụ thuộc gì ────────────────────────→ Task 9 (Bình) ──┘
                                                                       ↓
                                                     Task 10 (Linh) → app.py → RAGAS (Liễu)
```

### Ba việc làm được ngay từ phút 0, không cần chờ ai

Đây là chìa khoá để không ai ngồi không:

| Việc | Ai | Vì sao không cần chờ |
|---|---|---|
| **Task 7** — `rerank_rrf()` + `rerank()` | Vũ | Test truyền thẳng list candidates viết tay ([test dòng 378–382](tests/test_individual.py#L378-L382)), không đụng data hay ChromaDB. Pass 3/3 test ngay. |
| **Task 10** — `reorder_for_llm()` + `format_context()` | Linh | Test cũng dùng chunks viết tay ([test dòng 526–543](tests/test_individual.py#L526-L543)). Pass 2/3 test Task 10 trước cả khi có pipeline. |
| **`golden_dataset.json`** | Liễu | Viết từ chính trang help center, không cần hệ thống chạy. |

### Điểm tháo nút thắt: Task 6 KHÔNG cần chờ ChromaDB

[task6 dòng 20–21](src/task6_lexical_search.py#L20-L21) cho phép load corpus **thẳng từ `data/standardized/`**. Vũ chỉ cần file `.md` của Bình, **không phải đợi Đăng embed xong**. Task 4 (embed, chậm) và Task 6 (BM25, nhanh) chạy song song hoàn toàn.

### Chiến thuật "seed data" — mốc phút 20

Đừng đợi đủ 3 PDF + 5 bài news rồi mới convert. **Ngay khi có 1 PDF và 1 JSON đầu tiên**, Bình chạy Task 3 luôn để đẻ ra 2 file `.md`. Đăng lập tức index "mini" để có `chroma_db/` sớm, Vũ có corpus cho BM25. Cuối CP1 khi data về đủ thì Đăng **xoá `chroma_db/` cũ rồi index lại toàn bộ** (bắt buộc — xem [cảnh báo task4 dòng 28–30](src/task4_chunking_indexing.py#L28-L30)).

---

## 3. Contract chốt ở CP0 (Bình chủ trì, 5 phút)

Test đã cố định sẵn schema — chốt lại để 5 người code song song mà ráp vào là khớp:

```python
# Mọi hàm search/retrieve đều trả về list các dict dạng:
{
    "content":  str,    # bắt buộc
    "score":    float,  # bắt buộc, sort GIẢM DẦN
    "metadata": {"source": str, "type": "legal"|"news", "chunk_index": int},
    "source":   str,    # CHỈ Task 8 ("pageindex") và Task 9 ("hybrid"|"pageindex")
}
```

Chữ ký hàm cố định, không ai được đổi:

```python
semantic_search(query: str, top_k: int = 10)              -> list[dict]   # Đăng
lexical_search (query: str, top_k: int = 10)              -> list[dict]   # Vũ
rerank_rrf(ranked_lists: list[list[dict]], top_k=5, k=60) -> list[dict]   # Vũ
rerank(query, candidates, top_k=5, method="rrf")          -> list[dict]   # Vũ
pageindex_search(query: str, top_k: int = 5)              -> list[dict]   # Vũ
retrieve(query, top_k=5, score_threshold=..., use_reranking=True) -> list[dict]  # Bình
generate_with_citation(query: str, top_k: int = 5)        -> dict         # Linh
```

Chốt luôn: `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`, `COLLECTION_NAME="ecommerce_support_docs"`, RRF `k=60`.

---

## 4. Timeline chi tiết theo checkpoint

### CP0 — 0:00–0:10 · Môi trường + chốt contract

| Người | Việc | Chờ ai |
|---|---|---|
| **Bình** | Tạo repo nhóm, push starter, chia `.env` (`OPENROUTER_API_KEY`). Chốt contract mục 3. Chốt chủ đề corpus. | — |
| **Đăng** | `python -m venv .venv` → `pip install -r requirements.txt`. **Chạy ngay lệnh tải embedding model** (mục 5, rủi ro #1). | — |
| **Vũ** | Cài env → **bắt tay Task 7 luôn**, không chờ CP3. | — |
| **Linh** | Cài env, `streamlit run app.py` xác nhận UI lên. Điền `ARTICLE_URLS`. | — |
| **Liễu** | Cài env, kiểm tra `import ragas, datasets`. Mở help center chọn 15–20 câu hỏi. | — |

✅ **Pass:** 5 máy `import chromadb, sentence_transformers` không lỗi.

### CP1 — 0:10–0:35 · Thu thập dữ liệu (4 người chạy song song)

| Người | Việc | Chờ ai |
|---|---|---|
| **Đăng** | **Task 1** — tải ≥3 PDF chính sách vào `data/landing/legal/`, mỗi file >1KB. **Đẩy file đầu tiên lên sớm nhất có thể.** | — |
| **Linh** | **Task 2** — implement `crawl_article()`, chạy ≥5 URL → JSON có key `url`, mỗi file >500 bytes. | — |
| **Bình** | **Task 3** — bỏ comment code convert. Chạy "seed" ngay khi có 1 PDF + 1 JSON, chạy lại full ở cuối CP1. | 1 file đầu của Đăng/Linh |
| **Vũ** | **Hoàn thành Task 7** → `pytest -k TestTask7` pass 3/3. | — |
| **Liễu** | Viết `golden_dataset.json` từ đúng các trang Linh crawl. | — |

✅ **Pass:** ≥3 file `legal/`, ≥5 file `news/`, có `.md` trong `standardized/`. → 4 test đang FAIL của Task 1/2/3 chuyển sang PASS.

### CP2 — 0:35–1:00 · Index + hai nhánh search (song song)

| Người | Việc | Chờ ai |
|---|---|---|
| **Đăng** | **Task 4** — chunk → embed → ChromaDB. Xong thì làm **Task 5** (cosine + HyDE). | `.md` của Bình |
| **Vũ** | **Task 6** — BM25 đọc thẳng `data/standardized/`, **không chờ Đăng**. | `.md` của Bình |
| **Bình** | Duyệt tham số chunking. Viết **khung Task 9** theo contract (chưa chạy được cũng viết). | — |
| **Linh** | **Task 10** phần thuần: `reorder_for_llm()` + `format_context()` → pass 2/3 test. | — |
| **Liễu** | Chốt 15–20 câu hỏi + đáp án tham chiếu (`ground_truth`). | — |

✅ **Pass:** có `chroma_db/`, test Task 4/5/6 pass.

### CP3 — 1:00–1:20 · Rerank + Fallback

| Người | Việc | Chờ ai |
|---|---|---|
| **Vũ** | **Task 8** — PageIndex, kết quả phải có `"source": "pageindex"`. | `.md` |
| **Đăng** | Tinh chỉnh Task 5. **Đo điểm cosine** cho ~3 câu liên quan và ~3 câu lạc đề → **báo số cho Bình**. | — |
| **Bình** | Ráp **Task 9**, dùng số của Đăng để calibrate `SCORE_THRESHOLD`. | Task 5, 6, 7 |
| **Linh** | Trau chuốt UI: sidebar `top_k`, khung nguồn tham khảo, câu hỏi gợi ý. | — |
| **Liễu** | Chuẩn bị câu ngoài domain để test fallback. | — |

✅ **Pass:** RRF gộp được 2 ranker; PageIndex trả kết quả.

### CP4 — 1:20–1:45 · Pipeline + Generation → **mốc 50 điểm cá nhân**

| Người | Việc | Chờ ai |
|---|---|---|
| **Bình** | Task 9 chạy thông, `source` ∈ `{hybrid, pageindex}`. Chạy `pytest` cho cả nhóm. | Task 5,6,7,8 |
| **Linh** | **Task 10** — gọi LLM sinh câu trả lời có citation. | Task 9 |
| **Đăng** / **Vũ** | Trực hỗ trợ debug; ai xong sớm thì viết phần kiến trúc cho slide. | — |
| **Liễu** | Dựng sẵn `eval_pipeline.py`, chờ pipeline xanh là chạy. | — |

✅ **Pass:** `pytest tests/test_individual.py` đạt **35/35**.

### CP5 — 1:45–2:15 · Chatbot UI + RAGAS

| Người | Việc |
|---|---|
| **Linh** | Hoàn thiện `app.py`, chạy demo thật, kiểm tra hiển thị nguồn. |
| **Liễu** | Chạy RAGAS 4 chỉ số, viết `results.md` với bảng **A/B: Hybrid vs Dense-only**. |
| **Đăng** | Cấp "chế độ dense-only" cho Liễu để so sánh — gọi thẳng `semantic_search()` là đủ, **không cần viết thêm code mới**. |
| **Bình** | Gom code tối ưu nhất vào `main`, theo dõi tiến độ báo cáo. |
| **Vũ** | Viết phần Hybrid Search / RRF cho slide. |

### CP6 — 2:15–3:00 · Demo

Bình thuyết trình kiến trúc → Linh demo live Streamlit → Vũ trả lời câu hỏi Hybrid/RRF/Fallback → Đăng trả lời chunking/embedding → Liễu báo cáo RAGAS + phân tích A/B.

---

## 5. Rủi ro đã lường trước

**#1 — Embedding model `BAAI/bge-m3` nặng ~2.2 GB.** Lúc cài `requirements.txt` trên máy này, tốc độ mạng đo được chỉ **30–60 kB/s**. Nếu giữ nguyên tốc độ đó, tải bge-m3 mất **nhiều giờ** và sẽ chết CP2. **Đăng phải chạy lệnh này ngay ở CP0**, song song với mọi việc khác:

```powershell
.\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```

Nếu sau 15 phút chưa xong: đổi sang `sentence-transformers/all-MiniLM-L6-v2` (~90 MB, `EMBEDDING_DIM=384`). Đánh đổi: MiniLM yếu tiếng Việt hơn hẳn — nếu đổi thì phải **xoá `chroma_db/` và index lại**, và ghi rõ lý do trong báo cáo.

**#2 — `chroma_db/` chưa có trong `.gitignore`.** Đây là DB nhị phân sinh lại được. 5 người cùng commit nó sẽ gây conflict liên miên và phình repo. **Bình thêm `chroma_db/` vào `.gitignore` ngay ở CP0.** Ngược lại, `data/` thì **phải commit** vì test chấm điểm đọc file thật trong đó.

**#3 — Chuỗi import dây chuyền.** `app.py` → `task10` → `task9` → `task5, 6, 7, 8` ([task9 dòng 28–31](src/task9_retrieval_pipeline.py#L28-L31)). Chỉ cần một người push file lỗi cú pháp là **cả nhóm không chạy được gì**. Quy tắc: chỉ push code `import` được; chưa làm thì để nguyên `raise NotImplementedError` (test tự skip, không fail).

**#4 — Bẫy ngưỡng RRF.** Điểm RRF luôn ≈ 0.0164 bất kể câu hỏi liên quan hay không, nên **không được** đem so với `SCORE_THRESHOLD`. Bình phải dùng **điểm cosine gốc từ `semantic_search`** để quyết định fallback ([giải thích task9 dòng 14–25](src/task9_retrieval_pipeline.py#L14-L25)).

**#5 — Đổi corpus mà quên xoá `chroma_db/`** → chunk cũ lẫn chunk mới, retrieval trả rác. Mỗi lần đổi/thêm tài liệu: xoá thư mục rồi index lại.
