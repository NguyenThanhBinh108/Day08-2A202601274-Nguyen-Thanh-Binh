# PROJECT OVERVIEW — Trợ Lý RAG Hỗ Trợ Thương Mại Điện Tử (Shopee Support Chatbot)

> File này mô tả toàn bộ dự án: mục tiêu, kiến trúc, luồng dữ liệu, từng file/folder làm gì, trạng thái hoàn thành thật (không tô hồng), và các khoảng hở kỹ thuật đang tồn tại. Dùng file này để (1) tự dựng slide thuyết trình, (2) dán cho ChatGPT/LLM khác đọc để tư vấn thiết kế frontend hoặc kiến trúc hệ thống — không cần giải thích lại từ đầu.

---

## 1. Đây là dự án gì

Một **RAG (Retrieval-Augmented Generation) pipeline end-to-end** cho chatbot hỗ trợ khách hàng thương mại điện tử, chủ đề: chính sách Shopee Việt Nam (thanh toán, đổi trả/hoàn tiền, quy định người bán, quyền riêng tư) + hướng dẫn hỗ trợ (theo dõi đơn hàng, bằng chứng hoàn tiền, đổi phương thức thanh toán...).

Bài lab gồm **10 Task kỹ thuật cá nhân** (thu thập dữ liệu → indexing → hybrid retrieval → generation có citation) cộng thêm **1 bài tập nhóm** (giao diện chatbot + đánh giá chất lượng bằng RAGAS). 4 thành viên chia nhau làm trên các nhánh git riêng rồi merge vào `haidang2425`.

---

## 2. Kiến trúc luồng xử lý (Data Flow)

```
┌─────────────┐   ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐
│  Task 1-2   │──▶│   Task 3    │──▶│    Task 4    │──▶│  ChromaDB       │
│ Thu thập PDF│   │ Convert MD  │   │ Chunk+Embed  │   │ (vector store)  │
│ + crawl JSON│   │ (MarkItDown)│   │ (local model)│   │  chroma_db/     │
└─────────────┘   └─────────────┘   └──────────────┘   └────────┬────────┘
                                                                  │
                        User query từ giao diện chat              │
                                  │                                 │
                                  ▼                                 ▼
                         ┌─────────────────────────────────────────────┐
                         │            Task 9 — retrieve()                │
                         │  ┌────────────┐      ┌────────────┐          │
                         │  │  Task 5    │      │  Task 6    │          │
                         │  │ Semantic   │      │ Lexical    │          │
                         │  │ (cosine)   │      │ (BM25)     │          │
                         │  └─────┬──────┘      └─────┬──────┘          │
                         │        └──────┬─────────────┘                │
                         │               ▼                              │
                         │       Task 7 — Rerank (RRF)                  │
                         │               │                              │
                         │   best cosine score < 0.3 ? ──▶ Task 8       │
                         │               │                PageIndex     │
                         │               │                (vectorless)  │
                         │               ▼                              │
                         └───────────────┬───────────────────────────────┘
                                          ▼
                         ┌─────────────────────────────────────┐
                         │        Task 10 — generate_with_citation()   │
                         │  reorder_for_llm() (chống lost-in-middle)   │
                         │  → format_context() → LLM (OpenRouter)      │
                         │  → answer kèm [Nguồn, Năm]                  │
                         └───────────────────┬───────────────────────┘
                                              ▼
                         ┌─────────────────────────────────────┐
                         │  server.py  (POST /api/chat)         │
                         │  index.html + styles.css + script.js │
                         │  → http://localhost:5500             │
                         └───────────────────────────────────────┘
```

**Ý tưởng lõi cần nhớ khi thiết kế UI**: hệ thống là **Hybrid Search (Semantic + BM25) → Rerank (RRF) → Fallback (PageIndex nếu điểm thấp) → Generation có trích dẫn nguồn**. UI nên thể hiện được các bước này (đã có sẵn khung "Luồng xử lý" trong `index.html`), không chỉ là ô chat đơn thuần.

---

## 3. Cây thư mục — file nào làm gì

```
├── index.html              # Cấu trúc trang chatbot (sidebar cấu hình + khung chat + panel nguồn)
├── assets/
│   ├── styles.css          # Toàn bộ style (591 dòng) — dùng CSS variables (:root), theme sáng
│   └── script.js           # Logic JS: gửi câu hỏi, render chat bubble, hiện nguồn tham khảo
├── server.py                # HTTP server thuần Python (http.server), route:
│                             #   GET  /            → index.html
│                             #   POST /api/chat     → gọi generate_with_citation()
├── app.py                   # Entry point tiện lợi, chỉ gọi server.main()
│
├── src/                      # 10 module lõi của pipeline (mỗi file = 1 Task)
│   ├── task1_collect_legal_docs.py   # Sinh 14 PDF chính sách (dùng fpdf2) → data/landing/legal/
│   ├── task2_crawl_news.py           # Sinh 10 JSON bài hướng dẫn → data/landing/news/
│   ├── task3_convert_markdown.py     # Convert PDF/JSON → Markdown (MarkItDown) → data/standardized/
│   ├── task4_chunking_indexing.py    # Chunk (RecursiveCharacterTextSplitter 800/100) + embed
│   │                                  #   (sentence-transformers/all-MiniLM-L6-v2, local, 384dim)
│   │                                  #   + index vào ChromaDB
│   ├── task5_semantic_search.py      # semantic_search() — cosine similarity trên ChromaDB + HyDE
│   ├── task6_lexical_search.py       # lexical_search() — BM25/TF-IDF (keyword-exact)
│   ├── task7_reranking.py            # rerank_rrf() — Reciprocal Rank Fusion (k=60) gộp 2 kết quả
│   │                                  #   trên; có thêm Jina cross-encoder reranker (cần JINA_API_KEY)
│   ├── task8_pageindex_vectorless.py # pageindex_search() — fallback không cần vector, dùng
│   │                                  #   PageIndex SDK/API (cần PAGEINDEX_API_KEY, có fallback nội bộ)
│   ├── task9_retrieval_pipeline.py   # retrieve() — hợp nhất Task5+6+7, quyết định fallback Task8
│   │                                  #   khi best cosine score gốc < SCORE_THRESHOLD (0.3)
│   └── task10_generation.py          # generate_with_citation() — reorder chống lost-in-middle,
│                                       #   format prompt, gọi LLM (OpenRouter/OpenAI), fallback
│                                       #   extractive nếu không có API key hoặc lỗi
│
├── data/
│   ├── landing/               # Dữ liệu thô (Task 1-2)
│   │   ├── legal/              # 14 file .pdf + documents_index.json (metadata)
│   │   └── news/                # 10 file .json (mỗi file: url, title, customer_role, content_markdown...)
│   └── standardized/           # Dữ liệu đã convert Markdown (Task 3), giữ cấu trúc con legal/ + news/
│
├── chroma_db/                 # Vector store ChromaDB (SINH RA khi chạy Task 4, KHÔNG commit — gitignored)
│
├── tests/
│   └── test_individual.py      # 35 test pytest chấm điểm Task 1-10 (50% điểm kỹ thuật cá nhân)
│
├── group_project/
│   ├── README.md                # Hướng dẫn + phân công bài nhóm
│   └── evaluation/
│       ├── golden_dataset.json   # Bộ câu hỏi Q&A mẫu để đánh giá (yêu cầu ≥15, HIỆN CHỈ CÓ 3)
│       ├── eval_pipeline.py      # Script chạy RAGAS/DeepEval/TruLens — HIỆN LÀ FILE KHUNG, CHƯA IMPLEMENT
│       └── results.md            # Báo cáo điểm A/B testing — HIỆN LÀ TEMPLATE RỖNG
│
├── requirements.txt            # Toàn bộ dependency Python (crawl4ai, chromadb, sentence-transformers,
│                                 #   rank-bm25, scikit-learn, pageindex, openai, streamlit*, ragas, pytest)
├── .env.example                 # Mẫu biến môi trường (API keys)
├── .env                         # File thật (KHÔNG trong git, chứa key thật của máy này)
├── .gitignore                   # Loại trừ .venv/, .env, __pycache__/, chroma_db/, ...
│
├── check_environment.py         # Script kiểm tra môi trường CP0 (Python version, packages, API key, dirs)
├── README.md                    # Tài liệu chính thức của bài lab (mục tiêu, chấm điểm, timeline)
├── LAB_GUIDE.md                  # Hướng dẫn chi tiết từng Task + lịch 7 Checkpoint (180 phút)
├── SETUP_GUIDE.md                # Hướng dẫn cài đặt & chạy dự án từ đầu cho người mới (mới viết lại)
├── day8-lab-rag-pipeline.md      # Tài liệu gốc mô tả đề bài (từ ban tổ chức)
└── checkpoint_timer.html         # Dashboard đếm ngược Checkpoint dùng lúc làm lab trực tiếp
```

*`streamlit` nằm trong requirements.txt nhưng thực tế **không dùng** — nhóm chọn xây UI bằng HTML/CSS/JS thuần (`index.html`) + `server.py` thay vì Streamlit như gợi ý mặc định của đề bài.

---

## 4. Trạng thái hoàn thành thật (đã verify bằng cách chạy thật, không đoán)

| Phần | Trạng thái | Bằng chứng |
|---|---|---|
| Task 1 — Thu thập PDF pháp lý | ✅ Xong | 14/14 PDF, test pass |
| Task 2 — Crawl bài viết | ✅ Xong | 10 JSON (yêu cầu ≥5), test pass |
| Task 3 — Convert Markdown | ✅ Xong | 24 file .md, test pass |
| Task 4 — Chunking & Indexing | ✅ Xong | ChromaDB có data, test pass |
| Task 5 — Semantic Search | ✅ Xong | test pass |
| Task 6 — Lexical Search (BM25) | ✅ Xong | test pass |
| Task 7 — Reranking (RRF) | ✅ Xong | test pass |
| Task 8 — PageIndex Fallback | ✅ Xong | test pass |
| Task 9 — Retrieval Pipeline | ✅ Xong | test pass, đúng bẫy threshold cosine gốc |
| Task 10 — Generation + Citation | ✅ Xong | test pass |
| **→ pytest tests/ tổng** | **✅ 35/35 pass** | chạy thật ngày cập nhật file này |
| Frontend (`index.html`/CSS/JS/`server.py`) | ✅ Chạy được | có sẵn, kết nối đúng Task 10 |
| **Bài nhóm — Golden dataset** | ⚠️ **Thiếu** | mới có **3/15** câu hỏi yêu cầu |
| **Bài nhóm — Evaluation pipeline** | ❌ **Chưa làm** | `eval_pipeline.py` chỉ là khung, in ra "Implement evaluation logic and run again!" |
| **Bài nhóm — results.md** | ❌ **Chưa làm** | template rỗng, chưa có số liệu |

**Tóm lại: 10/10 Task kỹ thuật cá nhân (50% điểm) đã xong và pass test. Phần Bài Nhóm — Evaluation (12/30 điểm quan trọng nhất của bài nhóm) CHƯA làm** — đây là việc còn thiếu lớn nhất, ai phụ trách Evaluation/QA cần hoàn thành gấp trước CP5.

---

## 5. Khoảng hở kỹ thuật đã biết (chưa phải bug chặn chạy, nhưng nên biết)

1. **LLM fallback chain nông**: `task10_generation.py` chỉ thử `OPENROUTER_API_KEY` → `OPENAI_API_KEY`, lỗi/rate-limit là rơi thẳng xuống câu trả lời trích dẫn thô (không gọi LLM). `GEMINI_API_KEY` đã có trong `.env` nhưng **không được code nào dùng tới** — dead config.
2. **Model mặc định không free**: `LLM_MODEL` default = `"openai/gpt-4o-mini"` qua OpenRouter — đây **không phải model `:free`**, cần credit thật. Muốn miễn phí phải set env `OPENROUTER_MODEL` sang model có hậu tố `:free`.
3. **`chroma_db/` không nằm trong git** (đúng chủ đích) — nghĩa là **máy nào cũng phải tự chạy Task 4** để có vector store, không tải sẵn từ repo được.
4. **Task 8 (PageIndex)** hoạt động qua fallback nội bộ nếu thiếu `PAGEINDEX_API_KEY` thật — cần biết khi demo để không bị hỏi vặn là "vectorless RAG thật hay giả lập".

---

## 6. Tech stack tóm tắt (để ghi slide)

| Layer | Công nghệ |
|---|---|
| Data ingestion | `fpdf2` (sinh PDF), JSON tĩnh (thay crawl4ai thật), `MarkItDown` (convert → Markdown) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` (size=800, overlap=100) |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` — local, 384 chiều, chạy CPU |
| Vector Store | ChromaDB (persistent, local, cosine similarity) |
| Sparse Search | BM25 (`rank-bm25`) |
| Reranking | Reciprocal Rank Fusion (RRF, k=60) tự implement + tùy chọn Jina cross-encoder API |
| Vectorless Fallback | PageIndex SDK/API |
| Generation | LLM qua OpenRouter (OpenAI-compatible SDK), prompt ép buộc citation `[Nguồn, Năm]` |
| Frontend | HTML/CSS/JS thuần (không framework) |
| Backend server | Python `http.server` thuần (không Flask/FastAPI) |
| Testing | `pytest` (35 test case) |
| Evaluation (bài nhóm) | RAGAS/DeepEval/TruLens — **chưa implement** |

---

## 7. Gợi ý dùng file này với ChatGPT

Khi dán file này cho ChatGPT để tư vấn frontend/UI hoặc kiến trúc hệ thống, có thể hỏi kèm theo, ví dụ:
- "Dựa vào luồng xử lý ở mục 2 và cấu trúc file ở mục 3, thiết kế lại `index.html` + `styles.css` cho chuyên nghiệp hơn, giữ nguyên các element id mà `script.js` đang bind vào."
- "File `assets/script.js` gọi `POST /api/chat` với body `{query, top_k, use_reranking, use_pageindex_fallback}` và nhận về `{answer, sources, retrieval_source}` — gợi ý cách hiển thị `retrieval_source` (hybrid vs pageindex) rõ ràng hơn cho người dùng."
- "Bài nhóm còn thiếu phần Evaluation (mục 4) — vạch lộ trình 30 phút để implement `eval_pipeline.py` bằng RAGAS."
