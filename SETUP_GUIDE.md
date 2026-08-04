# 🚀 SETUP GUIDE — Chạy Toàn Bộ Dự Án Từ Đầu

Hướng dẫn này dành cho người **chưa từng đụng vào project**, đi từ máy trống tới chatbot RAG chạy được ở `http://localhost:5500`. Làm theo đúng thứ tự các bước, đừng bỏ qua.

---

## 0. Yêu cầu trước khi bắt đầu

| Yêu cầu                         | Ghi chú                                                                                                                                            |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python**3.10+**             | Kiểm tra:`python --version`                                                                                                                      |
| ~3GB dung lượng trống          | Cho venv + model embedding (`sentence-transformers` tải model ~80MB, cache tại `~/.cache/huggingface`)                                        |
| Kết nối mạng                   | Cần cho: cài package, tải model embedding lần đầu, gọi LLM (OpenRouter)                                                                      |
| 1 API key OpenRouter (miễn phí) | Bắt buộc để chạy**Task 10 (sinh câu trả lời)** và chatbot. Đăng ký tại https://openrouter.ai/ → tạo key dạng `sk-or-v1-...` |

Không có API key vẫn chạy được Task 1-9 (thu thập dữ liệu, indexing, search, rerank) — chỉ Task 10 (LLM trả lời) và giao diện chat mới cần key.

---

## 1. Cài đặt môi trường (một lần duy nhất)

### Windows — PowerShell

```powershell
# 1. Vào thư mục project
cd d:\VINAI_Team_093\LAB\Day08-2A202601274-Nguyen-Thanh-Binh

# 2. Tạo virtual environment
python -m venv .venv

# 3. Kích hoạt venv (mỗi lần mở terminal mới đều phải chạy lại dòng này)
.\.venv\Scripts\Activate.ps1
# Nếu bị chặn bởi policy, chạy 1 lần: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 4. Upgrade pip rồi cài dependencies (mất 5-10 phút tùy mạng)
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux — Bash

```bash
cd /path/to/Day08-2A202601274-Nguyen-Thanh-Binh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Tạo file `.env`

```powershell
copy .env.example .env      # Windows
# hoặc
cp .env.example .env        # macOS/Linux
```

Mở `.env`, điền tối thiểu:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

Các key khác trong `.env.example` (`PAGEINDEX_API_KEY`, `JINA_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`) đều **tùy chọn** — thiếu thì code tự fallback (xem bảng troubleshooting bên dưới), không crash.

### 6. Kiểm tra môi trường

```powershell
# Windows: set UTF-8 trước để tránh UnicodeEncodeError khi in tiếng Việt ra console
$env:PYTHONIOENCODING="utf-8"
python check_environment.py
```

Kỳ vọng: `CP0 PASSED — Môi trường cơ bản đã sẵn sàng!` (15/15 kiểm tra đạt). Nếu thiếu package nào, chạy lại `pip install -r requirements.txt`.

---

## 2. Chạy pipeline dữ liệu — theo đúng thứ tự Task 1 → 4

Mỗi lệnh phải chạy **xong hẳn** trước khi sang lệnh kế tiếp (Task sau phụ thuộc output của Task trước).

```powershell
# Task 1 — Sinh 14 file PDF chính sách vào data/landing/legal/
python src/task1_collect_legal_docs.py

# Task 2 — Sinh 10 file JSON bài viết hướng dẫn vào data/landing/news/
python src/task2_crawl_news.py

# Task 3 — Convert toàn bộ PDF/JSON sang Markdown, lưu data/standardized/
python src/task3_convert_markdown.py

# Task 4 — Chunk + embed (model local, tải lần đầu ~80MB) + index vào ChromaDB (chroma_db/)
python src/task4_chunking_indexing.py
```

Kiểm tra nhanh sau khi chạy xong:

```powershell
dir data\landing\legal\      # kỳ vọng: 14 file .pdf + documents_index.json
dir data\landing\news\       # kỳ vọng: 10 file .json
dir data\standardized\legal  # kỳ vọng: 14 file .md
dir data\standardized\news   # kỳ vọng: 10 file .md
dir chroma_db\               # kỳ vọng: có file, không rỗng
```

> Task 5-9 (semantic search, lexical search BM25, reranking RRF, PageIndex fallback, retrieval pipeline) và Task 10 (generation) là **module Python** (`src/task5_*.py` → `src/task10_*.py`), không cần chạy riêng — chúng được gọi tự động khi bạn mở chatbot ở bước 4, hoặc khi pytest chạy.

---

## 3. Xác nhận toàn bộ pipeline đúng bằng pytest

```powershell
pytest tests/test_individual.py -v
```

Kỳ vọng: **35 passed**. Đây là bộ test chấm điểm chính thức Task 1-10 (50% điểm kỹ thuật). Nếu có test FAIL, đọc thông báo lỗi — thường chỉ ra chính xác thiếu file/dữ liệu gì.

---

## 4. Chạy chatbot (giao diện HTML/CSS/JS + backend Python)

```powershell
python app.py
# (tương đương: python server.py)
```

Mở trình duyệt tại **http://localhost:5500**. Nhấn `Ctrl+C` trong terminal để dừng server.

Giao diện gồm:

- `index.html` — cấu trúc trang
- `assets/styles.css` / `assets/script.js` — style & tương tác
- `server.py` — HTTP server local, expose API `POST /api/chat` gọi `src/task10_generation.py`

Nếu trả lời báo lỗi "Không thể chạy pipeline RAG" → kiểm tra `chroma_db/` đã index chưa (bước 2) và `OPENROUTER_API_KEY` trong `.env` đã điền đúng chưa.

---

## 5. (Bài nhóm — chưa xong) Đánh giá RAGAS

```powershell
python -m group_project.evaluation.eval_pipeline
```

⚠️ **Hiện tại `group_project/evaluation/eval_pipeline.py` mới là file khung (`TODO` + `NotImplementedError`)** — chạy lệnh trên chỉ in ra `⚠ Implement evaluation logic and run again!`, chưa thực sự đánh giá gì. Đây là phần việc của Role Evaluation & QA, cần được hoàn thiện trước CP5 (xem `group_project/README.md` và `LAB_GUIDE.md` mục Checkpoint 5):

1. Import `generate_with_citation` từ `src/task10_generation.py`.
2. Chọn 1 trong 3 framework có sẵn code mẫu trong file (`evaluate_with_ragas` / `evaluate_with_deepeval` / `evaluate_with_trulens`).
3. Implement `export_results()` để ghi bảng điểm ra `results.md`.

Sau khi implement xong: đọc `group_project/evaluation/golden_dataset.json` (bộ câu hỏi mẫu), chạy qua pipeline, xuất kết quả vào `group_project/evaluation/results.md`.

⚠️ Model OpenRouter free giới hạn **50 request/ngày cho cả tài khoản** (không phải theo key). RAGAS gọi LLM nhiều lần/câu hỏi — nếu chạy hết 15+ câu bị rate limit giữa chừng, giảm xuống 5 câu để test trước.

---

## 6. Reset dữ liệu khi cần chạy lại từ đầu

Nếu đổi corpus / đổi embedding model / dữ liệu bị lỗi giữa chừng:

```powershell
Remove-Item -Recurse -Force chroma_db
python src/task4_chunking_indexing.py
```

**Không** cần xóa `data/landing/` hay `data/standardized/` trừ khi muốn tạo lại từ đầu — Task 1-3 dùng `write_text`/`upsert` nên chạy lại an toàn (ghi đè, không nhân đôi).

---

## 🚨 Lỗi thường gặp

| # | Lỗi / Hiện tượng                                                                                     | Nguyên nhân                                                                                                             | Cách khắc phục                                                                                                                                                                |
| :-: | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | `FPDFException: Not enough horizontal space...` (Task 1)                                               | Bug đã biết trong`fpdf2` khi `multi_cell()` không reset `new_x` về lề trái                                   | Đã fix trong`src/task1_collect_legal_docs.py` (thêm `new_x="LMARGIN", new_y="NEXT"`). Nếu vẫn gặp, đảm bảo đang dùng đúng file đã fix (kéo code mới nhất). |
| 2 | `MissingDependencyException` (Task 3, convert PDF)                                                     | Thiếu extra`[pdf]` của `markitdown`                                                                                 | `pip install "markitdown[pdf]"`                                                                                                                                                |
| 3 | `Executable doesn't exist` (crawl4ai/playwright)                                                       | Chưa cài Chromium binary                                                                                                | `playwright install chromium`                                                                                                                                                  |
| 4 | `UnicodeEncodeError`/`UnicodeDecodeError` trên Windows console (vd. chạy `check_environment.py`) | Console mặc định cp1252/cp1258 thay vì UTF-8                                                                          | Chạy`$env:PYTHONIOENCODING="utf-8"` trước khi gọi `python ...`, hoặc dùng `python -X utf8 <script>`                                                                  |
| 5 | `pip`/`python` không nhận lệnh                                                                    | Chưa kích hoạt venv                                                                                                    | Chạy lại`.\.venv\Scripts\Activate.ps1` (thấy `(.venv)` ở đầu dòng lệnh là đã bật)                                                                                |
| 6 | Chatbot trả lời rỗng / "cannot verify" liên tục                                                     | `chroma_db/` chưa được index (chưa chạy Task 4), hoặc đang hỏi ngoài domain (đúng như thiết kế fallback) | Chạy lại`python src/task4_chunking_indexing.py`, kiểm tra `chroma_db/` không rỗng                                                                                       |
| 7 | `401 Unauthorized` khi gọi LLM                                                                        | `OPENROUTER_API_KEY` sai/thiếu trong `.env`                                                                          | Kiểm tra key bắt đầu bằng`sk-or-`, không có dấu cách/thừa dòng                                                                                                      |
| 8 | `429 Too Many Requests`                                                                                | Vượt quota free của OpenRouter (50 req/ngày/tài khoản)                                                              | Đợi reset ngày hôm sau, hoặc nạp $10 credit để lên 1000 req/ngày                                                                                                       |
| 9 | Task 8 (PageIndex) không trả kết quả thật                                                           | Thiếu`PAGEINDEX_API_KEY` (tùy chọn)                                                                                  | Bỏ qua nếu chỉ cần demo — code tự dùng fallback nội bộ. Muốn dùng PageIndex thật: đăng ký tại pageindex.ai rồi điền key vào`.env`                          |
| 10 | Test Task 4/5 fail vì model tải quá lâu / timeout                                                    | Lần đầu tải`sentence-transformers/all-MiniLM-L6-v2` (~80MB) qua mạng chậm                                         | Chạy`python src/task4_chunking_indexing.py` riêng 1 lần trước để model cache vào `~/.cache/huggingface`, các lần sau tức thì                                     |
| 11 | Đổi corpus nhưng kết quả search vẫn cũ                                                            | ChromaDB cũ chưa xóa                                                                                                   | Xem mục 6 — xóa`chroma_db/` rồi index lại                                                                                                                                 |

---

## 📋 Checklist "chạy được dự án"

- [ ] `python check_environment.py` → CP0 PASSED
- [ ] `data/landing/legal/` ≥ 3 file PDF, `data/landing/news/` ≥ 5 file JSON
- [ ] `data/standardized/` có đủ file `.md` tương ứng
- [ ] `chroma_db/` không rỗng
- [ ] `pytest tests/test_individual.py -v` → 35 passed
- [ ] `python app.py` → mở `http://localhost:5500`, hỏi thử 1 câu và nhận được câu trả lời kèm trích dẫn nguồn

```powershell
dir data\landing\legal\      # kỳ vọng: 14 file .pdf + documents_index.json
dir data\landing\news\       # kỳ vọng: 10 file .json
dir data\standardized\legal  # kỳ vọng: 14 file .md
dir data\standardized\news   # kỳ vọng: 10 file .md
dir chroma_db\               # kỳ vọng: có file, không rỗng
```
