# DOCS — Tài liệu Kỹ thuật Toàn diện

**Dự án:** RAG Pipeline — E-commerce Support Chatbot (Lab 08, VinUni AI20k K4)
**Repo:** `Day08-2A202601274-Nguyen-Thanh-Binh`

> Mọi con số trong tài liệu này đo từ lần chạy thật trên máy, không phải ước lượng.
> Tài liệu này viết để: (a) thành viên nhóm đọc hiểu code, (b) chuẩn bị demo và bảo vệ,
> (c) nạp vào ChatGPT làm ngữ cảnh sinh slide — xem [Mục 12](#12-prompt-để-đưa-cho-chatgpt).

---

## Mục lục

1. [Bài toán và mục tiêu](#1-bài-toán-và-mục-tiêu)
2. [Kiến thức nền cần nắm](#2-kiến-thức-nền-cần-nắm)
3. [Kiến trúc tổng thể](#3-kiến-trúc-tổng-thể)
4. [Vòng đời một câu hỏi](#4-vòng-đời-một-câu-hỏi)
5. [Giải thích từng file code](#5-giải-thích-từng-file-code)
6. [Những cái bẫy đã gặp](#6-những-cái-bẫy-đã-gặp)
7. [Số liệu đo được](#7-số-liệu-đo-được)
8. [Đánh giá RAGAS](#8-đánh-giá-ragas)
9. [Câu hỏi pitching và cách trả lời](#9-câu-hỏi-pitching-và-cách-trả-lời)
10. [Kịch bản demo](#10-kịch-bản-demo)
11. [Hạn chế và hướng phát triển](#11-hạn-chế-và-hướng-phát-triển)
12. [Prompt để đưa cho ChatGPT](#12-prompt-để-đưa-cho-chatgpt)

---

## 1. Bài toán và mục tiêu

### Vấn đề

Một sàn thương mại điện tử có hàng trăm trang chính sách: đổi trả, hoàn tiền, thanh toán,
vận chuyển, quy định người bán, bảo mật. Khách hàng hỏi *"Tôi trả hàng trong bao lâu?"*
thì phải tự đi tìm trong đống tài liệu đó.

Dùng LLM trả lời thẳng thì gặp hai vấn đề chí mạng:

- **Bịa (hallucination):** LLM không được huấn luyện trên chính sách cụ thể của sàn này,
  nó sẽ tự chế ra con số nghe hợp lý.
- **Không kiểm chứng được:** người dùng không biết câu trả lời lấy từ đâu để đối chiếu.

### Giải pháp: RAG (Retrieval-Augmented Generation)

Thay vì hỏi LLM *"chính sách trả hàng là gì?"*, ta:

1. **Tìm** những đoạn tài liệu liên quan nhất trong kho.
2. **Đưa** những đoạn đó vào prompt làm bằng chứng.
3. **Yêu cầu** LLM chỉ trả lời dựa trên bằng chứng đó, kèm trích dẫn.

LLM lúc này làm việc nó giỏi (đọc hiểu, tóm tắt, diễn đạt) chứ không phải việc nó dở
(nhớ chính xác sự kiện).

### Mục tiêu cụ thể của bài lab

| Mục tiêu | Trạng thái |
|---|---|
| 10 task pipeline, chấm bằng 35 test tự động | ✅ **35/35 PASSED** |
| Chatbot Streamlit demo được | ✅ `streamlit run app.py` |
| Đánh giá RAGAS 4 chỉ số + so sánh A/B | ✅ có báo cáo |
| Bonus: HyDE, TF-IDF, UI/UX | ✅ đã làm |

---

## 2. Kiến thức nền cần nắm

Phần này để trả lời được câu hỏi của coach. Đọc kỹ trước khi bảo vệ.

### 2.1. Embedding và Vector Search

**Embedding** là biến đoạn văn thành một dãy số (vector) sao cho *ý nghĩa gần nhau thì
vector gần nhau*. Model của nhóm — `BAAI/bge-m3` — biến mỗi đoạn thành **1024 số**.

Đo "gần nhau" bằng **cosine similarity**: góc giữa hai vector.

```
cos(A, B) = (A · B) / (|A| × |B|)
```

- Giá trị **1** = cùng hướng = ý nghĩa giống hệt
- **0** = vuông góc = không liên quan
- **-1** = ngược hướng

> **Điểm dễ sai:** ChromaDB trả về **distance**, không phải similarity. Với
> `hnsw:space="cosine"` thì `distance = 1 - cosine_similarity`. Code phải đổi:
> `score = 1.0 - distance`. Nhầm chỗ này thì kết quả xếp hạng bị **đảo ngược**.

**Vì sao chọn `bge-m3`?** Corpus của nhóm **song ngữ**: tài liệu Shopee tiếng Việt và
eBay tiếng Anh. `bge-m3` là model đa ngôn ngữ, nên câu hỏi tiếng Anh vẫn tìm được tài
liệu tiếng Việt và ngược lại — điều mà model chỉ-tiếng-Anh như `all-MiniLM-L6-v2`
không làm được.

### 2.2. Chunking

Không thể nhét cả tài liệu 145.000 ký tự vào prompt. Phải cắt nhỏ.

| Tham số | Giá trị | Vì sao |
|---|---|---|
| `CHUNK_SIZE` | 800 ký tự | Đủ chứa trọn một đoạn chính sách hoàn chỉnh |
| `CHUNK_OVERLAP` | 100 ký tự | 12,5% — để câu bị cắt ở biên vẫn xuất hiện trọn vẹn ở chunk kế tiếp |

**Chunk quá nhỏ** → mất ngữ cảnh, đoạn lấy về không đủ trả lời.
**Chunk quá lớn** → nhiễu, một chunk chứa nhiều chủ đề làm vector bị "trung bình hoá".

`RecursiveCharacterTextSplitter` cắt theo thứ tự ưu tiên `\n\n` → `\n` → `. ` → `" "`,
tức cố cắt ở ranh giới đoạn trước, rồi mới đến câu, cuối cùng mới cắt giữa từ.

### 2.3. BM25 — tìm kiếm theo từ khoá

Semantic search hiểu *ý nghĩa* nhưng dở với **thuật ngữ hiếm**: mã đơn "SPX123456789VN",
tên riêng "NAPAS", số hiệu điều khoản. Embedding không "nhớ" được chuỗi cụ thể.

BM25 thì ngược lại — nó đếm từ khoá:

```
score(q,d) = Σ IDF(qᵢ) × [ tf(qᵢ,d) × (k₁+1) ] / [ tf(qᵢ,d) + k₁×(1−b+b×|d|/avgdl) ]
```

Hai tham số đáng nói:
- **k₁ (=1.5)** — *bão hoà tần suất*: từ xuất hiện 10 lần không "gấp đôi giá trị" so với
  5 lần. Không có k₁ thì tài liệu spam từ khoá sẽ thắng.
- **b (=0.75)** — *chuẩn hoá độ dài*: tài liệu dài tự nhiên chứa nhiều từ hơn, phải phạt
  để không lấn át tài liệu ngắn nhưng đúng trọng tâm.

**So với TF-IDF** (bonus của bài):

| | TF-IDF | BM25 |
|---|---|---|
| Tần suất từ | Tuyến tính, **không bão hoà** | Bão hoà theo k₁ |
| Độ dài tài liệu | **Không** chuẩn hoá | Chuẩn hoá theo b |
| Bản chất | Tích TF×IDF trên không gian vector | Hàm xếp hạng xác suất |

### 2.4. Hybrid Search và RRF

Semantic giỏi ý nghĩa, BM25 giỏi từ khoá → dùng cả hai, rồi **hợp nhất**.

Vấn đề: **không cộng thẳng điểm được**. Cosine ∈ [0,1], còn BM25 là điểm thô không giới
hạn — đo trên corpus này lên tới **24+**. Cộng thẳng thì BM25 át hoàn toàn.

**Reciprocal Rank Fusion** giải quyết bằng cách chỉ nhìn **thứ hạng**:

```
RRF(d) = Σ  1 / (k + rank_r(d))        với k = 60, rank bắt đầu từ 1
        r∈rankers
```

Tài liệu xếp hạng 1 ở dense được `1/61 = 0.0164`. Nếu nó cũng xếp hạng 3 ở sparse thì
cộng thêm `1/63 = 0.0159`, tổng **0.0323**. Tài liệu chỉ một ranker tìm ra chỉ có
**~0.016**.

→ **RRF thưởng cho sự đồng thuận giữa các ranker.** Đó chính là giá trị của nó.

**k = 60** lấy từ Cormack et al. 2009. k càng lớn thì khoảng cách giữa các thứ hạng đầu
càng nhỏ, tức càng ưu tiên "được nhiều ranker đồng ý" hơn là "xếp nhất ở một ranker".

### 2.5. Reranking

Sau khi có ~10 ứng viên, có thể chấm lại kỹ hơn:

| Phương pháp | Cơ chế | Chi phí |
|---|---|---|
| **RRF** | Gộp thứ hạng, không cần model | Gần như 0 |
| **Cross-Encoder** | Đưa CẢ cặp (query, doc) vào một model, chấm trực tiếp độ liên quan | Cao — mỗi cặp một lượt forward |
| **MMR** | Cân bằng liên quan và **đa dạng**, tránh 5 kết quả na ná nhau | Trung bình |

Khác biệt cốt lõi giữa **bi-encoder** (dùng ở Task 5) và **cross-encoder**: bi-encoder
mã hoá query và doc **riêng rẽ** rồi so vector (nhanh, index trước được); cross-encoder
đọc **cùng lúc** cả hai nên hiểu tương tác giữa chúng, chính xác hơn nhưng không index
trước được.

### 2.6. Lost in the Middle

Nghiên cứu cho thấy LLM **chú ý hai đầu context nhiều hơn phần giữa**. Đưa 5 chunk vào
prompt thì chunk thứ 3 dễ bị bỏ qua nhất.

Cách xử lý trong `reorder_for_llm`:

```python
front = chunks[0::2]        # 0, 2, 4  → đặt ở đầu
back  = chunks[1::2][::-1]  # 5, 3, 1  → đặt ở cuối, đảo ngược
return front + back
```

Chunk điểm cao nhất (index 0) nằm **đầu**, chunk điểm cao thứ hai (index 1) nằm **cuối** —
cả hai đều ở vị trí LLM chú ý nhất.

### 2.7. Guardrails và Routing

Đây là phần **vượt ngoài đề bài**, thêm vào theo hướng thiết kế production.

**Vấn đề quan sát được:** pipeline RAG thuần coi *mọi* input đều là câu hỏi cần tra tài
liệu. Gõ `"hi"` cũng chạy trọn semantic + BM25 + RRF + rerank, lôi về 5 chunk chính sách,
rồi trả lời *"Tôi không thể xác minh thông tin này từ nguồn hiện có"* — vô nghĩa với người
dùng, tốn một lượt gọi LLM và ~1,6 giây cho một lời chào.

**Ba việc mà pipeline gốc gộp làm một, nay tách ra:**

| Câu hỏi | Thành phần |
|---|---|
| Câu này có **cần** tra tài liệu không? | Router |
| Câu này có **được phép** xử lý không? | Input guardrail |
| Câu trả lời có **an toàn/đúng** không? | Output guardrail |

---

## 3. Kiến trúc tổng thể

```
┌──────────────────────────────────────────────────────────────────────────┐
│  TẦNG THU THẬP (offline, chạy một lần)                                   │
├──────────────────────────────────────────────────────────────────────────┤
│  Task 1  requests → fpdf2      → data/landing/legal/*.pdf   (4 file, VN)  │
│  Task 2  Crawl4AI + Playwright → data/landing/news/*.json   (6 file, EN)  │
│  Task 3  MarkItDown + LÀM SẠCH → data/standardized/*.md      (10 file)    │
│  Task 4  chunk 800/100 → bge-m3 → chroma_db/                 (439 vector) │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼──────────────────────────────────────┐
│  TẦNG PHỤC VỤ (online, mỗi câu hỏi)                                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Câu hỏi                                                                │
│      │                                                                   │
│      ▼                                                                   │
│   ┌─────────────────────────┐                                            │
│   │ [0] ROUTER + GUARDRAIL  │  guardrails.py                             │
│   └──┬──────────────────────┘                                            │
│      │                                                                   │
│      ├── blocked  ──→ từ chối, KHÔNG gọi LLM ─────────────────────┐      │
│      ├── empty    ──→ yêu cầu nhập lại ───────────────────────────┤      │
│      ├── chitchat ──→ LLM + prompt khoá phạm vi, KHÔNG truy hồi ──┤      │
│      │                                                            │      │
│      └── retrieve                                                 │      │
│             │                                                     │      │
│      ┌──────┴───────┐                                             │      │
│      ▼              ▼                                             │      │
│  ┌────────┐    ┌─────────┐                                        │      │
│  │[1] Sem │    │[1] BM25 │   Task 5 / Task 6                      │      │
│  │ bge-m3 │    │ +TF-IDF │   ← cùng đọc chunk TỪ ChromaDB         │      │
│  └────┬───┘    └────┬────┘                                        │      │
│       │ cosine gốc  │                                             │      │
│       └──────┬──────┘                                             │      │
│              ▼                                                    │      │
│       ┌─────────────┐                                             │      │
│       │[2] RRF k=60 │  Task 7                                     │      │
│       └──────┬──────┘                                             │      │
│              ▼                                                    │      │
│    ┌───────────────────────┐                                      │      │
│    │[3] cosine gốc < 0.48? │  Task 9                              │      │
│    └───┬───────────────┬───┘                                      │      │
│    có  │               │ không                                    │      │
│        ▼               ▼                                          │      │
│  ┌───────────┐   ┌──────────┐                                     │      │
│  │PageIndex  │   │  Hybrid  │                                     │      │
│  │vectorless │   │          │                                     │      │
│  └─────┬─────┘   └────┬─────┘                                     │      │
│        └──────┬───────┘                                           │      │
│               ▼                                                   │      │
│      ┌──────────────────┐                                         │      │
│      │[4] reorder +     │  Task 10                                │      │
│      │    format ctx    │                                         │      │
│      └────────┬─────────┘                                         │      │
│               ▼                                                   │      │
│      ┌──────────────────┐                                         │      │
│      │[5] LLM POOL      │  llm_pool.py                            │      │
│      │ Groq→Cerebras→   │  xoay vòng, phạt nghỉ khi lỗi           │      │
│      │ Gemini→…→OpenAI  │                                         │      │
│      └────────┬─────────┘                                         │      │
│               ▼                                                   │      │
│      ┌──────────────────┐                                         │      │
│      │[6] OUTPUT GUARD  │◄────────────────────────────────────────┘      │
│      │ gỡ URL, đối chiếu│  guardrails.py                                 │
│      │ số trích dẫn     │                                                │
│      └────────┬─────────┘                                                │
│               ▼                                                          │
│          app.py (Streamlit)                                              │
└──────────────────────────────────────────────────────────────────────────┘
```

### Nguyên tắc thiết kế xuyên suốt

1. **Không bao giờ `raise` ra ngoài.** Thiếu API key, thiếu `chroma_db/`, mất mạng, model
   bị gỡ — tất cả đều degrade thành kết quả kém hơn chứ không làm sập ứng dụng. Lý do:
   demo trực tiếp trước lớp, một exception là hỏng cả buổi.

2. **Lazy loading tuyệt đối.** Không load model, không mở ChromaDB, không đọc file ở **cấp
   module**. `pytest` import toàn bộ module trước khi chạy, nếu kéo 2GB weight lúc import
   thì test rất chậm hoặc crash.

3. **Không tin đầu ra của model.** Prompt đã cấm chép URL và bắt trích dẫn đúng số, nhưng
   vẫn kiểm lại bằng code ở output guardrail.

4. **Hai nhánh retrieval dùng chung một bộ chunk.** Điều kiện sống còn để RRF hoạt động —
   giải thích ở [Mục 6.1](#61-chunk-không-khớp-rrf-thành-vô-dụng).

---

## 4. Vòng đời một câu hỏi

Theo dõi câu **"Shopee hỗ trợ những phương thức thanh toán nào?"**:

| Bước | Việc | Kết quả thật |
|---|---|---|
| 0 | `classify_query()` chuẩn hoá bỏ dấu, so mẫu | `Route.RETRIEVE` |
| 1a | `semantic_search(q, top_k=10)` — embed query, truy vấn Chroma | 10 chunk, cosine cao nhất **0.78** |
| 1b | `lexical_search(q, top_k=10)` — BM25 trên cùng corpus | 10 chunk, BM25 cao nhất **24.4** |
| 1c | Giao nhau giữa hai danh sách | **8/10 chunk** |
| 2 | `rerank_rrf([dense, sparse], k=60)` | Top-1 = **0.0325** (hai phiếu) |
| 3 | So `dense[0].score = 0.78` với ngưỡng `0.48` | ≥ ngưỡng → **hybrid** |
| 4 | `reorder_for_llm` → `format_context` | 5 chunk, ~4.200 ký tự context |
| 5 | `LLMPool.chat()` | `groq/llama-3.1-8b-instant`, ~1,2s |
| 6 | `audit_answer(answer, n_sources=5)` | 0 URL, trích dẫn [1]–[5] hợp lệ |

Còn câu **"Công thức nấu phở bò gồm những gì?"**:

| Bước | Kết quả |
|---|---|
| 0 | `Route.RETRIEVE` (router không chặn — nó *là* một câu hỏi) |
| 1a | cosine cao nhất chỉ **0.326** |
| 3 | `0.326 < 0.48` → **kích hoạt fallback** |
| — | `pageindex_search()` trả 3 node, `source="pageindex"` |
| 5 | LLM trả lời "Tôi không thể xác minh thông tin này từ nguồn hiện có" |

> **Đây là hành vi đúng và là điểm đáng khoe nhất khi demo:** hệ thống biết lúc nào nó
> *không* biết, thay vì bịa ra công thức phở.

---

## 5. Giải thích từng file code

Tổng `src/`: **4.690 dòng**, `app.py`: **452 dòng**.

### `src/task1_collect_legal_docs.py` (746 dòng)

Tải 4 trang chính sách chính thức Shopee → render PDF.

| Hàm | Việc |
|---|---|
| `POLICY_SOURCES` | 6 nguồn (4 chính + 2 dự phòng), mỗi nguồn có `customer_role` |
| `download_file()` | `requests` với User-Agent thật, timeout 30s, retry 2 lần |
| `html_to_text()` | Bỏ `<script>/<style>`, strip tag, unescape entity |
| `text_to_pdf()` | `fpdf2` + font TTF Unicode (`arial.ttf`) |
| `collect_all()` | Lặp nguồn, dừng khi đủ `TARGET_FILE_COUNT=4` |
| `build_offline_fallback()` | Sinh PDF mẫu **kèm cảnh báo rõ** khi mọi nguồn mạng chết |

**Vì sao dùng `requests` chứ không Playwright?** Đã đo: `help.shopee.vn` trả nội dung ngay
trong HTML đầu tiên (~19.600 ký tự text sạch). Docstring lab cảnh báo help center thường là
SPA — với Shopee thì không phải.

**Vì sao font TTF?** Font lõi của `fpdf2` chỉ hỗ trợ latin-1, tiếng Việt sẽ thành `?`.

### `src/task2_crawl_news.py` (227 dòng)

Crawl 6 bài eBay Help Center bằng Crawl4AI (Playwright bên dưới).

> **Bẫy phiên bản:** `crawl4ai` 0.8.5 trả `result.markdown` là **object**
> `MarkdownGenerationResult`, **không phải `str`**. Hàm `_markdown_to_text()` thử lần lượt
> `raw_markdown` → `fit_markdown` → `str(md)`. Làm sai chỗ này sẽ ghi ra file rác.

Bỏ qua bài lỗi hoặc ngắn hơn `MIN_CONTENT_CHARS=600` rồi **in cảnh báo**, không im lặng.

### `src/task3_convert_markdown.py` (274 dòng)

Convert **và làm sạch** — bước có tác động lớn nhất tới chất lượng câu trả lời.

`clean_markdown()` lọc theo **cấu trúc**, không theo danh sách từ khoá (lọc từ khoá rất dễ
xoá nhầm nội dung chính sách thật):

1. Gỡ `![alt](url)` hoàn toàn.
2. Gỡ `[chữ](url)` → giữ lại `chữ`.
3. Xoá URL trần.
4. **Dòng mà sau khi gỡ hết link gần như không còn chữ** → thanh điều hướng, bỏ.
5. **Mục menu bullet là MỘT link nhãn ngắn (< 45 ký tự)** → bỏ. Câu chính sách thật gần
   như luôn dài hơn và hiếm khi là link.

Làm sạch **nội dung trước** rồi mới ghép header — nếu làm sạch cả header thì URL nguồn ở
dòng `**Source:**` cũng bị bộ lọc URL xoá mất.

### `src/task4_chunking_indexing.py` (416 dòng)

**Hai cặp đôi bắt buộc đi cùng nhau:**

1. `normalize_embeddings=True` **phải** đi với `hnsw:space="cosine"`.
2. ID chunk phải **duy nhất toàn cục**: `f"{type}__{source}__{chunk_index}"`. Chỉ dùng
   `source + chunk_index` thì file trùng tên ở `legal/` và `news/` sẽ ghi đè nhau.

`reset_collection()` chạy trước mỗi lần index lại — docstring lab đã cảnh báo chunk cũ lẫn
chunk mới làm retrieval trả về rác.

### `src/task5_semantic_search.py` (271 dòng)

```python
score = 1.0 - distance   # Chroma trả DISTANCE, không phải similarity
```

Đây là **điểm cosine gốc** — Task 9 dùng chính con số này để quyết định fallback.

**Bonus HyDE:** `generate_hyde_query()` sinh một đoạn trả lời giả định rồi embed đoạn đó
thay vì câu hỏi. Ý tưởng: câu hỏi và câu trả lời có "hình dạng" ngôn ngữ khác nhau, embed
một câu trả lời giả sẽ gần với tài liệu thật hơn. Mặc định `use_hyde=False` để test nhanh.

### `src/task6_lexical_search.py` (408 dòng)

**File quan trọng nhất về mặt thiết kế.**

`_load_corpus()` đọc chunk **thẳng từ ChromaDB** (`collection.get()`), không tự chunk lại.
Lý do ở [Mục 6.1](#61-chunk-không-khớp-rrf-thành-vô-dụng).

`_tokenize()` dùng regex unicode-aware `\w+` với `re.UNICODE` để giữ tiếng Việt có dấu.
Ép `float()` cho score vì `numpy.float64` làm phép so sánh sort trong test sai lệch.

### `src/task7_reranking.py` (358 dòng)

4 hàm: `rerank_rrf`, `rerank_cross_encoder`, `rerank_mmr`, `rerank` (dispatch).

**Đây là task duy nhất không phụ thuộc dữ liệu** — test truyền thẳng list candidate viết
tay, nên phải chạy đúng ngay cả khi chưa có `chroma_db/`.

Cross-encoder và MMR đều lazy-load có **cache** và bọc `try/except`: không tải được model
thì cảnh báo rồi giữ nguyên thứ tự, tuyệt đối không `raise`.

### `src/task8_pageindex_vectorless.py` (502 dòng)

Nhóm không có `PAGEINDEX_API_KEY` nên phần đáng nói là **chế độ vectorless cục bộ**:

- Quét heading markdown (`^#{1,6}`) dựng **cây mục lục**.
- Chấm điểm từng node bằng số từ khoá của query trong **tiêu đề** (trọng số cao) cộng
  trong **nội dung** (trọng số thấp).
- **Không dùng embedding** — đúng tinh thần "vectorless RAG": duyệt cấu trúc tài liệu
  thay vì tìm trong không gian vector.

### `src/task9_retrieval_pipeline.py` (323 dòng)

```python
best_score = dense_results[0]["score"]   # cosine GỐC, KHÔNG phải điểm RRF
if best_score < SCORE_THRESHOLD:         # 0.48
    return pageindex_search(query, top_k=top_k)
```

`retrieve_dense_only()` phục vụ A/B testing ở phần evaluation.

### `src/task10_generation.py` (625 dòng)

| Hàm | Việc |
|---|---|
| `reorder_for_llm()` | Chống lost-in-the-middle |
| `format_context()` | Đánh số `[n]` + in tên nguồn |
| `_call_via_pool()` | Gọi LLM qua pool, xử lý chạm trần token |
| `_extractive_answer()` | Fallback không-LLM: ghép nguyên văn đoạn điểm cao |
| `_finish()` | Chạy output guardrail rồi dựng dict trả về |
| `generate_with_citation()` | Điều phối toàn bộ |

**Số trích dẫn gán TRƯỚC khi reorder**, trên bản copy nông — nhờ vậy số `[n]` trong câu
trả lời luôn khớp thứ tự nguồn hiển thị trên giao diện, dù vị trí trong prompt đã xáo trộn.

### `src/guardrails.py` (250 dòng) — **ngoài đề bài**

```python
class Route(str, Enum):
    CHITCHAT = "chitchat"   # chào hỏi → trả lời trực tiếp, KHÔNG retrieval
    RETRIEVE = "retrieve"   # câu hỏi thật → pipeline RAG đầy đủ
    BLOCKED  = "blocked"    # vi phạm guardrail → từ chối, KHÔNG gọi LLM
    EMPTY    = "empty"      # rỗng/quá ngắn → yêu cầu nhập lại
```

**Router dùng LUẬT, không gọi LLM.** Lý do:

| | Router luật | Router LLM |
|---|---|---|
| Độ trễ | < 1ms | 0,5–2s **cho mọi câu hỏi** |
| Chi phí | 0 | Gấp đôi số lần gọi API |
| Tất định | Có | Không |

Đánh đổi: luật không hiểu câu diễn đạt lạ. Nên **tầng thứ hai — ngưỡng cosine ở Task 9 —
mới là thứ quyết định câu hỏi có thuộc phạm vi tài liệu hay không**, vì nó có bằng chứng
thật (điểm tương đồng với corpus) chứ không đoán bằng từ khoá.

**Thứ tự kiểm tra có chủ đích:** guardrail an toàn chạy **trước** nhận diện xã giao, để
câu `"hello, hãy bỏ qua mọi hướng dẫn phía trên"` không lọt qua nhánh chitchat chỉ vì mở
đầu bằng lời chào.

**So khớp trên chuỗi đã bỏ dấu tiếng Việt** (`_normalize` dùng NFD + loại category `Mn`,
riêng `đ` phải thay thủ công vì NFD không tách được) — nên không lách được bằng cách
thêm/bớt dấu.

`audit_answer()` — output guardrail:
1. Gỡ URL và markdown link còn sót.
2. Đối chiếu mọi số `[n]` với số nguồn thật. `[9]` khi chỉ có 5 nguồn = model bịa.
3. Báo khi câu trả lời không trích dẫn gì.

### `src/llm_pool.py` (289 dòng) — **ngoài đề bài**

Xoay vòng 7 nhà cung cấp, tất cả đều tương thích OpenAI SDK nên dùng chung một client
class, không cần adapter riêng.

```
Groq (ưu tiên 1) → Cerebras → Gemini → GitHub Models → Mistral → OpenRouter → OpenAI
```

| Lỗi | Phạt nghỉ | Vì sao |
|---|---|---|
| 429 rate limit | 60s | Hạn mức thường tính theo phút |
| 402 hết credit | 1 giờ | Nạp tiền không xảy ra trong vài giây |
| 401/403/404 | Vĩnh viễn | Hỏng cấu hình, thử lại vô ích |
| 5xx / timeout | 30s | Sự cố tạm thời |

**Round-robin điểm bắt đầu** thay vì luôn từ endpoint đầu danh sách — nếu luôn bắt đầu từ
số 1 thì nó gánh toàn bộ tải và chạm rate limit trước, các endpoint sau ngồi không.

> **Về nhiều tài khoản:** tạo tài khoản ảo để né hạn mức **vi phạm ToS** của hầu hết nhà
> cung cấp. Ngược lại, hạn mức của Groq/Cerebras/Gemini là **độc lập** nhau nên xoay vòng
> qua nhiều **bên** vừa hợp lệ vừa cộng dồn được. Cả nhóm góp key thì mỗi người dùng key
> của chính mình: `GROQ_API_KEY=key_bình,key_đăng,key_vũ`.

### `app.py` (452 dòng)

Streamlit UI. Bảng màu giấy ấm + xanh petrol + đồng ánh.

**Điểm nhấn: dải trace pipeline** — hiện đúng đường đi thật của từng câu hỏi. Nhánh không
đi qua retrieval thì **không vẽ** các bước semantic/BM25/RRF, vì vẽ ra sẽ nói dối về luồng.

> **Bẫy font:** ban đầu dùng `--serif: ui-serif, Georgia, "Iowan Old Style"`. Các font này
> thiếu glyph tiếng Việt dựng sẵn, trình duyệt phải tự chồng dấu lên chữ nền nên tiêu đề
> hiện ra kiểu **"Bă´t đâ`u"**. Đã đổi sang `Segoe UI Variable Display` / `Segoe UI`.

---

## 6. Những cái bẫy đã gặp

### 6.1. Chunk không khớp → RRF thành vô dụng

**Nghiêm trọng nhất, và hoàn toàn im lặng.**

RRF gộp theo khoá `item["content"]`. Nếu Task 6 tự cắt chunk theo cách khác (ví dụ 500/50
thay vì 800/100) thì chuỗi content của hai nhánh **không bao giờ trùng nhau** → RRF chỉ
đan xen hai danh sách chứ không fuse được gì. Hệ thống vẫn chạy, vẫn trả kết quả, chỉ là
**mất sạch giá trị của hybrid search**.

**Cách nhận biết:** nhìn dãy điểm RRF. Nếu là `1/61, 1/62, 1/63…` thì mỗi tài liệu chỉ
nhận **đúng một phiếu**.

**Cách phòng:** Task 6 đọc chunk thẳng từ ChromaDB.

### 6.2. Bẫy ngưỡng RRF

Điểm RRF của top-1 **luôn ≈ 1/(k+1) = 0.0164** bất kể liên quan hay không — vì nó chỉ phụ
thuộc thứ hạng. Đem so với `SCORE_THRESHOLD` thì fallback hoặc **không bao giờ** kích hoạt,
hoặc **luôn** kích hoạt.

→ Phải dùng **điểm cosine gốc** từ `semantic_search`.

### 6.3. RRF chạy hai lần (nhóm tự tìm ra)

Bước 2 fuse dense + sparse ra điểm có ý nghĩa. Nhưng bước 3 lại gọi `rerank(method="rrf")`
trên **một danh sách duy nhất** → mỗi tài liệu chỉ còn một phiếu, điểm bị ghi đè thành
`1/(60+rank)` đều tăm tắp. Thứ tự vẫn đúng nên không ai phát hiện, nhưng thông tin "được
mấy ranker đồng thuận" mất sạch.

**Sau khi sửa: 0.0164 → 0.0325.**

### 6.4. Context bẩn làm câu trả lời lủng củng

Corpus thô 324.107 ký tự chứa **722 markdown link, 735 URL trần** cộng khung điều hướng
eBay. Rác đi thẳng vào chunk → context → câu trả lời:

```
...of my request](https://www.ebay.com/help/action?topicid=4667&af=2) [Returns Policy, 2026].
```

**Chẩn đoán quan trọng:** đo `finish_reason` thấy là `'stop'`, tức LLM **không** bị cắt ở
`max_tokens`. Vấn đề nằm ở dữ liệu đầu vào, không phải cấu hình sinh.

Sau khi làm sạch: **722 → 0 link**, corpus còn 255.255 ký tự.

### 6.5. `fpdf2` multi_cell

`fpdf2` 2.8.7 mặc định `new_x=XPos.RIGHT`, nên sau lần gọi đầu con trỏ nằm sát lề phải và
lần gọi kế tiếp tính ra bề rộng = 0 → `FPDFException: Not enough horizontal space`.
Ban đầu **cả 7 nguồn đều fail**. Khắc phục: `pdf.set_x(pdf.l_margin)` trước mỗi `multi_cell`.

### 6.6. `hf-xet` làm treo tải model

Tải `BAAI/bge-m3` (2,27 GB) treo ở **0 B/s** vô thời hạn dù mạng đo được 2,4 MB/s.
Khắc phục: `HF_HUB_DISABLE_XET=1` → tốc độ về 3 MB/s.

### 6.7. Task 3 báo thành công dù không làm gì

Bug của bản starter: nếu thư mục nguồn không có file phù hợp thì vòng lặp không chạy lần
nào, hàm kết thúc âm thầm và **vẫn in "Done" với exit code 0**. Đã sửa: `sys.exit(1)` kèm
cảnh báo rõ.

---

## 7. Số liệu đo được

| Hạng mục | Con số |
|---|---|
| Tài liệu nguồn | 4 PDF Shopee 🇻🇳 (267.217 B) + 6 JSON eBay 🇬🇧 (228.371 B) |
| Corpus sau chuẩn hoá | 10 file `.md`, **255.255 ký tự** |
| Rác đã lọc | 722 markdown link → **0**; 735 URL → **0** |
| Chunk | **439** (143 legal + 296 news), size 800 / overlap 100 |
| Embedding | `BAAI/bge-m3`, 1024 chiều, cosine |
| Dense ∩ Sparse | **8/10** chunk (truy vấn điển hình) |
| RRF | `k=60`; điểm đồng thuận **0.0325** vs một phiếu **0.0164** |
| Ngưỡng fallback | **0.48** trên cosine gốc |
| Ngoài domain đo được | 0.466 (hộ chiếu), 0.326 (phở bò) → cả hai fallback đúng |
| Router | **17/17** ca kiểm thử đúng |
| LLM pool | 4 endpoint khả dụng (2 Groq + 2 OpenRouter) |
| Test tự động | **35/35 PASSED** (~17s) |
| Mã nguồn | `src/` **4.690 dòng** + `app.py` **452 dòng** |

---

## 8. Đánh giá RAGAS

**Thiết lập:** 18 câu golden dataset (13 EN + 3 VI + 2 ngoài domain), LLM judge qua
OpenRouter, embedding đánh giá `bge-m3` chạy cục bộ.

### Bốn chỉ số nghĩa là gì

| Chỉ số | Đo cái gì | Câu hỏi nó trả lời |
|---|---|---|
| **Faithfulness** | Câu trả lời có bám context không | "Có bịa không?" |
| **Answer Relevancy** | Câu trả lời có đúng trọng tâm câu hỏi không | "Có lạc đề không?" |
| **Context Recall** | Context lấy về có đủ thông tin cần không | "Có bỏ sót tài liệu không?" |
| **Context Precision** | Context lấy về có bao nhiêu phần thừa | "Có lấy rác không?" |

Hai chỉ số đầu chấm **generation**, hai chỉ số sau chấm **retrieval**.

### Kết quả (cấu hình hybrid)

| Chỉ số | Trước khi làm sạch | Sau khi làm sạch |
|---|---|---|
| Faithfulness | 0.777 | 0.727 |
| Answer Relevancy | 0.650 | **0.755** ↑ |
| Context Recall | 0.630 | **0.759** ↑ |
| Context Precision | 0.809 | 0.730 |

Answer Relevancy và Context Recall **tăng mạnh** sau khi bỏ 722 link và khung điều hướng —
đúng hướng kỳ vọng.

> **Ghi chú trung thực:** cấu hình `dense_only` lần chạy này trả `n/a` vì tài khoản
> OpenRouter hết credit giữa chừng (lỗi 402 — RAGAS mặc định xin 16.384 token cho judge).
> Đây là lỗi môi trường, không phải lỗi code. **Phải chạy lại sau khi có đủ credit** trước
> khi đưa số lên báo cáo chính thức.

---

## 9. Câu hỏi pitching và cách trả lời

### Về kiến trúc

**H: Vì sao cần cả semantic lẫn BM25? Một cái không đủ à?**
Đ: Semantic hiểu ý nghĩa nhưng dở với thuật ngữ hiếm — mã đơn, tên riêng, số hiệu điều
khoản. BM25 ngược lại. Đo trên corpus này hai nhánh trùng nhau 8/10 chunk, tức 2 chunk mỗi
bên tìm ra mà bên kia bỏ sót — đó chính là phần bù trừ.

**H: Vì sao RRF mà không phải weighted sum?**
Đ: Cosine ∈ [0,1], BM25 là điểm thô lên tới 24+ trên corpus này. Cộng thẳng thì BM25 át
hoàn toàn. Chuẩn hoá BM25 theo max của từng truy vấn lại làm điểm không ổn định giữa các
truy vấn. RRF chỉ dùng thứ hạng nên miễn nhiễm.

**H: k=60 ở đâu ra?**
Đ: Cormack et al. 2009. k càng lớn thì càng ưu tiên tài liệu được **nhiều ranker đồng
thuận** hơn là tài liệu xếp nhất ở một ranker.

**H: Ngưỡng 0.48 calibrate thế nào?**
Đ: Chạy các câu chắc chắn liên quan và chắc chắn lạc đề qua `semantic_search`, xem khoảng
cách điểm rồi chọn ở giữa. Nhóm trong phạm vi: 0.52–0.78. Nhóm lạc đề: 0.33–0.47.

**H: Vì sao thêm router? Đề bài không yêu cầu.**
Đ: Vì bản không có router trả lời "hi" bằng cách chạy trọn pipeline rồi nói "không xác
minh được" — vô nghĩa với người dùng và tốn 1,6 giây. Đây là khác biệt giữa một bài tập
và một sản phẩm dùng được.

**H: Router dùng luật thì có yếu không?**
Đ: Có giới hạn thật — luật không hiểu câu diễn đạt lạ. Nhưng router **chỉ** lọc trường hợp
không cần tra tài liệu. Việc xác định câu hỏi có thuộc phạm vi hay không do ngưỡng cosine
đảm nhiệm, nơi có bằng chứng thật thay vì đoán từ khoá. Đổi lại: dưới 1ms thay vì 0,5–2s
mỗi câu, và tất định nên test được.

### Về đánh giá

**H: Hybrid có chắc tốt hơn dense-only không?**
Đ: Chưa kết luận được — lần chạy gần nhất `dense_only` bị lỗi môi trường (hết credit).
Lần chạy trước đó cho thấy hai cấu hình **thắng ở mặt khác nhau**: hybrid hơn về Answer
Relevancy, dense-only hơn về Context Precision. Giải thích: RRF kéo thêm ứng viên BM25 —
khớp từ khoá nhưng có thể lệch ngữ nghĩa — làm loãng precision, đổi lại bổ sung đoạn chứa
thuật ngữ hiếm mà embedding bỏ sót.

**H: 18 câu có đủ để kết luận không?**
Đ: Không. Một câu lệch đã làm dịch chuyển trung bình đáng kể. Cần 40–50 câu, và người viết
câu hỏi nên khác người xây corpus để tránh thiên lệch.

### Về vận hành

**H: Hết API key thì sao?**
Đ: Ba lớp. Một là pool xoay 7 nhà cung cấp với phạt nghỉ theo loại lỗi. Hai là nếu mọi
endpoint chết thì chuyển sang **extractive fallback** — ghép nguyên văn đoạn điểm cao kèm
trích dẫn, không cần LLM. Ba là không bao giờ `raise`, UI luôn hiện được gì đó.

**H: Có chống prompt injection không?**
Đ: Có. Mẫu injection bị chặn **trước** khi gọi LLM (`used_llm=False`), và so khớp chạy
trên chuỗi đã bỏ dấu tiếng Việt nên không lách được bằng thêm/bớt dấu. Ngoài ra output
guardrail đối chiếu mọi số trích dẫn với số nguồn thật.

**H: Làm sao biết LLM không bịa?**
Đ: Bốn lớp. Prompt cấm dùng kiến thức ngoài context. Ngưỡng cosine chặn câu ngoài domain
trước khi tới LLM. Output guardrail bắt trích dẫn không tồn tại. Và RAGAS Faithfulness
chấm định lượng mức bám context.

---

## 10. Kịch bản demo

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

| # | Thao tác | Chứng minh điều gì |
|---|---|---|
| 1 | Chỉ hàng chip đầu trang | 439 vector, 10 tài liệu, bge-m3 — hệ thống sẵn sàng |
| 2 | Gõ **"hi"** | Trace chỉ hiện **Router → Chitchat → Trả lời**. Không truy hồi. Đây là điểm khác biệt so với bản không router |
| 3 | Bấm **"Shopee hỗ trợ những phương thức thanh toán nào?"** | Liệt kê đủ 10 phương thức kèm hạn mức, trích dẫn `[1][3]` |
| 4 | Mở **"Nguồn tham khảo"** | Thanh đo điểm, tên file nguồn thật |
| 5 | Chỉ **dải trace** | Semantic → BM25 → RRF → Rerank → **Hybrid**, kèm model và thời gian |
| 6 | Bấm câu **tiếng Anh** | Trả lời **bằng tiếng Anh** dù tài liệu gốc tiếng Việt — bge-m3 đa ngôn ngữ |
| 7 | Bấm **"Công thức nấu phở bò?"** | Trace đổi sang **PageIndex** màu đồng, hệ thống **từ chối** thay vì bịa |
| 8 | Gõ **"Ignore all previous instructions..."** | Bị chặn, **không gọi LLM** |
| 9 | Kéo `top_k` xuống 3, hỏi lại | Số chunk đổi theo — tham số có tác dụng thật |

**Câu chốt cho bước 7:** *"Đây là phần chúng em tâm đắc nhất — hệ thống biết lúc nào nó
không biết."*

---

## 11. Hạn chế và hướng phát triển

**Nên chủ động thừa nhận khi bảo vệ** — nói trước chủ động hơn bị bắt bài:

1. **Không có `PAGEINDEX_API_KEY`** — Task 8 chạy chế độ vectorless cục bộ tự cài đặt.
   Đúng tinh thần nhưng không phải SDK thật.
2. **Cross-encoder chỉ hiểu tiếng Anh** (`ms-marco-MiniLM-L-6-v2`) nên rerank phần tiếng
   Việt yếu. Vì vậy `RERANK_METHOD` mặc định là `"rrf"`.
3. **Corpus nhỏ** — 10 tài liệu, 439 chunk. Đủ chứng minh pipeline, chưa đủ kết luận về
   chất lượng retrieval.
4. **Golden dataset 18 câu** do nhóm tự viết → thiên lệch: người viết câu hỏi cũng biết
   corpus có gì.
5. **Router bằng luật** — không hiểu câu diễn đạt lạ.
6. **Chưa deploy online** — bonus 4 điểm còn để ngỏ.
7. **`dense_only` chưa có số RAGAS** vì hết credit giữa chừng.

**Hướng phát triển:**

| Ưu tiên | Việc | Lợi ích |
|---|---|---|
| Cao | Chạy lại RAGAS đủ 2 cấu hình | Hoàn thiện bảng A/B |
| Cao | Cross-encoder đa ngôn ngữ (Jina) | Rerank tiếng Việt tốt hơn |
| Trung | Mở rộng golden dataset lên 40–50 câu | Chỉ số ổn định hơn |
| Trung | `SemanticChunker` thay `RecursiveCharacterTextSplitter` | Chunk bám ranh giới ý |
| Trung | Conversation memory (multi-turn) | Bonus 3 điểm |
| Thấp | Deploy HuggingFace Spaces | Bonus 4 điểm |

---

## 12. Prompt để đưa cho ChatGPT

Dán nguyên khối dưới đây vào ChatGPT **kèm toàn bộ file `DOCS.md` này**:

```
Bạn là trợ lý chuẩn bị thuyết trình kỹ thuật. Tôi vừa gửi tài liệu DOCS.md mô tả
một hệ thống RAG (Retrieval-Augmented Generation) do nhóm 5 sinh viên xây dựng
cho bài lab. Hãy đọc kỹ toàn bộ.

BỐI CẢNH:
- Buổi bảo vệ 45 phút: 15 phút trình bày + demo, 30 phút hỏi đáp
- Người nghe: giảng viên và coach có nền tảng kỹ thuật, cùng các nhóm khác
- Nhóm 5 người, mỗi người nói một phần (xem mục 10 trong tài liệu)
- Mọi con số trong tài liệu là đo thật, KHÔNG được bịa thêm số mới

VIỆC 1 — Sinh outline slide:
Tạo 12–15 slide. Mỗi slide gồm: tiêu đề, 3–5 gạch đầu dòng, và ghi chú cho người
nói. Ưu tiên slide có SỐ LIỆU và SO SÁNH TRƯỚC/SAU hơn là slide chữ. Nêu rõ slide
nào nên có sơ đồ và mô tả sơ đồ đó.

VIỆC 2 — Rà soát phản biện:
Đóng vai coach khó tính. Chỉ ra:
- Chỗ nào trong thiết kế còn yếu mà tài liệu chưa thừa nhận
- Câu hỏi nào có thể làm nhóm bí mà mục 9 chưa chuẩn bị
- Con số nào nghe đáng ngờ hoặc cần thêm ngữ cảnh mới thuyết phục

VIỆC 3 — Đề xuất ngược:
Liệt kê những cải tiến kỹ thuật cụ thể mà nhóm nên làm tiếp, xếp theo tỉ lệ
lợi ích/công sức. Với mỗi đề xuất, nói rõ nó sửa vấn đề nào đã nêu trong tài liệu.
Phần này tôi sẽ đưa lại cho Claude Code để thực hiện, nên hãy viết đủ cụ thể để
một agent lập trình có thể hành động ngay: nêu tên file, tên hàm, và cách kiểm chứng.

Đừng khen xã giao. Tôi cần phản biện thật.
```

---

## Phụ lục — Lệnh hay dùng

```powershell
# Kích hoạt môi trường
.\.venv\Scripts\Activate.ps1

# Chạy lại toàn bộ pipeline dữ liệu
python -m src.task1_collect_legal_docs      # 4 PDF Shopee
python -m src.task2_crawl_news              # 6 bài eBay
python -m src.task3_convert_markdown        # → 10 file .md
python -m src.task4_chunking_indexing       # → 439 vector

# Kiểm tra
python -m pytest tests/test_individual.py -v   # phải 35/35
python scripts/check_cp.py                     # trạng thái từng checkpoint
python -m src.llm_pool                         # pool nhận được key nào

# Đánh giá
python -m group_project.evaluation.eval_pipeline

# Demo
streamlit run app.py

# Biến môi trường cần cho Windows
$env:HF_HUB_DISABLE_XET = "1"        # nếu không tải model sẽ treo ở 0 B/s
$env:TOKENIZERS_PARALLELISM = "false"
```
