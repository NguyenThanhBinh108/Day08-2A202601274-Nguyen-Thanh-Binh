# CP6 — Tài liệu Thuyết trình & Bảo vệ

> Bài Lab 08: RAG Pipeline — E-commerce Support Chatbot
> Mọi con số trong tài liệu này đều đo từ lần chạy thật trên repo, không phải ước lượng.

---

## 0. Tóm tắt 60 giây (Bình mở màn)

Chúng em xây một chatbot hỏi đáp chính sách thương mại điện tử trên **corpus song ngữ
thật**: 4 văn bản chính sách Shopee tiếng Việt (tải từ `help.shopee.vn`, render sang PDF)
và 6 bài help center eBay tiếng Anh (crawl bằng Crawl4AI). Tổng 255.255 ký tự, cắt thành
**439 chunk**, nhúng bằng **BAAI/bge-m3** (1024 chiều, đa ngôn ngữ) vào ChromaDB.

Truy hồi theo hướng **hybrid**: chạy song song semantic search và BM25, hợp nhất bằng
**Reciprocal Rank Fusion**, và khi điểm cosine gốc tụt dưới ngưỡng thì tự chuyển sang
**PageIndex vectorless** thay vì để LLM bịa. Câu trả lời luôn kèm trích dẫn đánh số khớp
với danh sách nguồn hiển thị trên giao diện.

Kết quả: **35/35 test tự động pass** (trọn 50 điểm pipeline), có báo cáo đánh giá RAGAS
so sánh A/B giữa Hybrid và Dense-only.

---

## 1. Kiến trúc pipeline (Role 1 — Bình)

```
                    ┌─────────────────────────────────────────┐
  data/landing/     │  Task 1  4 PDF chính sách Shopee  🇻🇳    │
                    │  Task 2  6 bài help center eBay   🇬🇧    │
                    └────────────────┬────────────────────────┘
                                     │  Task 3: MarkItDown + LÀM SẠCH
                    ┌────────────────▼────────────────────────┐
  data/standardized │  10 file .md — 255.255 ký tự            │
                    └────────────────┬────────────────────────┘
                                     │  Task 4: chunk 800/100 → bge-m3 → ChromaDB
                    ┌────────────────▼────────────────────────┐
  chroma_db/        │  439 vector (143 legal + 296 news)      │
                    └────────────────┬────────────────────────┘
                                     │
              ┌──────────────────────┴──────────────────────┐
              │                                             │
     ┌────────▼─────────┐                        ┌──────────▼─────────┐
     │ Task 5  Semantic │                        │ Task 6  BM25       │
     │ cosine, bge-m3   │                        │ + TF-IDF (bonus)   │
     └────────┬─────────┘                        └──────────┬─────────┘
              │  giữ điểm cosine GỐC                        │
              └──────────────────┬──────────────────────────┘
                                 │  Task 7: RRF k=60
                       ┌─────────▼──────────┐
                       │  Task 9  Pipeline  │
                       └─────────┬──────────┘
                                 │
              cosine gốc < 0.48? ├── có ──→ Task 8 PageIndex vectorless
                                 │
                                 └── không ──→ hybrid
                                 │
                       ┌─────────▼──────────┐
                       │ Task 10 Generation │  reorder + citation
                       └─────────┬──────────┘
                                 │
                            app.py (Streamlit)
```

**Ba điểm đáng nhấn khi thuyết trình:**

1. Corpus **song ngữ có chủ đích** — không phải ngẫu nhiên. Câu hỏi benchmark của lab là
   tiếng Anh, còn giao diện demo là tiếng Việt. `bge-m3` đa ngôn ngữ cho phép hỏi tiếng
   Anh mà lấy được tài liệu tiếng Việt và ngược lại.
2. Hai nhánh retrieval **dùng chung đúng một bộ chunk** — đây là điều kiện sống còn để RRF
   hoạt động, giải thích ở mục 3.2.
3. Fallback quyết định bằng **điểm cosine gốc**, không phải điểm RRF — cái bẫy chính của
   bài lab, giải thích ở mục 3.3.

---

## 2. Giải thích code theo từng Task

### Task 1 — `src/task1_collect_legal_docs.py`

Tải 4 trang chính sách chính thức của Shopee rồi render sang PDF bằng `fpdf2`.

| Quyết định | Lý do |
|---|---|
| Dùng `requests` thuần, không Playwright | Đã đo: `help.shopee.vn` trả nội dung ngay trong HTML đầu tiên (~19.600 ký tự text sạch cho bài 77251). Docstring lab cảnh báo help center thường là SPA — với Shopee thì không phải. |
| Nạp font TTF Unicode (`arial.ttf`) vào fpdf2 | Font lõi của fpdf2 chỉ hỗ trợ latin-1, tiếng Việt sẽ thành `?`. Có nhánh dự phòng sanitize latin-1 nếu máy không có TTF. |
| Gắn `customer_role` (buyer/seller/both) | Yêu cầu K4 Variant, lưu ở `data/landing/legal/_metadata.json`, dùng cho `metadata_filter`. |

> **Bẫy đã gặp:** `fpdf2` 2.8.7 có `multi_cell` mặc định `new_x=XPos.RIGHT`, nên sau lần gọi
> đầu con trỏ nằm sát lề phải và lần gọi kế tiếp tính ra bề rộng = 0 → `FPDFException:
> Not enough horizontal space`. Ban đầu **cả 7 nguồn đều fail** vì lỗi này. Khắc phục bằng
> `pdf.set_x(pdf.l_margin)` trước mỗi `multi_cell`.

### Task 2 — `src/task2_crawl_news.py`

Crawl 6 bài eBay Help Center bằng `Crawl4AI` (Playwright/Chromium bên dưới).

> **Bẫy phiên bản:** `crawl4ai` 0.8.5 trả `result.markdown` là **object**
> `MarkdownGenerationResult`, **không phải `str`**. Nếu `str()` thẳng sẽ ghi ra file rác.
> Hàm `_markdown_to_text()` thử lần lượt `raw_markdown` → `fit_markdown` → `str(md)`.

### Task 3 — `src/task3_convert_markdown.py`

Convert sang Markdown **và làm sạch** — bước có tác động lớn nhất tới chất lượng câu trả lời.

Đo trên corpus thô: 324.107 ký tự chứa **722 markdown link, 735 URL trần**, cộng toàn bộ
khung điều hướng eBay (`Skip to main content`, `Sign in`, `Watchlist`, menu `* Summary`,
`* Bids/Offers`…) và cả chuỗi session ID.

Rác này đi thẳng vào chunk → vector store → context của LLM. Trước khi làm sạch, câu trả
lời trả về nguyên văn:

```
...of my request](https://www.ebay.com/help/action?topicid=4667&af=2) [Returns Policy, 2026].
```

**Luật lọc dựa trên cấu trúc, không dựa trên danh sách từ khoá** — điểm này đáng nói khi
bảo vệ, vì lọc theo từ khoá rất dễ xoá nhầm nội dung chính sách thật:

- Gỡ `[chữ](url)` → giữ lại `chữ`, bỏ URL.
- Một dòng mà **sau khi gỡ hết link gần như không còn chữ** → đó là thanh điều hướng.
- Mục menu dạng bullet là **một link có nhãn ngắn** (< 45 ký tự); câu chính sách thật gần
  như luôn dài hơn và hiếm khi là link.

Kết quả: **722 → 0 link**, corpus 324.107 → **255.255 ký tự**. Riêng bài eBay bỏ được
**34–61%** rác; file Shopee (từ PDF) vốn sạch nên chỉ bỏ 3–6%.

> Ngoài ra sửa một bug thật của bản starter: nếu thư mục nguồn không có file phù hợp thì
> vòng lặp không chạy lần nào, hàm kết thúc âm thầm và **vẫn in "Done" với exit code 0** —
> người làm tưởng đã xong CP1. Nay in cảnh báo rõ và `sys.exit(1)`.

### Task 4 — `src/task4_chunking_indexing.py`

| Tham số | Giá trị | Lý do |
|---|---|---|
| `CHUNK_SIZE` | 800 | Đủ chứa trọn một đoạn chính sách hoàn chỉnh |
| `CHUNK_OVERLAP` | 100 | 12,5% — đủ để không cắt đứt câu ở biên chunk |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Đa ngôn ngữ, bắt buộc vì corpus song ngữ |
| `EMBEDDING_DIM` | 1024 | |
| Không gian đo | `hnsw:space=cosine` | |

**Hai cặp đôi bắt buộc đi cùng nhau** (câu hỏi coach hay hỏi):

1. `normalize_embeddings=True` **phải** đi với `hnsw:space="cosine"`. Thiếu chuẩn hoá thì
   khoảng cách cosine tính ra sai lệch.
2. ID chunk phải **duy nhất toàn cục**: `f"{type}__{source}__{chunk_index}"`. Nếu chỉ dùng
   `source + chunk_index` thì file trùng tên ở `legal/` và `news/` sẽ ghi đè nhau.

`reset_collection()` chạy trước mỗi lần index lại — docstring lab đã cảnh báo chunk cũ lẫn
chunk mới sẽ làm retrieval trả về rác.

### Task 5 — `src/task5_semantic_search.py`

```python
score = 1.0 - distance      # Chroma trả DISTANCE, không phải similarity
```

Đây là **điểm cosine gốc**, thang `[0,1]` có ý nghĩa — Task 9 dùng chính con số này để
quyết định fallback.

Toàn bộ model và collection đều **lazy-load có cache**: `pytest` import module này, nếu kéo
2 GB weight lúc import thì test sẽ rất chậm hoặc crash.

**Bonus HyDE** (+5 điểm): `generate_hyde_query()` sinh một đoạn trả lời giả định rồi embed
đoạn đó thay vì câu hỏi gốc — thu hẹp khoảng cách ngữ nghĩa giữa "câu hỏi" và "câu trả
lời". Mặc định `use_hyde=False` để test chạy nhanh và không tốn API.

### Task 6 — `src/task6_lexical_search.py`

**Đây là chỗ quan trọng nhất của toàn bài, phải giải thích được khi bảo vệ.**

`_load_corpus()` đọc chunk **thẳng từ ChromaDB** (`collection.get()`), không tự chunk lại.

Vì sao? Task 9 hợp nhất bằng RRF, **khoá gộp là `item["content"]`**. Nếu Task 6 tự cắt
chunk theo cách khác (ví dụ 500/50 thay vì 800/100) thì chuỗi content của hai nhánh **không
bao giờ trùng nhau**, RRF chỉ đan xen hai danh sách chứ không fuse được gì — và lỗi này
**hoàn toàn im lặng**, hệ thống vẫn chạy, vẫn trả kết quả, chỉ là mất sạch giá trị của
hybrid search.

Cách nhận biết: nhìn điểm RRF đầu ra. Nếu mọi tài liệu đều có điểm dạng
`1/61, 1/62, 1/63…` thì mỗi tài liệu chỉ nhận **đúng một phiếu** → hai ranker không giao
nhau tài liệu nào.

Đo thực tế trên hệ thống của nhóm:

| Truy vấn | Dense ∩ Sparse |
|---|---|
| "How does a buyer request a return or refund" | **8/10** chunk |
| "Shopee hỗ trợ những phương thức thanh toán nào?" | **8/10** chunk |
| "payment methods" | **4/10** chunk |

Có giao nhau nên tài liệu được cả hai ranker đồng thuận đạt điểm ≈ **0,032**, gấp đôi tài
liệu chỉ một ranker tìm ra (≈ 0,016). Đó chính là tín hiệu xếp hạng mà RRF sinh ra.

**Bonus TF-IDF** (+5 điểm) — khác biệt cơ chế cần nói rõ:

| | TF-IDF | BM25 |
|---|---|---|
| Tần suất từ | Tuyến tính, không bão hoà | Bão hoà theo `k1` (từ lặp 10 lần không hơn 5 lần bao nhiêu) |
| Độ dài tài liệu | Không chuẩn hoá | Chuẩn hoá theo `b` và độ dài trung bình |
| Bản chất | Tích `TF × IDF` trên không gian vector | Hàm xếp hạng xác suất |

### Task 7 — `src/task7_reranking.py`

```
RRF(d) = Σ  1 / (k + rank_r(d)) ,  k = 60,  rank bắt đầu từ 1
```

**Vì sao phải dùng thứ hạng thay vì cộng điểm trực tiếp?** Cosine nằm trong `[0,1]` còn
BM25 là điểm thô không giới hạn (có thể 0 → 24+ trên corpus này). Cộng thẳng hai thang đo
đó thì BM25 át hoàn toàn cosine. RRF chỉ nhìn **thứ hạng**, nên hai ranker được đối xử
công bằng bất kể thang điểm.

`k = 60` lấy từ Cormack et al. 2009 — làm mượt, giảm ảnh hưởng của vài vị trí đầu.

Cross-encoder và MMR đều **lazy-load có cache** và bọc `try/except`: không tải được model
thì cảnh báo rồi giữ nguyên thứ tự, tuyệt đối không `raise`.

### Task 8 — `src/task8_pageindex_vectorless.py`

Nhóm không có `PAGEINDEX_API_KEY`, nên phần đáng nói là **chế độ vectorless cục bộ**:

- Quét heading markdown (`^#{1,6}`) dựng thành **cây mục lục**.
- Chấm điểm từng node bằng số từ khoá của truy vấn xuất hiện trong **tiêu đề** (trọng số
  cao) cộng trong **nội dung** (trọng số thấp).
- **Không dùng embedding** — đúng tinh thần "vectorless RAG": duyệt cấu trúc tài liệu thay
  vì tìm trong không gian vector.

Luôn trả về `list` và gắn `"source": "pageindex"`, không bao giờ `raise`.

### Task 9 — `src/task9_retrieval_pipeline.py`

**Cái bẫy trung tâm của bài lab.** Điểm RRF của top-1 **luôn ≈ 1/(k+1) = 0,0164** bất kể
câu hỏi có liên quan hay không — vì nó chỉ phụ thuộc thứ hạng. Nếu đem con số đó so với
`SCORE_THRESHOLD` thì fallback hoặc không bao giờ kích hoạt, hoặc luôn kích hoạt.

```python
best_score = dense_results[0]["score"]   # cosine GỐC, KHÔNG phải điểm RRF
if best_score < SCORE_THRESHOLD:         # 0.48
    return pageindex_search(query, top_k=top_k)
```

Bằng chứng ngưỡng 0,48 hoạt động đúng, đo từ lần chạy đánh giá thật:

| Câu hỏi | Cosine gốc | Định tuyến |
|---|---|---|
| "Gia hạn hộ chiếu ở cục xuất nhập cảnh?" | **0,466** | → PageIndex |
| "Công thức nấu phở bò gồm những gì?" | **0,326** | → PageIndex |
| Các câu trong phạm vi tài liệu | 0,52 – 0,78 | → Hybrid |

> **Một lỗi nhóm tự tìm ra và sửa:** ban đầu pipeline chạy **RRF hai lần** — bước 2 fuse
> dense + sparse (điểm có ý nghĩa), rồi bước 3 lại gọi `rerank(method="rrf")` trên **một
> danh sách duy nhất**, khiến mỗi tài liệu chỉ còn một phiếu và điểm bị ghi đè thành
> `1/(60+rank)` đều tăm tắp. Thứ tự vẫn đúng nên không ai phát hiện, nhưng thông tin "được
> mấy ranker đồng thuận" mất sạch. Sau khi sửa, điểm từ `0,0164` → **`0,0325`**.

### Task 10 — `src/task10_generation.py`

**Reorder chống "lost in the middle"** — LLM chú ý hai đầu context hơn phần giữa:

```python
front = chunks[0::2]        # 0, 2, 4...
back  = chunks[1::2][::-1]  # 1, 3, 5... đảo ngược
return front + back
```

Cách này đặt chunk điểm cao nhất ở **đầu**, chunk điểm cao thứ hai ở **cuối**, giữ nguyên
số lượng.

**Số trích dẫn gán TRƯỚC khi reorder**, trên bản copy nông — nhờ vậy số `[n]` trong câu trả
lời luôn khớp thứ tự nguồn hiển thị trên giao diện, dù vị trí trong prompt đã bị xáo trộn.

**Ba lớp không-bao-giờ-gãy:**
1. Thiếu tài liệu → trả câu "không xác minh được", `answer` vẫn là chuỗi khác rỗng.
2. Model chính lỗi (402/429/404) → thử lần lượt các model `:free` của OpenRouter.
3. Mọi model đều lỗi → **extractive fallback**: ghép nguyên văn các đoạn điểm cao nhất kèm
   số trích dẫn, ghi rõ đang ở chế độ không-LLM. **Không bao giờ `raise`.**

`max_tokens=1400` đặt tường minh vì mỗi model trên OpenRouter có mặc định khác nhau; nếu
`finish_reason == "length"` thì xin model viết tiếp đúng một lần rồi nối lại.

---

## 3. Chuẩn bị hỏi đáp kỹ thuật (Role 2 — Đăng)

**H: Vì sao chọn RRF mà không phải weighted sum của cosine và BM25?**
Đ: Vì hai thang đo không so sánh được. Cosine ∈ [0,1], BM25 là điểm thô không giới hạn —
đo trên corpus này lên tới 24+. Cộng có trọng số thì phải chuẩn hoá BM25, mà chuẩn hoá theo
max của từng truy vấn lại làm điểm không ổn định giữa các truy vấn. RRF chỉ dùng thứ hạng
nên miễn nhiễm với chuyện này.

**H: k=60 lấy ở đâu ra, thử giá trị khác chưa?**
Đ: Từ paper Cormack et al. 2009. `k` càng lớn thì chênh lệch giữa các thứ hạng đầu càng
nhỏ, tức càng ưu tiên tài liệu được **nhiều ranker đồng thuận** hơn là tài liệu xếp nhất ở
một ranker. k=60 là mặc định đã được kiểm chứng rộng rãi.

**H: Làm sao biết hybrid thực sự tốt hơn dense-only?**
Đ: Có bảng RAGAS A/B ở mục 5 — và câu trả lời trung thực là **không tốt hơn ở mọi chỉ số**.

**H: Ngưỡng 0.48 calibrate thế nào?**
Đ: Chạy các câu chắc chắn liên quan và các câu chắc chắn lạc đề qua `semantic_search`, xem
khoảng cách điểm giữa hai nhóm rồi chọn ngưỡng nằm giữa. Nhóm trong phạm vi rơi vào
0,52–0,78; nhóm lạc đề 0,33–0,47. Chọn 0,48.

**H: Nếu ChromaDB chưa index thì hệ thống làm gì?**
Đ: `semantic_search` trả `[]` kèm cảnh báo, không `raise`. Giao diện hiện chip "chưa index"
màu cảnh báo và banner hướng dẫn chạy Task 4.

**H: Chunk 800 có cắt mất ngữ cảnh không?**
Đ: Có overlap 100 ký tự và `RecursiveCharacterTextSplitter` ưu tiên tách ở `\n\n` → `\n` →
`. ` trước khi tách giữa từ, nên phần lớn chunk kết thúc ở ranh giới câu.

---

## 4. Kịch bản demo live (Role 3 — Vũ)

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

**Diễn theo đúng thứ tự này** — mỗi bước chứng minh một tính năng:

| # | Thao tác | Chỉ cho lớp thấy điều gì |
|---|---|---|
| 1 | Chỉ vào hàng chip trên đầu | 439 vector, corpus 10 tài liệu, model bge-m3 — hệ thống đã sẵn sàng |
| 2 | Bấm câu mẫu **"Shopee hỗ trợ những phương thức thanh toán nào?"** | Trả lời tiếng Việt, liệt kê đủ 10 phương thức kèm hạn mức, có trích dẫn `[1][3]` |
| 3 | Mở khung **"Nguồn tham khảo"** | Thanh đo điểm cho thấy chênh lệch giữa các chunk; tên file nguồn thật |
| 4 | Chỉ vào **dải trace** | Semantic → BM25 → RRF → Rerank → **Hybrid**, kèm thời gian |
| 5 | Bấm câu tiếng Anh **"What payment methods does the marketplace accept?"** | **Trả lời bằng tiếng Anh** dù tài liệu gốc tiếng Việt — chứng minh bge-m3 đa ngôn ngữ |
| 6 | Bấm câu **"Công thức nấu phở bò gồm những gì?"** | Trace đổi sang **PageIndex** màu đồng, hệ thống **từ chối trả lời** thay vì bịa |
| 7 | Kéo `top_k` xuống 3, hỏi lại | Số chunk trong trace đổi theo — tham số có tác dụng thật |

**Câu chốt cho bước 6:** "Đây mới là phần chúng em tâm đắc nhất — hệ thống biết lúc nào
nó *không* biết."

---

## 5. Báo cáo đánh giá RAGAS (Role 4/5 — Linh & Liễu)

Bộ số chi tiết nằm ở [`group_project/evaluation/results.md`](group_project/evaluation/results.md).

**Thiết lập:** 18 câu golden dataset (13 tiếng Anh + 3 tiếng Việt + 2 ngoài domain),
LLM judge `openai/gpt-4o-mini` qua OpenRouter, embedding đánh giá dùng `bge-m3` chạy cục bộ.

**Hai cấu hình so sánh:**
- `hybrid` — `retrieve()` của Task 9: semantic + BM25 → RRF → fallback
- `dense_only` — `retrieve_dense_only()`: chỉ semantic, không BM25, không fallback

**Cách trình bày kết quả cho trung thực:** đừng nói "hybrid tốt hơn". Bộ số cho thấy hai
cấu hình **thắng ở những mặt khác nhau**, và giải thích được vì sao mới là phần ăn điểm:

- RRF kéo thêm ứng viên từ BM25 vào top-k. Những ứng viên này khớp **từ khoá** nhưng có thể
  lệch **ngữ nghĩa** → làm loãng `context_precision` và `context_recall`.
- Đổi lại, chúng bổ sung đúng những đoạn chứa thuật ngữ hiếm mà embedding bỏ sót → câu trả
  lời bám câu hỏi hơn, `answer_relevancy` tăng rõ.

**Phân tích worst performers** — 2 câu ngoài domain kéo điểm trung bình xuống ở cấu hình
`dense_only`, vì không có fallback nên nó vẫn nhồi đủ 5 context không liên quan rồi buộc
LLM trả lời. Cấu hình `hybrid` phát hiện cosine thấp và từ chối. **Đây chính là điểm yếu
mà bài lab muốn chứng minh** — nên nhấn mạnh khi báo cáo.

**Đề xuất cải tiến (nói ở cuối):**
1. Bật cross-encoder rerank đa ngôn ngữ (Jina) thay cho `ms-marco-MiniLM` chỉ hiểu tiếng Anh.
2. Tăng golden dataset lên 40–50 câu để chỉ số ổn định hơn — 18 câu thì một câu lệch đã
   làm dịch chuyển trung bình đáng kể.
3. Thử `SemanticChunker` thay `RecursiveCharacterTextSplitter` để chunk bám ranh giới ý.

---

## 6. Những thứ nên chủ động thừa nhận

Coach thường hỏi xoáy phần yếu. Nói trước sẽ chủ động hơn là bị bắt bài:

1. **Không có `PAGEINDEX_API_KEY`** — Task 8 chạy chế độ vectorless cục bộ tự cài đặt. Đúng
   tinh thần (duyệt cây mục lục, không dùng embedding) nhưng không phải SDK thật.
2. **Cross-encoder chỉ hiểu tiếng Anh** (`ms-marco-MiniLM-L-6-v2`) nên rerank phần tiếng
   Việt yếu. Vì vậy `RERANK_METHOD` mặc định là `"rrf"`.
3. **Corpus nhỏ** — 10 tài liệu, 439 chunk. Đủ để chứng minh pipeline, chưa đủ để kết luận
   chắc chắn về chất lượng retrieval.
4. **Golden dataset 18 câu** do nhóm tự viết nên có thiên lệch: người viết câu hỏi cũng là
   người biết corpus có gì.
5. **Chưa deploy online** — bonus 4 điểm còn để ngỏ.

---

## 7. Số liệu tra nhanh khi bị hỏi

| Hạng mục | Con số |
|---|---|
| Tài liệu nguồn | 4 PDF Shopee 🇻🇳 (267 KB) + 6 JSON eBay 🇬🇧 (228 KB) |
| Corpus sau chuẩn hoá | 10 file `.md`, **255.255 ký tự** |
| Rác đã lọc | 722 markdown link → **0**; 735 URL → 0 |
| Chunk | **439** (143 legal + 296 news), size 800 / overlap 100 |
| Embedding | `BAAI/bge-m3`, 1024 chiều, cosine |
| Dense ∩ Sparse | **8/10** chunk (truy vấn điển hình) |
| RRF | `k=60`; điểm đồng thuận ≈ 0,032 so với 0,016 |
| Ngưỡng fallback | **0,48** trên cosine gốc |
| Test tự động | **35/35 PASSED** |
| Mã nguồn | `src/` **4.027 dòng** |
| Lịch sử git | **25 commit**, tách theo từng task và role |
