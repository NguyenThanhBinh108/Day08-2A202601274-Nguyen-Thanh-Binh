"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import re
import sys
from pathlib import Path
from typing import Any, Optional

# Fallback config — chỉ dùng khi không đọc được hằng số từ Task 4 (ví dụ Task 4
# chưa implement xong). Giữ trùng giá trị với task4_chunking_indexing.py.
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
DEFAULT_COLLECTION_NAME = "ecommerce_support_docs"

# Corpus dùng chung cho BM25 và TF-IDF. List of {'content': str, 'metadata': dict}.
# KHÔNG load ở cấp module (pytest import module này, import phải nhanh và không được
# crash khi chưa có data/ hoặc chroma_db/). Tất cả đều lazy qua các getter bên dưới.
CORPUS: list[dict] = []

# Cache cấp module — build một lần rồi tái sử dụng cho mọi query.
_CORPUS_LOADED: bool = False
_BM25_INDEX: Optional[Any] = None
_TFIDF_VECTORIZER: Optional[Any] = None
_TFIDF_MATRIX: Optional[Any] = None


# =============================================================================
# Helper
# =============================================================================

def _safe_print(message: str) -> None:
    """
    In thông báo mà KHÔNG BAO GIỜ raise.

    Console Windows mặc định là cp1252, không encode được ký tự ⚠ hay chữ Việt có dấu
    ("ế", "ỗ"...) → print() thường sẽ ném UnicodeEncodeError và làm hỏng cả test run.
    Ở đây fallback sang ASCII (ký tự lạ thành '?') thay vì để lỗi lan ra ngoài.
    """
    try:
        print(message)
    except UnicodeEncodeError:
        try:
            print(message.encode("ascii", errors="replace").decode("ascii"))
        except Exception:
            pass
    except Exception:
        pass


# =============================================================================
# Tokenizer — dùng chung cho cả BM25 lẫn TF-IDF
# =============================================================================

def _tokenize(text: str) -> list[str]:
    """
    Tách token cho lexical search, unicode-aware để không làm hỏng tiếng Việt.

    Vì sao không dùng text.lower().split() như gợi ý trong starter:
        - split() giữ lại dấu câu dính vào từ ("refund." != "refund"), làm hụt match.
        - Regex r"\\w+" với cờ re.UNICODE giữ nguyên chữ có dấu (ả, ộ, ữ...) thay vì
          cắt vụn như [a-z]+ — bắt buộc vì corpus song ngữ Anh/Việt.
        - Bỏ token độ dài 1 để loại nhiễu (chữ cái lẻ, số lẻ, ký tự markdown sót lại).

    Args:
        text: Chuỗi đầu vào (có thể rỗng hoặc None-safe qua str())

    Returns:
        List token đã lowercase, độ dài >= 2.
    """
    if not text:
        return []
    return [tok for tok in re.findall(r"\w+", str(text).lower(), flags=re.UNICODE) if len(tok) > 1]


# =============================================================================
# Corpus loading — PHẢI khớp chunk với Task 4/Task 5
# =============================================================================

def _import_task4() -> Optional[Any]:
    """
    Import module Task 4 một cách "mềm", thử đủ các kiểu chạy thường gặp:
        - pytest / `import src.task6_lexical_search`  → relative import
        - `python -m src.task6_lexical_search`        → src.task4_chunking_indexing
        - `python src/task6_lexical_search.py`        → task4_chunking_indexing

    Import nằm TRONG hàm (không phải cấp module) để module này vẫn import được kể cả
    khi Task 4 lỗi/chưa xong. Trả về None thay vì raise.
    """
    import importlib

    candidates = [
        (".task4_chunking_indexing", __package__ or "src"),
        ("src.task4_chunking_indexing", None),
        ("task4_chunking_indexing", None),
    ]
    for name, package in candidates:
        try:
            return importlib.import_module(name, package=package)
        except Exception:
            continue
    return None


def _get_chroma_config() -> tuple[Path, str]:
    """
    Lấy (CHROMA_DIR, COLLECTION_NAME) từ Task 4 nếu import được, ngược lại dùng default.

    Luôn đọc config từ Task 4 để nếu bạn đổi COLLECTION_NAME ở đó thì Task 6 tự bám theo,
    không bị lệch collection giữa semantic search và lexical search.
    """
    task4 = _import_task4()
    if task4 is None:
        return CHROMA_DIR, DEFAULT_COLLECTION_NAME
    try:
        chroma_dir = Path(getattr(task4, "CHROMA_DIR", CHROMA_DIR))
        collection_name = str(getattr(task4, "COLLECTION_NAME", DEFAULT_COLLECTION_NAME))
        return chroma_dir, collection_name
    except Exception:
        return CHROMA_DIR, DEFAULT_COLLECTION_NAME


def _normalize_metadata(meta: Any) -> dict:
    """Chuẩn hoá metadata về đúng schema {'source', 'type', 'chunk_index'}."""
    meta = dict(meta) if isinstance(meta, dict) else {}
    meta.setdefault("source", "unknown")
    meta.setdefault("type", "legal")
    meta.setdefault("chunk_index", 0)
    return meta


def _load_from_chroma() -> list[dict]:
    """
    Ưu tiên 1: đọc thẳng chunk đã index từ ChromaDB.

    Đây là đường đi QUAN TRỌNG NHẤT của module. Task 9 gộp kết quả semantic + lexical
    bằng RRF với khoá gộp là item["content"]. Nếu Task 6 tự chunk lại theo cách khác
    Task 4 thì content sẽ không bao giờ trùng nhau → RRF chỉ nối hai danh sách chứ
    không thật sự fuse (mất toàn bộ lợi ích của hybrid search). Đọc lại đúng document
    đã nằm trong vector store đảm bảo chunk giống hệt Task 5 tới từng ký tự.
    """
    chroma_dir, collection_name = _get_chroma_config()
    if not chroma_dir.exists():
        return []

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection(name=collection_name)
        data = collection.get(include=["documents", "metadatas"])
    except Exception as exc:  # collection chưa tồn tại, DB hỏng, version lệch...
        _safe_print(f"  ⚠ Không đọc được ChromaDB ({type(exc).__name__}: {exc}) — thử fallback Task 4")
        return []

    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or [None] * len(documents)

    corpus: list[dict] = []
    for doc, meta in zip(documents, metadatas):
        if not doc:
            continue
        corpus.append({"content": doc, "metadata": _normalize_metadata(meta)})

    # collection.get() không cam kết thứ tự → sort ổn định theo (source, chunk_index)
    # để mọi lần chạy đều cho cùng thứ hạng khi điểm bằng nhau.
    corpus.sort(key=lambda c: (str(c["metadata"].get("source", "")), _safe_int(c["metadata"].get("chunk_index"))))
    return corpus


def _safe_int(value: Any) -> int:
    """Ép chunk_index về int an toàn (Chroma có thể trả str/float)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _load_from_task4() -> list[dict]:
    """
    Ưu tiên 2: chưa có chroma_db/ thì chunk lại bằng chính hàm của Task 4.

    Dùng lại load_documents() + chunk_documents() (KHÔNG viết splitter riêng) nên chunk
    vẫn giống hệt những gì Task 4 sẽ index — content khớp, RRF ở Task 9 vẫn fuse đúng.
    """
    task4 = _import_task4()
    if task4 is None or not hasattr(task4, "load_documents") or not hasattr(task4, "chunk_documents"):
        _safe_print("  ⚠ Không import được load_documents/chunk_documents của Task 4")
        return []

    try:
        documents = task4.load_documents()
        chunks = task4.chunk_documents(documents)
    except NotImplementedError:
        _safe_print("  ⚠ Task 4 chưa implement — không dựng được corpus cho BM25")
        return []
    except Exception as exc:
        _safe_print(f"  ⚠ Lỗi khi chunk lại bằng Task 4 ({type(exc).__name__}: {exc})")
        return []

    corpus: list[dict] = []
    for chunk in chunks or []:
        content = chunk.get("content") if isinstance(chunk, dict) else None
        if not content:
            continue
        corpus.append({"content": content, "metadata": _normalize_metadata(chunk.get("metadata"))})
    return corpus


def _load_corpus() -> list[dict]:
    """
    Dựng corpus cho lexical search theo thứ tự ưu tiên:
        1) ChromaDB (chunk y hệt Task 5) → tốt nhất cho hybrid fusion
        2) load_documents() + chunk_documents() của Task 4 → khi chưa index
        3) [] kèm cảnh báo → không bao giờ raise (test phải import/chạy được)
    """
    corpus = _load_from_chroma()
    if corpus:
        return corpus

    corpus = _load_from_task4()
    if corpus:
        return corpus

    _safe_print("  ⚠ Corpus rỗng: chưa có chroma_db/ và cũng chưa có data/standardized/ — "
                "lexical_search sẽ trả về []")
    return []


def get_corpus(force_reload: bool = False) -> list[dict]:
    """Getter có cache cho corpus (lazy — chỉ load ở lần gọi search đầu tiên)."""
    global CORPUS, _CORPUS_LOADED
    if force_reload or not _CORPUS_LOADED:
        CORPUS = _load_corpus()
        _CORPUS_LOADED = True
    return CORPUS


def reset_index() -> None:
    """
    Xoá toàn bộ cache (corpus, BM25 index, TF-IDF matrix) để build lại từ đầu.

    Gọi hàm này sau khi re-index Task 4 (đổi corpus/chunk size), nếu không module
    vẫn dùng corpus cũ đang nằm trong bộ nhớ.
    """
    global CORPUS, _CORPUS_LOADED, _BM25_INDEX
    global _TFIDF_VECTORIZER, _TFIDF_MATRIX
    CORPUS = []
    _CORPUS_LOADED = False
    _BM25_INDEX = None
    _TFIDF_VECTORIZER = None
    _TFIDF_MATRIX = None


# =============================================================================
# BM25
# =============================================================================

def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    if not corpus:
        _safe_print("  ⚠ build_bm25_index: corpus rỗng → không dựng được index")
        return None

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        _safe_print("  ⚠ Thiếu package rank-bm25 (pip install rank-bm25) → bỏ qua BM25")
        return None

    # Giữ nguyên vị trí từng document (kể cả doc tokenize ra rỗng) để index i của
    # bm25.get_scores() luôn ánh xạ đúng corpus[i].
    tokenized_corpus = [
        _tokenize(doc.get("content", "") if isinstance(doc, dict) else doc) for doc in corpus
    ]

    # BM25Okapi chia cho avgdl → corpus toàn document rỗng sẽ gây ZeroDivisionError.
    if not any(tokenized_corpus):
        _safe_print("  ⚠ build_bm25_index: toàn bộ document tokenize ra rỗng → bỏ qua BM25")
        return None

    try:
        # k1=1.5, b=0.75 là default của BM25Okapi — đúng với công thức mô tả ở docstring.
        return BM25Okapi(tokenized_corpus)
    except Exception as exc:
        _safe_print(f"  ⚠ Lỗi khi dựng BM25 index ({type(exc).__name__}: {exc})")
        return None


def get_bm25_index():
    """Getter có cache cho BM25 index (build một lần cho toàn bộ vòng đời process)."""
    global _BM25_INDEX
    if _BM25_INDEX is None:
        _BM25_INDEX = build_bm25_index(get_corpus())
    return _BM25_INDEX


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if not query or not str(query).strip() or top_k <= 0:
        return []

    corpus = get_corpus()
    if not corpus:
        return []

    bm25 = get_bm25_index()
    if bm25 is None:
        return []

    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    try:
        scores = bm25.get_scores(tokenized_query)
    except Exception as exc:
        _safe_print(f"  ⚠ Lỗi khi tính BM25 score ({type(exc).__name__}: {exc})")
        return []

    results: list[dict] = []
    for idx, raw_score in enumerate(scores):
        # float() bắt buộc: get_scores() trả numpy.float64, để nguyên thì test so sánh
        # scores == sorted(scores, reverse=True) có thể lệch kiểu và khó debug.
        score = round(float(raw_score), 4)
        if score <= 0:
            continue  # không trùng term nào → không phải lexical match, loại bỏ
        results.append({
            "content": corpus[idx]["content"],
            "score": score,
            "metadata": corpus[idx]["metadata"],
        })

    # sort ổn định: điểm bằng nhau thì giữ thứ tự corpus (đã sort theo source/chunk_index)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# =============================================================================
# BONUS (+5) — TF-IDF search
#
# Khác biệt cơ chế TF-IDF vs BM25 (giải thích cho phần demo):
#   - TF-IDF chấm điểm bằng tích TF * IDF trên không gian vector rồi đo cosine giữa
#     vector query và vector document. Term frequency vào thẳng công thức (dạng thô
#     hoặc log), KHÔNG bão hoà: một từ lặp 20 lần đóng góp gấp ~20 lần so với lặp 1
#     lần. Độ dài tài liệu chỉ được xử lý gián tiếp qua chuẩn hoá L2 của vector, không
#     có tham số điều khiển riêng.
#   - BM25 thêm hai tham số: k1 làm BÃO HOÀ term frequency (tf/(tf+k1), lặp thêm thì
#     điểm tăng chậm dần rồi tiệm cận trần) và b CHUẨN HOÁ ĐỘ DÀI theo tỉ lệ |d|/avgdl
#     (tài liệu dài hơn trung bình bị phạt, ngắn hơn được thưởng).
#   → Với corpus chunk dài ngắn lệch nhau như bài này, BM25 ổn định hơn nên được chọn
#     làm ranker chính; TF-IDF giữ lại để đối chiếu.
# =============================================================================

def _get_tfidf_index() -> tuple[Optional[Any], Optional[Any]]:
    """Getter có cache cho (vectorizer, ma trận TF-IDF của corpus)."""
    global _TFIDF_VECTORIZER, _TFIDF_MATRIX
    if _TFIDF_VECTORIZER is not None and _TFIDF_MATRIX is not None:
        return _TFIDF_VECTORIZER, _TFIDF_MATRIX

    corpus = get_corpus()
    if not corpus:
        return None, None

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        _safe_print("  ⚠ Thiếu package scikit-learn (pip install scikit-learn) → bỏ qua TF-IDF")
        return None, None

    try:
        # Dùng chung _tokenize với BM25 để hai bảng xếp hạng so sánh được công bằng.
        # token_pattern=None: bắt buộc khi truyền tokenizer riêng (sklearn >= 1.2).
        vectorizer = TfidfVectorizer(tokenizer=_tokenize, token_pattern=None, lowercase=True)
        matrix = vectorizer.fit_transform([str(doc.get("content", "")) for doc in corpus])
    except Exception as exc:
        _safe_print(f"  ⚠ Lỗi khi dựng TF-IDF index ({type(exc).__name__}: {exc})")
        return None, None

    _TFIDF_VECTORIZER, _TFIDF_MATRIX = vectorizer, matrix
    return _TFIDF_VECTORIZER, _TFIDF_MATRIX


def tfidf_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa bằng TF-IDF + cosine similarity (bonus, để đối chiếu với BM25).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # cosine similarity trên không gian TF-IDF, [0, 1]
            'metadata': dict
        }
        Sorted by score descending.
    """
    if not query or not str(query).strip() or top_k <= 0:
        return []

    corpus = get_corpus()
    if not corpus:
        return []

    vectorizer, matrix = _get_tfidf_index()
    if vectorizer is None or matrix is None:
        return []

    try:
        from sklearn.metrics.pairwise import cosine_similarity

        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, matrix)[0]
    except Exception as exc:
        _safe_print(f"  ⚠ Lỗi khi tính TF-IDF similarity ({type(exc).__name__}: {exc})")
        return []

    results: list[dict] = []
    for idx, raw_score in enumerate(similarities):
        score = round(float(raw_score), 4)  # numpy.float64 → float thường
        if score <= 0:
            continue
        results.append({
            "content": corpus[idx]["content"],
            "score": score,
            "metadata": corpus[idx]["metadata"],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Console Windows mặc định cp1252 → ép UTF-8 để in được corpus tiếng Việt.
    # Chạy: python -m src.task6_lexical_search (từ thư mục gốc dự án)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # Test — demo cả BM25 lẫn TF-IDF trên vài query song ngữ
    demo_queries = [
        "phương thức thanh toán shopee",
        "What payment methods are supported?",
        "return refund policy evidence",
        "seller listing regulations",
        "order tracking",
    ]

    _safe_print("=" * 60)
    _safe_print(f"Task 6: Lexical Search — corpus: {len(get_corpus())} chunks")
    _safe_print("=" * 60)

    for query in demo_queries:
        _safe_print(f"\nQuery: {query}")
        _safe_print("-" * 60)

        _safe_print("  [BM25] — có bão hoà TF (k1) + chuẩn hoá độ dài (b)")
        results = lexical_search(query, top_k=3)
        if not results:
            _safe_print("    (không có kết quả)")
        for r in results:
            _safe_print(f"    [{r['score']:.3f}] {r['content'][:100]}...")

        _safe_print("  [TF-IDF] — cosine trên vector TF*IDF, không bão hoà TF")
        tfidf_results = tfidf_search(query, top_k=3)
        if not tfidf_results:
            _safe_print("    (không có kết quả)")
        for r in tfidf_results:
            _safe_print(f"    [{r['score']:.3f}] {r['content'][:100]}...")
