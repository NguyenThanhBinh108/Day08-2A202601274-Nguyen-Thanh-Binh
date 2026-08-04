"""
Task 6 — Lexical Search Module (BM25 + TF-IDF bonus).

Sparse Retrieval: tìm kiếm theo từ khoá CHÍNH XÁC (khác Semantic Search — Task 5 —
vốn dựa trên ý nghĩa). Hiệu quả với số hiệu, mã tài liệu, tên riêng, thuật ngữ kỹ
thuật mà embedding dễ "pha loãng".

Cài đặt:
    pip install rank-bm25 scikit-learn   # scikit-learn chỉ cần cho bonus TF-IDF

BM25 hoạt động thế nào (Okapi BM25):
    score(q,d) = Σ_{qᵢ ∈ q} IDF(qᵢ) * ( tf(qᵢ,d) * (k₁+1) ) / ( tf(qᵢ,d) + k₁*(1-b+b*|d|/avgdl) )
    - TF (Term Frequency): từ xuất hiện nhiều trong doc → điểm cao, nhưng bão hoà
      nhờ k₁=1.5 (từ xuất hiện 100 lần không gấp 100 lần 1 lần).
    - IDF (Inverse Document Frequency): từ hiếm → quan trọng hơn.
    - Length normalization: |d|/avgdl với b=0.75 → doc dài không bị ưu tiên quá mức.

TF-IDF (bonus, dùng scikit-learn):
    tfidf(t,d) = tf(t,d) * idf(t)   với idf(t) = ln((N+1)/(df(t)+1)) + 1
    Truy vấn được vector-hoá cùng vocab, tính cosine similarity với từng chunk.
    Khác BM25 ở chỗ không có length-normalization nhưng dễ hiểu, thuần TF*IDF.
    (Giải thích sự khác biệt này trong buổi demo → +5 điểm bonus.)

Tokenization cho tiếng Việt: tiếng Việt tách từ bằng khoảng trắng (khác tiếng
Trung/Nhật), nên `split()` trên token hợp lệ (letters + digits) là đủ cho BM25.
Có thể nâng cấp bằng `underthesea` (word tokenizer) nếu muốn, nhưng sẽ làm chậm
và thêm dependency không cần thiết.
"""

import re
from pathlib import Path

from rank_bm25 import BM25Okapi

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Chunking config — khớp với Task 4 để index BM25 trên CÙNG đơn vị văn bản
# mà vector store dùng (nếu corpus khác nhau, kết quả hybrid merge sẽ lệch).
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Các hằng số BM25 chuẩn từ paper Robertson & Zaragoza (2009)
BM25_K1 = 1.5     # term saturation
BM25_B = 0.75     # document length normalization

# =============================================================================
# Tiền xử lý văn bản (normalize + tokenize)
# =============================================================================

_TOKEN_RE = re.compile(r"[a-zA-Z0-9\u00C0-\u1EF9]+")


def normalize_text(text: str) -> str:
    """
    Chuẩn hoá văn bản: lowercase + giữ lại letters/digits (bao gồm ký tự có
    dấu tiếng Việt trong dải Unicode \u00C0-\u1EF9), loại bỏ dấu câu.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    return " ".join(tokens)


def tokenize(text: str) -> list[str]:
    """Tokenize văn bản tiếng Việt: normalize rồi tách theo khoảng trắng."""
    return normalize_text(text).split()


# =============================================================================
# Đọc corpus
# =============================================================================

def load_corpus() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/ và chunk chúng.

    Self-contained: không phụ thuộc vào Task 4 (dễ chạy độc lập, không crash
    khi Role khác chưa xong Task 4). Nếu sau này Task 4 đã index vào
    chroma_db/, vẫn ưu tiên đọc từ standardized/ để tránh coupling cứng.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents: list[dict] = []
    md_files = sorted(STANDARDIZED_DIR.rglob("*.md")) if STANDARDIZED_DIR.exists() else []
    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })

    # Chunk từng document giống cấu hình Task 4 (recursive character splitter)
    chunks: list[dict] = []
    for doc in documents:
        text = doc["content"]
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            if end < len(text):
                end = text.rfind(" ", start, end)
                if end == -1:
                    end = min(start + CHUNK_SIZE, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "content": chunk_text,
                    "metadata": {**doc["metadata"], "chunk_index": idx},
                })
                idx += 1
            if end >= len(text):
                break
            start = end - CHUNK_OVERLAP
    return chunks


# =============================================================================
# Index (lazy singleton — build 1 lần, tái sử dụng)
# =============================================================================

_corpus: list[dict] | None = None
_bm25: BM25Okapi | None = None
_tfidf_vectorizer = None     # TfidfVectorizer
_tfidf_matrix = None         # sparse matrix (n_chunks x vocab)


def _ensure_index():
    """Build lazily BM25 + TF-IDF index trên corpus hiện tại (cache global)."""
    global _corpus, _bm25, _tfidf_vectorizer, _tfidf_matrix
    if _corpus is None:
        _corpus = load_corpus()
        if not _corpus:
            return
        _bm25 = build_bm25_index(_corpus)
        _tfidf_vectorizer, _tfidf_matrix = build_tfidf_index(_corpus)


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus, k1=BM25_K1, b=BM25_B)


def build_tfidf_index(corpus: list[dict]):
    """
    (Bonus) Xây dựng TF-IDF index bằng scikit-learn.

    Returns:
        (TfidfVectorizer, sparse_matrix): matrix shape (n_chunks, n_terms)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
    matrix = vectorizer.fit_transform([doc["content"] for doc in corpus])
    return vectorizer, matrix


# =============================================================================
# Search
# =============================================================================

def lexical_search(
    query: str,
    top_k: int = 10,
    method: str = "bm25",   # "bm25" | "tfidf"
) -> list[dict]:
    """
    Tìm kiếm từ khoá — mặc định BM25, hỗ trợ TF-IDF (bonus).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa
        method: "bm25" (mặc định) hoặc "tfidf"

    Returns:
        List of {
            'content': str,
            'score': float,       # BM25 hoặc cosine-similarity score
            'metadata': dict
        }
        Sorted by score descending. Bỏ các kết quả score <= 0 (không match từ nào).
    """
    global _corpus
    _ensure_index()
    if not _corpus or _bm25 is None:
        return []

    if method == "tfidf":
        return _search_tfidf(query, top_k)
    return _search_bm25(query, top_k)


def _search_bm25(query: str, top_k: int) -> list[dict]:
    tokenized_query = tokenize(query)
    if not tokenized_query:
        return []
    scores = _bm25.get_scores(tokenized_query)

    # Lấy top_k indices có score > 0
    ranked = sorted(
        ((i, float(s)) for i, s in enumerate(scores) if s > 0),
        key=lambda x: x[1], reverse=True,
    )
    results = []
    for idx, score in ranked[:top_k]:
        results.append({
            "content": _corpus[idx]["content"],
            "score": score,
            "metadata": _corpus[idx]["metadata"],
        })
    return results


def _search_tfidf(query: str, top_k: int) -> list[dict]:
    import numpy as np

    q_vec = _tfidf_vectorizer.transform([query])
    sims = (_tfidf_matrix @ q_vec.T).toarray().ravel()  # cosine (đã chuẩn hoá L2)

    ranked = sorted(
        ((i, float(s)) for i, s in enumerate(sims) if s > 0),
        key=lambda x: x[1], reverse=True,
    )
    results = []
    for idx, score in ranked[:top_k]:
        results.append({
            "content": _corpus[idx]["content"],
            "score": round(score, 4),
            "metadata": _corpus[idx]["metadata"],
        })
    return results


def reset_index():
    """Reset cache index — gọi khi corpus thay đổi (thêm/sửa documents)."""
    global _corpus, _bm25, _tfidf_vectorizer, _tfidf_matrix
    _corpus = None
    _bm25 = None
    _tfidf_vectorizer = None
    _tfidf_matrix = None


if __name__ == "__main__":
    # Test
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    print(f"BM25 — {len(results)} results:")
    for r in results:
        print(f"  [{r['score']:.3f}] {r['content'][:100]}...")

    print("\nTF-IDF (bonus):")
    for r in lexical_search("phương thức thanh toán shopee", top_k=5, method="tfidf"):
        print(f"  [{r['score']:.3f}] {r['content'][:100]}...")
