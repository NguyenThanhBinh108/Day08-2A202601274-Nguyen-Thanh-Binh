"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results

⚠️ BẪY THƯỜNG GẶP — đọc kỹ trước khi code:
    Nếu bạn dùng điểm RRF đã fuse (Task 7) để so với score_threshold, bạn sẽ gặp bug
    thật: RRF max score luôn ≈ 1/(k+1) ≈ 0.0164 (k=60) BẤT KỂ nội dung có liên quan
    hay không. Nếu đặt threshold thấp (như 0.005) để "hợp" với thang điểm RRF, thực
    chất KHÔNG câu hỏi nào đủ thấp để trigger fallback nữa — kể cả query hoàn toàn vô
    nghĩa vẫn trả về kết quả "hybrid" (rác) thay vì fallback đúng như thiết kế.

    Cách sửa đúng: giữ điểm cosine similarity GỐC của semantic_search (trước khi qua
    RRF) làm căn cứ quyết định fallback, tách biệt khỏi điểm RRF dùng để sắp xếp kết
    quả cuối cùng. Calibrate threshold bằng cách tự đo: chạy vài câu hỏi chắc chắn
    liên quan và vài câu chắc chắn lạc đề/rác qua semantic_search, xem khoảng cách
    điểm số giữa hai nhóm rồi chọn ngưỡng nằm giữa.
"""

from typing import Any, Callable

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# Calibrate threshold này bằng cách tự đo điểm cosine của semantic_search
# cho câu hỏi liên quan vs câu hỏi lạc đề (xem ghi chú ở trên) — ĐỪNG copy nguyên
# giá trị mẫu, mỗi corpus/embedding model sẽ cho khoảng điểm khác nhau.
#
# 0.48 là con số lấy từ tài liệu lab (BAAI/bge-m3, corpus e-commerce support), CHƯA
# phải con số đúng cho corpus của nhóm. Cách calibrate bắt buộc phải làm:
#   1. Chọn ~5 câu chắc chắn liên quan ("What payment methods are supported?", ...)
#      và ~5 câu chắc chắn lạc đề/rác ("xyzabc123nonsense", "công thức nấu phở", ...)
#   2. Chạy semantic_search(q, top_k=1) cho từng câu, ghi lại score cosine GỐC
#   3. Nhóm liên quan thường tụ ở một khoảng cao, nhóm lạc đề tụ ở khoảng thấp →
#      đặt SCORE_THRESHOLD vào giữa hai khoảng đó (nếu hai khoảng chồng lấn thì
#      embedding/chunking đang có vấn đề, sửa Task 4 trước khi chỉnh threshold).
SCORE_THRESHOLD = 0.48  # Nếu best score (cosine gốc) < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # "cross_encoder" | "mmr" | "rrf"

# Hệ số mở rộng candidate pool trước khi rerank: lấy dư (top_k * 2) ở tầng retrieval
# để reranker còn "đất" mà đảo thứ hạng, sau đó mới cắt về đúng top_k.
CANDIDATE_MULTIPLIER = 2


# =============================================================================
# HELPERS — mọi thứ ở đây tồn tại để pipeline KHÔNG BAO GIỜ raise ra ngoài.
# Task 9 nằm ở giữa chuỗi phụ thuộc (5, 6, 7, 8): chỉ cần một module con chưa
# implement / thiếu API key / thiếu chroma_db là cả pipeline sẽ chết nếu không
# bọc lỗi. Thiết kế ở đây là degrade gracefully: nhánh nào hỏng thì bỏ nhánh đó.
# =============================================================================

def _warn(message: str) -> None:
    """
    In cảnh báo ra stdout mà KHÔNG BAO GIỜ raise.

    Console Windows mặc định dùng codepage cp1252, không encode được "⚠" hay ký tự
    tiếng Việt có dấu → print() thường sẽ ném UnicodeEncodeError và giết cả pipeline
    (đúng ngay trong except block, chỗ khó debug nhất). Fallback sang dạng escape
    ASCII để thông điệp vẫn tới được người dùng.
    """
    try:
        print(message)
    except UnicodeEncodeError:
        try:
            print(message.encode("ascii", "backslashreplace").decode("ascii"))
        except Exception:  # noqa: BLE001 — log lỗi không bao giờ được làm hỏng pipeline
            pass
    except Exception:  # noqa: BLE001
        pass


def _normalize_results(raw: Any, source: str | None = None) -> list[dict]:
    """
    Chuẩn hoá output của một retriever con về đúng schema của lab.

    Bỏ qua mọi phần tử không hợp lệ (không phải dict, content rỗng) thay vì raise,
    ép 'score' về float và đảm bảo 'metadata' luôn là dict.

    Args:
        raw: Giá trị thô do retriever con trả về (có thể là bất cứ thứ gì)
        source: Nếu khác None thì ghi đè key 'source' cho mọi item

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, ...}
    """
    if not isinstance(raw, list):
        return []

    normalized: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue

        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0

        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        clean = dict(item)
        clean["content"] = content
        clean["score"] = score
        clean["metadata"] = metadata
        if source is not None:
            clean["source"] = source
        normalized.append(clean)

    return normalized


def _safe_call(
    fn: Callable[..., Any],
    description: str,
    *args: Any,
    source: str | None = None,
    **kwargs: Any,
) -> list[dict]:
    """
    Gọi một retriever/reranker con và nuốt mọi lỗi thành list rỗng.

    NotImplementedError được tách riêng vì đó là trạng thái bình thường khi bạn
    làm lab theo thứ tự (Task 9 xong trước Task 8 chẳng hạn) — in cảnh báo rồi đi tiếp.

    Args:
        fn: Hàm cần gọi
        description: Tên hiển thị trong cảnh báo
        source: Truyền xuống _normalize_results để gắn nhãn nguồn

    Returns:
        Kết quả đã chuẩn hoá, hoặc [] nếu hàm lỗi.
    """
    try:
        raw = fn(*args, **kwargs)
    except NotImplementedError:
        _warn(f"  ⚠ {description}: chưa được implement — bỏ qua nhánh này.")
        return []
    except Exception as exc:  # noqa: BLE001 — pipeline phải sống sót mọi lỗi con
        _warn(f"  ⚠ {description}: lỗi {type(exc).__name__}: {exc} — bỏ qua nhánh này.")
        return []

    return _normalize_results(raw, source=source)


def _coerce_top_k(top_k: Any) -> int:
    """
    Ép top_k về int dương; trả 0 nếu giá trị vô nghĩa (None, "abc", -3, 3.7 → 3).

    Chấp nhận cả float/str số để hàm dễ dùng từ notebook hay Streamlit (app.py đọc
    slider có thể ra float), nhưng không bao giờ raise ValueError ra ngoài.
    """
    try:
        value = int(top_k)
    except (TypeError, ValueError):
        return 0
    return value if value > 0 else 0


def _sort_desc(results: list[dict]) -> list[dict]:
    """
    Sort giảm dần theo score (yêu cầu bắt buộc của schema).

    Dùng sorted() (stable) nên các item cùng điểm giữ nguyên thứ tự tương đối.
    Bỏ qua khi RERANK_METHOD == "mmr": MMR cố ý xếp theo thứ tự chọn (relevance +
    diversity), sort lại theo score sẽ phá mất tính đa dạng vừa đạt được.
    """
    if RERANK_METHOD == "mmr":
        return results
    return sorted(results, key=lambda item: item.get("score", 0.0), reverse=True)


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → dense_results (giữ điểm cosine gốc)
          ├→ Lexical Search  → sparse_results
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If dense_results[0]["score"] < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu (KHÔNG phải điểm RRF)
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Guard đầu vào: query rỗng / không phải str, hoặc top_k <= 0 → không có gì để làm.
    if not isinstance(query, str) or not query.strip():
        _warn("  ⚠ Query rỗng hoặc không hợp lệ — trả về list rỗng.")
        return []
    top_k = _coerce_top_k(top_k)
    if top_k <= 0:
        _warn("  ⚠ top_k không hợp lệ (phải là số nguyên dương) — trả về list rỗng.")
        return []

    try:
        threshold = float(score_threshold)
    except (TypeError, ValueError):
        threshold = SCORE_THRESHOLD

    candidate_k = top_k * CANDIDATE_MULTIPLIER

    # Step 1: Song song chạy semantic + lexical
    # (chạy tuần tự trong process này; hai nhánh độc lập nhau nên nhánh nào lỗi
    #  thì nhánh còn lại vẫn dùng được)
    dense_results = _safe_call(
        semantic_search, "semantic_search (Task 5)", query, top_k=candidate_k
    )
    sparse_results = _safe_call(
        lexical_search, "lexical_search (Task 6)", query, top_k=candidate_k
    )

    # Step 2: Merge bằng RRF
    # RRF chỉ dùng THỨ HẠNG nên gộp được hai thang điểm khác nhau hoàn toàn
    # (cosine ∈ [0,1] vs BM25 ∈ [0, +∞)) mà không cần normalize thủ công.
    ranked_lists = [lst for lst in (dense_results, sparse_results) if lst]
    if ranked_lists:
        merged = _safe_call(
            rerank_rrf,
            "rerank_rrf (Task 7)",
            ranked_lists,
            top_k=candidate_k,
            source="hybrid",
        )
    else:
        merged = []

    # Nếu RRF chưa implement / lỗi mà vẫn còn kết quả dense hoặc sparse thì đừng
    # vứt đi — dùng tạm nhánh dense (rồi tới sparse) làm merged.
    if not merged:
        fallback_merge = dense_results or sparse_results
        merged = _normalize_results(fallback_merge, source="hybrid")

    # Step 3: Rerank
    #
    # KHÔNG chạy RRF lần thứ hai. Bước 2 đã fuse dense + sparse, và điểm sau khi
    # fuse mang thông tin thật: tài liệu được CẢ HAI ranker bình chọn sẽ cộng hai
    # phiếu (≈ 1/61 + 1/64 ≈ 0,032), gấp đôi tài liệu chỉ một ranker tìm ra
    # (≈ 0,016). Đo trên corpus này, dense và sparse trùng nhau 8/10 chunk nên
    # chênh lệch đó là tín hiệu xếp hạng có giá trị.
    #
    # Nếu đem `merged` (một danh sách duy nhất) chạy qua rerank_rrf lần nữa thì
    # mỗi tài liệu lại chỉ còn đúng một phiếu, điểm bị ghi đè thành 1/(60+rank)
    # đều tăm tắp — thứ tự vẫn giữ nhưng thông tin "được mấy ranker đồng thuận"
    # mất sạch, và thanh đo điểm trên giao diện thành vô nghĩa.
    #
    # Cross-encoder và MMR thì khác: chúng chấm lại bằng ngữ nghĩa nên vẫn chạy.
    if use_reranking and merged and RERANK_METHOD != "rrf":
        final_results = _safe_call(
            rerank,
            f"rerank(method={RERANK_METHOD}) (Task 7)",
            query,
            merged,
            top_k=top_k,
            method=RERANK_METHOD,
        )
        if not final_results:
            # Reranker chưa sẵn sàng → giữ nguyên thứ tự RRF thay vì trả rỗng.
            final_results = merged[:top_k]
    else:
        final_results = merged[:top_k]

    # rerank() có quyền trả về dict MỚI (ví dụ cross-encoder dựng lại item từ API
    # response) nên key 'source' có thể bị mất — set lại cho chắc.
    final_results = _normalize_results(final_results, source="hybrid")
    final_results = _sort_desc(final_results)

    # Step 4: Check threshold DÙNG ĐIỂM COSINE GỐC (dense_results), KHÔNG PHẢI RRF
    # Đây chính là cái bẫy mô tả ở docstring đầu file: điểm RRF của top-1 luôn
    # ≈ 1/(k+1) ≈ 0.0164 bất kể liên quan hay không, so với threshold là vô nghĩa.
    best_score = dense_results[0]["score"] if dense_results else 0.0
    if best_score < threshold:
        _warn(f"  ⚠ Semantic best score ({best_score:.3f}) < threshold ({threshold})")
        fallback = _safe_call(
            pageindex_search,
            "pageindex_search (Task 8)",
            query,
            top_k=top_k,
            source="pageindex",
        )
        if fallback:
            return _sort_desc(fallback)[:top_k]
        # Fallback rỗng (chưa có API key / chưa upload document) → vẫn trả kết quả
        # hybrid đang có, đừng trả rỗng khi trong tay còn dữ liệu dùng được.
        _warn("  ⚠ PageIndex fallback không có kết quả — giữ kết quả hybrid.")

    return final_results[:top_k]


def retrieve_dense_only(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Chỉ chạy semantic search (bỏ lexical + RRF + rerank + fallback).

    Mục đích: A/B testing ở phần evaluation. So sánh metric (recall@k, MRR, nDCG)
    giữa baseline dense-only này và retrieve() đầy đủ để CHỨNG MINH bằng số rằng
    hybrid + reranking thực sự tốt hơn, thay vì chỉ khẳng định suông trong báo cáo.
    Giữ nguyên nhãn source='hybrid' để downstream (Task 10) không phải xử lý thêm
    một giá trị source lạ — schema chỉ chấp nhận 'hybrid' | 'pageindex'.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': 'hybrid'}
        Sorted by cosine score descending.
    """
    if not isinstance(query, str) or not query.strip():
        _warn("  ⚠ Query rỗng hoặc không hợp lệ — trả về list rỗng.")
        return []
    top_k = _coerce_top_k(top_k)
    if top_k <= 0:
        _warn("  ⚠ top_k không hợp lệ (phải là số nguyên dương) — trả về list rỗng.")
        return []

    dense_results = _safe_call(
        semantic_search,
        "semantic_search (Task 5)",
        query,
        top_k=top_k,
        source="hybrid",
    )
    # semantic_search đã sort sẵn, sort lại một lần nữa cho chắc chắn đúng schema.
    return sorted(
        dense_results, key=lambda item: item.get("score", 0.0), reverse=True
    )[:top_k]


if __name__ == "__main__":
    test_queries = [
        "What payment methods does Shopee support?",
        "How do I request a return or refund?",
        "What evidence do I need for a refund request?",
        "xyzabc123nonsense",  # Query không có kết quả → test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
