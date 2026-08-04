"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement — khuyến nghị vì không cần API key

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó.
"""

import math
import sys
from typing import Any, Optional, Sequence, Union


# =============================================================================
# CONFIG + CACHE MODEL (LAZY)
# =============================================================================
# Nguyên tắc: KHÔNG load model ở cấp module. File này bị import bởi pytest và bởi
# Task 9, nên import phải nhanh và không bao giờ crash khi máy offline / chưa cache
# model. Model chỉ được tải lần đầu tiên khi thật sự cần, rồi giữ lại ở biến
# module-level để các lần gọi sau dùng lại (tránh load đi load lại).

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Dùng lại đúng embedding model của Task 4 để vector query/doc nằm cùng không gian
# (và để không phải tải thêm model thứ hai về máy).
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"

_cross_encoder: Optional[Any] = None
_cross_encoder_name: Optional[str] = None
_embedding_model: Optional[Any] = None
_embedding_model_name: Optional[str] = None


def _get_cross_encoder(model_name: str = DEFAULT_CROSS_ENCODER_MODEL) -> Any:
    """
    Lazy-load cross-encoder và cache lại.

    Có thể raise (offline, thiếu model, lỗi mạng) — người gọi PHẢI bọc try/except.
    """
    global _cross_encoder, _cross_encoder_name

    if _cross_encoder is not None and _cross_encoder_name == model_name:
        return _cross_encoder

    # import nằm trong hàm: sentence_transformers rất nặng (kéo theo torch),
    # import ở cấp module sẽ làm mọi lần `import src.task7_reranking` chậm vài giây.
    from sentence_transformers import CrossEncoder

    _cross_encoder = CrossEncoder(model_name)
    _cross_encoder_name = model_name
    return _cross_encoder


def _get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Any:
    """
    Lazy-load SentenceTransformer dùng cho MMR và cache lại.

    Có thể raise — người gọi PHẢI bọc try/except.
    """
    global _embedding_model, _embedding_model_name

    if _embedding_model is not None and _embedding_model_name == model_name:
        return _embedding_model

    from sentence_transformers import SentenceTransformer

    _embedding_model = SentenceTransformer(model_name)
    _embedding_model_name = model_name
    return _embedding_model


# =============================================================================
# HELPERS
# =============================================================================

def _ensure_score(item: dict, default: float = 0.0) -> dict:
    """
    Copy nông 1 candidate và đảm bảo có key 'score' kiểu float.

    Mọi hàm rerank đều phải trả về dict có 'score' (test kiểm tra trực tiếp),
    kể cả khi đi vào nhánh fallback.
    """
    result = dict(item)
    try:
        result["score"] = float(result.get("score", default))
    except (TypeError, ValueError):
        result["score"] = default
    return result


def _fallback(candidates: list[dict], top_k: int) -> list[dict]:
    """Fallback an toàn: giữ nguyên thứ tự đầu vào, chỉ cắt top_k và chuẩn hoá score."""
    return [_ensure_score(c) for c in candidates[: max(0, top_k)] if isinstance(c, dict)]


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """
    Cosine similarity thuần Python (không cần numpy) — đủ nhanh với vài chục candidates.

    Trả về 0.0 nếu vector rỗng, lệch chiều, hoặc có norm = 0 (tránh chia cho 0).
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _as_vector(value: Any) -> Optional[list[float]]:
    """Ép 1 giá trị bất kỳ về list[float] nếu nó trông giống vector, ngược lại None."""
    if value is None or isinstance(value, (str, bytes, dict)):
        return None
    try:
        vector = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    return vector or None


def _encode(texts: list[str], model_name: str = DEFAULT_EMBEDDING_MODEL) -> list[list[float]]:
    """Embed danh sách text bằng SentenceTransformer (lazy). Có thể raise."""
    model = _get_embedding_model(model_name)
    encoded = model.encode(texts, normalize_embeddings=True)
    return [[float(x) for x in vector] for vector in encoded]


# =============================================================================
# RERANKERS
# =============================================================================

def rerank_cross_encoder(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank
        model_name: Tên cross-encoder trên HuggingFace Hub

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    # Ghi chú lựa chọn kỹ thuật — 3 hướng khả thi:
    #   Option A: Jina Reranker API (POST https://api.jina.ai/v1/rerank,
    #             model "jina-reranker-v2-base-multilingual") → cần JINA_API_KEY.
    #   Option B: Local model Qwen3-Reranker qua transformers → rất nặng.
    #   Option C (chọn): CrossEncoder của sentence-transformers, model MiniLM nhỏ
    #             (~90MB), chạy offline sau lần tải đầu, không cần API key.
    # Cross-encoder đọc CẶP (query, document) cùng lúc nên chính xác hơn bi-encoder,
    # nhưng chi phí O(n) forward pass → chỉ dùng để rerank vài chục candidate.
    if not candidates:
        return []

    usable = [c for c in candidates if isinstance(c, dict)]
    if not usable:
        return []

    try:
        model = _get_cross_encoder(model_name)
    except Exception as exc:  # offline, thiếu dependency, lỗi mạng...
        # KHÔNG raise: degrade gracefully về thứ tự retrieval gốc.
        print(
            f"  [WARN] rerank_cross_encoder: khong tai duoc model '{model_name}' "
            f"({type(exc).__name__}: {exc}). Giu nguyen thu tu candidates goc."
        )
        return _fallback(usable, top_k)

    try:
        pairs = [(query, str(c.get("content", ""))) for c in usable]
        raw_scores = model.predict(pairs)

        reranked: list[dict] = []
        for candidate, raw_score in zip(usable, raw_scores):
            item = dict(candidate)
            item["score"] = float(raw_score)
            reranked.append(item)
    except Exception as exc:
        print(
            f"  [WARN] rerank_cross_encoder: loi khi predict "
            f"({type(exc).__name__}: {exc}). Giu nguyen thu tu candidates goc."
        )
        return _fallback(usable, top_k)

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked[: max(0, top_k)]


def rerank_mmr(
    query: Union[str, Sequence[float]],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.5,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query: Câu truy vấn (str) HOẶC vector embedding của query đã tính sẵn
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates:
        return []

    usable = [c for c in candidates if isinstance(c, dict)]
    if not usable:
        return []

    limit = min(max(0, top_k), len(usable))
    if limit == 0:
        return []

    # --- Bước 1: chuẩn bị vector -------------------------------------------
    # Ưu tiên embedding có sẵn trong candidate (khỏi encode lại). Nếu thiếu dù chỉ
    # 1 candidate thì encode toàn bộ để mọi vector nằm cùng không gian.
    try:
        doc_vectors: list[Optional[list[float]]] = [
            _as_vector(c.get("embedding")) for c in usable
        ]
        query_vector = _as_vector(query) if not isinstance(query, str) else None

        need_encode = query_vector is None or any(v is None for v in doc_vectors)
        if need_encode:
            texts = [str(c.get("content", "")) for c in usable]
            if isinstance(query, str):
                vectors = _encode([query] + texts)
                query_vector = vectors[0]
                doc_vectors = list(vectors[1:])
            else:
                doc_vectors = list(_encode(texts))

        # Query là vector cho sẵn nhưng lệch chiều với doc → không so sánh được.
        if query_vector is None or any(
            v is None or len(v) != len(query_vector) for v in doc_vectors
        ):
            raise ValueError("query/document embedding khong cung so chieu")
    except Exception as exc:
        # KHÔNG raise: thiếu model/offline thì trả về thứ tự gốc.
        print(
            f"  [WARN] rerank_mmr: khong tao duoc embedding "
            f"({type(exc).__name__}: {exc}). Giu nguyen thu tu candidates goc."
        )
        return _fallback(usable, limit)

    # --- Bước 2: vòng lặp MMR ----------------------------------------------
    # Mỗi lượt chọn 1 doc tối đa hoá: λ*relevance - (1-λ)*max_sim_to_selected.
    # λ→1 nghiêng về relevance thuần, λ→0 nghiêng về diversity thuần.
    relevance = [_cosine_similarity(query_vector, v) for v in doc_vectors]

    selected: list[int] = []
    selected_scores: list[float] = []
    remaining = list(range(len(usable)))

    for _ in range(limit):
        best_idx: Optional[int] = None
        best_score = float("-inf")

        for idx in remaining:
            # Độ tương đồng lớn nhất với các doc ĐÃ chọn = mức độ trùng lặp.
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = _cosine_similarity(doc_vectors[idx], doc_vectors[sel_idx])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = (
                lambda_param * relevance[idx]
                - (1.0 - lambda_param) * max_sim_to_selected
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break

        selected.append(best_idx)
        selected_scores.append(best_score)
        remaining.remove(best_idx)

    results: list[dict] = []
    for idx, score in zip(selected, selected_scores):
        item = dict(usable[idx])
        item["score"] = float(score)  # score = điểm MMR tại lúc được chọn
        results.append(item)

    # Điểm MMR của lượt sau có thể cao hơn lượt trước (penalty thay đổi theo tập đã
    # chọn), nên sort lại giảm dần cho đúng hợp đồng chung "score sorted descending".
    # sorted() ổn định → các điểm bằng nhau vẫn giữ đúng thứ tự MMR đã chọn.
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    # RRF chỉ nhìn THỨ HẠNG, không nhìn điểm gốc → gộp được các ranker có thang điểm
    # khác nhau hoàn toàn (cosine 0..1 của dense vs BM25 0..30 của sparse) mà không
    # cần normalize. Đổi lại: điểm ra KHÔNG mang ý nghĩa độ liên quan (xem docstring
    # module + cảnh báo ở Task 9).
    if not ranked_lists:
        return []

    rrf_scores: dict[str, float] = {}   # content -> điểm RRF cộng dồn
    content_map: dict[str, dict] = {}   # content -> dict đầy đủ (lần gặp ĐẦU TIÊN)

    for ranked_list in ranked_lists:
        if not ranked_list:  # bỏ qua list rỗng / None
            continue

        for rank, item in enumerate(ranked_list, start=1):
            if not isinstance(item, dict):
                continue

            key = item.get("content")
            if not isinstance(key, str):
                continue

            denominator = k + rank
            if denominator <= 0:  # phòng trường hợp truyền k âm
                continue

            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / denominator
            # Giữ metadata của lần gặp đầu tiên (ranker đứng trước trong ranked_lists).
            if key not in content_map:
                content_map[key] = item

    if not rrf_scores:
        return []

    # sorted() ổn định → cùng điểm thì giữ thứ tự xuất hiện, kết quả tất định.
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results: list[dict] = []
    for content, score in sorted_items[: max(0, top_k)]:
        item = dict(content_map[content])  # copy để không sửa dict đầu vào
        item["score"] = float(score)
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

VALID_METHODS = ("cross_encoder", "mmr", "rrf")


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if not candidates:
        return []

    if method not in VALID_METHODS:
        print(
            f"  [WARN] rerank: method '{method}' khong hop le "
            f"(hop le: {', '.join(VALID_METHODS)}). Dung mac dinh 'rrf'."
        )
        method = "rrf"

    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k=top_k)

    if method == "mmr":
        # rerank_mmr tự embed query (lazy) nên ở đây truyền thẳng chuỗi query.
        return rerank_mmr(query, candidates, top_k=top_k)

    # method == "rrf": interface này chỉ có MỘT danh sách, coi nó như 1 ranker duy
    # nhất → RRF giữ nguyên thứ tự đầu vào và chỉ đổi thang điểm. Muốn fuse nhiều
    # nguồn (dense + sparse như Task 9) thì gọi thẳng rerank_rrf([list1, list2]).
    results = rerank_rrf([candidates], top_k=top_k)
    if not results:
        # candidates thiếu key 'content' → vẫn phải trả kết quả có 'score'.
        return _fallback([c for c in candidates if isinstance(c, dict)], top_k)
    return results


if __name__ == "__main__":
    # Console Windows mặc định dùng cp1252 → print tiếng Việt sẽ UnicodeEncodeError.
    # Chỉ ép stdout về utf-8 trong nhánh demo, không ảnh hưởng khi module bị import.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError, ValueError):
        pass

    # Test with dummy data
    dummy_candidates = [
        {"content": "Chính sách trả hàng và hoàn tiền Shopee trong 15 ngày", "score": 0.8, "metadata": {}},
        {"content": "Các phương thức thanh toán hỗ trợ trên Shopee Vietnam", "score": 0.6, "metadata": {}},
        {"content": "Quy định đăng bán sản phẩm dành cho người bán", "score": 0.5, "metadata": {}},
    ]
    results = rerank("chính sách trả hàng shopee", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
