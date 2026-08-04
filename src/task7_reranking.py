"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp (mỗi phương pháp đều đã implement):
    - Cross-encoder reranker: Jina Reranker v2 (API) hoặc local CrossEncoder
    - MMR (Maximal Marginal Relevance): tự implement — cân bằng relevance & diversity
    - RRF (Reciprocal Rank Fusion): tự implement — gộp thứ hạng từ nhiều ranker

Khuyến nghị: RRF (không cần API key, được dùng làm merge step trong Task 9).

⚠️ LƯU Ý QUAN TRỌNG về RRF (sẽ dùng lại ở Task 9):
    Điểm RRF fused CHỈ phụ thuộc thứ hạng, không phải độ tương đồng thật.
    Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60), bất kể nội dung đó có
    thật sự liên quan hay không. Đừng dùng điểm RRF để quyết định fallback ở
    Task 9 — hãy dùng điểm cosine similarity GỐC của semantic_search.
"""

import math
import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY", "")

_TOKEN_RE = re.compile(r"[a-zA-Z0-9\u00C0-\u1EF9]+")


def _tokenize(text: str) -> set[str]:
    """Tokenize (giống Task 6) — dùng cho lexical relevance fallback."""
    return set(_TOKEN_RE.findall(text.lower()))


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity giữa 2 vector."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _lexical_relevance(query: str, content: str) -> float:
    """Tỷ lệ từ khoá query xuất hiện trong content (dùng cho offline fallback)."""
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0
    d_tokens = _tokenize(content)
    if not d_tokens:
        return 0.0
    return len(q_tokens & d_tokens) / len(q_tokens)


# =============================================================================
# RRF — Reciprocal Rank Fusion (khuyến nghị cho hybrid merge)
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker, đã sort)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    # Sort by RRF score
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)
    return results


# =============================================================================
# MMR — Maximal Marginal Relevance
# =============================================================================

def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content', 'score', 'embedding', 'metadata'}
        top_k: Số lượng kết quả
        lambda_param: Trade-off relevance (1.0) ↔ diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR (có key 'mmr_score').
    """
    if top_k <= 0 or not candidates:
        return []

    selected_idx: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx: Optional[int] = None
        best_mmr = float("-inf")

        for idx in remaining:
            relevance = _cosine_sim(query_embedding, candidates[idx].get("embedding", []))

            max_sim_to_selected = 0.0
            for sel_idx in selected_idx:
                sim = _cosine_sim(
                    candidates[idx].get("embedding", []),
                    candidates[sel_idx].get("embedding", []),
                )
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx is None:
            break
        selected_idx.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for i in selected_idx:
        item = candidates[i].copy()
        mmr = lambda_param * _cosine_sim(
            query_embedding, candidates[i].get("embedding", [])
        ) - (1 - lambda_param) * max(
            (_cosine_sim(candidates[i].get("embedding", []), candidates[j].get("embedding", []))
             for j in selected_idx if j != i),
            default=0.0,
        )
        item["mmr_score"] = mmr
        results.append(item)
    return results


# =============================================================================
# Cross-encoder reranker (Jina API + local model)
# =============================================================================

def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Ưu tiên:
        1. Jina Reranker v2 (multilingual, tốt cho tiếng Việt) — cần JINA_API_KEY
        2. Local CrossEncoder (sentence-transformers) nếu đã cài
        3. Offline fallback: lexical relevance re-score (không cần API key)

    Returns:
        List of top_k candidates, re-scored (key 'score') và sorted descending.
    """
    if not candidates:
        return []

    # 1) Jina Reranker API
    if JINA_API_KEY:
        try:
            import requests

            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {JINA_API_KEY}"},
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": [c["content"] for c in candidates],
                    "top_n": top_k,
                },
                timeout=30,
            )
            response.raise_for_status()
            reranked = response.json()["results"]
            return [
                {**candidates[r["index"]], "score": r["relevance_score"]}
                for r in sorted(reranked, key=lambda x: x["relevance_score"], reverse=True)
            ]
        except Exception as e:  # network / API error → fall through
            print(f"  ⚠ Jina rerank failed ({e}), fallback: local/lexical")

    # 2) Local CrossEncoder
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, c["content"]) for c in candidates]
        scores = model.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [
            {**c, "score": float(s)} for s, c in ranked[:top_k]
        ]
    except Exception as e:
        print(f"  ⚠ Local CrossEncoder unavailable ({e}), fallback: lexical")

    # 3) Offline lexical relevance fallback
    print("  ⚠ No cross-encoder available — dùng lexical relevance re-score")
    scored = []
    for c in candidates:
        item = c.copy()
        item["score"] = round(_lexical_relevance(query, item["content"]), 4)
        scored.append(item)
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


# =============================================================================
# Unified rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
    alpha: float = 0.3,
) -> list[dict]:
    """
    Unified reranking interface.

    - method="rrf":
        * Nếu `candidates` là list các ranked lists (list[list[dict]]) → RRF fusion thật.
        * Nếu `candidates` là 1 list phẳng → re-score theo mức độ liên quan từ khoá
          với query (hybrid: điểm gốc + lexical overlap). Lý do: RRF trên 1 list chỉ
          là sắp theo thứ hạng có sẵn, không phải "rerank"; sau khi Task 9 đã merge
          xong, bước này thực sự nâng hạng kết quả liên quan query.
    - method="cross_encoder": gọi Jina API / local model (fallback lexical).
    - method="mmr": cần 'embedding' trong candidates — gọi rerank_mmr() riêng.

    Args:
        query: Câu truy vấn
        candidates: 1 ranked list (list[dict]) hoặc nhiều ranked lists (list[list[dict]])
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking
        alpha: Trọng số lexical overlap trong hybrid re-score (0 → chỉ dùng điểm gốc)

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)

    if method == "mmr":
        raise NotImplementedError(
            "Call rerank_mmr with query_embedding (candidates cần có key 'embedding')"
        )

    if method == "rrf":
        # Nhiều ranked lists → RRF fusion thật
        if candidates and isinstance(candidates[0], list):
            return rerank_rrf(candidates, top_k=top_k, k=60)

        # 1 list phẳng → hybrid re-score dựa trên query relevance
        if not candidates:
            return []
        max_orig = max((c.get("score", 0.0) or 0.0) for c in candidates)
        max_orig = max_orig or 1.0

        scored = []
        for c in candidates:
            item = c.copy()
            orig_norm = (item.get("score", 0.0) or 0.0) / max_orig
            lex = _lexical_relevance(query, item["content"])
            item["score"] = round((1 - alpha) * orig_norm + alpha * lex, 4)
            scored.append(item)
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]

    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test với dummy data
    dummy_candidates = [
        {"content": "Chính sách trả hàng và hoàn tiền Shopee trong 15 ngày", "score": 0.8, "metadata": {}},
        {"content": "Các phương thức thanh toán hỗ trợ trên Shopee Vietnam", "score": 0.6, "metadata": {}},
        {"content": "Quy định đăng bán sản phẩm dành cho người bán", "score": 0.5, "metadata": {}},
    ]
    results = rerank("chính sách trả hàng shopee", dummy_candidates, top_k=2)
    print("Hybrid re-score rerank:")
    for r in results:
        print(f"  [{r['score']:.3f}] {r['content']}")

    # RRF fusion demo với 2 ranker giả
    dense = [
        {"content": "Phương thức thanh toán", "score": 0.9, "metadata": {}},
        {"content": "Chính sách trả hàng", "score": 0.7, "metadata": {}},
    ]
    sparse = [
        {"content": "Chính sách trả hàng", "score": 12.3, "metadata": {}},
        {"content": "Quy định người bán", "score": 8.1, "metadata": {}},
    ]
    fused = rerank_rrf([dense, sparse], top_k=2)
    print("\nRRF fusion:")
    for r in fused:
        print(f"  [{r['score']:.4f}] {r['content']}")
