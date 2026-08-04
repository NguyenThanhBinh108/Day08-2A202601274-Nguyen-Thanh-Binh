"""
Task 5 — Semantic Search Module (Dense Retrieval + HyDE).

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4

Bonus — HyDE (Hypothetical Document Embeddings):
    - Sinh ra "tài liệu giả định" từ query trước khi embed
    - Cải thiện retrieval cho query ngắn hoặc mơ hồ
"""

import os
from pathlib import Path

# Dùng chung config và helpers từ Task 4
from src.task4_chunking_indexing import (
    COLLECTION_NAME,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    get_embedding_model,
    get_collection,
)


# =============================================================================
# SEMANTIC SEARCH (DENSE RETRIEVAL)
# =============================================================================

def semantic_search(query: str, top_k: int = 10, use_hyde: bool = False) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity (cosine).

    Args:
        query   : Câu truy vấn người dùng.
        top_k   : Số lượng kết quả tối đa trả về.
        use_hyde: Nếu True, dùng HyDE (sinh hypothetical doc trước khi embed).

    Returns:
        List of {
            'content' : str,    # Nội dung chunk
            'score'   : float,  # Cosine similarity [0, 1]
            'metadata': dict    # source, doc_id, type, customer_role, category, ...
        }
        Sorted by score descending.
    """
    # Chọn text để embed
    if use_hyde:
        search_text = _generate_hypothetical_document(query)
    else:
        search_text = query

    # Embed query
    model = get_embedding_model()
    query_vector = model.encode(
        search_text, normalize_embeddings=True
    ).tolist()

    # Query ChromaDB
    collection = get_collection()
    if collection.count() == 0:
        # ChromaDB chưa có data → trả rỗng, không crash
        return []

    n_results = min(top_k, collection.count())
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # Chuyển đổi kết quả
    output = []
    docs_list = results.get("documents", [[]])[0]
    metas_list = results.get("metadatas", [[]])[0]
    dists_list = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs_list, metas_list, dists_list):
        # ChromaDB trả cosine DISTANCE [0, 2] với hnsw:space=cosine
        # Convert sang similarity [0, 1]: score = 1 - distance/2
        score = max(0.0, 1.0 - dist / 2.0) if dist <= 2.0 else max(0.0, 1.0 - dist)
        output.append({
            "content" : doc,
            "score"   : round(score, 4),
            "metadata": meta,
        })

    # Sort descending theo score
    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


# =============================================================================
# HyDE — Hypothetical Document Embeddings (Bonus)
# =============================================================================

def _generate_hypothetical_document(query: str) -> str:
    """
    Sinh tài liệu giả định từ query dùng LLM (OpenRouter).
    Nếu không có API key → fallback về query gốc.

    HyDE logic:
        Thay vì embed câu hỏi → embed "câu trả lời giả định" → tìm tài liệu gần hơn.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key or not api_key.startswith("sk-or-"):
        # Fallback: mở rộng query thủ công bằng keyword expansion
        return _expand_query(query)

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        system_prompt = (
            "Ban la tro ly ho tro khach hang cua Shopee Vietnam. "
            "Hay sinh ra doan van ngan (100-150 tu) mo ta chinh sach hoac "
            "huong dan lien quan den cau hoi sau day. "
            "Chi sinh ra noi dung, khong giai thich them."
        )
        resp = client.chat.completions.create(
            model="google/gemini-flash-1.5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Cau hoi: {query}"},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        hyp_doc = resp.choices[0].message.content.strip()
        return hyp_doc if len(hyp_doc) > 50 else _expand_query(query)

    except Exception:
        return _expand_query(query)


def _expand_query(query: str) -> str:
    """
    Mở rộng query thủ công bằng cách thêm từ ngữ liên quan.
    Fallback khi không có LLM.
    """
    expansions = {
        "hoan tien": "chinh sach hoan tien tra hang shopee nguoi mua yeu cau",
        "refund": "return refund policy shopee buyer request procedure",
        "tra hang": "quy trinh tra hang hoan tien shopee thoi han ly do",
        "van chuyen": "don vi van chuyen giao hang shipper phi van chuyen",
        "thanh toan": "phuong thuc thanh toan shopeepay the tin dung COD",
        "payment": "payment methods shopee wallet credit card COD",
        "voucher": "ma giam gia voucher shopee ap dung dieu kien",
        "bao mat": "chinh sach bao mat du lieu ca nhan quyen nguoi dung",
        "nguoi ban": "quy dinh nguoi ban dang san pham phi hoa hong",
        "seller": "seller policy listing fee commission shopee",
    }
    for kw, expansion in expansions.items():
        if kw in query.lower():
            return f"{query} {expansion}"
    return query


# =============================================================================
# FILTER SEARCH — Tìm kiếm có lọc theo metadata
# =============================================================================

def semantic_search_filtered(
    query: str,
    top_k: int = 10,
    customer_role: str = None,
    category: str = None,
    use_hyde: bool = False,
) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa có lọc metadata.

    Args:
        query         : Câu truy vấn.
        top_k         : Số kết quả tối đa.
        customer_role : Lọc theo role: 'buyer', 'seller', hoặc None (lấy hết).
        category      : Lọc theo danh mục: 'returns', 'payment', ... hoặc None.
        use_hyde      : Dùng HyDE hay không.

    Returns:
        List dicts như semantic_search().
    """
    if use_hyde:
        search_text = _generate_hypothetical_document(query)
    else:
        search_text = query

    model = get_embedding_model()
    query_vector = model.encode(
        search_text, normalize_embeddings=True
    ).tolist()

    collection = get_collection()
    if collection.count() == 0:
        return []

    # Xây dựng where clause
    where = {}
    conditions = []
    if customer_role in ("buyer", "seller", "both"):
        conditions.append({"customer_role": {"$eq": customer_role}})
    if category:
        conditions.append({"category": {"$eq": category}})

    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    n_results = min(top_k, collection.count())
    query_kwargs = dict(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        query_kwargs["where"] = where

    try:
        results = collection.query(**query_kwargs)
    except Exception:
        # Fallback: tìm không lọc nếu lỗi where
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    output = []
    for doc, meta, dist in zip(
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    ):
        score = max(0.0, 1.0 - dist / 2.0)
        output.append({"content": doc, "score": round(score, 4), "metadata": meta})

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


# =============================================================================
# MAIN — Test nhanh
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 60)
    print("Task 5: Semantic Search Test")
    print("=" * 60)

    test_queries = [
        ("hoan tien khi hang bi loi", 5),
        ("phuong thuc thanh toan tren shopee", 3),
        ("quy dinh nguoi ban dang san pham", 3),
    ]

    for query, k in test_queries:
        print(f"\nQuery: '{query}' (top_k={k})")
        results = semantic_search(query, top_k=k)
        if results:
            for i, r in enumerate(results, 1):
                meta = r.get("metadata", {})
                print(f"  [{i}] score={r['score']:.3f} | role={meta.get('customer_role','?')} | {r['content'][:80]}...")
        else:
            print("  (No results — chua index hoac ChromaDB rong)")

    # Test HyDE
    print("\n--- HyDE Test ---")
    hyde_results = semantic_search("khi nao duoc doi hang", top_k=3, use_hyde=True)
    print(f"HyDE results: {len(hyde_results)} ket qua")
