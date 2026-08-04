"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4

Ghi chú thiết kế (implementation notes):
    - Mọi thứ "nặng" (SentenceTransformer, ChromaDB client) đều được load LAZY và cache
      ở cấp module. Lý do: pytest import toàn bộ module để chấm điểm, nên `import` phải
      nhanh và không được crash khi máy chưa có chroma_db/ hoặc chưa tải model.
    - Không bao giờ raise khi thiếu dữ liệu: in cảnh báo rồi trả về [] (degrade gracefully).
    - BONUS HyDE: sinh một đoạn trả lời giả định bằng LLM rồi embed đoạn đó thay vì câu hỏi
      gốc — giúp thu hẹp khoảng cách ngữ nghĩa giữa "câu hỏi ngắn" và "đoạn văn tài liệu".
      Mặc định TẮT để test chạy nhanh và không tốn API.
"""

from __future__ import annotations

import os
from typing import Any, Optional

# Dùng chung cấu hình với Task 4 để đảm bảo query vector cùng "không gian" với index vector.
# Chỉ import HẰNG SỐ (không gọi hàm nặng) nên import này rất nhẹ.
from .task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

# --- Cấu hình cho BONUS HyDE (Hypothetical Document Embeddings) ---
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
HYDE_MODEL = "openai/gpt-4o-mini"  # hoặc model ":free" nếu chưa có credit
HYDE_MAX_TOKENS = 180

# --- Lazy singletons (module-level cache) ---
_embedding_model: Optional[Any] = None
_embedding_model_failed: bool = False
_collection: Optional[Any] = None
_warned_missing_store: bool = False


def _safe_print(message: str) -> None:
    """
    In ra stdout mà không bao giờ làm chết chương trình.

    Console Windows mặc định dùng codepage cp1252 → print() chuỗi có dấu tiếng Việt
    sẽ ném UnicodeEncodeError. Cảnh báo mà lại crash thì còn tệ hơn không cảnh báo,
    nên ở đây fallback sang dạng ASCII escape khi console không encode được.
    """
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", "backslashreplace").decode("ascii"))


# =============================================================================
# LAZY LOADERS — không chạy lúc import, chỉ chạy khi thực sự cần
# =============================================================================

def _get_model() -> Optional[Any]:
    """
    Load (và cache) embedding model đúng bằng model đã dùng ở Task 4.

    Returns:
        SentenceTransformer instance, hoặc None nếu không load được (in cảnh báo, không raise).
    """
    global _embedding_model, _embedding_model_failed

    if _embedding_model is not None:
        return _embedding_model
    if _embedding_model_failed:
        # Đã thử và thất bại → không thử lại (tránh tải model chậm nhiều lần).
        return None

    # Ưu tiên tái dùng instance đã cache của Task 4 (get_embedding_model). Hai lý do:
    #   1) Đảm bảo query được embed bằng ĐÚNG object model đã dùng lúc index — không
    #      có cách nào lệch model/phiên bản giữa index và query.
    #   2) bge-m3 chiếm ~2GB RAM. Nếu Task 5 tự load bản thứ hai thì một process vừa
    #      index vừa search (Streamlit app, eval_pipeline) sẽ giữ 2 bản trong bộ nhớ.
    # Import nằm trong hàm để giữ nguyên tắc lazy (không kéo model lúc import module).
    try:
        from .task4_chunking_indexing import get_embedding_model

        shared = get_embedding_model()
        if shared is not None:
            _embedding_model = shared
            return _embedding_model
    except Exception:  # Task 4 chưa sẵn sàng → tự load bản riêng ở dưới
        pass

    try:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        return _embedding_model
    except Exception as exc:  # thiếu thư viện, không có mạng để tải model, hết RAM...
        _embedding_model_failed = True
        _safe_print(f"[task5] WARNING: Không load được embedding model '{EMBEDDING_MODEL}': {exc}")
        return None


def _get_collection() -> Optional[Any]:
    """
    Mở (và cache) ChromaDB collection đã được index ở Task 4.

    Returns:
        Chroma collection, hoặc None nếu chưa có chroma_db/ hoặc collection rỗng.
        Chỉ cache khi thành công — để nếu user chạy Task 4 sau đó thì lần gọi sau vẫn thấy dữ liệu.
    """
    global _collection, _warned_missing_store

    if _collection is not None:
        return _collection

    if not CHROMA_DIR.exists():
        if not _warned_missing_store:
            _warned_missing_store = True
            _safe_print(
                f"[task5] WARNING: Chưa có vector store tại '{CHROMA_DIR}'. "
                "Hãy chạy Task 4 (run_pipeline) trước. Trả về [] cho mọi query."
            )
        return None

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(name=COLLECTION_NAME)
        if collection.count() == 0:
            if not _warned_missing_store:
                _warned_missing_store = True
                _safe_print(
                    f"[task5] WARNING: Collection '{COLLECTION_NAME}' đang rỗng — "
                    "hãy chạy lại Task 4 để index dữ liệu."
                )
            return None
    except Exception as exc:  # chưa cài chromadb, collection chưa tồn tại, DB hỏng...
        if not _warned_missing_store:
            _warned_missing_store = True
            _safe_print(f"[task5] WARNING: Không mở được collection '{COLLECTION_NAME}': {exc}")
        return None

    _collection = collection
    return _collection


# =============================================================================
# BONUS — HyDE (Hypothetical Document Embeddings)
# =============================================================================

def generate_hyde_query(query: str) -> str:
    """
    Sinh một "tài liệu giả định" (hypothetical answer) cho câu hỏi, dùng cho HyDE.

    Ý tưởng: câu hỏi ngắn ("phương thức thanh toán?") và đoạn tài liệu dài có phân bố
    embedding khá khác nhau. Nếu ta để LLM viết trước một đoạn trả lời *giả định* rồi
    embed đoạn đó, vector truy vấn sẽ "giống văn phong tài liệu" hơn → recall tốt hơn.

    Args:
        query: Câu truy vấn gốc

    Returns:
        Đoạn văn giả định (str). Trả về "" nếu thiếu API key hoặc lỗi API
        → caller sẽ im lặng dùng query gốc (không raise, không làm hỏng test).
    """
    if not query or not query.strip():
        return ""

    try:
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv()  # đọc .env ở thư mục gốc dự án
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return ""  # thiếu key → im lặng fallback về query gốc

        client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        response = client.chat.completions.create(
            model=HYDE_MODEL,
            messages=[
                {
                    # Prompt viết bằng tiếng Anh vì corpus SONG NGỮ: prompt tiếng Việt làm
                    # model trả lời tiếng Việt cho cả câu hỏi tiếng Anh, khiến vector HyDE
                    # lệch ngôn ngữ so với chunk cần tìm và giảm chất lượng ranking.
                    "role": "system",
                    "content": (
                        "You write e-commerce customer-support policy excerpts. "
                        "Given a question, write a SHORT hypothetical passage (2-3 sentences) "
                        "that looks like an excerpt from a help-center or policy document "
                        "answering it. No preamble, no disclaimers, never say you don't know. "
                        "CRITICAL: write in exactly the same language as the question "
                        "(English question -> English answer, Vietnamese question -> Vietnamese answer)."
                    ),
                },
                {"role": "user", "content": query.strip()},
            ],
            temperature=0.3,
            max_tokens=HYDE_MAX_TOKENS,
        )
        hypothetical = (response.choices[0].message.content or "").strip()
        return hypothetical
    except Exception:
        # Lỗi mạng / hết quota / 429 rate limit / chưa cài openai → im lặng dùng query gốc.
        return ""


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def semantic_search(query: str, top_k: int = 10, use_hyde: bool = False) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa
        use_hyde: (BONUS) Bật HyDE — embed đoạn trả lời giả định thay vì câu hỏi gốc.
                  Mặc định False để test chạy nhanh và không tốn API.

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []

    collection = _get_collection()
    if collection is None:
        return []

    model = _get_model()
    if model is None:
        return []

    # Bước 1: Embed query bằng cùng model ở Task 4.
    # normalize_embeddings=True phải KHỚP với cách index ở Task 4, nếu không thì
    # khoảng cách cosine trả về sẽ lệch thang đo.
    search_text = query.strip()
    if use_hyde:
        hyde_text = generate_hyde_query(query)
        if hyde_text:
            # HyDE: embed đoạn tài liệu giả định thay cho câu hỏi gốc.
            search_text = hyde_text

    try:
        query_vector = model.encode(search_text, normalize_embeddings=True)
        query_vector = (
            query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
        )
    except Exception as exc:
        _safe_print(f"[task5] WARNING: Lỗi khi embed query: {exc}")
        return []

    # Bước 2: Query vector store (cosine similarity).
    # Clamp n_results theo số chunk thật sự có, tránh cảnh báo/lỗi khi collection nhỏ.
    try:
        available = collection.count()
        n_results = max(1, min(int(top_k), available))
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        _safe_print(f"[task5] WARNING: Lỗi khi query vector store: {exc}")
        return []

    # Bước 3: Chuẩn hoá kết quả về schema chung + sort descending.
    documents = (results.get("documents") or [[]])[0] or []
    metadatas = (results.get("metadatas") or [[]])[0] or []
    distances = (results.get("distances") or [[]])[0] or []

    output: list[dict] = []
    for idx, doc in enumerate(documents):
        if doc is None:
            continue
        meta = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
        distance = distances[idx] if idx < len(distances) else 1.0

        # ChromaDB trả về DISTANCE cosine (0 = giống hệt), không phải similarity.
        # Đây là điểm cosine GỐC mà Task 9 dùng để quyết định fallback → phải đúng thang [0, 1].
        try:
            score = 1.0 - float(distance)
        except (TypeError, ValueError):
            score = 0.0
        score = min(1.0, max(0.0, score))

        output.append(
            {
                "content": doc,
                "score": round(score, 4),
                "metadata": {
                    "source": meta.get("source", "unknown"),
                    "type": meta.get("type", "legal"),
                    "chunk_index": meta.get("chunk_index", idx),
                },
            }
        )

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    # Test
    demo_queries = [
        "quy định trả hàng hoàn tiền shopee",
        "payment methods",
        "seller listing regulations",
        "how to track my order",
    ]
    for q in demo_queries:
        _safe_print(f"\n=== Query: {q} ===")
        results = semantic_search(q, top_k=5)
        if not results:
            _safe_print("  (không có kết quả — kiểm tra đã chạy Task 4 chưa)")
        for r in results:
            src = r["metadata"].get("source", "?")
            _safe_print(f"[{r['score']:.3f}] ({src}) {r['content'][:100]}...")
