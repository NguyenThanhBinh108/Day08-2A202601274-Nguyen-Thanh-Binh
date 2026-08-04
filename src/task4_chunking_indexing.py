"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho cả tiếng Việt lẫn tiếng Anh)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

from pathlib import Path
from typing import Any, Optional

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


def _safe_print(message: str) -> None:
    """
    In ra stdout mà KHÔNG BAO GIỜ raise (dùng chung cho toàn module thay cho print()).

    Console Windows mặc định là codepage cp1252: các ký tự "⚠", "✓", "→", "📊" và chữ
    tiếng Việt có dấu đều không encode được → print() thường sẽ ném UnicodeEncodeError
    và giết cả pipeline (thường ngay trong except block, chỗ khó debug nhất). Ở đây
    fallback sang dạng ASCII escape để thông điệp vẫn tới được người dùng.
    Cùng cơ chế với _safe_print/_warn ở Task 5, 6, 8, 9 — giữ hành vi nhất quán.
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


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chunking strategy: RecursiveCharacterTextSplitter.
# Lý do: corpus là markdown chính sách TMĐT song ngữ Anh/Việt, cấu trúc heading không
# đồng đều giữa file legal (PDF convert ra, heading lộn xộn) và file news (heading chuẩn).
# MarkdownHeaderTextSplitter sẽ tạo chunk dài ngắn rất chênh lệch trên corpus như vậy,
# còn SemanticChunker phải gọi embedding model khi split (chậm và tốn tài nguyên).
# Recursive cắt theo thứ tự ưu tiên đoạn → dòng → câu → từ nên giữ được ranh giới ngữ
# nghĩa mà vẫn đảm bảo chunk có kích thước ổn định.
CHUNK_SIZE = 800        # 800 ký tự ≈ 1 đoạn chính sách hoàn chỉnh (vd: toàn bộ mục
                        # "điều kiện đổi trả"), đủ ngữ cảnh để LLM trả lời trọn vẹn mà
                        # không nhồi quá nhiều chủ đề khác vào cùng 1 vector.
CHUNK_OVERLAP = 100     # 100/800 = 12.5% overlap — đủ để câu bị cắt ở biên chunk vẫn
                        # xuất hiện nguyên vẹn ở chunk kế tiếp, nhưng không lặp lại quá
                        # nhiều làm phình index và gây trùng lặp kết quả retrieval.
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Embedding model: BAAI/bge-m3.
# Lý do: corpus song ngữ (query tiếng Anh có thể phải match nội dung tiếng Việt và ngược
# lại). bge-m3 được train multilingual nên cùng một không gian vector phục vụ được cả
# hai ngôn ngữ; all-MiniLM-L6-v2 chỉ mạnh tiếng Anh, còn OpenAI embedding cần API key
# (bài lab phải chạy được offline).
EMBEDDING_MODEL = "BAAI/bge-m3"  # Vì sao? Multilingual, tốt cho tiếng Việt lẫn tiếng Anh
EMBEDDING_DIM = 1024

# Vector store: ChromaDB — persistent local, không cần Docker, API upsert đơn giản.
VECTOR_STORE = "chromadb"  # "chromadb" | "weaviate" | "faiss"
COLLECTION_NAME = "ecommerce_support_docs"

# Chunk ngắn hơn ngưỡng này gần như chỉ là heading/dòng phân cách "---", không mang đủ
# thông tin để trả lời câu hỏi nhưng vẫn chiếm slot trong top_k → loại bỏ.
MIN_CHUNK_CHARS = 50

# Upsert theo lô để tránh lỗi payload quá lớn khi corpus nhiều nghìn chunk.
UPSERT_BATCH_SIZE = 256

# Batch size khi encode — 16 đủ nhanh mà không nổ RAM/VRAM với model 1024 chiều.
EMBED_BATCH_SIZE = 16


# =============================================================================
# LAZY LOADERS — không load model / mở DB ở cấp module
# =============================================================================
# pytest import toàn bộ module này, nên import phải nhanh và không được crash khi
# chưa có data/, chưa có chroma_db/, hoặc chưa tải model về máy. Mọi thứ nặng đều
# nằm sau getter có cache ở dưới.

_model: Optional[Any] = None          # cache SentenceTransformer
_chroma_client: Optional[Any] = None  # cache chromadb.PersistentClient
_collection: Optional[Any] = None     # cache collection object


def _get_model() -> Optional[Any]:
    """
    Load SentenceTransformer 1 lần duy nhất rồi cache lại (lazy singleton).

    Returns:
        SentenceTransformer instance, hoặc None nếu không load được
        (thiếu thư viện / chưa tải được weights) — không raise để pipeline
        và các task phía sau vẫn degrade gracefully.
    """
    global _model
    if _model is not None:
        return _model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        _safe_print(f"⚠ Chưa cài sentence-transformers ({exc}). Bỏ qua bước embedding.")
        return None

    try:
        _safe_print(f"→ Loading embedding model: {EMBEDDING_MODEL} (lần đầu sẽ tải về, hơi lâu)")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as exc:  # noqa: BLE001 — mạng lỗi, thiếu weights, hết RAM...
        _safe_print(f"⚠ Không load được model '{EMBEDDING_MODEL}': {exc}")
        return None

    return _model


def get_embedding_model() -> Optional[Any]:
    """
    Public getter cho embedding model — Task 5 / Task 9 dùng chung hàm này
    để đảm bảo query được embed bằng ĐÚNG model đã dùng lúc index.
    """
    return _get_model()


def _get_chroma_client() -> Optional[Any]:
    """Mở PersistentClient 1 lần rồi cache (lazy). Trả None nếu ChromaDB lỗi."""
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client

    try:
        import chromadb
    except ImportError as exc:
        _safe_print(f"⚠ Chưa cài chromadb ({exc}).")
        return None

    try:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    except Exception as exc:  # noqa: BLE001
        _safe_print(f"⚠ Không mở được ChromaDB tại {CHROMA_DIR}: {exc}")
        return None

    return _chroma_client


def get_collection() -> Optional[Any]:
    """
    Lấy (hoặc tạo) collection ChromaDB dùng chung cho toàn bộ pipeline.

    metadata={"hnsw:space": "cosine"} là bắt buộc: embedding được normalize nên
    cosine mới cho khoảng cách đúng ngữ nghĩa (mặc định của Chroma là L2).

    Returns:
        Collection object, hoặc None nếu không mở được vector store.
    """
    global _collection
    if _collection is not None:
        return _collection

    client = _get_chroma_client()
    if client is None:
        return None

    try:
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:  # noqa: BLE001
        _safe_print(f"⚠ Không tạo/mở được collection '{COLLECTION_NAME}': {exc}")
        return None

    return _collection


def reset_collection() -> bool:
    """
    Xóa collection cũ trước khi reindex.

    Như cảnh báo ở docstring đầu file: nếu không xóa, chunk của corpus cũ vẫn nằm
    trong collection và retrieval sẽ trả về kết quả rác lẫn lộn giữa 2 lần index.

    Returns:
        True nếu đã xóa (hoặc collection vốn không tồn tại), False nếu lỗi.
    """
    global _collection

    client = _get_chroma_client()
    if client is None:
        _safe_print("⚠ Bỏ qua reset: không mở được ChromaDB.")
        return False

    _collection = None  # invalidate cache vì object cũ sẽ stale sau khi delete
    try:
        client.delete_collection(name=COLLECTION_NAME)
        _safe_print(f"✓ Đã xóa collection cũ: {COLLECTION_NAME}")
    except Exception:  # noqa: BLE001 — collection chưa tồn tại là chuyện bình thường
        _safe_print(f"  (Collection '{COLLECTION_NAME}' chưa tồn tại, không cần xóa)")
    return True


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents: list[dict] = []

    if not STANDARDIZED_DIR.exists():
        _safe_print(f"⚠ Chưa có thư mục {STANDARDIZED_DIR} — chạy Task 3 trước. Trả về list rỗng.")
        return documents

    # sorted() để thứ tự chunk_index ổn định giữa các lần chạy (reproducible index).
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if not md_file.is_file():
            continue

        try:
            content = md_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            _safe_print(f"⚠ Bỏ qua {md_file.name}: đọc lỗi ({exc})")
            continue

        # Bỏ file rỗng / chỉ có whitespace — chunk từ file rỗng chỉ làm nhiễu index.
        if not content.strip():
            _safe_print(f"⚠ Bỏ qua {md_file.name}: file rỗng")
            continue

        # Phân loại theo thư mục con thay vì substring trên cả đường dẫn tuyệt đối,
        # tránh nhận nhầm khi thư mục cha của project vô tình chứa chữ "legal".
        try:
            rel_parts = md_file.relative_to(STANDARDIZED_DIR).parts
        except ValueError:
            rel_parts = md_file.parts
        doc_type = "legal" if any(p.lower() == "legal" for p in rel_parts) else "news"

        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    chunks: list[dict] = []
    if not documents:
        return chunks

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        _safe_print(f"⚠ Chưa cài langchain-text-splitters ({exc}). Trả về list rỗng.")
        return chunks

    # separators xếp theo độ "ưu tiên giữ nguyên vẹn": đoạn văn → dòng → câu → từ → ký tự.
    # "" ở cuối là fallback bắt buộc: đảm bảo một khối văn bản dài không có khoảng trắng
    # (bảng markdown, URL dài) vẫn bị cắt xuống dưới CHUNK_SIZE thay vì tràn ra.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    for doc in documents:
        content = (doc.get("content") or "").strip()
        doc_meta = dict(doc.get("metadata") or {})
        if not content:
            continue

        try:
            splits = splitter.split_text(content)
        except Exception as exc:  # noqa: BLE001
            _safe_print(f"⚠ Split lỗi cho {doc_meta.get('source', '?')}: {exc}")
            continue

        kept: list[str] = []
        for raw in splits:
            piece = raw.strip()
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            # Chốt chặn cuối: dù splitter đã tôn trọng chunk_size, vẫn hard-split lần nữa
            # để chắc chắn không chunk nào vượt CHUNK_SIZE (yêu cầu bắt buộc của test).
            if len(piece) > CHUNK_SIZE:
                for start in range(0, len(piece), CHUNK_SIZE):
                    sub = piece[start:start + CHUNK_SIZE].strip()
                    if len(sub) >= MIN_CHUNK_CHARS:
                        kept.append(sub)
            else:
                kept.append(piece)

        # Tài liệu quá ngắn (mọi chunk đều dưới ngưỡng) vẫn phải được index,
        # nếu không sẽ mất hẳn nội dung của file đó khỏi corpus.
        if not kept:
            kept = [content[:CHUNK_SIZE]]

        # enumerate trên danh sách ĐÃ lọc → chunk_index liên tục 0..n-1 trong mỗi document,
        # nhờ vậy id ghép từ (type, source, chunk_index) không bị trùng.
        for i, chunk_text in enumerate(kept):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc_meta, "chunk_index": i},
            })

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    if not chunks:
        return chunks

    model = _get_model()
    if model is None:
        _safe_print("⚠ Không có embedding model — chunks được trả về nguyên trạng (chưa có 'embedding').")
        return chunks

    texts = [c.get("content", "") for c in chunks]

    try:
        # normalize_embeddings=True là BẮT BUỘC: collection dùng "hnsw:space"="cosine",
        # vector đã chuẩn hóa L2 thì cosine distance = 1 - dot product, score mới đúng
        # và nằm gọn trong [0, 2]. Thiếu normalize → score sai lệch theo độ dài chunk.
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=True,
        )
    except Exception as exc:  # noqa: BLE001
        _safe_print(f"⚠ Encode lỗi: {exc}. Trả về chunks chưa embed.")
        return chunks

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist() if hasattr(emb, "tolist") else list(emb)

    # Cảnh báo sớm nếu model trả về số chiều khác EMBEDDING_DIM đã khai báo: index bằng
    # model này rồi query bằng model khác chiều sẽ làm Chroma báo lỗi ở Task 5, lúc đó
    # rất khó lần ra nguyên nhân.
    actual_dim = len(chunks[0].get("embedding") or [])
    if actual_dim and actual_dim != EMBEDDING_DIM:
        _safe_print(f"⚠ EMBEDDING_DIM khai báo {EMBEDDING_DIM} nhưng model trả về {actual_dim} chiều.")

    return chunks


def _build_chunk_id(chunk: dict, fallback_index: int) -> str:
    """
    Sinh id DUY NHẤT TOÀN CỤC cho 1 chunk.

    Chỉ dùng (source, chunk_index) là chưa đủ: data/standardized/legal/ và
    data/standardized/news/ hoàn toàn có thể chứa file trùng tên, khi đó chunk của
    file này sẽ ghi đè chunk của file kia lúc upsert. Thêm 'type' vào id để tách hẳn.
    """
    meta = chunk.get("metadata") or {}
    doc_type = str(meta.get("type", "unknown"))
    source = str(meta.get("source", "unknown"))
    chunk_index = meta.get("chunk_index", fallback_index)
    return f"{doc_type}__{source}__{chunk_index}"


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    if not chunks:
        _safe_print("⚠ Không có chunk nào để index.")
        return

    # Chỉ index chunk đã có vector — chunk thiếu 'embedding' (do model load lỗi)
    # mà đẩy vào Chroma sẽ khiến collection thiếu nhất quán về chiều vector.
    ready = [c for c in chunks if c.get("embedding")]
    if not ready:
        _safe_print("⚠ Không chunk nào có embedding — bỏ qua bước index.")
        return

    if len(ready) < len(chunks):
        _safe_print(f"⚠ Bỏ qua {len(chunks) - len(ready)} chunk thiếu embedding.")

    collection = get_collection()
    if collection is None:
        _safe_print("⚠ Không mở được collection — bỏ qua bước index.")
        return

    ids: list[str] = []
    documents: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict] = []

    seen: dict[str, int] = {}
    for i, chunk in enumerate(ready):
        chunk_id = _build_chunk_id(chunk, i)
        # Phòng trường hợp id vẫn trùng (metadata bất thường): thêm hậu tố thay vì
        # để upsert âm thầm ghi đè mất dữ liệu.
        if chunk_id in seen:
            seen[chunk_id] += 1
            chunk_id = f"{chunk_id}__dup{seen[chunk_id]}"
        else:
            seen[chunk_id] = 0

        ids.append(chunk_id)
        documents.append(chunk.get("content", ""))
        embeddings.append(chunk["embedding"])
        # Chroma chỉ nhận metadata value là str/int/float/bool → ép kiểu cho an toàn.
        meta = chunk.get("metadata") or {}
        metadatas.append({
            "source": str(meta.get("source", "unknown")),
            "type": str(meta.get("type", "unknown")),
            "chunk_index": int(meta.get("chunk_index", i)),
        })

    total = 0
    for start in range(0, len(ids), UPSERT_BATCH_SIZE):
        end = start + UPSERT_BATCH_SIZE
        try:
            collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )
            total += len(ids[start:end])
        except Exception as exc:  # noqa: BLE001
            _safe_print(f"⚠ Upsert lô {start}-{end} lỗi: {exc}")

    _safe_print(f"  Upserted {total}/{len(ids)} chunks vào '{COLLECTION_NAME}'")


def run_pipeline(reset: bool = True):
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    _safe_print("=" * 50)
    _safe_print("Task 4: Chunking & Indexing")
    _safe_print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    _safe_print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    _safe_print(f"  Vector Store: {VECTOR_STORE}")
    _safe_print("=" * 50)

    docs = load_documents()
    _safe_print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    _safe_print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    embedded = sum(1 for c in chunks if c.get("embedding"))
    _safe_print(f"✓ Embedded {embedded} chunks")

    # Xóa collection cũ TRƯỚC khi index (xem cảnh báo ở docstring đầu file):
    # trộn chunk của 2 lần index khác nhau làm retrieval trả về kết quả rác.
    if reset:
        reset_collection()

    index_to_vectorstore(chunks)
    _safe_print("✓ Indexed to vector store")

    collection = get_collection()
    if collection is not None:
        try:
            _safe_print(f"\n📊 Collection '{COLLECTION_NAME}' hiện có {collection.count()} vectors")
        except Exception as exc:  # noqa: BLE001
            _safe_print(f"⚠ Không đếm được số vector: {exc}")


if __name__ == "__main__":
    # Console Windows mặc định cp1252 → ép UTF-8 để log tiếng Việt hiển thị đúng dấu.
    # Chỉ làm trong nhánh demo, KHÔNG làm lúc import (pytest/Streamlit tự quản stdout).
    try:
        import sys

        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 — stream không hỗ trợ reconfigure thì bỏ qua
        pass

    run_pipeline()
