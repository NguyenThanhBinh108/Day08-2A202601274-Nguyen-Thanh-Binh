"""
Task 4 — Chunking & Indexing vào Vector Store.

LÝ DO LỰA CHỌN:
  - Chunking: RecursiveCharacterTextSplitter (CHUNK_SIZE=800, OVERLAP=100)
    * Ưu tiên tách theo đoạn văn → câu → từ (hierarchy).
    * CHUNK_SIZE=800 đủ lớn để giữ ngữ cảnh, không quá lớn làm loãng embedding.
    * OVERLAP=100 đảm bảo không mất thông tin tại ranh giới chunk.
  - Embedding: BAAI/bge-m3 (1024 dim, multilingual)
    * Tốt cho cả tiếng Việt lẫn tiếng Anh.
    * Hỗ trợ dense + sparse + colbert retrieval.
  - Vector Store: ChromaDB (persistent, local, không cần Docker)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý: nếu đổi corpus, xóa chroma_db/ cũ trước khi reindex.
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Chunking: RecursiveCharacterTextSplitter
# CHUNK_SIZE=800: đủ ngữ cảnh, phù hợp mô hình nhẹ.
# CHUNK_OVERLAP=100: ~12.5% overlap, tránh mất thông tin biên chunk
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"

# Embedding: sentence-transformers/all-MiniLM-L6-v2 (nhẹ ~80MB, 384 dim)
# Tải siêu nhanh, không cần API, phù hợp làm Lab.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

VECTOR_STORE = "chromadb"
COLLECTION_NAME = "ecommerce_support_docs"

# Cache model toàn cục
_embedding_model = None
_chroma_collection = None


# =============================================================================
# HELPERS
# =============================================================================

def _extract_metadata_from_content(content: str, filepath: Path) -> dict:
    """
    Trích xuất metadata từ nội dung markdown.
    Đọc frontmatter-like lines: **Customer Role:**, **Source:**, etc.
    """
    meta = {
        "source": filepath.name,
        "doc_id": filepath.stem,
        "type": "legal" if "legal" in str(filepath) else "news",
        "customer_role": "both",  # default
        "category": "general",
        "language": "vi",
    }

    for line in content.splitlines()[:20]:
        line_lower = line.lower()
        if "customer role:" in line_lower:
            role = line.split(":")[-1].strip().strip("*").strip()
            if role in ("buyer", "seller", "both"):
                meta["customer_role"] = role
        elif "**category:**" in line_lower or "category:" in line_lower:
            cat = line.split(":")[-1].strip().strip("*").strip()
            if cat:
                meta["category"] = cat
        elif "doc id:" in line_lower:
            doc_id = line.split(":")[-1].strip().strip("*").strip()
            if doc_id:
                meta["doc_id"] = doc_id
    return meta


def get_embedding_model():
    """Trả về (và cache) SentenceTransformer model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[Embedding] Loading {EMBEDDING_MODEL} ...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"[Embedding] Model loaded OK")
    return _embedding_model


def get_collection():
    """Trả về (và cache) ChromaDB collection."""
    global _chroma_collection
    if _chroma_collection is None:
        import chromadb
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _chroma_collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collection


# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str, ...}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        print(f"[WARN] Thu muc khong ton tai: {STANDARDIZED_DIR}")
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8")
            if len(content.strip()) < 100:
                continue  # bỏ qua file rỗng / quá ngắn
            meta = _extract_metadata_from_content(content, md_file)
            documents.append({"content": content, "metadata": meta})
        except Exception as e:
            print(f"[WARN] Khong doc duoc {md_file.name}: {e}")

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents dùng RecursiveCharacterTextSplitter.

    Args:
        documents: List of {'content': str, 'metadata': dict}
    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": i,
                    "chunk_total": len(splits),
                }
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng BAAI/bge-m3.

    Args:
        chunks: List of {'content': str, 'metadata': dict}
    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    model = get_embedding_model()
    texts = [c["content"] for c in chunks]

    batch_size = 32
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        embs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        embeddings.extend(embs.tolist())

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Upsert chunks vào ChromaDB.
    """
    collection = get_collection()

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        ids = [
            f"{c['metadata']['doc_id']}_chunk_{c['metadata']['chunk_index']}"
            for c in batch
        ]
        # Làm sạch metadata (chỉ giữ string/int/float/bool)
        clean_metas = []
        for c in batch:
            clean = {
                k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                for k, v in c["metadata"].items()
            }
            clean_metas.append(clean)

        collection.upsert(
            ids=ids,
            documents=[c["content"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            metadatas=clean_metas,
        )
        print(f"  Indexed batch {i//batch_size + 1}: {len(batch)} chunks")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 60)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking  : {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding : {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector DB : {VECTOR_STORE}")
    print("=" * 60)

    # Bước 1: Load
    docs = load_documents()
    print(f"\n[1/4] Loaded {len(docs)} documents from {STANDARDIZED_DIR}")
    if not docs:
        print("[ERROR] Khong co document nao! Chay task3 truoc.")
        return

    # Bước 2: Chunk
    chunks = chunk_documents(docs)
    print(f"[2/4] Created {len(chunks)} chunks (avg size ~{CHUNK_SIZE} chars)")

    # Bước 3: Embed
    print(f"[3/4] Embedding {len(chunks)} chunks voi {EMBEDDING_MODEL} ...")
    chunks = embed_chunks(chunks)
    print(f"[3/4] Embedding xong!")

    # Bước 4: Index
    print(f"[4/4] Indexing vao ChromaDB: {CHROMA_DIR}")
    index_to_vectorstore(chunks)

    # Verify
    col = get_collection()
    total_in_db = col.count()
    print(f"\n[OK] ChromaDB: {total_in_db} chunks da index")
    print(f"     Path    : {CHROMA_DIR}")
    print(f"\nTask 4 PASS!" if total_in_db > 0 else "Task 4 FAIL")


if __name__ == "__main__":
    run_pipeline()
