"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.

---------------------------------------------------------------------------
GHI CHÚ TRIỂN KHAI (nhóm KHÔNG có PAGEINDEX_API_KEY)
---------------------------------------------------------------------------
Module chạy ở HAI chế độ, tự chọn tại runtime:

  (A) Chế độ API (chỉ khi có PAGEINDEX_API_KEY): upload PDF trong data/landing/legal/
      lên PageIndex, lưu doc_id ra pageindex_doc_ids.json, rồi query qua SDK.
      Toàn bộ nhánh này được bọc try/except — lỗi mạng/quota chỉ in cảnh báo.

  (B) Chế độ VECTORLESS CỤC BỘ (mặc định của lab): tự dựng "cây mục lục" (TOC) từ
      heading markdown trong data/standardized/, rồi duyệt cây và chấm điểm bằng
      TỪ KHÓA + CẤU TRÚC — không embedding, không vector store. Đây chính là tinh
      thần "vectorless RAG": mô hình hiểu tài liệu qua cấu trúc mục lục (tiêu đề,
      breadcrumb cha-con) thay vì qua khoảng cách vector.

Công thức chấm điểm (mọi thành phần đều đã nằm trong [0, 1] nên tổng có trọng số
cũng nằm trong [0, 1] — không cần chia cho max, giữ được ý nghĩa tuyệt đối của điểm):

    score = W_TITLE   * (tỉ lệ từ khóa query khớp TIÊU ĐỀ section)
          + W_PATH    * (tỉ lệ từ khóa khớp BREADCRUMB các heading cha)
          + W_CONTENT * (tỉ lệ từ khóa xuất hiện trong NỘI DUNG section)
          + W_DENSITY * (mật độ lặp lại của từ khóa trong nội dung, đã chặn trên)

Trọng số tiêu đề >> nội dung: trong tài liệu chính sách, heading ("Payment Methods",
"Return & Refund Policy") là tín hiệu chủ đề mạnh hơn nhiều so với việc từ khóa tình
cờ xuất hiện đâu đó trong thân bài.
"""

import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Optional

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# --- Đường dẫn phụ trợ cho chế độ API -------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
LANDING_LEGAL_DIR = PROJECT_ROOT / "data" / "landing" / "legal"
DOC_IDS_FILE = PROJECT_ROOT / "pageindex_doc_ids.json"

# --- Tham số chế độ vectorless cục bộ -------------------------------------
W_TITLE = 0.55      # từ khóa khớp tiêu đề section — tín hiệu mạnh nhất
W_PATH = 0.10       # từ khóa khớp breadcrumb (các heading cha) — tín hiệu cấu trúc
W_CONTENT = 0.25    # từ khóa xuất hiện trong thân section
W_DENSITY = 0.10    # mật độ lặp lại (đã chặn trên) để phá thế hòa điểm
DENSITY_CAP = 3     # mỗi từ khóa chỉ được tính tối đa 3 lần xuất hiện
MAX_SECTION_CHARS = 2000  # cắt bớt section quá dài trước khi đưa vào LLM context

# Regex heading markdown: "# Tiêu đề" .. "###### Tiêu đề" (bỏ '#' đuôi nếu có).
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
# Hàng rào code block ``` hoặc ~~~ — heading giả bên trong code phải bị bỏ qua.
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
# Tách token unicode-aware: \w trên chuỗi str của Python 3 đã bao gồm chữ tiếng Việt
# có dấu (miễn là văn bản đã được chuẩn hóa về NFC trước đó).
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Stopword tối thiểu (EN + VI) — loại bỏ để tỉ lệ "khớp từ khóa" không bị pha loãng
# bởi các từ chức năng trong câu hỏi tự nhiên ("What payment methods does ... ?").
_STOPWORDS = frozenset({
    # English
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or", "the",
    "to", "what", "when", "where", "which", "who", "why", "with", "you", "your",
    # Vietnamese
    "và", "là", "của", "có", "cho", "khi", "nào", "gì", "thì", "được", "các",
    "những", "một", "này", "đó", "tôi", "bạn", "ở", "về", "với", "như", "để",
    "làm", "sao", "thế", "nếu", "trong", "trên", "bị", "phải",
})

# Cache cây mục lục ở cấp module — LAZY, chỉ dựng ở lần search đầu tiên.
# (Không đọc file lúc import: pytest import cả module, import phải nhanh và không crash.)
_TOC_CACHE: Optional[list[dict]] = None

# Cache API key — đọc LAZY ở lần dùng đầu tiên, không đọc .env lúc import module.
_API_KEY_CACHE: Optional[str] = None


def _get_api_key() -> str:
    """Đọc PAGEINDEX_API_KEY (lazy, có cache).

    Không gọi load_dotenv() ở cấp module: pytest import cả module này (và import
    gián tiếp qua Task 9), import phải nhanh và không được chạm đĩa. Ngoài ra đọc
    lazy còn giúp nhóm điền key vào .env rồi chạy lại mà không phải reload module.
    Thiếu key KHÔNG raise — chỉ khiến module chạy ở chế độ vectorless cục bộ.
    """
    global _API_KEY_CACHE
    if _API_KEY_CACHE is not None:
        return _API_KEY_CACHE

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # thiếu python-dotenv → vẫn đọc được biến môi trường hệ thống
        pass

    _API_KEY_CACHE = (os.getenv("PAGEINDEX_API_KEY") or "").strip()
    return _API_KEY_CACHE


# =============================================================================
# TIỆN ÍCH TEXT (unicode-aware, dùng chung cho cả tiếng Việt lẫn tiếng Anh)
# =============================================================================

def _warn(message: str) -> None:
    """In thông báo ra stdout một cách an toàn trên mọi console.

    Console Windows mặc định dùng codepage cp1252, nên `print("⚠ ...")` sẽ ném
    UnicodeEncodeError và làm gãy cả hàm gọi nó. Module này là nhánh fallback
    cuối cùng của pipeline (Task 9) — không được phép chết chỉ vì một ký tự
    không in được, nên mọi thông báo đều đi qua đây.
    """
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(message.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def _normalize(text: str) -> str:
    """Chuẩn hóa NFC + lowercase.

    NFC là bắt buộc: nếu văn bản ở dạng NFD (ký tự cơ sở + dấu tổ hợp) thì `\\w`
    sẽ cắt rời dấu thanh khỏi chữ, làm hỏng toàn bộ việc so khớp tiếng Việt.
    """
    return unicodedata.normalize("NFC", text or "").lower()


def _stem(token: str) -> str:
    """Chuẩn hóa số nhiều tiếng Anh ở mức tối thiểu ("methods" -> "method").

    Chỉ áp dụng cho token thuần ASCII để không đụng vào tiếng Việt; bỏ qua các
    đuôi dễ gây lỗi ("address" -> "addres", "status" -> "statu").
    """
    if len(token) >= 4 and token.isascii() and token.endswith("s"):
        if not token.endswith(("ss", "us", "is", "as")):
            return token[:-1]
    return token


def _tokenize(text: str, drop_stopwords: bool = False) -> list[str]:
    """Tách text thành list token đã chuẩn hóa.

    Args:
        text: Chuỗi đầu vào (có thể None/rỗng).
        drop_stopwords: True khi tokenize QUERY — bỏ từ chức năng để tỉ lệ khớp
            phản ánh đúng các từ khóa mang nghĩa.
    """
    tokens = [_stem(t) for t in _TOKEN_RE.findall(_normalize(text))]
    if not drop_stopwords:
        return tokens
    filtered = [t for t in tokens if t not in _STOPWORDS]
    # Nếu query toàn stopword thì giữ nguyên bản gốc, tránh trả về rỗng.
    return filtered or tokens


# =============================================================================
# CHẾ ĐỘ VECTORLESS CỤC BỘ — DỰNG CÂY MỤC LỤC TỪ HEADING MARKDOWN
# =============================================================================

def _infer_doc_type(md_file: Path) -> str:
    """Suy ra type ('legal' | 'news') từ thư mục chứa file.

    Chỉ xét phần đường dẫn TƯƠNG ĐỐI so với data/standardized/, không xét cả đường
    dẫn tuyệt đối: nếu một thư mục cha của project vô tình tên "news"/"legal" thì mọi
    tài liệu sẽ bị gán sai type. Quy tắc ("legal" khi nằm dưới legal/, ngược lại
    "news") giữ ĐÚNG như Task 4 để metadata['type'] của nhánh hybrid và nhánh
    pageindex fallback không mâu thuẫn nhau.
    """
    try:
        rel_parts = md_file.relative_to(STANDARDIZED_DIR).parts
    except ValueError:
        rel_parts = md_file.parts
    return "legal" if any(p.lower() == "legal" for p in rel_parts) else "news"


def _parse_markdown_tree(md_file: Path, raw_text: str) -> list[dict]:
    """Tách một file markdown thành các node mục lục theo heading.

    Mỗi node = một section: nội dung nằm giữa heading của nó và heading kế tiếp.
    `path` là breadcrumb tổ tiên (ví dụ "Shopee Policy > Payments > Methods"),
    tái tạo quan hệ cha-con của cây mục lục mà không cần cấu trúc lồng nhau.
    """
    text = unicodedata.normalize("NFC", raw_text)
    doc_type = _infer_doc_type(md_file)
    source = md_file.name

    nodes: list[dict] = []
    # Ngăn xếp (level, title) của các heading cha đang mở.
    stack: list[tuple[int, str]] = []
    # Node "preamble": phần text đứng trước heading đầu tiên.
    current: dict[str, Any] = {
        "title": md_file.stem,
        "level": 0,
        "path": md_file.stem,
        "lines": [],
    }
    in_fence = False

    def flush(node: dict) -> None:
        content = "\n".join(node.pop("lines")).strip()
        # Bỏ node rỗng hoàn toàn (không tiêu đề thật, không nội dung).
        # Bỏ luôn cả heading KHÔNG có thân bài (level > 0 mà content rỗng): tiêu đề
        # kiểu "## Payment methods" đứng ngay trên một heading con sẽ ăn trọn trọng số
        # W_TITLE và leo lên đầu bảng xếp hạng với content = "" — chiếm mất một slot
        # top_k mà không mang được chữ nào vào context cho LLM. Nội dung thật vẫn được
        # giữ ở các node con (breadcrumb `path` đã tái tạo quan hệ cha-con).
        if not content:
            return
        if len(content) > MAX_SECTION_CHARS:
            content = content[:MAX_SECTION_CHARS].rstrip() + " …"
        node["content"] = content
        node["source"] = source
        node["doc_type"] = doc_type
        node["file"] = str(md_file)
        nodes.append(node)

    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            current["lines"].append(line)
            continue

        match = None if in_fence else _HEADING_RE.match(line)
        if match is None:
            current["lines"].append(line)
            continue

        # Gặp heading mới -> đóng section đang mở, mở section mới.
        flush(current)
        level = len(match.group(1))
        title = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        breadcrumb = [t for _, t in stack] + [title]
        stack.append((level, title))
        current = {
            "title": title,
            "level": level,
            "path": " > ".join(breadcrumb),
            "lines": [],
        }

    flush(current)
    return nodes


def _build_toc() -> list[dict]:
    """Quét data/standardized/**/*.md và dựng cây mục lục phẳng hóa.

    Returns:
        List of {'title', 'level', 'path', 'content', 'source', 'doc_type',
                 'file', 'chunk_index'}.
        Trả về [] (kèm cảnh báo) nếu chưa có data — KHÔNG raise.
    """
    if not STANDARDIZED_DIR.exists():
        _warn(f"⚠ Chưa có thư mục {STANDARDIZED_DIR} — hãy chạy Task 3 trước.")
        _warn("  PageIndex vectorless sẽ trả về danh sách rỗng.")
        return []

    md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
    if not md_files:
        _warn(f"⚠ Không tìm thấy file .md nào trong {STANDARDIZED_DIR}.")
        return []

    nodes: list[dict] = []
    for md_file in md_files:
        try:
            raw = md_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            _warn(f"  ⚠ Bỏ qua {md_file.name}: {exc}")
            continue
        nodes.extend(_parse_markdown_tree(md_file, raw))

    for idx, node in enumerate(nodes):
        node["chunk_index"] = idx
    return nodes


def _get_toc(force_rebuild: bool = False) -> list[dict]:
    """Getter có cache cho cây mục lục (lazy — chỉ dựng khi thực sự search)."""
    global _TOC_CACHE
    if _TOC_CACHE is None or force_rebuild:
        _TOC_CACHE = _build_toc()
    return _TOC_CACHE


def _node_index(node: dict) -> tuple[set[str], set[str], Counter]:
    """Lấy (title_tokens, path_tokens, content_counts) của node, có cache trong node."""
    cached = node.get("_index")
    if cached is None:
        title_tokens = set(_tokenize(node.get("title", "")))
        path_tokens = set(_tokenize(node.get("path", "")))
        content_counts = Counter(_tokenize(node.get("content", "")))
        cached = (title_tokens, path_tokens, content_counts)
        node["_index"] = cached
    return cached


def _score_node(query_tokens: list[str], node: dict) -> float:
    """Chấm điểm một node mục lục theo từ khóa + cấu trúc (không embedding).

    Điểm luôn nằm trong [0, 1] vì mỗi thành phần là một tỉ lệ đã chặn trên và
    tổng trọng số W_TITLE + W_PATH + W_CONTENT + W_DENSITY = 1.0.
    """
    unique_terms = set(query_tokens)
    if not unique_terms:
        return 0.0

    title_tokens, path_tokens, content_counts = _node_index(node)
    n_terms = len(unique_terms)

    title_cov = sum(1 for t in unique_terms if t in title_tokens) / n_terms
    path_cov = sum(1 for t in unique_terms if t in path_tokens) / n_terms
    content_cov = sum(1 for t in unique_terms if content_counts.get(t, 0) > 0) / n_terms
    density = sum(
        min(content_counts.get(t, 0), DENSITY_CAP) for t in unique_terms
    ) / (DENSITY_CAP * n_terms)

    score = (
        W_TITLE * title_cov
        + W_PATH * path_cov
        + W_CONTENT * content_cov
        + W_DENSITY * density
    )
    # Kẹp lại cho chắc (phòng khi ai đó chỉnh trọng số cho tổng > 1).
    return max(0.0, min(1.0, score))


def _search_local_toc(query: str, top_k: int) -> list[dict]:
    """Duyệt cây mục lục cục bộ và trả về top_k section liên quan nhất."""
    toc = _get_toc()
    if not toc:
        return []

    query_tokens = _tokenize(query, drop_stopwords=True)
    if not query_tokens:
        return []

    scored: list[tuple[float, int, dict]] = []
    for node in toc:
        score = _score_node(query_tokens, node)
        if score > 0.0:
            # chunk_index làm khóa phụ -> thứ tự ổn định khi điểm bằng nhau.
            scored.append((score, node["chunk_index"], node))

    # Sort GIẢM DẦN theo score, tie-break tăng dần theo vị trí trong tài liệu.
    scored.sort(key=lambda item: (-item[0], item[1]))

    results: list[dict] = []
    for score, _, node in scored[:top_k]:
        results.append({
            "content": node["content"],
            "score": round(float(score), 6),
            "metadata": {
                "source": node["source"],
                "type": node["doc_type"],
                "section": node["path"],
                "chunk_index": node["chunk_index"],
            },
            "source": "pageindex",
        })
    return results


# =============================================================================
# CHẾ ĐỘ API THẬT (chỉ chạy khi có PAGEINDEX_API_KEY)
# =============================================================================

def _get_client() -> Optional[Any]:
    """Khởi tạo PageIndexClient (lazy import để không làm chậm/crash lúc import module)."""
    api_key = _get_api_key()
    if not api_key:
        return None
    try:
        from pageindex.client import PageIndexClient  # import trong hàm: LAZY

        return PageIndexClient(api_key=api_key)
    except Exception as exc:  # SDK thiếu / key sai định dạng
        _warn(f"⚠ Không khởi tạo được PageIndexClient: {exc}")
        return None


def _load_doc_ids() -> dict[str, str]:
    """Đọc mapping {filename: doc_id} đã lưu từ lần upload trước."""
    if not DOC_IDS_FILE.exists():
        return {}
    try:
        data = json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except (OSError, ValueError) as exc:
        _warn(f"⚠ Không đọc được {DOC_IDS_FILE.name}: {exc}")
        return {}


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    # PageIndex nhận PDF, không nhận .md trực tiếp -> upload thẳng PDF gốc trong
    # data/landing/legal/ (đây là nguồn của các file markdown ở data/standardized/).
    # Không có key = trường hợp mặc định của lab -> cảnh báo rồi thoát êm, KHÔNG raise.
    client = _get_client()
    if client is None:
        _warn("⚠ Bỏ qua upload: chưa có PAGEINDEX_API_KEY (module sẽ chạy vectorless cục bộ).")
        return {}

    if not LANDING_LEGAL_DIR.exists():
        _warn(f"⚠ Chưa có thư mục {LANDING_LEGAL_DIR} — hãy chạy Task 1 trước.")
        return {}

    doc_ids: dict[str, str] = _load_doc_ids()
    pdf_files = sorted(
        p for p in LANDING_LEGAL_DIR.rglob("*") if p.suffix.lower() == ".pdf"
    )
    if not pdf_files:
        _warn(f"⚠ Không tìm thấy file PDF nào trong {LANDING_LEGAL_DIR}.")
        return doc_ids

    for pdf_path in pdf_files:
        if pdf_path.name in doc_ids:
            _warn(f"  • Bỏ qua (đã upload): {pdf_path.name}")
            continue
        try:
            resp = client.submit_document(str(pdf_path))
            doc_id = resp.get("doc_id") or resp.get("id")
            if not doc_id:
                _warn(f"  ⚠ Response không có doc_id cho {pdf_path.name}: {resp}")
                continue
            doc_ids[pdf_path.name] = str(doc_id)
            _warn(f"  ✓ Uploaded: {pdf_path.name} -> {doc_id}")
        except Exception as exc:  # lỗi mạng / quota / định dạng — không được làm gãy pipeline
            _warn(f"  ⚠ Upload thất bại {pdf_path.name}: {exc}")

    try:
        DOC_IDS_FILE.write_text(
            json.dumps(doc_ids, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _warn(f"✓ Đã lưu {len(doc_ids)} doc_id vào {DOC_IDS_FILE.name}")
    except OSError as exc:
        _warn(f"⚠ Không ghi được {DOC_IDS_FILE.name}: {exc}")

    return doc_ids


def _search_pageindex_api(query: str, top_k: int) -> list[dict]:
    """Query PageIndex API thật và parse "retrieved_nodes" về schema chuẩn."""
    client = _get_client()
    if client is None:
        return []

    doc_ids = _load_doc_ids()
    if not doc_ids:
        _warn("⚠ Chưa có pageindex_doc_ids.json — hãy chạy upload_documents() trước.")
        return []

    raw_items: list[dict] = []
    for doc_id in doc_ids.values():
        try:
            resp = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = resp.get("retrieval_id") or resp.get("id")
            if not retrieval_id:
                continue

            # Poll cho đến khi status == "completed" (có trần thời gian để không treo).
            retrieval: dict = {}
            for _ in range(15):
                retrieval = client.get_retrieval(retrieval_id) or {}
                status = str(retrieval.get("status", "")).lower()
                if status in ("completed", "failed", "error"):
                    break
                time.sleep(2)

            # Schema: retrieved_nodes[i]["relevant_contents"] là list[list[dict]].
            for node in retrieval.get("retrieved_nodes", []) or []:
                for group in node.get("relevant_contents", []) or []:
                    for item in group or []:
                        content = (item.get("relevant_content") or "").strip()
                        if content:
                            raw_items.append({
                                "content": content,
                                "section": item.get("section_title") or "",
                                "doc_id": doc_id,
                            })
        except Exception as exc:
            _warn(f"  ⚠ PageIndex query lỗi (doc_id={doc_id}): {exc}")

    # PageIndex không trả score trực tiếp -> gán điểm giảm dần theo thứ hạng,
    # vẫn nằm trong [0, 1] để đồng nhất thang điểm với chế độ cục bộ.
    results: list[dict] = []
    total = max(1, min(len(raw_items), top_k))
    for rank, item in enumerate(raw_items[:top_k]):
        results.append({
            "content": item["content"],
            "score": round(max(0.0, 1.0 - rank * (0.5 / total)), 6),
            "metadata": {
                "source": item["doc_id"],
                "type": "legal",
                "section": item["section"],
                "chunk_index": rank,
            },
            "source": "pageindex",
        })
    return results


# =============================================================================
# PUBLIC API
# =============================================================================

def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    # Hợp đồng của hàm này: LUÔN trả về list, KHÔNG BAO GIỜ raise. Nó là nhánh
    # fallback cuối cùng của pipeline (Task 9) — nếu nó gãy thì cả pipeline gãy.
    try:
        if not isinstance(query, str) or not query.strip():
            return []
        if top_k <= 0:
            return []

        # (A) Có API key -> thử PageIndex thật trước.
        if _get_api_key():
            api_results = _search_pageindex_api(query, top_k)
            if api_results:
                return api_results
            _warn("⚠ PageIndex API không trả kết quả — chuyển sang vectorless cục bộ.")

        # (B) Mặc định: duyệt cây mục lục cục bộ (keyword + structure, no embedding).
        return _search_local_toc(query, top_k)
    except Exception as exc:  # degrade gracefully, không làm gãy Task 9
        _warn(f"⚠ pageindex_search lỗi ({type(exc).__name__}: {exc}) — trả về [].")
        return []


if __name__ == "__main__":
    if not _get_api_key():
        _warn("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        _warn("  Đăng ký tại: https://pageindex.ai/")
        _warn("  → Đang chạy chế độ VECTORLESS CỤC BỘ (duyệt cây mục lục markdown).\n")
    else:
        _warn("Uploading documents...")
        upload_documents()

    toc = _get_toc()
    _warn(f"TOC: {len(toc)} section từ {len({n['file'] for n in toc})} file markdown.\n")

    demo_queries = [
        "payment methods",
        "return and refund policy",
        "seller listing regulations",
        "order tracking",
        "danh sách sản phẩm cấm đăng bán",
        "xyzabc123nonsense",
    ]
    for q in demo_queries:
        _warn(f"Query: {q}")
        results = pageindex_search(q, top_k=3)
        if not results:
            _warn("  (không có kết quả)")
        for r in results:
            section = r["metadata"].get("section", "")
            _warn(f"  [{r['score']:.3f}] ({section}) {r['content'][:100]}...")
        _warn("-" * 60)
