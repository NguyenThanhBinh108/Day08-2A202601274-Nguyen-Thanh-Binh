"""
Task 8 — PageIndex Vectorless RAG (Fallback khi hybrid search không đủ tốt).

Đăng ký tài khoản: https://pageindex.ai/  →  Developer Dashboard → API Key
SDK:            https://github.com/VectifyAI/PageIndex
Docs:           https://docs.pageindex.ai/sdk

PageIndex khác RAG truyền thống:
    - KHÔNG dùng vector store: xây "tree index" theo cấu trúc (chương/mục/tiêu đề)
      của tài liệu, sau đó LLM "suy luận" (reasoning) điều hướng trên cây để tìm
      đúng phần liên quan → relevance thay vì similarity.
    - KHÔNG cần chunking: giữ nguyên tính toàn vẹn ngữ nghĩa của từng section.

Quy trình:
    1. upload_documents(): convert markdown → PDF (PageIndex nhận PDF), submit lên
       cloud, cache doc_id vào pageindex_doc_ids.json (đã gitignore).
    2. pageindex_search(): với mỗi doc_id → submit_query → poll get_retrieval
       → parse retrieved_nodes thành danh sách kết quả.

⚠️ API `/retrieval` là legacy (vẫn hoạt động, response đôi khi kèm field
"deprecation"). Response trả "retrieved_nodes"; mỗi node có "relevant_contents"
dạng list[dict] hoặc list[list[dict]] tuỳ phiên bản — parser bên dưới xử lý cả hai.
"""

import json
import time
import unicodedata
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PDF_DIR = Path(__file__).parent.parent / "pageindex_pdfs"      # gitignored
DOC_IDS_FILE = Path(__file__).parent.parent / "pageindex_doc_ids.json"  # gitignored

POLL_INTERVAL_SEC = 2.0
POLL_TIMEOUT_SEC = 120.0

# Danh sách font Unicode TTF để render được tiếng Việt (macOS / Linux)
_UNICODE_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",       # Debian/Ubuntu
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",                # Fedora/RHEL
    "/usr/share/fonts/TTF/DejaVuSans.ttf",                   # Arch
]


def strip_diacritics(text: str) -> str:
    """
    Bỏ dấu tiếng Việt (fallback khi không tìm thấy font Unicode).
    Chỉ dùng khi bắt buộc — bỏ dấu làm giảm độ chính xác của PageIndex OCR nên
    luôn ưu tiên font Unicode.
    """
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def _get_client():
    """Khởi tạo PageIndexClient — raise nếu thiếu API key."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError(
            "PAGEINDEX_API_KEY chưa được set trong .env — đăng ký tại https://pageindex.ai/"
        )
    try:
        from pageindex.client import PageIndexClient
    except ImportError as e:
        raise RuntimeError(
            "Chưa cài SDK pageindex: pip install pageindex"
        ) from e
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _find_unicode_font() -> str | None:
    """Tìm font TTF hỗ trợ Unicode trên hệ thống."""
    for path in _UNICODE_FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def convert_markdown_to_pdf(markdown_path: Path, pdf_path: Path):
    """
    Convert 1 file markdown → PDF đơn giản bằng fpdf2.

    PageIndex chỉ nhận PDF. Ưu tiên render tiếng Việt đúng dấu bằng font Unicode
    (Arial Unicode / DejaVu Sans); nếu không có font → strip dấu (vẫn chạy được).
    """
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    font_path = _find_unicode_font()
    needs_strip = font_path is None

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    if font_path:
        pdf.add_font("VNFont", "", font_path)
        pdf.set_font("VNFont", size=11)
    else:
        print(f"  ⚠ Không tìm thấy font Unicode — sẽ bỏ dấu tiếng Việt: {pdf_path.name}")
        pdf.set_font("helvetica", size=11)

    # Gộp toàn bộ dòng thành MỘT text rồi gọi multi_cell đúng 1 lần.
    # Lý do: fpdf2 2.8.x gọi multi_cell lặp lại trên cùng page với font TTF Unicode
    # (Arial Unicode) sẽ lỗi "Not enough horizontal space to render a single character";
    # gọi 1 lần duy nhất cho cả text (có auto page-break) thì hoạt động bình thường.
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    body_lines = []
    for line in lines:
        if needs_strip:
            line = strip_diacritics(line)
        body_lines.append(line.strip())

    pdf.multi_cell(
        0, 5, "\n".join(body_lines),
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.output(str(pdf_path))


def upload_documents(force: bool = False) -> dict[str, str]:
    """
    Upload toàn bộ markdown documents lên PageIndex.

    Args:
        force: True → upload lại kể cả khi doc_id đã cache

    Returns:
        dict {source_filename: doc_id}
    """
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY chưa được set trong .env")

    # Load cache cũ
    cache: dict[str, str] = {}
    if DOC_IDS_FILE.exists() and not force:
        try:
            cache = json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}

    md_files = sorted(STANDARDIZED_DIR.rglob("*.md")) if STANDARDIZED_DIR.exists() else []
    if not md_files:
        print("  ⚠ Không tìm thấy file .md nào trong data/standardized/")
        return cache

    client = _get_client()
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    uploaded: dict[str, str] = {}
    for md_file in md_files:
        if md_file.name in cache and not force:
            print(f"  ⏭ Skipped (đã upload): {md_file.name} -> {cache[md_file.name]}")
            uploaded[md_file.name] = cache[md_file.name]
            continue

        pdf_path = PDF_DIR / f"{md_file.stem}.pdf"
        print(f"  → Convert: {md_file.name}")
        convert_markdown_to_pdf(md_file, pdf_path)

        print(f"  → Upload: {md_file.name}")
        resp = client.submit_document(str(pdf_path))
        doc_id = resp.get("doc_id") or resp.get("id")
        if not doc_id:
            print(f"  ✗ Không nhận được doc_id cho {md_file.name}: {resp}")
            continue
        print(f"  ✓ {md_file.name} -> {doc_id}")
        uploaded[md_file.name] = doc_id

    # Lưu cache
    if uploaded:
        cache.update(uploaded)
        DOC_IDS_FILE.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return cache


def _parse_relevant_contents(node: dict) -> list[dict]:
    """
    Parse "relevant_contents" của 1 retrieved_node.

    Format hiện tại: list[{page_index, relevant_content}]
    Format cũ:        list[list[{section_title, relevant_content}]]
    """
    parsed: list[dict] = []
    relevant = node.get("relevant_contents") or []
    for group in relevant:
        if isinstance(group, list):        # format cũ: list of lists
            for item in group:
                if isinstance(item, dict):
                    parsed.append(item)
        elif isinstance(group, dict):      # format hiện tại: list of dicts
            parsed.append(group)
    return parsed


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt (Task 9).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,        # PageIndex không trả score → gán theo thứ hạng
            'metadata': dict,      # section title, page_index, doc_id
            'source': 'pageindex'
        }
    """
    if not PAGEINDEX_API_KEY:
        print("  ⚠ pageindex_search: PAGEINDEX_API_KEY chưa set → trả về rỗng")
        return []

    # Lấy doc_id từ cache (đã upload ở bước trước)
    if not DOC_IDS_FILE.exists():
        print("  ⚠ pageindex_search: chưa upload documents (chạy upload_documents() trước)")
        return []
    try:
        doc_ids: dict[str, str] = json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    client = _get_client()
    all_results: list[dict] = []

    for source, doc_id in doc_ids.items():
        try:
            if not client.is_retrieval_ready(doc_id):
                print(f"  ⏳ Document chưa sẵn sàng retrieval: {source}")
                continue

            resp = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = resp.get("retrieval_id") or resp.get("id")
            if not retrieval_id:
                print(f"  ⚠ Không nhận được retrieval_id cho {source}: {resp}")
                continue

            # Poll cho đến khi completed (timeout 120s)
            retrieval: dict = {}
            deadline = time.time() + POLL_TIMEOUT_SEC
            while time.time() < deadline:
                retrieval = client.get_retrieval(retrieval_id)
                status = retrieval.get("status", "processing")
                if status == "completed":
                    break
                time.sleep(POLL_INTERVAL_SEC)
            else:
                print(f"  ⚠ Timeout khi retrieval document {source}")
                continue

            for node in retrieval.get("retrieved_nodes", []):
                for item in _parse_relevant_contents(node):
                    content = (
                        item.get("relevant_content")
                        or item.get("content")
                        or ""
                    ).strip()
                    if not content:
                        continue
                    all_results.append({
                        "content": content,
                        "score": 0.0,  # set theo rank bên dưới
                        "metadata": {
                            "section": item.get("section_title") or node.get("title"),
                            "page_index": item.get("page_index"),
                            "node_id": node.get("node_id"),
                            "doc_id": doc_id,
                            "source_file": source,
                        },
                        "source": "pageindex",
                    })
        except Exception as e:
            print(f"  ⚠ PageIndex query lỗi ({source}): {e}")
            continue

    # PageIndex không trả score — gán score theo thứ hạng (descending)
    for rank, result in enumerate(all_results):
        result["score"] = round(1.0 / (rank + 1), 4)
    return all_results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents(force=True)

        print("\nTest query:")
        results = pageindex_search("danh sách sản phẩm cấm đăng bán", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
