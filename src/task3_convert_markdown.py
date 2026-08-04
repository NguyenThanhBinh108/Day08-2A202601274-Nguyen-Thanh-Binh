"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[pdf]"
    # Lưu ý: cần extra [pdf] để convert được file PDF. Chỉ "pip install markitdown"
    # (không có extra) sẽ báo MissingDependencyException khi convert PDF, dù JSON/DOCX
    # vẫn convert bình thường.

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Ngưỡng tối thiểu của file markdown đầu ra. File ngắn hơn ngưỡng này gần như chắc chắn
# là convert lỗi (PDF scan không có text layer, SPA chỉ crawl được tiêu đề...) nên bị bỏ
# qua kèm cảnh báo thay vì ghi ra một file rỗng gây nhiễu cho các task sau.
MIN_CONTENT_CHARS = 200

# Định dạng nguồn được xử lý ở mỗi nhánh.
LEGAL_EXTENSIONS = (".pdf", ".docx", ".doc")
NEWS_EXTENSIONS = (".json",)

# Lazy singleton: KHÔNG khởi tạo MarkItDown ở cấp module. pytest import cả file này,
# import phải nhanh và không được crash khi thiếu dependency của markitdown.
_markitdown: Optional[Any] = None


def _print(message: str) -> None:
    """
    In ra stdout, an toàn với console Windows mặc định (cp1252 không encode được ✓ / ⚠).

    Fallback thay ký tự không encode được bằng '?' thay vì để UnicodeEncodeError
    làm chết cả lô convert.
    """
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(message.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def _get_markitdown() -> Optional[Any]:
    """
    Trả về instance MarkItDown dùng chung (khởi tạo một lần, cache lại).

    Trả về None nếu chưa cài markitdown — gọi hàm vẫn chạy được và degrade
    thành "convert 0 file" thay vì raise.
    """
    global _markitdown
    if _markitdown is None:
        try:
            from markitdown import MarkItDown

            _markitdown = MarkItDown()
        except Exception as exc:  # ImportError hoặc lỗi khởi tạo dependency
            _print(f"⚠ Không khởi tạo được MarkItDown: {exc}")
            _print('  Cài đặt: pip install "markitdown[pdf]"')
            return None
    return _markitdown


# ---------------------------------------------------------------------------
# LÀM SẠCH MARKDOWN
# ---------------------------------------------------------------------------
# Vì sao bước này quan trọng: trang help center crawl về mang theo toàn bộ khung
# điều hướng (Skip to main content / Sign in / Watchlist / Deals...), cú pháp
# markdown link và URL trần. Đo trên corpus thật: 324.107 ký tự chứa tới 722
# markdown link và 735 URL.
#
# Rác này đi thẳng vào chunk → vector store → context của LLM, khiến câu trả lời
# lẫn nguyên chuỗi "[Check the status](https://www.ebay.com/help/action?topicid=...)"
# và đọc như bị đứt đoạn. Làm sạch ở đây là rẻ nhất: sửa một lần, mọi task phía
# sau (4, 5, 6, 8, 9, 10) đều hưởng.

_RE_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_RE_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_RE_BARE_URL = re.compile(r"https?://\S+")
# Chuỗi định danh phiên kiểu 924430616488a6f264e0-09af-46bb-...:19fcbc840a01
_RE_SESSION_ID = re.compile(r"^[0-9a-f]{12,}[0-9a-f:\-]*$", re.IGNORECASE)
_RE_BLANK_RUNS = re.compile(r"\n{3,}")
_RE_INLINE_SPACES = re.compile(r"[ \t]{2,}")

# Mẩu chữ do widget động của trang sinh ra, dính vào giữa câu nội dung thật.
# Ví dụ: "Open a return request - opens in new window or tab" -> bỏ phần đuôi.
_RE_WIDGET_NOISE = re.compile(
    r"\s*[-–]\s*opens in (?:a )?new (?:window|tab)(?: or tab)?", re.IGNORECASE
)

# Dòng rác của khung tìm kiếm / trạng thái tải, không mang nội dung chính sách.
_NOISE_MARKERS = ("rlogid", "loading...", "javascript is required")

# Chỉ khớp khi CẢ DÒNG đúng bằng cụm này (so sánh sau khi lower + strip), nên không
# xoá nhầm đoạn nội dung thật có chứa cùng từ khoá.
_NAV_EXACT = frozenset(
    {
        "ship to", "sell", "watchlist", "expand watchlist", "my ebay",
        "expand my ebay", "notifications", "expand cart", "hi!",
        "back to home page", "skip to main content", "help & contact",
        "was this article helpful?", "yes", "no",
        "related help topics", "still have questions?",
        "bạn có hài lòng với bài viết này?", "hài lòng", "không hài lòng",
        "xem thêm:", "xem thêm", "chia sẻ bài viết",
        "xin chào, shopee có thể giúp gì cho bạn?",
    }
)


def _strip_bullet(line: str) -> str:
    """Bỏ ký hiệu bullet/đánh số đầu dòng để đo độ dài phần chữ thật."""
    return re.sub(r"^\s*(?:[-*+•]|\d+[.)])\s*", "", line).strip()


def clean_markdown(text: str) -> str:
    """
    Bỏ khung điều hướng, cú pháp link và URL trần khỏi markdown đã crawl.

    Nguyên tắc giữ nội dung: KHÔNG xoá theo danh sách từ khoá (dễ xoá nhầm đoạn
    chính sách thật). Quy tắc chính là cấu trúc — một dòng mà sau khi gỡ hết link
    gần như không còn chữ nào thì đó là thanh điều hướng, không phải nội dung.

    Args:
        text: Markdown thô từ MarkItDown hoặc Crawl4AI.

    Returns:
        Markdown đã làm sạch, giữ nguyên heading và đoạn văn.
    """
    if not text:
        return ""

    kept: list[str] = []
    for raw_line in text.splitlines():
        line = _RE_IMAGE.sub("", raw_line)

        # Đếm link TRƯỚC khi gỡ, để nhận ra dòng vốn chỉ toàn link.
        link_count = len(_RE_LINK.findall(line))
        # Giữ lại phần chữ hiển thị của link, bỏ URL: [Trả hàng](http://...) -> Trả hàng
        line = _RE_LINK.sub(r"\1", line)
        line = _RE_BARE_URL.sub("", line)
        line = _RE_WIDGET_NOISE.sub("", line)
        line = _RE_INLINE_SPACES.sub(" ", line).strip()

        if not line:
            kept.append("")
            continue

        lowered = line.lower().strip(" *_#:|-")
        if lowered in _NAV_EXACT:
            continue
        if any(marker in lowered for marker in _NOISE_MARKERS):
            continue
        if _RE_SESSION_ID.match(line.replace(" ", "")):
            continue

        # Dòng chỉ toàn link và còn lại rất ít chữ → thanh điều hướng.
        if link_count >= 2 and len(line) < 90:
            continue

        # Mục menu dạng bullet ("* Summary", "* Bids/Offers"): bản chất là MỘT link
        # với nhãn ngắn. Câu chính sách thật gần như luôn dài hơn 45 ký tự và hiếm
        # khi là link, nên ngưỡng này cắt được menu mà giữ nguyên nội dung.
        if link_count >= 1 and len(_strip_bullet(line)) < 45:
            continue

        # Dòng quá ngắn và không phải heading/bullet thì không mang thông tin gì.
        if len(line) < 3 and not line.startswith("#"):
            continue

        kept.append(line)

    cleaned = "\n".join(kept)
    return _RE_BLANK_RUNS.sub("\n\n", cleaned).strip()


def _write_markdown(
    output_path: Path, text: str, source_name: str, clean: bool = True
) -> bool:
    """
    Làm sạch rồi ghi text ra file markdown nếu đủ dài. Trả về True nếu đã ghi.

    Args:
        clean: Đặt False khi caller đã tự làm sạch phần nội dung và ghép header
            vào sau (nhánh news) — tránh làm sạch hai lần và xoá nhầm URL nguồn.

    Bỏ qua (kèm cảnh báo rõ tên file) khi nội dung rỗng hoặc ngắn hơn MIN_CONTENT_CHARS.
    """
    if clean:
        raw_len = len((text or "").strip())
        text = clean_markdown(text or "")
        if raw_len and text:
            removed = raw_len - len(text)
            if removed > 0:
                _print(f"    · làm sạch: bỏ {removed:,} ký tự rác ({removed / raw_len:.0%})")
    if not text:
        _print(f"  ⚠ Bỏ qua {source_name}: convert ra nội dung RỖNG")
        return False
    if len(text) <= MIN_CONTENT_CHARS:
        _print(
            f"  ⚠ Bỏ qua {source_name}: chỉ {len(text)} ký tự "
            f"(cần > {MIN_CONTENT_CHARS}) — nhiều khả năng convert lỗi"
        )
        return False

    output_path.write_text(text, encoding="utf-8")
    _print(f"  ✓ Saved: {output_path} ({len(text)} chars)")
    return True


def _extract_news_content(data: Any) -> str:
    """
    Lấy phần nội dung markdown từ JSON bài viết đã crawl.

    Crawl4AI có thể trả `markdown` dạng object đã serialize thành dict
    (raw_markdown / fit_markdown) chứ không phải string thuần — xử lý cả hai dạng.
    """
    if not isinstance(data, dict):
        return ""

    content = data.get("content_markdown") or data.get("markdown") or data.get("content") or ""
    if isinstance(content, dict):
        content = content.get("raw_markdown") or content.get("fit_markdown") or ""
    return content if isinstance(content, str) else str(content)


def convert_legal_docs() -> int:
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.is_dir():
        _print(f"⚠ Không tìm thấy thư mục nguồn: {legal_dir} (chạy task 1 trước)")
        return 0

    md = _get_markitdown()
    if md is None:
        return 0

    converted = 0
    # sorted() để thứ tự convert ổn định giữa các lần chạy / các máy.
    for filepath in sorted(legal_dir.iterdir()):
        if filepath.is_file() and filepath.suffix.lower() in LEGAL_EXTENSIONS:
            print(f"Converting: {filepath.name}")
            # Bắt lỗi TỪNG FILE: một PDF hỏng / thiếu extra [pdf] không được làm chết cả lô.
            try:
                result = md.convert(str(filepath))
                output_path = output_dir / f"{filepath.stem}.md"
                if _write_markdown(output_path, result.text_content, filepath.name):
                    converted += 1
            except Exception as exc:
                _print(f"  ✗ Lỗi khi convert {filepath.name}: {type(exc).__name__}: {exc}")
                continue

    return converted


def convert_news_articles() -> int:
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.is_dir():
        _print(f"⚠ Không tìm thấy thư mục nguồn: {news_dir} (chạy task 2 trước)")
        return 0

    converted = 0
    for filepath in sorted(news_dir.iterdir()):
        if filepath.is_file() and filepath.suffix.lower() in NEWS_EXTENSIONS:
            print(f"Converting: {filepath.name}")
            # Bắt lỗi TỪNG FILE: JSON hỏng / thiếu field không được làm chết cả lô.
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                output_path = output_dir / f"{filepath.stem}.md"

                # Làm sạch phần NỘI DUNG trước, rồi mới ghép header vào.
                # Thứ tự này quan trọng: nếu làm sạch cả header thì URL trong dòng
                # "**Source:**" cũng bị bộ lọc URL trần xoá mất, kéo theo mất luôn
                # thông tin nguồn gốc mà Task 8 và phần trích dẫn cần đến.
                body = clean_markdown(_extract_news_content(data))

                header = f"# {data.get('title', 'Unknown')}\n\n"
                header += f"**Source:** {data.get('url', 'N/A')}\n"
                header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

                content = header + body
                if _write_markdown(output_path, content, filepath.name, clean=False):
                    converted += 1
            except Exception as exc:
                _print(f"  ✗ Lỗi khi convert {filepath.name}: {type(exc).__name__}: {exc}")
                continue

    return converted


def convert_all() -> int:
    """Convert toàn bộ files. Trả về tổng số file markdown đã ghi thành công."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    legal_count = convert_legal_docs()

    print("\n--- News Articles ---")
    news_count = convert_news_articles()

    total = legal_count + news_count

    print("\n--- Tổng kết ---")
    print(f"Legal : {legal_count} file")
    print(f"News  : {news_count} file")
    print(f"Tổng  : {total} file")

    # Bug của bản starter: khi thư mục nguồn không có file nào phù hợp thì vòng lặp
    # không chạy lần nào, hàm kết thúc âm thầm và vẫn in "Done" như thể thành công.
    # Phải báo lỗi thật rõ để người chạy biết pipeline chưa có dữ liệu đầu vào.
    if total == 0:
        _print("\n⚠ CẢNH BÁO: không convert được file nào - kiểm tra lại data/landing/")
        _print(f"  - Nguồn legal: {LANDING_DIR / 'legal'} (cần .pdf/.docx/.doc)")
        _print(f"  - Nguồn news : {LANDING_DIR / 'news'} (cần .json)")
        _print("  - Chạy task 1 (tải văn bản) và task 2 (crawl bài viết) trước.")
        return 0

    _print(f"\n✓ Done! Output tại: {OUTPUT_DIR}")
    return total


if __name__ == "__main__":
    # Exit code != 0 khi không convert được gì, để CI / script gọi ngoài phát hiện được lỗi.
    sys.exit(0 if convert_all() > 0 else 1)
