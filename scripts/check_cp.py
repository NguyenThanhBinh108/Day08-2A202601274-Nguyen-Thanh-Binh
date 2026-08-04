"""
Công cụ QA checkpoint cho Lab 08 — RAG Pipeline (E-commerce Support).

Mục tiêu: chạy MỘT lệnh là biết cả nhóm đang đứng ở checkpoint nào và còn thiếu gì.
Script KHÔNG phụ thuộc pytest (nhiều thành viên chạy song song sẽ đụng nhau), chỉ
kiểm tra bằng filesystem + import + gọi hàm nhẹ.

Cách dùng (chạy từ thư mục gốc dự án):
    .\\.venv\\Scripts\\python.exe scripts\\check_cp.py            # kiểm tra CP0 → CP6
    .\\.venv\\Scripts\\python.exe scripts\\check_cp.py --cp 2     # chỉ kiểm tra CP2

Exit code:
    0 — tất cả checkpoint được kiểm tra đều PASS
    1 — còn ít nhất một checkpoint FAIL (dùng được trong CI / pre-push hook)

Nguyên tắc thiết kế:
    - Mỗi bước bọc try/except riêng: một bước hỏng KHÔNG được làm chết cả script.
      Nhóm cần biết "CP4 hỏng vì lý do X" chứ không phải một traceback rồi mất
      toàn bộ thông tin của CP5, CP6.
    - Không làm việc nặng ở cấp module: mọi import của src.* và chromadb đều nằm
      TRONG hàm, nên `python scripts/check_cp.py --cp 6` không cần load model.
    - Mọi kết quả in ra kèm SỐ LIỆU CỤ THỂ (bao nhiêu file, bao nhiêu vector,
      score bao nhiêu) — "PASS" chung chung không giúp gỡ lỗi.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# =============================================================================
# ĐƯỜNG DẪN & HẰNG SỐ
# =============================================================================

# resolve() để script chạy đúng dù được gọi từ bất kỳ thư mục nào.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Cho phép `import src.task5_semantic_search` khi chạy trực tiếp file này.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ENV_FILE = PROJECT_ROOT / ".env"
DATA_DIR = PROJECT_ROOT / "data"
LANDING_LEGAL_DIR = DATA_DIR / "landing" / "legal"
LANDING_NEWS_DIR = DATA_DIR / "landing" / "news"
STANDARDIZED_DIR = DATA_DIR / "standardized"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
APP_FILE = PROJECT_ROOT / "app.py"
EVALUATION_DIR = PROJECT_ROOT / "group_project" / "evaluation"
GOLDEN_DATASET_FILE = EVALUATION_DIR / "golden_dataset.json"
RESULTS_FILE = EVALUATION_DIR / "results.md"

# CP0 — các package bắt buộc: (tên module để import, tên distribution để lấy version)
REQUIRED_PACKAGES: tuple[tuple[str, str], ...] = (
    ("chromadb", "chromadb"),
    ("sentence_transformers", "sentence-transformers"),
    ("ragas", "ragas"),
    ("streamlit", "streamlit"),
    ("crawl4ai", "crawl4ai"),
)

# Chuỗi thường gặp trong API key mẫu — coi là "chưa điền key thật".
PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "...", "your", "xxx", "changeme", "<", ">", "todo", "paste", "here",
)
MIN_API_KEY_LEN = 20

# CP1 — ngưỡng số lượng / kích thước tối thiểu
LEGAL_SUFFIXES = (".pdf", ".docx", ".doc")
NEWS_SUFFIXES = (".json", ".html", ".md", ".txt")
MIN_LEGAL_FILES = 3
MIN_NEWS_FILES = 5
MIN_STANDARDIZED_FILES = 1
MIN_LEGAL_BYTES = 1024       # > 1 KB
MIN_NEWS_BYTES = 500         # > 500 B
MIN_STANDARDIZED_CHARS = 200  # > 200 ký tự

# CP2 — collection mặc định (sẽ ưu tiên đọc từ src/task4 nếu import được)
DEFAULT_COLLECTION_NAME = "ecommerce_support_docs"
PROBE_QUERY_EN = "payment methods"

# CP4/CP5 — truy vấn & ngưỡng dùng để probe
PROBE_QUERY_RETRIEVE = "return refund policy"
HIGH_THRESHOLD = 0.99   # ép fallback để kiểm tra pipeline không crash
MIN_GOLDEN_ITEMS = 15

GIT_TIMEOUT = 20  # giây — tránh treo script nếu git bị kẹt (credential prompt...)
LABEL_WIDTH = 40


# =============================================================================
# TIỆN ÍCH IN ẤN (an toàn với console Windows không hỗ trợ UTF-8)
# =============================================================================

def _init_stdout() -> bool:
    """
    Cố gắng bật UTF-8 cho stdout và cho biết console có in được ký tự tick/cross không.

    Console Windows mặc định là cp1252/cp437 → in "✓" sẽ ném UnicodeEncodeError và
    làm chết script. Ta thử reconfigure sang UTF-8 trước; nếu không được thì rơi về
    ký hiệu ASCII.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "✓✗⚠".encode(encoding)
        return True
    except Exception:
        return False


_UNICODE_OK = _init_stdout()
TICK = "✓" if _UNICODE_OK else "v"
CROSS = "✗" if _UNICODE_OK else "x"
INFO = "·" if _UNICODE_OK else "-"
ARROW = "→" if _UNICODE_OK else "->"


def _pad(text: str, width: int = LABEL_WIDTH) -> str:
    """Căn trái theo số ký tự (không phải byte) để bảng thẳng hàng cả khi có dấu tiếng Việt."""
    if len(text) >= width:
        return text
    return text + " " * (width - len(text))


# =============================================================================
# MÔ HÌNH DỮ LIỆU KẾT QUẢ
# =============================================================================

@dataclass
class CheckRow:
    """Một dòng kiểm tra trong bảng kết quả của checkpoint."""

    ok: bool
    label: str
    detail: str = ""
    required: bool = True  # required=False → dòng thông tin, không ảnh hưởng PASS/FAIL


@dataclass
class CheckpointResult:
    """Kết quả tổng hợp của một checkpoint."""

    cp: int
    title: str
    rows: list[CheckRow] = field(default_factory=list)
    error: Optional[str] = None   # lỗi ngoài dự kiến của chính hàm kiểm tra
    elapsed: float = 0.0

    @property
    def passed(self) -> bool:
        if self.error is not None:
            return False
        required_rows = [r for r in self.rows if r.required]
        return bool(required_rows) and all(r.ok for r in required_rows)

    def add(self, ok: bool, label: str, detail: str = "", required: bool = True) -> None:
        self.rows.append(CheckRow(ok=ok, label=label, detail=detail, required=required))


def print_checkpoint(result: CheckpointResult) -> None:
    """In một checkpoint dạng bảng: [tick/cross] nhãn | số liệu cụ thể."""
    print()
    print("=" * 74)
    print(f" CP{result.cp} — {result.title}")
    print("=" * 74)

    for row in result.rows:
        if not row.required:
            mark = INFO
        else:
            mark = TICK if row.ok else CROSS
        line = f"  [{mark}] {_pad(row.label)}"
        if row.detail:
            line += f" | {row.detail}"
        print(line)

    if result.error:
        print(f"  [{CROSS}] {_pad('LỖI NGOÀI DỰ KIẾN')} | {result.error}")

    verdict = "PASS" if result.passed else "FAIL"
    print(f"  {ARROW} CP{result.cp}: {verdict}  ({result.elapsed:.1f}s)")


# =============================================================================
# TIỆN ÍCH KIỂM TRA (đều "an toàn": không bao giờ ném ra ngoài)
# =============================================================================

def _safe_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[bool, Any, str]:
    """
    Gọi một hàm và bắt MỌI lỗi.

    Returns:
        (thành công, giá trị trả về, mô tả lỗi)
    """
    try:
        return True, fn(*args, **kwargs), ""
    except NotImplementedError as exc:
        return False, None, f"chưa implement ({exc})"
    except Exception as exc:  # noqa: BLE001 — chủ đích bắt rộng, đây là công cụ QA
        return False, None, f"{type(exc).__name__}: {exc}"


def _safe_import(module_name: str) -> tuple[Optional[Any], str]:
    """Import một module và bắt mọi lỗi (module của bạn cùng nhóm có thể đang dở dang)."""
    try:
        return importlib.import_module(module_name), ""
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _package_version(dist_name: str) -> str:
    """Lấy version từ metadata (không import package → rất nhẹ)."""
    try:
        return importlib.metadata.version(dist_name)
    except Exception:  # noqa: BLE001
        return "?"


def _read_env_values(path: Path) -> dict[str, str]:
    """
    Đọc file .env thủ công (utf-8), bỏ qua dòng trống và dòng comment.

    Tự parse thay vì dùng python-dotenv để: (1) không phụ thuộc thư viện, (2) không
    ghi đè biến môi trường của tiến trình đang chạy.
    """
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return values
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _looks_like_placeholder(value: str) -> bool:
    """Key mẫu (sk-or-v1-..., your_key_here) phải bị coi là CHƯA có key."""
    if len(value) < MIN_API_KEY_LEN:
        return True
    lowered = value.lower()
    return any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def _mask(value: str) -> str:
    """Che bớt API key khi in ra màn hình (script này hay được chụp màn hình gửi nhóm)."""
    if len(value) <= 12:
        return "*" * len(value)
    return f"{value[:8]}...{value[-4:]}"


def _scan_files(
    folder: Path, suffixes: tuple[str, ...], min_bytes: int
) -> tuple[list[Path], list[Path]]:
    """
    Quét file theo đuôi mở rộng trong folder (đệ quy).

    Returns:
        (file đạt kích thước, file quá nhỏ) — tách riêng để báo cáo cụ thể file nào lỗi.
    """
    if not folder.is_dir():
        return [], []
    big: list[Path] = []
    small: list[Path] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        (big if size > min_bytes else small).append(path)
    return big, small


def _scan_markdown(folder: Path, min_chars: int) -> tuple[list[Path], list[Path]]:
    """Như _scan_files nhưng đo bằng SỐ KÝ TỰ (đọc utf-8) — chuẩn hơn cho file tiếng Việt."""
    if not folder.is_dir():
        return [], []
    big: list[Path] = []
    small: list[Path] = []
    for path in sorted(folder.rglob("*.md")):
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        (big if len(content) > min_chars else small).append(path)
    return big, small


def _names(paths: list[Path], limit: int = 3) -> str:
    """Liệt kê vài tên file để người đọc biết ngay file nào có vấn đề."""
    if not paths:
        return "-"
    shown = ", ".join(p.name for p in paths[:limit])
    if len(paths) > limit:
        shown += f", +{len(paths) - limit} file nữa"
    return shown


def _describe_results(results: Any) -> tuple[bool, str]:
    """
    Kiểm tra nhanh một list kết quả retrieval theo schema chung của lab.

    Returns:
        (đúng schema và có kết quả, mô tả để in ra)
    """
    if not isinstance(results, list):
        return False, f"trả về {type(results).__name__}, mong đợi list"
    if not results:
        return False, "0 kết quả"
    first = results[0]
    if not isinstance(first, dict):
        return False, f"phần tử là {type(first).__name__}, mong đợi dict"
    missing = [k for k in ("content", "score", "metadata") if k not in first]
    if missing:
        return False, f"{len(results)} kết quả nhưng thiếu key: {', '.join(missing)}"
    try:
        scores = [float(r.get("score", 0.0)) for r in results if isinstance(r, dict)]
    except (TypeError, ValueError):
        return False, f"{len(results)} kết quả nhưng score không phải số"
    sorted_desc = all(a >= b for a, b in zip(scores, scores[1:]))
    top_source = ""
    meta = first.get("metadata")
    if isinstance(meta, dict):
        top_source = str(meta.get("source", ""))
    detail = f"{len(results)} kết quả, top score={scores[0]:.4f}"
    if top_source:
        detail += f", top source={top_source}"
    if not sorted_desc:
        return False, detail + " — CHƯA sort giảm dần"
    return True, detail


# =============================================================================
# CP0 — Môi trường & API key
# =============================================================================

def check_cp0() -> CheckpointResult:
    """CP0: venv đã cài đủ package chính và .env đã có OPENROUTER_API_KEY thật."""
    result = CheckpointResult(cp=0, title="Setup môi trường (venv + .env)")

    result.add(True, "Python interpreter",
               f"{sys.version.split()[0]} @ {sys.executable}", required=False)

    for module_name, dist_name in REQUIRED_PACKAGES:
        try:
            spec = importlib.util.find_spec(module_name)
        except Exception as exc:  # noqa: BLE001 — find_spec có thể ném nếu package hỏng
            result.add(False, f"package {module_name}", f"lỗi khi dò tìm: {exc}")
            continue
        if spec is None:
            result.add(False, f"package {module_name}",
                       f"THIẾU — pip install {dist_name}")
        else:
            result.add(True, f"package {module_name}", f"v{_package_version(dist_name)}")

    # --- .env ---
    if not ENV_FILE.is_file():
        result.add(False, ".env", f"KHÔNG tồn tại tại {ENV_FILE}")
        return result

    result.add(True, ".env", f"tồn tại ({ENV_FILE.stat().st_size} B)")
    env_values = _read_env_values(ENV_FILE)
    api_key = env_values.get("OPENROUTER_API_KEY", "")
    if not api_key:
        result.add(False, "OPENROUTER_API_KEY", "chưa khai báo trong .env")
    elif _looks_like_placeholder(api_key):
        result.add(False, "OPENROUTER_API_KEY",
                   f"vẫn là placeholder ({_mask(api_key)})")
    else:
        result.add(True, "OPENROUTER_API_KEY",
                   f"{_mask(api_key)} ({len(api_key)} ký tự)")

    # Các key tùy chọn: chỉ báo thông tin, không tính PASS/FAIL.
    for optional_key in ("PAGEINDEX_API_KEY", "JINA_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        value = env_values.get(optional_key, "")
        has_key = bool(value) and not _looks_like_placeholder(value)
        result.add(has_key, f"{optional_key} (tùy chọn)",
                   _mask(value) if has_key else "chưa có → module phải degrade gracefully",
                   required=False)

    return result


# =============================================================================
# CP1 — Thu thập & chuẩn hóa dữ liệu
# =============================================================================

def check_cp1() -> CheckpointResult:
    """CP1: đủ file legal/news ở landing zone và đã convert sang markdown."""
    result = CheckpointResult(cp=1, title="Thu thập dữ liệu & chuẩn hóa Markdown")

    legal_ok, legal_small = _scan_files(LANDING_LEGAL_DIR, LEGAL_SUFFIXES, MIN_LEGAL_BYTES)
    detail = f"{len(legal_ok)}/{MIN_LEGAL_FILES} file >1KB ({_names(legal_ok)})"
    if legal_small:
        detail += f" | {len(legal_small)} file quá nhỏ: {_names(legal_small)}"
    if not LANDING_LEGAL_DIR.is_dir():
        detail = f"thư mục không tồn tại: {LANDING_LEGAL_DIR}"
    result.add(len(legal_ok) >= MIN_LEGAL_FILES, "data/landing/legal (.pdf/.docx/.doc)", detail)

    news_ok, news_small = _scan_files(LANDING_NEWS_DIR, NEWS_SUFFIXES, MIN_NEWS_BYTES)
    detail = f"{len(news_ok)}/{MIN_NEWS_FILES} file >500B ({_names(news_ok)})"
    if news_small:
        detail += f" | {len(news_small)} file quá nhỏ: {_names(news_small)}"
    if not LANDING_NEWS_DIR.is_dir():
        detail = f"thư mục không tồn tại: {LANDING_NEWS_DIR}"
    result.add(len(news_ok) >= MIN_NEWS_FILES, "data/landing/news (.json/.html/.md/.txt)", detail)

    md_ok, md_small = _scan_markdown(STANDARDIZED_DIR, MIN_STANDARDIZED_CHARS)
    detail = f"{len(md_ok)}/{MIN_STANDARDIZED_FILES} file .md >200 ký tự ({_names(md_ok)})"
    if md_small:
        detail += f" | {len(md_small)} file quá ngắn: {_names(md_small)}"
    if not STANDARDIZED_DIR.is_dir():
        detail = f"thư mục không tồn tại: {STANDARDIZED_DIR}"
    result.add(len(md_ok) >= MIN_STANDARDIZED_FILES, "data/standardized (.md)", detail)

    # Corpus SONG NGỮ: test dùng query tiếng Anh nên phải có các từ khóa tiếng Anh.
    keywords = ("payment", "methods", "return", "refund", "seller",
                "listing", "order", "tracking")
    found: set[str] = set()
    for path in md_ok:
        try:
            lowered = path.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            continue
        found.update(kw for kw in keywords if kw in lowered)
    missing = [kw for kw in keywords if kw not in found]
    result.add(
        not missing and bool(md_ok),
        "Từ khóa tiếng Anh trong corpus",
        f"{len(found)}/{len(keywords)} từ khóa" + (f" | THIẾU: {', '.join(missing)}" if missing else ""),
        required=False,
    )

    return result


# =============================================================================
# CP2 — Vector store + Semantic/Lexical search
# =============================================================================

def _collection_name() -> str:
    """Ưu tiên đọc COLLECTION_NAME từ Task 4 để luôn khớp với code thật của nhóm."""
    module, _ = _safe_import("src.task4_chunking_indexing")
    if module is not None:
        return str(getattr(module, "COLLECTION_NAME", DEFAULT_COLLECTION_NAME))
    return DEFAULT_COLLECTION_NAME


def _count_chroma_vectors(collection_name: str) -> tuple[bool, str]:
    """Đếm số vector trong collection ChromaDB. Không bao giờ ném lỗi ra ngoài."""
    try:
        import chromadb  # import trong hàm: chromadb khá nặng
    except Exception as exc:  # noqa: BLE001
        return False, f"không import được chromadb: {exc}"
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(name=collection_name)
        count = collection.count()
    except Exception as exc:  # noqa: BLE001 — collection chưa tồn tại là trường hợp thường gặp
        return False, f"không đọc được collection '{collection_name}': {type(exc).__name__}: {exc}"
    return count > 0, f"{count} vector trong collection '{collection_name}'"


def check_cp2() -> CheckpointResult:
    """CP2: chroma_db/ đã index và cả hai nhánh search đều trả kết quả."""
    result = CheckpointResult(cp=2, title="Indexing + Semantic Search + Lexical BM25")

    chroma_exists = CHROMA_DIR.is_dir()
    result.add(chroma_exists, "chroma_db/",
               str(CHROMA_DIR) if chroma_exists
               else "chưa tồn tại — chạy `python -m src.task4_chunking_indexing`")

    collection_name = _collection_name()
    if chroma_exists:
        ok, detail = _count_chroma_vectors(collection_name)
        result.add(ok, "Số vector đã index", detail)
    else:
        result.add(False, "Số vector đã index", "bỏ qua vì chưa có chroma_db/")

    # --- semantic_search ---
    module, import_error = _safe_import("src.task5_semantic_search")
    if module is None:
        result.add(False, "semantic_search()", f"không import được: {import_error}")
    else:
        ok, results, error = _safe_call(module.semantic_search, PROBE_QUERY_EN, top_k=3)
        if not ok:
            result.add(False, f"semantic_search('{PROBE_QUERY_EN}')", error)
        else:
            schema_ok, detail = _describe_results(results)
            result.add(schema_ok, f"semantic_search('{PROBE_QUERY_EN}')", detail)

    # --- lexical_search ---
    module, import_error = _safe_import("src.task6_lexical_search")
    if module is None:
        result.add(False, "lexical_search()", f"không import được: {import_error}")
    else:
        ok, results, error = _safe_call(module.lexical_search, PROBE_QUERY_EN, top_k=3)
        if not ok:
            result.add(False, f"lexical_search('{PROBE_QUERY_EN}')",
                       f"{error} — kiểm tra build_bm25_index(corpus) có được gọi lazy không")
        else:
            schema_ok, detail = _describe_results(results)
            result.add(schema_ok, f"lexical_search('{PROBE_QUERY_EN}')", detail)

    return result


# =============================================================================
# CP3 — Reranking + PageIndex
# =============================================================================

def _fake_ranked_lists() -> list[list[dict]]:
    """
    Hai ranked list giả lập (có phần tử trùng nhau) để thử RRF mà không cần vector store.

    RRF chỉ dựa vào THỨ HẠNG nên dữ liệu giả là đủ để xác nhận thuật toán chạy đúng:
    tài liệu xuất hiện ở top của cả hai list phải được đẩy lên đầu sau khi fuse.
    """
    def make(content: str, score: float, index: int) -> dict:
        return {
            "content": content,
            "score": score,
            "metadata": {"source": "fake_doc.md", "type": "legal", "chunk_index": index},
        }

    dense = [
        make("Return and refund policy: buyers may request a refund within 15 days.", 0.91, 0),
        make("Supported payment methods include ShopeePay wallet and credit cards.", 0.78, 1),
        make("Seller listing regulations prohibit counterfeit goods.", 0.64, 2),
    ]
    sparse = [
        make("Supported payment methods include ShopeePay wallet and credit cards.", 12.4, 1),
        make("Order tracking is available in the My Purchases section.", 9.1, 3),
        make("Return and refund policy: buyers may request a refund within 15 days.", 7.7, 0),
    ]
    return [dense, sparse]


def check_cp3() -> CheckpointResult:
    """CP3: RRF fuse được 2 ranked list; PageIndex trả về list (kể cả khi thiếu API key)."""
    result = CheckpointResult(cp=3, title="Reranking (RRF) + PageIndex Vectorless")

    module, import_error = _safe_import("src.task7_reranking")
    if module is None:
        result.add(False, "rerank_rrf()", f"không import được: {import_error}")
    else:
        ranked_lists = _fake_ranked_lists()
        ok, fused, error = _safe_call(module.rerank_rrf, ranked_lists, top_k=3)
        if not ok:
            result.add(False, "rerank_rrf(dữ liệu giả)", error)
        else:
            schema_ok, detail = _describe_results(fused)
            result.add(schema_ok, "rerank_rrf(2 ranked list giả)", detail)
            if isinstance(fused, list) and fused:
                # Tài liệu xuất hiện ở cả 2 list phải được xếp cao hơn tài liệu chỉ ở 1 list.
                contents = [r.get("content", "") for r in fused if isinstance(r, dict)]
                overlap_first = bool(contents) and "payment methods" in contents[0].lower()
                result.add(overlap_first, "RRF ưu tiên tài liệu trùng ở cả 2 list",
                           f"top-1 = {contents[0][:60]}..." if contents else "-",
                           required=False)

        # rerank() dispatcher — thử luôn cho chắc, nhưng không bắt buộc để PASS CP3.
        ok, reranked, error = _safe_call(
            module.rerank, PROBE_QUERY_RETRIEVE, _fake_ranked_lists()[0], top_k=2, method="rrf"
        )
        result.add(ok and isinstance(reranked, list), "rerank(method='rrf')",
                   f"{len(reranked)} kết quả" if ok and isinstance(reranked, list) else error,
                   required=False)

    module, import_error = _safe_import("src.task8_pageindex_vectorless")
    if module is None:
        result.add(False, "pageindex_search()", f"không import được: {import_error}")
    else:
        ok, results, error = _safe_call(module.pageindex_search, PROBE_QUERY_RETRIEVE, top_k=3)
        if not ok:
            result.add(False, "pageindex_search()",
                       f"{error} — thiếu API key PHẢI trả [] chứ không được raise")
        else:
            # Yêu cầu tối thiểu: trả về list (rỗng cũng chấp nhận khi chưa có API key).
            is_list = isinstance(results, list)
            detail = f"trả về list ({len(results)} kết quả)" if is_list \
                else f"trả về {type(results).__name__}, mong đợi list"
            result.add(is_list, "pageindex_search()", detail)
            if is_list and results:
                sources = {r.get("source") for r in results if isinstance(r, dict)}
                result.add(sources == {"pageindex"}, "pageindex_search(): key 'source'",
                           f"source={sources}", required=False)

    return result


# =============================================================================
# CP4 — Retrieval pipeline hoàn chỉnh
# =============================================================================

def check_cp4() -> CheckpointResult:
    """CP4: retrieve() trả đúng schema và fallback (threshold cao) không làm crash."""
    result = CheckpointResult(cp=4, title="Retrieval Pipeline hoàn chỉnh (Task 9)")

    module, import_error = _safe_import("src.task9_retrieval_pipeline")
    if module is None:
        result.add(False, "retrieve()", f"không import được: {import_error}")
        return result

    threshold = getattr(module, "SCORE_THRESHOLD", None)
    result.add(True, "SCORE_THRESHOLD", str(threshold), required=False)

    ok, results, error = _safe_call(module.retrieve, PROBE_QUERY_RETRIEVE, top_k=3)
    if not ok:
        result.add(False, f"retrieve('{PROBE_QUERY_RETRIEVE}')", error)
    else:
        schema_ok, detail = _describe_results(results)
        result.add(schema_ok, f"retrieve('{PROBE_QUERY_RETRIEVE}')", detail)
        if isinstance(results, list) and results:
            # Task 9 bắt buộc có thêm key 'source' ('hybrid' | 'pageindex').
            missing_source = [i for i, r in enumerate(results)
                              if not isinstance(r, dict) or "source" not in r]
            sources = {r.get("source") for r in results if isinstance(r, dict)}
            result.add(not missing_source, "retrieve(): key 'source' ở mọi kết quả",
                       f"source={sources}" if not missing_source
                       else f"thiếu ở vị trí {missing_source}")

    ok, fallback_results, error = _safe_call(
        module.retrieve, PROBE_QUERY_RETRIEVE, top_k=3, score_threshold=HIGH_THRESHOLD
    )
    fallback_label = f"retrieve(threshold={HIGH_THRESHOLD}) fallback"
    if not ok:
        result.add(False, fallback_label,
                   f"{error} — nhánh fallback phải degrade gracefully, không raise")
    else:
        is_list = isinstance(fallback_results, list)
        sources = {r.get("source") for r in fallback_results if isinstance(r, dict)} if is_list else set()
        detail = f"không crash, {len(fallback_results)} kết quả, source={sources or '-'}" \
            if is_list else f"trả về {type(fallback_results).__name__}, mong đợi list"
        result.add(is_list, fallback_label, detail)

    return result


# =============================================================================
# CP5 — UI + Evaluation
# =============================================================================

def check_cp5() -> CheckpointResult:
    """CP5: app.py chạy được, golden dataset đủ câu hỏi, results.md đã viết."""
    result = CheckpointResult(cp=5, title="Chatbot UI + Evaluation (group project)")

    if APP_FILE.is_file():
        size = APP_FILE.stat().st_size
        result.add(size > 0, "app.py", f"{size} B")
    else:
        result.add(False, "app.py", f"không tồn tại tại {APP_FILE}")

    # golden_dataset.json — tìm ở vị trí chuẩn trước, sau đó quét toàn dự án.
    golden_path: Optional[Path] = GOLDEN_DATASET_FILE if GOLDEN_DATASET_FILE.is_file() else None
    if golden_path is None:
        candidates = [p for p in PROJECT_ROOT.rglob("golden_dataset.json")
                      if ".venv" not in p.parts and ".git" not in p.parts]
        golden_path = candidates[0] if candidates else None

    if golden_path is None:
        result.add(False, "golden_dataset.json", "không tìm thấy trong dự án")
    else:
        try:
            data = json.loads(golden_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — JSON hỏng là lỗi hay gặp khi merge tay
            result.add(False, "golden_dataset.json",
                       f"đọc/parse lỗi: {type(exc).__name__}: {exc}")
        else:
            if isinstance(data, dict):  # cho phép dạng {"questions": [...]}
                items = next((v for v in data.values() if isinstance(v, list)), [])
            elif isinstance(data, list):
                items = data
            else:
                items = []
            result.add(len(items) >= MIN_GOLDEN_ITEMS, "golden_dataset.json",
                       f"{len(items)}/{MIN_GOLDEN_ITEMS} mục ({golden_path.relative_to(PROJECT_ROOT)})")

    if not RESULTS_FILE.is_file():
        result.add(False, "results.md", f"không tồn tại tại {RESULTS_FILE}")
    else:
        try:
            content = RESULTS_FILE.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            result.add(False, "results.md", f"không đọc được: {exc}")
        else:
            result.add(bool(content.strip()), "results.md",
                       f"{len(content)} ký tự, {len(content.splitlines())} dòng")
            # Cảnh báo (không tính FAIL): bảng điểm vẫn còn ô trống của template.
            looks_blank = "| Faithfulness | |" in content.replace("  ", " ")
            result.add(not looks_blank, "results.md đã điền số liệu",
                       "bảng điểm vẫn là template rỗng" if looks_blank else "đã có nội dung",
                       required=False)

    return result


# =============================================================================
# CP6 — Git repository
# =============================================================================

def _git(*args: str) -> tuple[bool, str, str]:
    """Chạy một lệnh git. Returns (chạy được, stdout, mô tả lỗi)."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=GIT_TIMEOUT,
        )
    except FileNotFoundError:
        return False, "", "không tìm thấy lệnh `git` trong PATH"
    except subprocess.TimeoutExpired:
        return False, "", f"git treo quá {GIT_TIMEOUT}s"
    except Exception as exc:  # noqa: BLE001
        return False, "", f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return False, proc.stdout or "", (proc.stderr or "").strip() or f"git exit code {proc.returncode}"
    return True, proc.stdout or "", ""


def check_cp6() -> CheckpointResult:
    """CP6: repo là git repo hợp lệ và working tree đã sạch (đã commit hết)."""
    result = CheckpointResult(cp=6, title="Git repository sạch & đã push")

    ok, out, error = _git("rev-parse", "--is-inside-work-tree")
    if not ok:
        result.add(False, "git repository", error)
        return result
    result.add(out.strip() == "true", "git repository", str(PROJECT_ROOT))

    ok, branch_out, error = _git("rev-parse", "--abbrev-ref", "HEAD")
    result.add(ok, "Branch hiện tại", branch_out.strip() if ok else error, required=False)

    ok, status_out, error = _git("status", "--porcelain")
    if not ok:
        result.add(False, "git status --porcelain", error)
        return result

    lines = [ln for ln in status_out.splitlines() if ln.strip()]
    if not lines:
        result.add(True, "Working tree", "sạch (0 file thay đổi)")
    else:
        modified = [ln for ln in lines if not ln.startswith("??")]
        untracked = [ln for ln in lines if ln.startswith("??")]
        preview = ", ".join(ln[3:] for ln in lines[:3])
        if len(lines) > 3:
            preview += f", +{len(lines) - 3} nữa"
        result.add(False, "Working tree",
                   f"{len(modified)} file thay đổi + {len(untracked)} file chưa track: {preview}")

    # Số commit chưa push — chỉ là thông tin (có thể chưa cấu hình upstream).
    ok, ahead_out, error = _git("rev-list", "--count", "@{u}..HEAD")
    if ok:
        count = ahead_out.strip() or "0"
        result.add(count == "0", "Commit chưa push", f"{count} commit", required=False)
    else:
        result.add(False, "Commit chưa push", f"không xác định được ({error})", required=False)

    return result


# =============================================================================
# ĐIỀU PHỐI
# =============================================================================

CHECKS: dict[int, Callable[[], CheckpointResult]] = {
    0: check_cp0,
    1: check_cp1,
    2: check_cp2,
    3: check_cp3,
    4: check_cp4,
    5: check_cp5,
    6: check_cp6,
}

CP_TITLES: dict[int, str] = {
    0: "Setup môi trường (venv + .env)",
    1: "Thu thập dữ liệu & chuẩn hóa Markdown",
    2: "Indexing + Semantic Search + Lexical BM25",
    3: "Reranking (RRF) + PageIndex Vectorless",
    4: "Retrieval Pipeline hoàn chỉnh (Task 9)",
    5: "Chatbot UI + Evaluation (group project)",
    6: "Git repository sạch & đã push",
}


def run_checkpoint(cp: int) -> CheckpointResult:
    """
    Chạy một checkpoint, bọc try/except để một CP hỏng không làm chết cả script.
    """
    started = time.perf_counter()
    try:
        result = CHECKS[cp]()
    except Exception as exc:  # noqa: BLE001 — lưới an toàn cuối cùng
        result = CheckpointResult(cp=cp, title=CP_TITLES.get(cp, ""))
        result.error = f"{type(exc).__name__}: {exc}"
    result.elapsed = time.perf_counter() - started
    return result


def print_summary(results: list[CheckpointResult]) -> None:
    """In tổng kết cuối cùng: CP nào đã qua, CP nào chưa."""
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    print()
    print("=" * 74)
    print(" TỔNG KẾT")
    print("=" * 74)
    print(f"  CP đã qua : {', '.join(f'CP{r.cp}' for r in passed) if passed else '(chưa có)'}")
    print(f"  CP chưa qua: {', '.join(f'CP{r.cp}' for r in failed) if failed else '(không còn)'}")

    for res in failed:
        blockers = [r for r in res.rows if r.required and not r.ok]
        for row in blockers:
            print(f"    {CROSS} CP{res.cp} · {row.label}: {row.detail or 'chưa đạt'}")
        if res.error:
            print(f"    {CROSS} CP{res.cp} · lỗi ngoài dự kiến: {res.error}")

    total = len(results)
    print()
    print(f"  {ARROW} {len(passed)}/{total} checkpoint PASS")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="check_cp.py",
        description="Kiểm tra tiến độ checkpoint CP0–CP6 của Lab 08 RAG Pipeline.",
    )
    parser.add_argument(
        "--cp",
        type=int,
        choices=sorted(CHECKS.keys()),
        default=None,
        help="Chỉ kiểm tra một checkpoint (mặc định: kiểm tra tất cả CP0–CP6).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    targets = [args.cp] if args.cp is not None else sorted(CHECKS.keys())

    print("=" * 74)
    print(" LAB 08 — RAG PIPELINE | CHECKPOINT QA")
    print(f" Project: {PROJECT_ROOT}")
    print(f" Kiểm tra: {', '.join(f'CP{c}' for c in targets)}")
    print("=" * 74)

    results: list[CheckpointResult] = []
    for cp in targets:
        result = run_checkpoint(cp)
        print_checkpoint(result)
        results.append(result)

    print_summary(results)
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
