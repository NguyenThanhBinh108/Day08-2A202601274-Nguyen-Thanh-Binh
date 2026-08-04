"""
RAG Evaluation Pipeline.

Sử dụng DeepEval / RAGAS / TruLens để đánh giá chất lượng RAG pipeline.
Chọn 1 framework và implement đầy đủ.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Lưu ý rate limit nếu dùng model OpenRouter ":free": RAGAS/DeepEval gọi LLM RẤT NHIỀU LẦN
(không phải 1 lần/câu hỏi mà nhiều lần/metric/câu hỏi). Model free của OpenRouter giới hạn
50 request/ngày CHO CẢ TÀI KHOẢN (không phải theo model hay theo API key — đổi model free
khác hay tạo key mới KHÔNG reset quota). Nếu chạy full 15+ câu hỏi mà bị rate limit giữa
chừng, thử giảm xuống subset 5 câu để chạy kịp trong buổi, hoặc nạp $10 credit để mở khóa
1000 request/ngày.
"""

# =============================================================================
# FRAMEWORK ĐÃ CHỌN: RAGAS 0.1.21 (đã ghim sẵn trong requirements.txt)
# -----------------------------------------------------------------------------
# Lý do chọn RAGAS:
#   - Có đủ 4 metric mà đề bài yêu cầu (faithfulness, answer_relevancy,
#     context_recall, context_precision) trong cùng một lần evaluate().
#   - Chỉ cần 1 dataset dạng bảng (question/answer/contexts/ground_truth) nên rất
#     hợp để chạy A/B: dựng 2 dataset từ 2 retriever rồi so cùng bộ metric.
#   - LLM/embedding có thể trỏ sang OpenRouter (OpenAI-compatible) qua
#     langchain_openai.ChatOpenAI, không bị khoá cứng vào OpenAI.
#
# DeepEval / TruLens vẫn được giữ lại bên dưới dưới dạng optional (import trong
# hàm, thiếu thư viện thì báo hướng dẫn chứ không raise) để tham khảo.
#
# QUY ƯỚC CHUNG CỦA FILE:
#   - Không làm việc nặng ở cấp module (không load model, không mở ChromaDB,
#     không gọi API lúc import) → mọi thứ nặng đều lazy + cache.
#   - Không bao giờ raise khi thiếu API key / thiếu dữ liệu / task khác chưa xong:
#     in cảnh báo rồi degrade (trả về dict/list rỗng, vẫn ghi được report).
# =============================================================================

import importlib
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Thư mục gốc repo (…/Day08-…): cần add vào sys.path để import được package `src`
# khi chạy file này trực tiếp (`python group_project/evaluation/eval_pipeline.py`).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

FRAMEWORK = "RAGAS 0.1.21"

# Tên metric của RAGAS 0.1.x — đồng thời là key trong Result và cột trong to_pandas()
METRIC_NAMES: tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
)
METRIC_LABELS: dict[str, str] = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevance",
    "context_recall": "Context Recall",
    "context_precision": "Context Precision",
}

# Số chunk đưa vào context khi eval (giữ bằng TOP_K của Task 10 để đo đúng hệ thống thật)
EVAL_TOP_K = 5

# LLM dùng làm "judge" cho RAGAS + để sinh answer. Đổi bằng biến môi trường EVAL_LLM_MODEL.
EVAL_LLM_MODEL = os.getenv("EVAL_LLM_MODEL", "openai/gpt-4o-mini")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"


def _int_env(name: str, default: int) -> int:
    """Đọc biến môi trường dạng int, sai định dạng thì dùng default (không raise)."""
    try:
        return int(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    """Đọc biến môi trường dạng float, sai định dạng thì dùng default (không raise)."""
    try:
        return float(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        return default


# Giới hạn số câu hỏi để né rate limit của model ":free" (0 = chạy hết dataset).
# Ví dụ: EVAL_MAX_QUESTIONS=5 python group_project/evaluation/eval_pipeline.py
MAX_QUESTIONS = _int_env("EVAL_MAX_QUESTIONS", 0)

# Nghỉ giữa 2 lần gọi LLM sinh answer (giây) — tăng lên nếu bị 429 liên tục.
REQUEST_DELAY = _float_env("EVAL_REQUEST_DELAY", 0.0)

# 2 config đem ra so sánh A/B
CONFIGS: dict[str, str] = {
    "hybrid": (
        "Hybrid — semantic (bge-m3) + BM25 → hợp nhất bằng RRF → rerank → "
        "fallback PageIndex khi điểm cosine gốc thấp (`retrieve()` của Task 9)"
    ),
    "dense_only": (
        "Dense-only — chỉ semantic search bge-m3, KHÔNG BM25, KHÔNG rerank, "
        "KHÔNG fallback (`retrieve_dense_only()` của Task 9)"
    ),
}

# Prompt dự phòng: chỉ dùng khi Task 10 chưa có SYSTEM_PROMPT (giữ nguyên tinh thần
# của Task 10: chỉ dùng context, có citation, thiếu evidence thì từ chối trả lời).
FALLBACK_SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về chính sách thương mại điện tử.
Chỉ dùng thông tin trong context, mỗi khẳng định kèm citation dạng [Source].
Nếu context không đủ thông tin, trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có"."""

PLACEHOLDER_CONTEXT = "(no context retrieved)"
PLACEHOLDER_ANSWER = "(no answer generated)"


# =============================================================================
# LAZY STATE — mọi thứ nặng chỉ khởi tạo lần đầu khi thật sự cần
# =============================================================================

_STDOUT_READY = False
_MODULE_CACHE: dict[str, Any] = {}
_ENV_LOADED = False
_LLM_CLIENT: Any = None
_RAGAS_LLM: Any = None
_RAGAS_EMBEDDINGS: Any = None
_RUNTIME_NOTES: list[str] = []


def _ensure_utf8_stdout() -> None:
    """
    Ép stdout/stderr sang UTF-8.

    Trên Windows, khi output bị pipe/redirect thì Python dùng cp1252 → in tiếng Việt
    hoặc ký tự ✓/⚠ sẽ ném UnicodeEncodeError và làm chết cả script eval.
    """
    global _STDOUT_READY
    if _STDOUT_READY:
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass  # stream không hỗ trợ reconfigure → bỏ qua, không phải lỗi chặn
    _STDOUT_READY = True


def _log(message: str) -> None:
    """In thông báo tiến trình (an toàn với tiếng Việt trên Windows)."""
    _ensure_utf8_stdout()
    print(message)


def _warn(message: str, note: bool = False) -> None:
    """In cảnh báo; `note=True` thì ghi thêm vào phần 'Ghi chú' của report."""
    _log(f"⚠ {message}")
    if note and message not in _RUNTIME_NOTES:
        _RUNTIME_NOTES.append(message)


def _load_env() -> None:
    """Nạp .env ở thư mục gốc repo (chỉ 1 lần, thiếu dotenv cũng không sao)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:
        pass  # không có python-dotenv → vẫn đọc được biến môi trường hệ thống
    _ENV_LOADED = True


def _get_api_credentials() -> tuple[Optional[str], str, str]:
    """
    Lấy API key cho LLM.

    Ưu tiên OpenRouter (bài lab dùng key này), fallback sang OpenAI thuần.

    Returns:
        (api_key hoặc None, base_url, tên provider)
    """
    _load_env()
    key = os.getenv("OPENROUTER_API_KEY")
    if key and key.strip():
        return key.strip(), OPENROUTER_BASE_URL, "openrouter"
    key = os.getenv("OPENAI_API_KEY")
    if key and key.strip():
        return key.strip(), OPENAI_BASE_URL, "openai"
    return None, OPENROUTER_BASE_URL, "none"


def _print_setup_help() -> None:
    """In hướng dẫn khi thiếu API key — KHÔNG raise, chỉ dừng sạch."""
    _log("")
    _log("=" * 70)
    _log("⚠ Chưa có API key cho LLM → không chạy được RAGAS evaluation.")
    _log("=" * 70)
    _log("Cách khắc phục:")
    _log(f"  1. Mở file {PROJECT_ROOT / '.env'}")
    _log("  2. Điền OPENROUTER_API_KEY=sk-or-v1-... (lấy tại https://openrouter.ai/keys)")
    _log("     hoặc OPENAI_API_KEY=sk-... nếu dùng OpenAI trực tiếp")
    _log("  3. Chạy lại: .\\.venv\\Scripts\\python.exe group_project/evaluation/eval_pipeline.py")
    _log("")
    _log("Mẹo né rate limit của model ':free' (50 request/ngày cho cả tài khoản):")
    _log("  EVAL_MAX_QUESTIONS=5 để chỉ eval 5 câu đầu")
    _log("  EVAL_REQUEST_DELAY=2 để nghỉ 2 giây giữa các lần gọi LLM")
    _log("=" * 70)


def _ensure_project_root_on_path() -> None:
    """Đưa thư mục gốc repo vào sys.path để `import src.taskX...` chạy được."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _import_project_module(dotted: str) -> Any:
    """
    Import module trong repo (vd 'src.task9_retrieval_pipeline') một cách an toàn.

    Trả về None nếu module lỗi/chưa tồn tại — eval vẫn chạy tiếp và ghi note.
    """
    if dotted in _MODULE_CACHE:
        return _MODULE_CACHE[dotted]
    _ensure_project_root_on_path()
    try:
        module = importlib.import_module(dotted)
    except Exception as exc:  # ImportError, NotImplementedError lúc import, ...
        _warn(f"Không import được `{dotted}`: {type(exc).__name__}: {exc}", note=True)
        module = None
    _MODULE_CACHE[dotted] = module
    return module


def _to_float(value: Any) -> Optional[float]:
    """Ép về float; None/NaN/không phải số → None (để phân biệt 'không đo được' với 0.0)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _mean(values: list[Optional[float]]) -> Optional[float]:
    """Trung bình bỏ qua None; rỗng → None."""
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _fmt(value: Optional[float]) -> str:
    """Format điểm cho bảng markdown."""
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_delta(a: Optional[float], b: Optional[float]) -> str:
    """Format chênh lệch A - B, kèm dấu."""
    if a is None or b is None:
        return "n/a"
    return f"{a - b:+.3f}"


def _md_cell(text: str, limit: int = 70) -> str:
    """Làm sạch text để nhét vào ô bảng markdown (bỏ xuống dòng, escape dấu |)."""
    flat = " ".join(str(text).split()).replace("|", "\\|")
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _ground_truth(item: dict) -> str:
    """
    Lấy ground truth từ 1 item golden dataset.

    Chấp nhận cả key mới ('ground_truth') lẫn key cũ ('expected_answer') để dataset
    của các nhóm khác nhau vẫn dùng được.
    """
    for key in ("ground_truth", "expected_answer", "answer"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _load_questions() -> list[dict]:
    """Load golden dataset an toàn + áp dụng giới hạn EVAL_MAX_QUESTIONS."""
    try:
        questions = load_golden_dataset()
    except FileNotFoundError:
        _warn(f"Không tìm thấy {GOLDEN_DATASET_PATH}", note=True)
        return []
    except json.JSONDecodeError as exc:
        _warn(f"golden_dataset.json sai định dạng JSON: {exc}", note=True)
        return []
    if not isinstance(questions, list):
        _warn("golden_dataset.json phải là một list các object", note=True)
        return []
    questions = [q for q in questions if isinstance(q, dict) and str(q.get("question", "")).strip()]
    if MAX_QUESTIONS > 0 and len(questions) > MAX_QUESTIONS:
        _warn(
            f"EVAL_MAX_QUESTIONS={MAX_QUESTIONS} → chỉ eval {MAX_QUESTIONS}/{len(questions)} câu hỏi",
            note=True,
        )
        questions = questions[:MAX_QUESTIONS]
    return questions


# =============================================================================
# RETRIEVER ADAPTERS — 2 nhánh của A/B test
# =============================================================================

def get_retriever(config_name: str) -> Optional[Callable[..., list[dict]]]:
    """
    Lấy hàm retrieval tương ứng với 1 config A/B.

    - "hybrid"     → `retrieve()` của Task 9 (semantic + BM25 + rerank + fallback)
    - "dense_only" → `retrieve_dense_only()` của Task 9; nếu Task 9 chưa expose hàm này
                     thì dựng tạm từ `semantic_search()` của Task 5 để A/B vẫn chạy được.

    Returns:
        Callable(query, top_k) -> list[dict], hoặc None nếu không dựng được.
    """
    task9 = _import_project_module("src.task9_retrieval_pipeline")

    if config_name == "hybrid":
        fn = getattr(task9, "retrieve", None) if task9 else None
        if fn is None:
            _warn("Task 9 chưa có `retrieve()` → bỏ qua config 'hybrid'", note=True)
        return fn

    if config_name == "dense_only":
        fn = getattr(task9, "retrieve_dense_only", None) if task9 else None
        if fn is not None:
            return fn

        task5 = _import_project_module("src.task5_semantic_search")
        semantic_search = getattr(task5, "semantic_search", None) if task5 else None
        if semantic_search is None:
            _warn("Không dựng được retriever 'dense_only' (thiếu cả Task 9 và Task 5)", note=True)
            return None

        _warn(
            "Task 9 chưa có `retrieve_dense_only()` → dùng tạm `semantic_search()` của Task 5 "
            "làm nhánh dense-only",
            note=True,
        )

        def _dense_only(query: str, top_k: int = EVAL_TOP_K) -> list[dict]:
            """Dense-only fallback: chỉ semantic search, không fusion/rerank."""
            results = semantic_search(query, top_k=top_k)
            for item in results:
                if isinstance(item, dict):
                    item.setdefault("source", "dense")
            return results[:top_k]

        return _dense_only

    _warn(f"Config không hợp lệ: {config_name}")
    return None


def _safe_retrieve(
    retriever_fn: Optional[Callable[..., list[dict]]],
    query: str,
    top_k: int,
) -> list[dict]:
    """
    Gọi retriever và luôn trả về list[dict] hợp lệ.

    Nuốt mọi lỗi (task chưa implement, ChromaDB chưa index, ...) để 1 câu hỏi hỏng
    không làm chết cả pipeline eval.
    """
    if retriever_fn is None:
        return []

    results: Any = None
    try:
        results = retriever_fn(query, top_k=top_k)
    except TypeError:
        # Retriever của nhóm khác có thể nhận top_k theo vị trí
        try:
            results = retriever_fn(query, top_k)
        except Exception as exc:
            _warn(f"Retriever lỗi ở câu '{query[:40]}...': {type(exc).__name__}: {exc}")
            return []
    except NotImplementedError:
        _warn("Retriever chưa được implement (NotImplementedError)", note=True)
        return []
    except Exception as exc:
        _warn(f"Retriever lỗi ở câu '{query[:40]}...': {type(exc).__name__}: {exc}")
        return []

    if not isinstance(results, list):
        return []
    return [item for item in results if isinstance(item, dict)][:top_k]


# =============================================================================
# GENERATION ADAPTER — sinh answer cho từng câu hỏi theo đúng config đang test
# =============================================================================

def _get_llm_client() -> Any:
    """Tạo (và cache) OpenAI-compatible client trỏ tới OpenRouter/OpenAI."""
    global _LLM_CLIENT
    if _LLM_CLIENT is not None:
        return _LLM_CLIENT
    api_key, base_url, _ = _get_api_credentials()
    if not api_key:
        return None
    try:
        from openai import OpenAI

        _LLM_CLIENT = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)
    except Exception as exc:
        _warn(f"Không khởi tạo được LLM client: {type(exc).__name__}: {exc}", note=True)
        _LLM_CLIENT = None
    return _LLM_CLIENT


def _format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string.

    Ưu tiên dùng `reorder_for_llm` + `format_context` của Task 10 để eval đúng hệ thống
    thật; nếu Task 10 chưa xong thì format tại chỗ theo cùng layout.
    """
    task10 = _import_project_module("src.task10_generation")
    if task10 is not None:
        try:
            ordered = task10.reorder_for_llm(chunks)
            return task10.format_context(ordered)
        except Exception:
            pass  # Task 10 chưa implement → dùng bản dự phòng bên dưới

    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata") or {}
        parts.append(
            f"[Document {i} | Source: {meta.get('source', f'Source {i}')} | "
            f"Type: {meta.get('type', 'unknown')}]\n{chunk.get('content', '')}\n"
        )
    return "\n---\n".join(parts)


def _generate_answer(query: str, chunks: list[dict]) -> str:
    """
    Sinh câu trả lời từ context của MỘT config cụ thể.

    Không gọi thẳng `generate_with_citation()` của Task 10 vì hàm đó tự retrieve bên
    trong (luôn dùng config mặc định) → sẽ làm hỏng A/B test. Ở đây ta tái sử dụng
    prompt/tham số của Task 10 nhưng bơm context do retriever của config hiện tại lấy về.
    """
    client = _get_llm_client()
    if client is None or not chunks:
        return ""

    task10 = _import_project_module("src.task10_generation")
    system_prompt = getattr(task10, "SYSTEM_PROMPT", FALLBACK_SYSTEM_PROMPT)
    model = getattr(task10, "LLM_MODEL", EVAL_LLM_MODEL) or EVAL_LLM_MODEL
    temperature = getattr(task10, "TEMPERATURE", 0.3)
    top_p = getattr(task10, "TOP_P", 0.9)

    user_message = f"Context:\n{_format_context(chunks)}\n\n---\n\nQuestion: {query}"

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                top_p=top_p,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            message = str(exc)
            is_rate_limit = "429" in message or "rate" in message.lower()
            if is_rate_limit and attempt == 0:
                _warn("Bị rate limit khi sinh answer → nghỉ 10s rồi thử lại 1 lần")
                time.sleep(10)
                continue
            _warn(f"Lỗi gọi LLM: {type(exc).__name__}: {message[:160]}", note=is_rate_limit)
            return ""
    return ""


# =============================================================================
# BUILD EVAL DATASET
# =============================================================================

def build_eval_dataset(
    retriever_fn: Optional[Callable[..., list[dict]]],
    questions: list[dict],
    top_k: int = EVAL_TOP_K,
) -> list[dict]:
    """
    Chạy RAG pipeline trên toàn bộ golden dataset và gom thành dataset cho RAGAS.

    Args:
        retriever_fn: Hàm retrieval của config đang test (xem `get_retriever`)
        questions: List item golden dataset (cần key 'question' + 'ground_truth')
        top_k: Số chunk đưa vào context

    Returns:
        List of {
            'question': str,
            'answer': str,
            'contexts': list[str],   # nội dung các chunk đã retrieve
            'ground_truth': str,
            ...metadata phụ ('id', 'category', 'in_domain', 'n_contexts', ...)
        }
    """
    _ensure_utf8_stdout()
    rows: list[dict] = []
    total = len(questions)

    for i, item in enumerate(questions, 1):
        question = str(item.get("question", "")).strip()
        if not question:
            continue

        chunks = _safe_retrieve(retriever_fn, question, top_k)
        contexts = [
            str(chunk.get("content", "")).strip()
            for chunk in chunks
            if str(chunk.get("content", "")).strip()
        ]
        answer = _generate_answer(question, chunks)

        _log(
            f"  [{i}/{total}] {question[:52]}{'...' if len(question) > 52 else ''} "
            f"→ {len(contexts)} contexts, answer {len(answer)} ký tự"
        )

        rows.append(
            {
                # 4 cột bắt buộc của RAGAS 0.1.x
                "question": question,
                "answer": answer or PLACEHOLDER_ANSWER,
                # RAGAS cần contexts khác rỗng, nếu không metric sẽ ném lỗi và mất luôn
                # cả dòng → dùng placeholder để vẫn chấm được (và điểm sẽ thấp đúng bản chất)
                "contexts": contexts or [PLACEHOLDER_CONTEXT],
                "ground_truth": _ground_truth(item),
                # metadata phụ (không đưa vào Dataset, chỉ dùng để phân tích worst performers)
                "id": item.get("id") or f"q{i:02d}",
                "category": item.get("category", ""),
                "language": item.get("language", ""),
                "in_domain": bool(item.get("in_domain", True)),
                "n_contexts": len(contexts),
                "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
                "generation_failed": not bool(answer),
            }
        )

        if REQUEST_DELAY > 0:
            time.sleep(REQUEST_DELAY)

    return rows


# =============================================================================
# Option 1: DeepEval
# =============================================================================

def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng DeepEval.

    pip install deepeval
    """
    # DeepEval KHÔNG nằm trong requirements.txt (nhóm chọn RAGAS) → import lazy và
    # degrade gracefully nếu chưa cài, thay vì raise.
    _ensure_utf8_stdout()
    try:
        from deepeval import evaluate
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            FaithfulnessMetric,
        )
        from deepeval.test_case import LLMTestCase
    except ImportError:
        _warn("Chưa cài DeepEval (`pip install deepeval`). Nhóm này dùng RAGAS — xem run_ragas().")
        return {}

    generate = getattr(rag_pipeline, "generate_with_citation", rag_pipeline)

    test_cases = []
    for item in golden_dataset:
        try:
            result = generate(item["question"])
        except Exception as exc:
            _warn(f"Bỏ qua câu '{item.get('question', '')[:40]}...': {type(exc).__name__}: {exc}")
            continue
        test_cases.append(
            LLMTestCase(
                input=item["question"],
                actual_output=result.get("answer", ""),
                expected_output=_ground_truth(item),
                retrieval_context=[c.get("content", "") for c in result.get("sources", [])],
            )
        )

    if not test_cases:
        _warn("Không tạo được test case nào cho DeepEval")
        return {}

    metrics = [
        FaithfulnessMetric(threshold=0.7),
        AnswerRelevancyMetric(threshold=0.7),
        ContextualRecallMetric(threshold=0.7),
        ContextualPrecisionMetric(threshold=0.7),
    ]

    try:
        return {"deepeval": evaluate(test_cases, metrics)}
    except Exception as exc:
        _warn(f"DeepEval thất bại: {type(exc).__name__}: {exc}")
        return {}


# =============================================================================
# Option 2: RAGAS  ← framework chính của nhóm
# =============================================================================

def _get_ragas_llm() -> Any:
    """
    LLM judge cho RAGAS: langchain_openai.ChatOpenAI trỏ về OpenRouter (hoặc OpenAI).

    temperature=0 để điểm ổn định giữa các lần chạy A/B.
    """
    global _RAGAS_LLM
    if _RAGAS_LLM is not None:
        return _RAGAS_LLM

    api_key, base_url, _ = _get_api_credentials()
    if not api_key:
        return None

    try:
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper

        chat = ChatOpenAI(
            model=EVAL_LLM_MODEL,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
            timeout=180,
            max_retries=3,
        )
        _RAGAS_LLM = LangchainLLMWrapper(chat)
    except Exception as exc:
        _warn(f"Không tạo được RAGAS LLM: {type(exc).__name__}: {exc}", note=True)
        _RAGAS_LLM = None
    return _RAGAS_LLM


def _get_ragas_embeddings() -> Any:
    """
    Embedding cho RAGAS (answer_relevancy cần embed câu hỏi sinh ngược).

    Thứ tự ưu tiên:
        1. OpenAI thật (nếu có OPENAI_API_KEY) — nhanh, rẻ.
        2. bge-m3 chạy local qua sentence-transformers — CÙNG model với Task 4 nên
           đo được cả tiếng Việt. OpenRouter KHÔNG có endpoint /embeddings nên không
           thể tái dùng OPENROUTER_API_KEY ở đây.
        3. None — answer_relevancy sẽ ra NaN, 3 metric còn lại vẫn chạy.
    """
    global _RAGAS_EMBEDDINGS
    if _RAGAS_EMBEDDINGS is not None:
        return _RAGAS_EMBEDDINGS

    _load_env()
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except Exception as exc:
        _warn(f"Không import được LangchainEmbeddingsWrapper: {exc}", note=True)
        return None

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key.strip():
        try:
            from langchain_openai import OpenAIEmbeddings

            _RAGAS_EMBEDDINGS = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key.strip())
            )
            return _RAGAS_EMBEDDINGS
        except Exception as exc:
            _warn(f"Không dùng được OpenAI embeddings: {type(exc).__name__}: {exc}")

    try:
        from langchain_core.embeddings import Embeddings

        model = _get_local_embedding_model()
        if model is None:
            raise RuntimeError("thiếu sentence-transformers hoặc model bge-m3")

        class _LocalEmbeddings(Embeddings):
            """Bọc SentenceTransformer (bge-m3) theo interface Embeddings của LangChain."""

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return model.encode(texts, normalize_embeddings=True).tolist()

            def embed_query(self, text: str) -> list[float]:
                return model.encode([text], normalize_embeddings=True)[0].tolist()

        _RAGAS_EMBEDDINGS = LangchainEmbeddingsWrapper(_LocalEmbeddings())
        _warn("Dùng embedding local bge-m3 cho RAGAS (không có OPENAI_API_KEY)", note=True)
    except Exception as exc:
        _warn(
            f"Không tạo được embeddings ({exc}) → answer_relevancy sẽ là n/a",
            note=True,
        )
        _RAGAS_EMBEDDINGS = None
    return _RAGAS_EMBEDDINGS


def _get_local_embedding_model() -> Any:
    """Lấy SentenceTransformer bge-m3, ưu tiên tái dùng cache của Task 4 (lazy)."""
    task4 = _import_project_module("src.task4_chunking_indexing")
    getter = getattr(task4, "get_embedding_model", None) if task4 else None
    if callable(getter):
        try:
            return getter()
        except Exception:
            pass  # Task 4 chưa implement getter → tự load bên dưới

    model_name = getattr(task4, "EMBEDDING_MODEL", "BAAI/bge-m3") if task4 else "BAAI/bge-m3"
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception:
        return None


def run_ragas(dataset: list[dict]) -> dict:
    """
    Chấm điểm 1 dataset bằng RAGAS 0.1.21 (4 metric).

    Args:
        dataset: Output của `build_eval_dataset()`

    Returns:
        {
            'aggregate':    {metric: float | None},   # điểm trung bình toàn dataset
            'per_question': [ {id, question, category, in_domain, <4 metric>} ],
            'n':            int,
            'error':        str | None,
        }
    """
    _ensure_utf8_stdout()
    empty = {"aggregate": {}, "per_question": [], "n": 0, "error": None}

    if not dataset:
        _warn("Dataset rỗng → bỏ qua RAGAS")
        return {**empty, "error": "dataset rỗng"}

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        _warn(f"Chưa cài RAGAS/datasets: {exc} → `pip install -r requirements.txt`", note=True)
        return {**empty, "n": len(dataset), "error": f"thiếu thư viện: {exc}"}

    llm = _get_ragas_llm()
    if llm is None:
        return {**empty, "n": len(dataset), "error": "thiếu API key cho LLM judge"}

    # Chỉ giữ đúng 4 cột RAGAS yêu cầu, metadata phụ khớp lại theo index sau
    rows = [
        {
            "question": row["question"],
            "answer": row["answer"],
            "contexts": list(row["contexts"]),
            "ground_truth": row["ground_truth"],
        }
        for row in dataset
    ]

    try:
        hf_dataset = Dataset.from_list(rows)
        result = evaluate(
            hf_dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=llm,
            embeddings=_get_ragas_embeddings(),
            # raise_exceptions=False: 1 câu lỗi (rate limit, parse fail) chỉ thành NaN
            # chứ không giết cả lượt eval
            raise_exceptions=False,
        )
    except Exception as exc:
        _warn(f"RAGAS evaluate() thất bại: {type(exc).__name__}: {exc}", note=True)
        return {**empty, "n": len(dataset), "error": f"{type(exc).__name__}: {exc}"}

    # ---- Điểm theo từng câu hỏi (ghép lại metadata để phân tích worst performers) ----
    per_question: list[dict] = []
    try:
        records = result.to_pandas().to_dict(orient="records")
    except Exception as exc:
        _warn(f"Không đọc được chi tiết từng câu: {exc}")
        records = []

    for idx, record in enumerate(records):
        source = dataset[idx] if idx < len(dataset) else {}
        row = {
            "id": source.get("id", f"q{idx + 1:02d}"),
            "question": source.get("question", record.get("question", "")),
            "category": source.get("category", ""),
            "language": source.get("language", ""),
            "in_domain": bool(source.get("in_domain", True)),
            "n_contexts": source.get("n_contexts", 0),
            "generation_failed": bool(source.get("generation_failed", False)),
        }
        for name in METRIC_NAMES:
            row[name] = _to_float(record.get(name))
        row["mean"] = _mean([row[name] for name in METRIC_NAMES])
        per_question.append(row)

    # ---- Điểm tổng hợp ----
    aggregate: dict[str, Optional[float]] = {}
    for name in METRIC_NAMES:
        value = _to_float(result.get(name)) if hasattr(result, "get") else None
        if value is None:  # RAGAS trả NaN → tự tính nanmean từ per_question
            value = _mean([row.get(name) for row in per_question])
        aggregate[name] = value

    return {
        "aggregate": aggregate,
        "per_question": per_question,
        "n": len(dataset),
        "error": None,
    }


def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng RAGAS.

    pip install ragas
    """
    # Biến thể "1 config": dùng thẳng generate_with_citation() của pipeline truyền vào
    # (retrieval nằm bên trong hàm đó). Dùng cho A/B thì gọi run_ab_comparison().
    _ensure_utf8_stdout()
    generate = getattr(rag_pipeline, "generate_with_citation", rag_pipeline)
    if not callable(generate):
        _warn("rag_pipeline không có `generate_with_citation()` gọi được")
        return {"aggregate": {}, "per_question": [], "n": 0, "error": "pipeline không hợp lệ"}

    dataset: list[dict] = []
    for i, item in enumerate(golden_dataset, 1):
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        try:
            result = generate(question)
        except Exception as exc:
            _warn(f"Lỗi generate câu '{question[:40]}...': {type(exc).__name__}: {exc}")
            result = {}
        sources = result.get("sources", []) if isinstance(result, dict) else []
        contexts = [
            str(chunk.get("content", "")).strip()
            for chunk in sources
            if isinstance(chunk, dict) and str(chunk.get("content", "")).strip()
        ]
        dataset.append(
            {
                "question": question,
                "answer": (result.get("answer") if isinstance(result, dict) else "") or PLACEHOLDER_ANSWER,
                "contexts": contexts or [PLACEHOLDER_CONTEXT],
                "ground_truth": _ground_truth(item),
                "id": item.get("id", f"q{i:02d}"),
                "category": item.get("category", ""),
                "language": item.get("language", ""),
                "in_domain": bool(item.get("in_domain", True)),
                "n_contexts": len(contexts),
            }
        )

    return run_ragas(dataset)


# =============================================================================
# Option 3: TruLens
# =============================================================================

def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng TruLens.

    pip install trulens
    """
    # TruLens cũng KHÔNG nằm trong requirements.txt → import lazy, thiếu thì báo hướng dẫn.
    _ensure_utf8_stdout()
    try:
        from trulens.apps.custom import TruCustomApp
        from trulens.core import Feedback
        from trulens.providers.openai import OpenAI as TruOpenAI
    except ImportError:
        _warn("Chưa cài TruLens (`pip install trulens`). Nhóm này dùng RAGAS — xem run_ragas().")
        return {}

    try:
        provider = TruOpenAI()

        f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
        f_relevance = Feedback(provider.relevance).on_input_output()
        f_context_relevance = Feedback(provider.context_relevance).on_input()

        tru_rag = TruCustomApp(
            rag_pipeline,
            app_name="EcommerceSupport_RAG",
            feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
        )

        with tru_rag as _recording:
            for item in golden_dataset:
                rag_pipeline.generate_with_citation(item["question"])
        # Dashboard: from trulens.dashboard import run_dashboard; run_dashboard()
        return {"trulens_app": tru_rag}
    except Exception as exc:
        _warn(f"TruLens thất bại: {type(exc).__name__}: {exc}")
        return {}


# =============================================================================
# A/B Comparison
# =============================================================================

def run_ab_comparison() -> dict:
    """
    Chạy eval cho 2 config rồi so sánh 4 metric.

        Config A "hybrid"     — `retrieve()` của Task 9 (semantic + BM25 + rerank + fallback)
        Config B "dense_only" — `retrieve_dense_only()` của Task 9 (chỉ dense, không rerank)

    Returns:
        dict kết quả đầy đủ cho `write_report()`; {} nếu không chạy được (thiếu key/dataset).
    """
    _ensure_utf8_stdout()

    questions = _load_questions()
    if not questions:
        _warn("Golden dataset rỗng → không có gì để eval")
        return {}

    api_key, _base_url, provider = _get_api_credentials()
    if not api_key:
        _print_setup_help()
        return {}

    _log(f"Framework: {FRAMEWORK} | LLM judge: {EVAL_LLM_MODEL} ({provider})")
    _log(f"Golden dataset: {len(questions)} câu hỏi | top_k = {EVAL_TOP_K}")

    results: dict[str, Any] = {
        "framework": FRAMEWORK,
        "llm_model": EVAL_LLM_MODEL,
        "provider": provider,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_questions": len(questions),
        "top_k": EVAL_TOP_K,
        "configs": {},
        "notes": [],
    }

    for name, description in CONFIGS.items():
        _log("")
        _log("=" * 70)
        _log(f"Config '{name}': {description}")
        _log("=" * 70)

        retriever = get_retriever(name)
        if retriever is None:
            results["configs"][name] = {
                "description": description,
                "aggregate": {},
                "per_question": [],
                "n": 0,
                "error": "không dựng được retriever",
            }
            continue

        dataset = build_eval_dataset(retriever, questions)
        _log(f"→ Chấm điểm {len(dataset)} câu bằng RAGAS (có thể mất vài phút)...")
        scored = run_ragas(dataset)
        scored["description"] = description
        results["configs"][name] = scored

        summary = " | ".join(
            f"{METRIC_LABELS[m]}: {_fmt(scored.get('aggregate', {}).get(m))}" for m in METRIC_NAMES
        )
        _log(f"✓ {name}: {summary}")

    results["notes"] = list(_RUNTIME_NOTES)
    return results


def compare_configs(rag_pipeline=None, golden_dataset: Optional[list[dict]] = None):
    """
    So sánh A/B giữa ít nhất 2 configs.

    Gợi ý configs để so sánh:
    - Config A: hybrid search + reranking
    - Config B: dense-only (không reranking)
    - Config C: hybrid search + PageIndex fallback
    """
    # Giữ lại signature cũ cho tương thích, phần thân uỷ quyền cho run_ab_comparison()
    # (config được lấy trực tiếp từ Task 9 nên không cần truyền pipeline vào).
    return run_ab_comparison()


# =============================================================================
# Export Results
# =============================================================================

def _diagnose(row: dict) -> tuple[str, str]:
    """
    Suy ra khâu hỏng + nguyên nhân gốc cho 1 câu hỏi điểm thấp.

    Thứ tự kiểm tra bám theo luồng pipeline: retrieval → ranking → generation.
    """
    if not row.get("in_domain", True):
        return (
            "Ngoài domain (kỳ vọng)",
            "Câu hỏi không có evidence trong corpus — điểm thấp là ĐÚNG, chỉ cần kiểm tra "
            "hệ thống có từ chối trả lời thay vì bịa.",
        )
    if row.get("generation_failed"):
        return ("Generation", "LLM không trả về nội dung (rate limit / lỗi API) nên không chấm được.")
    if not row.get("n_contexts"):
        return ("Retrieval", "Không retrieve được chunk nào — kiểm tra ChromaDB đã index chưa.")

    recall = row.get("context_recall")
    precision = row.get("context_precision")
    faith = row.get("faithfulness")
    relevancy = row.get("answer_relevancy")

    if recall is not None and recall < 0.5:
        return (
            "Retrieval",
            "Chunk chứa evidence không nằm trong top_k — câu hỏi dùng từ vựng khác tài liệu "
            "hoặc evidence bị cắt rời giữa 2 chunk (chunk_size/overlap).",
        )
    if precision is not None and precision < 0.5:
        return (
            "Ranking",
            "Context lấy về nhiều nhiễu, chunk đúng bị xếp dưới — reranking chưa đẩy được "
            "chunk liên quan lên đầu.",
        )
    if faith is not None and faith < 0.5:
        return (
            "Generation",
            "Câu trả lời chứa khẳng định không có trong context (hallucination) — cần siết "
            "system prompt và bắt buộc citation.",
        )
    if relevancy is not None and relevancy < 0.5:
        return (
            "Generation",
            "Câu trả lời lạc đề hoặc trả lời chung chung, không bám câu hỏi.",
        )
    return ("Hỗn hợp", "Điểm thấp rải đều ở nhiều metric — cần xem lại cả retrieval lẫn prompt.")


def _worst_performers(per_question: list[dict], limit: int = 3) -> list[dict]:
    """Lấy N câu điểm trung bình thấp nhất (bỏ các câu không chấm được metric nào)."""
    scored = [row for row in per_question if row.get("mean") is not None]
    scored.sort(key=lambda row: row["mean"])
    return scored[:limit]


def _recommendations(best_config: Optional[dict]) -> list[tuple[str, str, str]]:
    """
    Sinh 3 đề xuất cải tiến, ưu tiên metric đang yếu nhất.

    Returns:
        List of (tiêu đề, action, expected impact)
    """
    aggregate = (best_config or {}).get("aggregate", {}) or {}
    scored = {name: value for name, value in aggregate.items() if value is not None}
    weakest = min(scored, key=scored.get) if scored else None

    targeted = {
        "context_recall": (
            "Tăng Context Recall",
            "Bật query expansion / multi-query (sinh 2-3 biến thể câu hỏi Việt–Anh rồi hợp nhất "
            "bằng RRF) và tăng top_k của tầng retrieve lên 15-20 trước khi rerank xuống 5.",
            "Kéo evidence bị bỏ sót vào context; recall thường tăng 10-20 điểm mà precision "
            "không giảm nhiều nhờ rerank ở tầng sau.",
        ),
        "context_precision": (
            "Tăng Context Precision",
            "Đổi rerank sang cross-encoder (BAAI/bge-reranker-v2-m3) thay cho RRF thuần, và cắt "
            "chunk có điểm rerank dưới ngưỡng thay vì luôn lấy đủ top_k.",
            "Loại chunk nhiễu ra khỏi prompt, precision tăng và faithfulness cũng tăng theo vì "
            "LLM ít bị phân tán.",
        ),
        "faithfulness": (
            "Tăng Faithfulness",
            "Siết system prompt: bắt buộc mỗi câu phải kèm [Source] và trả lời 'Tôi không thể xác "
            "minh thông tin này từ nguồn hiện có' khi thiếu evidence; hạ temperature xuống 0.1.",
            "Giảm hallucination, các câu ngoài domain trả lời đúng kiểu từ chối thay vì bịa.",
        ),
        "answer_relevancy": (
            "Tăng Answer Relevance",
            "Thêm few-shot mẫu trả lời ngắn gọn đúng trọng tâm vào prompt và yêu cầu trả lời "
            "trực tiếp câu hỏi ở câu đầu tiên trước khi giải thích.",
            "Câu trả lời bám câu hỏi hơn, answer_relevancy tăng rõ nhất ở các câu hỏi 'how/what'.",
        ),
    }

    recs: list[tuple[str, str, str]] = []
    if weakest and weakest in targeted:
        recs.append(targeted[weakest])

    backlog = [
        (
            "Chuẩn hoá chunking cho tài liệu song ngữ",
            "Chunk theo heading (markdown-aware) thay vì cắt cứng 800 ký tự, và gắn thêm tiêu đề "
            "mục vào đầu mỗi chunk để chunk tự mang ngữ cảnh.",
            "Giảm trường hợp evidence bị cắt đôi giữa 2 chunk — cải thiện đồng thời recall và "
            "faithfulness.",
        ),
        (
            "Calibrate lại SCORE_THRESHOLD cho fallback",
            "Đo phân bố điểm cosine gốc của semantic_search trên nhóm câu in-domain và nhóm ngoài "
            "domain trong golden dataset, chọn ngưỡng nằm giữa 2 phân bố (KHÔNG dùng điểm RRF).",
            "Câu ngoài domain đi đúng nhánh PageIndex/từ chối trả lời, giảm hallucination mà không "
            "làm hỏng các câu in-domain.",
        ),
        (
            "Mở rộng golden dataset và tự động hoá eval",
            "Nâng lên 30-50 câu (thêm câu hỏi nhiều bước, câu hỏi có số liệu/thời hạn) và chạy eval "
            "định kỳ, lưu lịch sử điểm để phát hiện hồi quy.",
            "Điểm ổn định và có ý nghĩa thống kê hơn; phát hiện sớm khi thay đổi chunking/model "
            "làm chất lượng đi xuống.",
        ),
    ]
    for rec in backlog:
        if len(recs) >= 3:
            break
        if rec not in recs:
            recs.append(rec)
    return recs[:3]


def write_report(results: dict) -> Path:
    """
    Ghi kết quả evaluation ra group_project/evaluation/results.md.

    Nội dung: bảng điểm A/B (4 metric + trung bình), phân tích worst performers
    (3 câu điểm thấp nhất kèm nguyên nhân) và đề xuất cải tiến.

    Args:
        results: Output của `run_ab_comparison()` ({} nếu eval không chạy được)

    Returns:
        Đường dẫn file report đã ghi.
    """
    _ensure_utf8_stdout()
    lines: list[str] = ["# RAG Evaluation Results", ""]

    if not results or not results.get("configs"):
        # Vẫn ghi report để người chấm thấy trạng thái + cách chạy lại (không raise).
        # Giữ nguyên bộ khung bảng của template để file vẫn dùng được như form điền tay
        # nếu nhóm chạy eval thủ công.
        lines += [
            "> ⚠ **Chưa có kết quả evaluation.** Script đã chạy nhưng không thu được điểm.",
            "",
            "## Nguyên nhân có thể",
            "",
            "- Thiếu `OPENROUTER_API_KEY` / `OPENAI_API_KEY` trong `.env`",
            "- Chưa index dữ liệu vào ChromaDB (chạy `python -m src.task4_chunking_indexing`)",
            "- Task 9 (`retrieve`) hoặc Task 10 chưa implement xong",
            "",
            "## Cách chạy lại",
            "",
            "```bash",
            "# né rate limit của model :free bằng cách giảm số câu hỏi",
            "EVAL_MAX_QUESTIONS=5 python group_project/evaluation/eval_pipeline.py",
            "```",
            "",
        ]
        if results.get("notes") or _RUNTIME_NOTES:
            lines.append("## Ghi chú từ lần chạy gần nhất")
            lines.append("")
            for note in results.get("notes") or _RUNTIME_NOTES:
                lines.append(f"- {note}")
            lines.append("")
        lines += [
            "---",
            "",
            "## Framework sử dụng",
            "",
            f"> {FRAMEWORK} (xem `eval_pipeline.py` → `run_ragas`)",
            "",
            "## Overall Scores",
            "",
            "| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |",
            "|--------|---------------------------|----------------------|---|",
        ]
        for metric in METRIC_NAMES:
            lines.append(f"| {METRIC_LABELS[metric]} | | | |")
        lines += [
            "| **Average** | | | |",
            "",
            "## A/B Comparison Analysis",
            "",
            f"**Config A:** {CONFIGS['hybrid']}",
            "",
            f"**Config B:** {CONFIGS['dense_only']}",
            "",
            "**Kết luận:**",
            "",
            "> _(điền sau khi chạy được eval)_",
            "",
            "## Worst Performers (Bottom 3)",
            "",
            "| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |",
            "|---|----------|-------------|-----------|--------|---------------|------------|",
            "| 1 | | | | | | |",
            "| 2 | | | | | | |",
            "| 3 | | | | | | |",
            "",
            "## Recommendations",
            "",
        ]
        for i, (title, action, impact) in enumerate(_recommendations(None), 1):
            lines += [
                f"### Cải tiến {i} — {title}",
                f"**Action:** {action}",
                "",
                f"**Expected impact:** {impact}",
                "",
            ]
        RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
        _log(f"✓ Đã ghi report (trạng thái rỗng): {RESULTS_PATH}")
        return RESULTS_PATH

    configs: dict[str, dict] = results["configs"]
    names = list(configs.keys())
    name_a = names[0]
    name_b = names[1] if len(names) > 1 else names[0]
    config_a, config_b = configs[name_a], configs[name_b]

    # ---------------- Header ----------------
    lines += [
        "## Framework sử dụng",
        "",
        f"> **{results.get('framework', FRAMEWORK)}** — LLM judge: "
        f"`{results.get('llm_model', EVAL_LLM_MODEL)}` qua {results.get('provider', 'openrouter')}.",
        "",
        f"- Golden dataset: **{results.get('n_questions', 0)} câu hỏi** "
        f"(song ngữ Anh–Việt, có câu ngoài domain để test fallback)",
        f"- top_k = {results.get('top_k', EVAL_TOP_K)} | Thời điểm chạy: "
        f"{results.get('generated_at', '')}",
        "",
        "---",
        "",
    ]

    # ---------------- Bảng điểm ----------------
    aggregate_a = config_a.get("aggregate", {}) or {}
    aggregate_b = config_b.get("aggregate", {}) or {}
    mean_a = _mean([aggregate_a.get(m) for m in METRIC_NAMES])
    mean_b = _mean([aggregate_b.get(m) for m in METRIC_NAMES])

    lines += [
        "## Overall Scores",
        "",
        f"| Metric | Config A — {name_a} | Config B — {name_b} | Δ (A − B) |",
        "|--------|--------------------|--------------------|-----------|",
    ]
    for metric in METRIC_NAMES:
        lines.append(
            f"| {METRIC_LABELS[metric]} | {_fmt(aggregate_a.get(metric))} | "
            f"{_fmt(aggregate_b.get(metric))} | {_fmt_delta(aggregate_a.get(metric), aggregate_b.get(metric))} |"
        )
    lines += [
        f"| **Average** | **{_fmt(mean_a)}** | **{_fmt(mean_b)}** | **{_fmt_delta(mean_a, mean_b)}** |",
        "",
    ]

    for name, config in configs.items():
        if config.get("error"):
            lines.append(f"> ⚠ Config `{name}` không chấm được: {config['error']}")
    lines.append("")
    lines += ["---", ""]

    # ---------------- Phân tích A/B ----------------
    lines += ["## A/B Comparison Analysis", "", f"**Config A — {name_a}:**", "",
              f"> {config_a.get('description', CONFIGS.get(name_a, ''))}", "",
              f"**Config B — {name_b}:**", "",
              f"> {config_b.get('description', CONFIGS.get(name_b, ''))}", "", "**Kết luận:**", ""]

    if mean_a is None or mean_b is None:
        lines.append("> Chưa đủ điểm ở cả 2 config để kết luận — xem phần Ghi chú bên dưới.")
    else:
        gap = mean_a - mean_b
        if abs(gap) < 0.02:
            lines.append(
                f"> Hai config gần như ngang nhau (chênh {gap:+.3f}). Với corpus nhỏ, dense search "
                f"đã đủ tìm đúng tài liệu nên phần fusion + rerank chưa tạo khác biệt rõ; nên ưu "
                f"tiên `{name_b}` nếu cần độ trễ thấp."
            )
        elif gap > 0:
            best_metric = max(
                (m for m in METRIC_NAMES if aggregate_a.get(m) is not None and aggregate_b.get(m) is not None),
                key=lambda m: aggregate_a[m] - aggregate_b[m],
                default=None,
            )
            detail = (
                f" Khác biệt lớn nhất ở **{METRIC_LABELS[best_metric]}** "
                f"({_fmt_delta(aggregate_a[best_metric], aggregate_b[best_metric])})."
                if best_metric
                else ""
            )
            lines.append(
                f"> **Config A ({name_a}) tốt hơn** (trung bình {_fmt(mean_a)} so với {_fmt(mean_b)}, "
                f"chênh {gap:+.3f}). BM25 bổ sung các câu hỏi chứa từ khoá chính xác (mã đơn, tên "
                f"chính sách) mà embedding dễ bỏ sót, còn rerank đẩy chunk đúng lên đầu context."
                f"{detail}"
            )
        else:
            lines.append(
                f"> **Config B ({name_b}) tốt hơn** (trung bình {_fmt(mean_b)} so với {_fmt(mean_a)}, "
                f"chênh {gap:+.3f}). Nhiều khả năng RRF đang pha loãng thứ hạng của dense search: "
                f"BM25 kéo lên các chunk trùng từ khoá nhưng lạc ngữ nghĩa. Nên hạ trọng số nhánh "
                f"lexical hoặc chuyển sang cross-encoder rerank."
            )
    lines += ["", "---", ""]

    # ---------------- Worst performers ----------------
    reference_name = name_a if (mean_a or 0) >= (mean_b or 0) else name_b
    reference = configs[reference_name]
    worst = _worst_performers(reference.get("per_question", []) or [], limit=3)

    lines += [
        "## Worst Performers (Bottom 3)",
        "",
        f"_Xếp hạng theo điểm trung bình 4 metric của config `{reference_name}`._",
        "",
        "| # | Question | Faithfulness | Relevance | Recall | Precision | Failure Stage | Root Cause |",
        "|---|----------|-------------|-----------|--------|-----------|---------------|------------|",
    ]
    if not worst:
        lines.append("| - | _Không có dữ liệu chi tiết theo câu hỏi_ | | | | | | |")
    else:
        for i, row in enumerate(worst, 1):
            stage, cause = _diagnose(row)
            lines.append(
                f"| {i} | {_md_cell(row.get('question', ''))} | "
                f"{_fmt(row.get('faithfulness'))} | {_fmt(row.get('answer_relevancy'))} | "
                f"{_fmt(row.get('context_recall'))} | {_fmt(row.get('context_precision'))} | "
                f"{stage} | {_md_cell(cause, limit=160)} |"
            )
    lines.append("")

    if worst:
        lines += ["**Phân tích chi tiết:**", ""]
        for i, row in enumerate(worst, 1):
            stage, cause = _diagnose(row)
            lines += [
                f"{i}. **{row.get('id', '')} — {_md_cell(row.get('question', ''), limit=110)}** "
                f"(mean = {_fmt(row.get('mean'))}, {row.get('n_contexts', 0)} contexts, "
                f"category: `{row.get('category', 'n/a')}`)",
                f"   - Khâu hỏng: **{stage}**",
                f"   - Nguyên nhân: {cause}",
            ]
        lines.append("")
    lines += ["---", ""]

    # ---------------- Recommendations ----------------
    lines += ["## Recommendations", ""]
    for i, (title, action, impact) in enumerate(_recommendations(reference), 1):
        lines += [
            f"### Cải tiến {i} — {title}",
            f"**Action:** {action}",
            "",
            f"**Expected impact:** {impact}",
            "",
        ]

    # ---------------- Ghi chú ----------------
    notes = results.get("notes") or _RUNTIME_NOTES
    if notes:
        lines += ["---", "", "## Ghi chú / hạn chế của lần chạy này", ""]
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    _log(f"✓ Đã ghi report: {RESULTS_PATH}")
    return RESULTS_PATH


def export_results(results: dict, comparison: Optional[dict] = None):
    """Export evaluation results to results.md"""
    # Giữ signature cũ; nếu truyền vào kết quả A/B (comparison) thì ưu tiên dùng nó.
    payload = comparison if comparison else results
    return write_report(payload or {})


if __name__ == "__main__":
    _ensure_utf8_stdout()
    try:
        golden_dataset = load_golden_dataset()
    except Exception as exc:
        _warn(f"Không đọc được golden dataset: {type(exc).__name__}: {exc}")
        golden_dataset = []
    print(f"Loaded {len(golden_dataset)} test cases")

    # RAGAS + A/B: Config A = hybrid (Task 9 retrieve), Config B = dense-only
    ab_results = run_ab_comparison()
    write_report(ab_results)

    if not ab_results:
        print("→ Report đã ghi ở trạng thái rỗng. Sửa nguyên nhân bên trên rồi chạy lại.")
