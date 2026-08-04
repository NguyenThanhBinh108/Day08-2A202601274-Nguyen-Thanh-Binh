"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"

Gợi ý LLM: OpenRouter có nhiều model gắn hậu tố ":free" không tính phí — xem
https://openrouter.ai/models?max_price=0 — phù hợp nếu chưa có credit trả phí.
Base URL: "https://openrouter.ai/api/v1", dùng chung interface với OpenAI SDK.
"""

import os
import re
from typing import Any, Callable, Optional

from .guardrails import (
    BLOCKED_ANSWER,
    CHITCHAT_FALLBACK_ANSWER,
    CHITCHAT_SYSTEM_PROMPT,
    EMPTY_ANSWER,
    Route,
    RouteDecision,
    audit_answer,
    classify_query,
)

# Import "chịu lỗi": nếu một module anh em (task5..task9) chưa được implement xong hoặc
# thiếu thư viện, việc `import src.task10_generation` vẫn phải thành công — pytest import
# toàn bộ module trước khi chạy test, import không được phép crash (quy tắc lazy loading).
try:
    from .task9_retrieval_pipeline import retrieve
except Exception as _err:  # pragma: no cover - chỉ xảy ra khi task9 chưa sẵn sàng
    retrieve = None  # type: ignore[assignment]
    _RETRIEVE_IMPORT_ERROR: Optional[Exception] = _err
else:
    _RETRIEVE_IMPORT_ERROR = None


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# max_tokens: trần độ dài câu trả lời.
# Đặt tường minh vì mỗi model trên OpenRouter có mặc định khác nhau, có model cắt
# rất sớm khiến câu trả lời đứt giữa chừng. 1400 token đủ cho một câu trả lời có
# liệt kê đầy đủ mà vẫn rẻ.
MAX_TOKENS = 1400

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

# LLM model chính (OpenRouter model ID). gpt-4o-mini rẻ, bám sát instruction và
# trích dẫn tốt — phù hợp RAG. Cần credit trả phí trên OpenRouter.
LLM_MODEL = "openai/gpt-4o-mini"  # hoặc model ":free" nếu chưa có credit

# Danh sách model dự phòng: các model ":free" của OpenRouter, thử lần lượt khi model
# chính lỗi (hết credit 402, rate limit 429, model bị gỡ 404...). Model free có thể
# thay đổi theo thời gian — kiểm tra lại tại https://openrouter.ai/models?max_price=0
FALLBACK_LLM_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free",
]

# Base URL của OpenRouter — tương thích hoàn toàn với OpenAI SDK 1.x
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Timeout (giây) cho mỗi lần gọi API, tránh treo pytest khi mạng chập chờn
REQUEST_TIMEOUT = 60.0

# Số đoạn tối đa dùng cho câu trả lời extractive (chế độ không-LLM)
FALLBACK_NUM_PASSAGES = 3

# Số câu tối đa trích ra từ mỗi đoạn trong chế độ extractive
FALLBACK_SENTENCES_PER_PASSAGE = 2

# Cờ lazy: .env chỉ được nạp ở lần cần API key đầu tiên (xem _load_env_once).
_ENV_LOADED: bool = False


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về chính sách thương mại điện tử và hỗ trợ
khách hàng (thanh toán, đổi trả, giao hàng, quyền riêng tư, quy định người bán).

Mỗi đoạn context được đánh số sẵn ở đầu dòng theo dạng "[3] Source: ... | Type: ...".
Con số đó chính là số trích dẫn bạn phải dùng.

QUY TẮC SỐ 0 — NGÔN NGỮ (quan trọng nhất, kiểm tra trước khi viết chữ đầu tiên):
Trả lời ĐÚNG ngôn ngữ của CÂU HỎI, không phải ngôn ngữ của tài liệu.
  - Câu hỏi tiếng Anh  -> toàn bộ câu trả lời bằng tiếng Anh.
  - Câu hỏi tiếng Việt -> toàn bộ câu trả lời bằng tiếng Việt.
Corpus là song ngữ, nên chuyện tài liệu tiếng Việt mà câu hỏi tiếng Anh (và ngược
lại) là bình thường — cứ dịch ý sang ngôn ngữ của người hỏi.

Quy tắc bắt buộc:
1. Chỉ dùng thông tin có trong context — KHÔNG bịa, KHÔNG lấy kiến thức bên ngoài.
2. Trích dẫn bằng ĐÚNG số của đoạn context, viết ngay sau khẳng định: [1], [3].
   Có thể gộp nhiều nguồn: [1][4]. TUYỆT ĐỐI không tự đặt tên hay năm cho nguồn
   (không viết kiểu "[Returns Policy, 2026]") — số phải khớp context thì giao diện
   mới liên kết được câu trả lời với tài liệu.
3. Trả lời CÙNG NGÔN NGỮ với câu hỏi: hỏi tiếng Việt thì đáp tiếng Việt, hỏi tiếng
   Anh thì đáp tiếng Anh.
4. Viết thành câu hoàn chỉnh, đủ ý, không bỏ lửng giữa chừng. Khi tài liệu liệt kê
   nhiều mục (các phương thức thanh toán, các bước thực hiện, các trường hợp được
   hoàn tiền) thì liệt kê ĐẦY ĐỦ bằng gạch đầu dòng, đừng tóm tắt cụt lủn.
   Nêu rõ con số cụ thể khi tài liệu có nêu (số ngày, mức phí, thời hạn).
5. TUYỆT ĐỐI không chép lại URL, đường dẫn hay cú pháp markdown link còn sót trong
   context. Chỉ diễn đạt lại nội dung bằng câu văn bình thường.
6. Nếu context không đủ để trả lời, nói thẳng: "Tôi không thể xác minh thông tin này
   từ nguồn hiện có", rồi nêu phần nào trả lời được (nếu có).
7. Bố cục: mở đầu một câu trả lời trực tiếp vào trọng tâm, sau đó mới đến chi tiết."""


# Câu trả lời chuẩn khi retrieval không tìm được tài liệu nào.
# Luôn là chuỗi KHÔNG rỗng để caller (app/eval) không phải xử lý None.
NO_EVIDENCE_ANSWER = (
    "Tôi không thể xác minh thông tin này từ nguồn hiện có: "
    "hệ thống không truy xuất được tài liệu nào liên quan tới câu hỏi. "
    "Vui lòng kiểm tra lại vector store (chroma_db/) và dữ liệu trong data/, "
    "hoặc diễn đạt lại câu hỏi cụ thể hơn."
)


# =============================================================================
# LAZY HELPERS — không load gì ở cấp module
# =============================================================================

def _safe_print(message: str) -> None:
    """
    In ra stdout mà KHÔNG BAO GIỜ raise (dùng chung cho toàn module thay cho print()).

    Console Windows mặc định là codepage cp1252: các ký tự "⚠", "ℹ" và chữ tiếng Việt
    có dấu đều không encode được → print() thường sẽ ném UnicodeEncodeError, thoát ra
    khỏi generate_with_citation() và làm gãy cả app Streamlit lẫn pipeline eval — đúng
    ngay ở nhánh cảnh báo (thiếu API key / không retrieve được), tức là lúc hệ thống
    đáng ra phải degrade êm nhất. Cùng cơ chế với _safe_print/_warn ở Task 4, 5, 6, 8, 9.
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


def _get_retrieve_fn() -> Optional[Callable[..., list[dict]]]:
    """
    Trả về hàm retrieve() của Task 9, hoặc None nếu module chưa import được.

    Tách ra thành getter để việc import task10 không bao giờ crash vì task9.
    """
    if retrieve is None:
        _safe_print(f"  ⚠ Không import được retrieve() từ task9: {_RETRIEVE_IMPORT_ERROR}")
        return None
    return retrieve


def _load_env_once() -> None:
    """
    Nạp .env MỘT LẦN, lazy (ở lần cần API key đầu tiên) thay vì ở cấp module.

    Vì sao không load_dotenv() ngay dưới phần import: pytest import toàn bộ module
    này (và import gián tiếp qua app.py / eval_pipeline), import phải nhanh và không
    được chạm đĩa — cùng nguyên tắc lazy với Task 5, 6, 8, 9.
    Chỉ chạy đúng 1 lần để không đọc lại file ở mỗi câu hỏi, và để test vẫn có thể
    unset biến môi trường sau lần gọi đầu mà không bị .env ghi đè lại.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # không có python-dotenv → vẫn đọc biến môi trường hệ thống
        pass


def _get_api_key() -> str:
    """
    Đọc API key tại thời điểm gọi (không cache ở cấp module) để test có thể
    set/unset biến môi trường mà không cần reload module.
    """
    _load_env_once()
    return (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()


def _get_llm_client(api_key: str) -> Optional[Any]:
    """
    Khởi tạo OpenAI SDK client trỏ vào OpenRouter.

    Import `openai` bên trong hàm (lazy) để module import nhanh và không phụ thuộc
    vào việc SDK đã được cài hay chưa.

    Returns:
        Client object, hoặc None nếu thiếu key / thiếu SDK.
    """
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError as e:
        _safe_print(f"  ⚠ Chưa cài openai SDK ({e}) — chuyển sang chế độ extractive.")
        return None

    try:
        return OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            timeout=REQUEST_TIMEOUT,
            max_retries=1,  # SDK tự retry 1 lần; nhiều hơn sẽ làm test chờ quá lâu
        )
    except Exception as e:  # cấu hình sai, key rỗng sau khi strip...
        _safe_print(f"  ⚠ Không khởi tạo được LLM client: {type(e).__name__}: {e}")
        return None


def _split_sentences(text: str) -> list[str]:
    """
    Tách câu theo cách unicode-aware (hoạt động với cả tiếng Việt có dấu).

    Không dùng thư viện ngoài: chỉ cắt sau dấu kết câu + khoảng trắng.
    """
    if not text:
        return []
    parts = re.split(r"(?<=[.!?…])\s+|\n{2,}", text.strip(), flags=re.UNICODE)
    return [p.strip() for p in parts if p and p.strip()]


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if not chunks:
        return []
    if len(chunks) <= 2:
        # 0-2 chunks thì không có "giữa" để mà quên → giữ nguyên thứ tự điểm số
        return list(chunks)

    # Vì sao chia chẵn/lẻ rồi đảo nửa sau:
    #   - front = index chẵn (0, 2, 4...) giữ chunk điểm cao nhất (index 0) ở NGAY ĐẦU
    #   - back  = index lẻ  (1, 3, 5...) đảo ngược nên chunk điểm cao thứ hai (index 1)
    #     rơi xuống VỊ TRÍ CUỐI — hai vùng LLM chú ý nhất
    #   - các chunk điểm thấp tự động dồn vào giữa, nơi bị "lost in the middle"
    # Phép chia này là một hoán vị nên GIỮ NGUYÊN số lượng chunk (không mất, không lặp).
    front = chunks[0::2]          # 0, 2, 4, ...  → đặt ở đầu
    back = chunks[1::2][::-1]     # 1, 3, 5, ... đảo ngược → đặt ở cuối
    return front + back


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    if not chunks:
        return ""

    context_parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        # Dùng .get() phòng thủ ở mọi tầng: chunk có thể thiếu metadata (ví dụ kết quả
        # PageIndex fallback chỉ có 'section'), không được để KeyError làm hỏng pipeline.
        metadata = chunk.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        # citation_id (nếu có) là số trích dẫn ỔN ĐỊNH theo thứ hạng điểm gốc, được
        # generate_with_citation gắn TRƯỚC khi reorder — nhờ vậy [n] trong câu trả lời
        # luôn khớp với thứ tự nguồn mà UI hiển thị, dù prompt đã bị xáo trộn vị trí.
        citation_no = chunk.get("citation_id", i)

        source = metadata.get("source") or metadata.get("section") or f"Document {citation_no}"
        doc_type = metadata.get("type", "unknown")
        content = str(chunk.get("content", "")).strip()

        header = f"[{citation_no}] Source: {source} | Type: {doc_type}"

        score = chunk.get("score")
        if isinstance(score, (int, float)):
            header += f" | Score: {float(score):.3f}"

        chunk_index = metadata.get("chunk_index")
        if chunk_index is not None:
            header += f" | Chunk: {chunk_index}"

        context_parts.append(f"{header}\n{content}\n")

    return "\n---\n".join(context_parts)


# =============================================================================
# EXTRACTIVE FALLBACK (khi không gọi được LLM)
# =============================================================================

def _extractive_answer(query: str, ordered_chunks: list[dict], reason: str) -> str:
    """
    Sinh câu trả lời trích xuất (extractive) trực tiếp từ context — KHÔNG cần LLM.

    Dùng khi thiếu API key hoặc mọi model đều lỗi. Nguyên tắc: chỉ ghép nguyên văn
    các đoạn điểm cao nhất kèm số trích dẫn [n] khớp với đánh số trong format_context,
    tuyệt đối không tự sinh thêm nội dung (đúng tinh thần "không bịa đặt" của RAG).

    Args:
        query: Câu hỏi gốc
        ordered_chunks: Chunks ĐÃ reorder (số [n] = vị trí trong context)
        reason: Lý do phải fallback, hiển thị cho người dùng biết

    Returns:
        Chuỗi answer khác rỗng.
    """
    # Số trích dẫn lấy từ citation_id (khớp với context), fallback về vị trí nếu thiếu.
    # Sắp xếp lại theo điểm giảm dần để đoạn liên quan nhất xuất hiện trước.
    numbered = [
        (chunk.get("citation_id", pos), chunk)
        for pos, chunk in enumerate(ordered_chunks, 1)
    ]
    numbered.sort(
        key=lambda pair: pair[1].get("score", 0.0)
        if isinstance(pair[1].get("score"), (int, float))
        else 0.0,
        reverse=True,
    )

    lines: list[str] = [
        f"[Chế độ KHÔNG-LLM — trích xuất trực tiếp từ tài liệu. Lý do: {reason}]",
        "",
        f"Câu hỏi: {query}",
        "",
        "Các đoạn tài liệu liên quan nhất truy xuất được:",
    ]

    used: list[str] = []
    for citation_no, chunk in numbered[:FALLBACK_NUM_PASSAGES]:
        content = str(chunk.get("content", "")).strip()
        if not content:
            continue
        sentences = _split_sentences(content)
        excerpt = " ".join(sentences[:FALLBACK_SENTENCES_PER_PASSAGE]) if sentences else content
        excerpt = excerpt.strip()
        if len(excerpt) > 600:
            excerpt = excerpt[:600].rstrip() + "..."

        metadata = chunk.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        source = metadata.get("source") or metadata.get("section") or f"Document {citation_no}"

        lines.append(f"  [{citation_no}] {excerpt} (Nguồn: {source})")
        used.append(f"[{citation_no}] {source}")

    if not used:
        # Có chunk nhưng toàn bộ content rỗng → vẫn phải trả chuỗi khác rỗng
        return (
            f"{NO_EVIDENCE_ANSWER}\n\n"
            f"(Chế độ KHÔNG-LLM. Lý do fallback: {reason})"
        )

    lines += [
        "",
        "Trích dẫn: " + "; ".join(used),
        "",
        "Lưu ý: đây là trích dẫn nguyên văn, CHƯA được LLM tổng hợp/diễn giải. "
        "Hãy đặt OPENROUTER_API_KEY trong .env để bật chế độ sinh câu trả lời đầy đủ.",
    ]
    return "\n".join(lines)


def _call_llm(client: Any, model: str, user_message: str) -> str:
    """
    Gọi một model cụ thể qua OpenRouter. Raise nếu lỗi để caller thử model kế tiếp.

    Returns:
        Nội dung answer (đã strip). Chuỗi rỗng nếu model trả về nội dung trống.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    content = (getattr(choices[0].message, "content", None) or "").strip()

    # Model dừng vì CHẠM TRẦN token, không phải vì đã nói xong → câu cuối đang bỏ
    # lửng. Xin viết tiếp một lần rồi nối lại, thay vì trả về câu cụt cho người dùng.
    # Chỉ nối tiếp đúng MỘT lần để không rơi vào vòng gọi API vô tận.
    if getattr(choices[0], "finish_reason", None) == "length" and content:
        _safe_print("  ⚠ Câu trả lời chạm trần token — đang xin model viết tiếp phần còn lại")
        try:
            follow_up = client.chat.completions.create(
                model=model,
                messages=messages
                + [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": "Viết tiếp phần còn dang dở, bắt đầu ngay từ chỗ bị "
                                   "cắt, không nhắc lại nội dung đã viết.",
                    },
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
            )
            extra = (follow_up.choices[0].message.content or "").strip()
            if extra:
                content = f"{content} {extra}".strip()
        except Exception as exc:  # noqa: BLE001 — nối tiếp lỗi thì giữ phần đã có
            _safe_print(f"  ⚠ Không viết tiếp được ({exc}) — giữ phần đã sinh")

    return content


# =============================================================================
# GENERATION
# =============================================================================

def _route_response(answer: str, decision: RouteDecision, **extra: Any) -> dict:
    """Dựng dict trả về chuẩn cho các nhánh KHÔNG chạy retrieval."""
    payload = {
        "answer": answer,
        "sources": [],
        "model": "none",
        "used_llm": False,
        "retrieval_source": "none",
        "route": decision.route.value,
        "route_reason": decision.reason,
        "warnings": [],
    }
    payload.update(extra)
    return payload


def _call_via_pool(user_message: str) -> tuple[str, str]:
    """
    Gọi LLM qua pool nhiều nhà cung cấp, có xử lý trường hợp chạm trần token.

    Returns:
        (answer, nhãn endpoint đã dùng) nếu thành công.
        ("", lý do thất bại) nếu mọi endpoint đều hỏng — caller chuyển sang
        chế độ extractive.
    """
    from .llm_pool import LLMPoolExhausted, get_pool

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    try:
        response, endpoint = get_pool().chat(
            messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
        )
    except LLMPoolExhausted as exc:
        return "", str(exc)
    except Exception as exc:  # noqa: BLE001 — pool lỗi bất thường
        return "", f"{type(exc).__name__}: {exc}"

    choices = getattr(response, "choices", None) or []
    if not choices:
        return "", f"{endpoint.label} trả về nội dung rỗng"

    content = (getattr(choices[0].message, "content", None) or "").strip()

    # Chạm trần token → câu cuối bỏ lửng. Xin viết tiếp đúng MỘT lần.
    if getattr(choices[0], "finish_reason", None) == "length" and content:
        _safe_print("  ⚠ Chạm trần token — xin model viết tiếp phần còn lại")
        try:
            follow, _ = get_pool().chat(
                messages
                + [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": "Viết tiếp phần còn dang dở, bắt đầu ngay từ chỗ "
                                   "bị cắt, không nhắc lại nội dung đã viết.",
                    },
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
            )
            extra = (follow.choices[0].message.content or "").strip()
            if extra:
                content = f"{content} {extra}".strip()
        except Exception as exc:  # noqa: BLE001 — giữ phần đã sinh
            _safe_print(f"  ⚠ Không viết tiếp được ({exc})")

    if not content:
        return "", f"{endpoint.label} trả về nội dung rỗng"
    _safe_print(f"  ℹ LLM: {endpoint.label}")
    return content, endpoint.label


def _finish(
    answer: str,
    chunks: list[dict],
    model: str,
    used_llm: bool,
    retrieval_source: str,
    decision: RouteDecision,
) -> dict:
    """
    Chạy output guardrail rồi dựng dict trả về cho nhánh RAG.

    Nguyên tắc "không tin đầu ra của model": prompt đã cấm chép URL và bắt trích
    dẫn đúng số, nhưng vẫn phải kiểm lại bằng code. Model có thể lờ chỉ dẫn, và
    một trích dẫn [7] khi chỉ có 5 nguồn sẽ khiến người dùng tin vào thứ không
    tồn tại.
    """
    audit = audit_answer(answer, n_sources=len(chunks))
    for warning in audit.warnings:
        _safe_print(f"  ⚠ Guardrail: {warning}")

    return {
        "answer": audit.answer,
        "sources": chunks,
        "model": model,
        "used_llm": used_llm,
        "retrieval_source": retrieval_source,
        "route": decision.route.value,
        "route_reason": decision.reason,
        "warnings": audit.warnings,
        "citations_used": audit.citations_used,
    }


def _chitchat_response(query: str, decision: RouteDecision) -> dict:
    """
    Trả lời câu xã giao bằng LLM nhưng KHÔNG có tài liệu.

    Dùng system prompt riêng (CHITCHAT_SYSTEM_PROMPT) khoá chặt phạm vi, nếu
    không bot sẽ sẵn sàng giải toán hay viết code — biến một hệ RAG kiểm chứng
    được thành trợ lý đa năng không nguồn gốc.

    Không gọi được LLM (thiếu key / hết credit) thì dùng câu chào soạn sẵn —
    nhánh này không cần LLM mới trả lời được.
    """
    from .llm_pool import LLMPoolExhausted, get_pool

    try:
        response, endpoint = get_pool().chat(
            [
                {"role": "system", "content": CHITCHAT_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.5,   # Cao hơn nhánh RAG: lời chào cần tự nhiên, không
            top_p=TOP_P,       # phải bám sát tài liệu như câu trả lời có trích dẫn.
            max_tokens=250,    # Prompt yêu cầu tối đa 3 câu.
        )
        answer = (response.choices[0].message.content or "").strip()
        if answer:
            return _route_response(
                answer, decision, model=endpoint.label, used_llm=True
            )
    except LLMPoolExhausted as exc:
        _safe_print(f"  ⚠ Pool cạn ({exc}) — dùng câu chào soạn sẵn")
    except Exception as exc:  # noqa: BLE001 — nhánh này không cần LLM mới trả lời được
        _safe_print(f"  ⚠ Chitchat lỗi ({exc}) — dùng câu chào soạn sẵn")

    return _route_response(CHITCHAT_FALLBACK_ANSWER, decision)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        0. Router + input guardrail  ← quyết định có cần retrieval hay không
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Output guardrail rồi return answer + sources

    Bước 0 là bổ sung so với đề bài. Lý do: pipeline RAG thuần coi mọi input đều
    là câu hỏi cần tra tài liệu, nên gõ "hi" cũng chạy hết semantic + BM25 + RRF
    + rerank rồi trả về "không xác minh được" — vô nghĩa với người dùng và lãng
    phí một lượt gọi LLM. Router chặn ngay ở đầu.

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str, # 'hybrid' | 'pageindex' | 'none'
            'route': str,            # 'retrieve' | 'chitchat' | 'blocked' | 'empty'
            'route_reason': str,     # Vì sao chọn nhánh đó (hiển thị lên trace)
            'warnings': list[str],   # Cảnh báo từ output guardrail
        }
    """
    # ---- Step 0: Router + input guardrail ---------------------------------
    decision = classify_query(query)

    if decision.route is Route.EMPTY:
        return _route_response(EMPTY_ANSWER, decision)

    if decision.route is Route.BLOCKED:
        _safe_print(f"  ⛔ Chặn truy vấn: {decision.reason} ({decision.matched!r})")
        # KHÔNG gọi LLM ở nhánh này — gửi nội dung tấn công cho model là vô nghĩa
        # và vẫn tốn tiền. Trả câu từ chối cố định.
        return _route_response(BLOCKED_ANSWER, decision)

    if decision.route is Route.CHITCHAT:
        _safe_print(f"  💬 Nhánh chitchat: {decision.reason} — bỏ qua retrieval")
        return _chitchat_response(query, decision)

    # ---- Step 1: Retrieve -------------------------------------------------
    chunks: list[dict] = []
    retrieve_fn = _get_retrieve_fn()
    if retrieve_fn is not None:
        try:
            chunks = retrieve_fn(query, top_k=top_k) or []
        except NotImplementedError:
            _safe_print("  ⚠ Task 9 retrieve() chưa được implement — không có context.")
        except Exception as e:
            # Không bao giờ raise ra ngoài: thiếu chroma_db/, thiếu data/, lỗi model...
            _safe_print(f"  ⚠ Retrieval lỗi ({type(e).__name__}: {e}) — không có context.")

    if not chunks:
        # Không có evidence → trả lời "không xác minh được", KHÔNG gọi LLM (tránh hallucination)
        _safe_print("  ⚠ Không truy xuất được tài liệu nào cho query này.")
        return _route_response(NO_EVIDENCE_ANSWER, decision)

    # ---- Step 2 + 3: Reorder rồi format context ---------------------------
    # Gắn citation_id THEO THỨ HẠNG ĐIỂM GỐC (trước khi reorder) trên bản copy nông,
    # để không mutate kết quả của retrieve(). Nhờ đó số [n] trong câu trả lời khớp
    # đúng với thứ tự phần tử trong 'sources' mà UI/eval liệt kê.
    numbered_chunks = [{**chunk, "citation_id": i} for i, chunk in enumerate(chunks, 1)]
    reordered = reorder_for_llm(numbered_chunks)
    context = format_context(reordered)

    # ---- Step 4: Build prompt --------------------------------------------
    user_message = (
        f"Context:\n{context}\n\n---\n\n"
        f"Question: {query}\n\n"
        "Hãy trả lời dựa HOÀN TOÀN trên context ở trên, mỗi khẳng định kèm trích dẫn "
        "nguồn tương ứng (ví dụ [1], [2] hoặc tên tài liệu)."
    )

    retrieval_source = chunks[0].get("source", "hybrid") if chunks else "none"

    # ---- Step 5: Call LLM qua pool nhiều nhà cung cấp ---------------------
    from .llm_pool import get_pool

    if not get_pool().available():
        reason = "chưa có API key nào khả dụng trong .env (xem src/llm_pool.py)"
        _safe_print(f"  ⚠ Fallback extractive: {reason}")
        return _finish(
            _extractive_answer(query, reordered, reason),
            chunks, "extractive-fallback", False, retrieval_source, decision,
        )

    # Gọi qua LLM pool: tự xoay vòng Groq → Cerebras → Gemini → ... → OpenRouter.
    # Một key duy nhất là điểm chết đơn lẻ — tài khoản OpenRouter của nhóm đã từng
    # hết credit giữa lúc chạy đánh giá (lỗi 402), kéo cả pipeline dừng theo.
    answer, model_used = _call_via_pool(user_message)
    if answer:
        return _finish(answer, chunks, model_used, True, retrieval_source, decision)
    last_error = model_used  # _call_via_pool trả lý do lỗi ở vị trí này khi thất bại

    # Mọi model đều thất bại → extractive fallback, KHÔNG raise
    reason = f"tất cả model LLM đều lỗi (lỗi cuối: {last_error})"
    _safe_print(f"  ⚠ Fallback extractive: {reason}")
    return _finish(
        _extractive_answer(query, reordered, reason),
        chunks, "extractive-fallback", False, retrieval_source, decision,
    )


if __name__ == "__main__":
    # Console Windows mặc định cp1252 → ép UTF-8 để câu trả lời tiếng Việt hiển thị
    # đúng dấu. Chỉ làm ở nhánh demo, KHÔNG làm lúc import (pytest/Streamlit tự quản stdout).
    try:
        import sys

        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 — stream không hỗ trợ reconfigure thì bỏ qua
        pass

    test_queries = [
        "Shopee hỗ trợ những phương thức thanh toán nào?",
        "Làm sao để yêu cầu đổi trả hay hoàn tiền?",
        "Cần chuẩn bị bằng chứng gì khi yêu cầu hoàn tiền?",
    ]

    for q in test_queries:
        _safe_print(f"\n{'='*70}")
        _safe_print(f"Q: {q}")
        _safe_print("=" * 70)
        result = generate_with_citation(q)
        _safe_print(f"\nA: {result['answer']}")
        _safe_print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
