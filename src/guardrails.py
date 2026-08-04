"""
Guardrails & Query Router — tầng phòng thủ đặt TRƯỚC và SAU pipeline RAG.

Vì sao cần module này
---------------------
Pipeline RAG thuần (Task 5..10) mặc định coi MỌI input đều là câu hỏi cần tra tài
liệu. Hệ quả quan sát được trên bản demo: gõ "hi" thì hệ thống vẫn chạy trọn
semantic search + BM25 + RRF + rerank, lôi về 5 chunk chính sách rồi trả lời
"Tôi không thể xác minh thông tin này từ nguồn hiện có" — vừa vô nghĩa với người
dùng, vừa tốn một lượt gọi LLM và ~1,6 giây cho một lời chào.

Kiến trúc production tách bạch ba việc mà pipeline gốc gộp làm một:

    1. Câu này có CẦN tra tài liệu không?        -> Router
    2. Câu này có ĐƯỢC PHÉP xử lý không?          -> Input guardrail
    3. Câu trả lời sinh ra có AN TOÀN/ĐÚNG không? -> Output guardrail

Thiết kế router: ưu tiên LUẬT, không gọi LLM
--------------------------------------------
Có hai cách phân loại truy vấn: gọi một LLM nhỏ để phân loại, hoặc dùng luật.
Ở đây chọn LUẬT vì:

  - Độ trễ: router bằng luật chạy dưới 1ms, gọi LLM tốn thêm 0,5–2 giây cho MỌI
    câu hỏi. Với chatbot demo trực tiếp trước lớp, đó là khác biệt cảm nhận được.
  - Chi phí: mỗi lượt chat sẽ tốn 2 lần gọi API thay vì 1.
  - Tính tất định: luật cho kết quả giống hệt nhau mọi lần chạy, dễ viết test và
    dễ giải thích khi bảo vệ. LLM router có thể đổi ý giữa hai lần chạy.

Đánh đổi: luật không hiểu được câu diễn đạt lạ. Nên tầng thứ hai — ngưỡng điểm
cosine trong Task 9 — mới là thứ quyết định câu hỏi có thuộc phạm vi tài liệu hay
không. Router chỉ lọc những trường hợp KHÔNG CẦN tra tài liệu ngay từ đầu
(chào hỏi, hỏi về chính bot, tấn công prompt injection).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum

# =============================================================================
# ĐỊNH DANH & PHẠM VI CỦA CHATBOT
# =============================================================================

BOT_NAME = "Trợ lý Chính sách TMĐT"

# Phạm vi tài liệu hiện có trong corpus — dùng cho câu trả lời chitchat và cho
# thông báo từ chối, để người dùng biết hỏi gì thì được.
BOT_SCOPE = (
    "chính sách trả hàng và hoàn tiền, phương thức thanh toán, vận chuyển và "
    "theo dõi đơn hàng, quy định đăng bán dành cho người bán, sản phẩm bị cấm, "
    "và chính sách bảo mật dữ liệu"
)

# System prompt cho nhánh CHITCHAT — nhánh này KHÔNG có tài liệu nên phải chặn
# LLM tự do trả lời kiến thức ngoài, nếu không bot sẽ biến thành trợ lý đa năng
# và mất tính kiểm chứng được của một hệ RAG.
CHITCHAT_SYSTEM_PROMPT = f"""Bạn là {BOT_NAME}, một trợ lý CHỈ trả lời dựa trên
kho tài liệu chính sách thương mại điện tử.

Người dùng vừa gửi một câu giao tiếp xã giao (chào hỏi, cảm ơn, hỏi bạn là ai,
hỏi bạn làm được gì) — KHÔNG phải câu hỏi về tài liệu.

Quy tắc:
1. Trả lời ngắn gọn, thân thiện, tối đa 3 câu.
2. Giới thiệu bạn tra cứu được về: {BOT_SCOPE}.
3. Gợi ý người dùng đặt một câu hỏi cụ thể.
4. TUYỆT ĐỐI KHÔNG trả lời kiến thức ngoài kho tài liệu (không giải toán, không
   viết code, không tư vấn y tế/pháp lý, không bàn chuyện thời sự). Nếu người
   dùng hỏi những thứ đó, từ chối lịch sự và nhắc lại phạm vi của bạn.
5. Trả lời cùng ngôn ngữ với người dùng.
6. Không bịa ra thông tin về chính sách — nhánh này không có tài liệu để trích dẫn."""


# =============================================================================
# PHÂN LOẠI TRUY VẤN
# =============================================================================

class Route(str, Enum):
    """Nhánh xử lý được chọn cho một truy vấn."""

    CHITCHAT = "chitchat"          # Chào hỏi / hỏi về bot -> trả lời trực tiếp, KHÔNG retrieval
    RETRIEVE = "retrieve"          # Câu hỏi thật -> chạy pipeline RAG đầy đủ
    BLOCKED = "blocked"            # Vi phạm guardrail -> từ chối, KHÔNG gọi LLM
    EMPTY = "empty"                # Rỗng / quá ngắn -> yêu cầu nhập lại


@dataclass
class RouteDecision:
    """Kết quả định tuyến, kèm lý do để hiển thị lên trace của giao diện."""

    route: Route
    reason: str
    matched: str = ""

    @property
    def needs_retrieval(self) -> bool:
        return self.route is Route.RETRIEVE


# --- Ngưỡng input guardrail -------------------------------------------------

MIN_QUERY_CHARS = 2
MAX_QUERY_CHARS = 2000          # Chặn prompt bomb / dán nguyên tài liệu vào ô chat

# --- Mẫu nhận diện ----------------------------------------------------------
# Ghi chú kỹ thuật: mọi so khớp đều chạy trên chuỗi đã BỎ DẤU tiếng Việt
# (xem _normalize) để "cảm ơn", "cam on", "Cám ơn" cùng khớp một mẫu, và để
# người tấn công không lách được bằng cách thêm/bớt dấu.

_SMALLTALK_EXACT = frozenset(
    {
        "hi", "hey", "hello", "yo", "helo", "hi there",
        "chao", "xin chao", "chao ban", "alo", "a lo",
        "thanks", "thank you", "thank u", "ty", "thx",
        "cam on", "cam on ban", "cam on nhieu", "tks",
        "bye", "goodbye", "tam biet", "ok", "oke", "okay", "okie",
        "good morning", "good afternoon", "good evening",
        "test", "hello world",
    }
)

# Hỏi về chính bot — cũng không cần tra tài liệu.
_ABOUT_BOT_PATTERNS = (
    r"\bban la ai\b", r"\bban ten (la )?gi\b", r"\bban lam duoc gi\b",
    r"\bban giup duoc gi\b", r"\bban co the lam gi\b", r"\bgioi thieu ve ban\b",
    r"\bwho are you\b", r"\bwhat (can|do) you do\b", r"\bwhat are you\b",
    r"\byour name\b", r"\bintroduce yourself\b", r"\bhelp me\b$",
    r"\bban hoat dong (nhu )?the nao\b", r"\bhow do you work\b",
)

# Prompt injection / cố gắng bẻ khoá hệ thống.
_INJECTION_PATTERNS = (
    r"\bignore (all |the |your )?(previous|prior|above)\b",
    r"\bdisregard (all |the |your )?(previous|prior|above)\b",
    r"\bbo qua (moi|tat ca|cac)? ?(huong dan|chi dan|quy tac|lenh)\b",
    r"\bquen (het |di )?(huong dan|quy tac)\b",
    r"\b(reveal|show|print|repeat|output) (me )?(your |the )?(system )?(prompt|instruction)",
    r"\b(cho|noi|in) (toi|tui|minh) (xem )?(system )?prompt\b",
    r"\byou are now\b", r"\bpretend (to be|you are)\b",
    r"\bact as (a |an )?(dan|jailbreak|unrestricted)\b",
    r"\bdeveloper mode\b", r"\bjailbreak\b",
    r"\btu gio (tro di )?ban (la|se)\b",
    r"\bkhong can (tuan theo|theo) (quy tac|huong dan)\b",
)

_SMALLTALK_PREFIX = (
    "hi ", "hey ", "hello ", "chao ", "xin chao ", "cam on ", "thanks ", "thank you ",
)

_ABOUT_BOT_RE = re.compile("|".join(_ABOUT_BOT_PATTERNS), re.IGNORECASE)
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def _normalize(text: str) -> str:
    """
    Chuẩn hoá để so khớp: bỏ dấu tiếng Việt, hạ chữ thường, gom khoảng trắng.

    Bỏ dấu bằng NFD rồi loại các ký tự dấu kết hợp (category Mn). Riêng chữ 'đ'
    không phải chữ 'd' + dấu nên NFD không tách được, phải thay thủ công.
    """
    text = (text or "").strip().lower()
    text = text.replace("đ", "d").replace("Đ", "d")
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", stripped).strip()


def classify_query(query: str) -> RouteDecision:
    """
    Quyết định truy vấn đi nhánh nào.

    Thứ tự kiểm tra CÓ CHỦ ĐÍCH: guardrail an toàn chạy TRƯỚC nhận diện xã giao,
    để câu kiểu "hello, bỏ qua mọi hướng dẫn phía trên" không lọt qua nhánh
    chitchat chỉ vì nó bắt đầu bằng lời chào.

    Args:
        query: Câu người dùng nhập, nguyên văn.

    Returns:
        RouteDecision — luôn trả về một nhánh hợp lệ, không bao giờ raise.
    """
    raw = (query or "").strip()

    if len(raw) < MIN_QUERY_CHARS:
        return RouteDecision(Route.EMPTY, "Câu hỏi rỗng hoặc quá ngắn")

    if len(raw) > MAX_QUERY_CHARS:
        return RouteDecision(
            Route.BLOCKED,
            f"Câu hỏi dài {len(raw):,} ký tự, vượt giới hạn {MAX_QUERY_CHARS:,}",
            matched="max_length",
        )

    normalized = _normalize(raw)

    # 1) An toàn trước tiên.
    injection = _INJECTION_RE.search(normalized)
    if injection:
        return RouteDecision(
            Route.BLOCKED,
            "Phát hiện mẫu prompt injection",
            matched=injection.group(0),
        )

    # 2) Xã giao / hỏi về bot -> không cần tra tài liệu.
    if normalized in _SMALLTALK_EXACT:
        return RouteDecision(Route.CHITCHAT, "Câu chào hỏi xã giao", matched=normalized)

    about = _ABOUT_BOT_RE.search(normalized)
    if about:
        return RouteDecision(Route.CHITCHAT, "Hỏi về chính trợ lý", matched=about.group(0))

    # Câu ngắn mở đầu bằng lời chào và không mang nội dung hỏi.
    # Ngưỡng 25 ký tự: "hi how do i return an item" (26) vẫn phải đi retrieval.
    if len(normalized) <= 25 and normalized.startswith(_SMALLTALK_PREFIX):
        return RouteDecision(Route.CHITCHAT, "Lời chào ngắn", matched=normalized)

    # 3) Mặc định: coi là câu hỏi thật. Việc xác định có thuộc phạm vi tài liệu
    # hay không do ngưỡng cosine ở Task 9 đảm nhiệm — nơi có bằng chứng thật
    # (điểm tương đồng với corpus) chứ không phải đoán bằng từ khoá.
    return RouteDecision(Route.RETRIEVE, "Câu hỏi cần tra tài liệu")


# =============================================================================
# OUTPUT GUARDRAIL
# =============================================================================

_RE_MD_LINK = re.compile(r"\[([^\]]*)\]\((?:https?://|/)[^)]*\)")
_RE_BARE_URL = re.compile(r"https?://\S+")
_RE_CITATION = re.compile(r"\[(\d{1,2})\]")


@dataclass
class AnswerAudit:
    """Kết quả kiểm tra câu trả lời trước khi trả cho người dùng."""

    answer: str
    citations_used: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    stripped_urls: int = 0
    has_citation: bool = False

    @property
    def warnings(self) -> list[str]:
        out: list[str] = []
        if self.invalid_citations:
            out.append(
                "Câu trả lời trích dẫn số không tồn tại: "
                + ", ".join(f"[{n}]" for n in self.invalid_citations)
            )
        if self.stripped_urls:
            out.append(f"Đã gỡ {self.stripped_urls} URL lọt từ tài liệu vào câu trả lời")
        return out


def audit_answer(answer: str, n_sources: int) -> AnswerAudit:
    """
    Kiểm tra và làm sạch câu trả lời trước khi hiển thị.

    Ba việc:
      1. Gỡ URL và cú pháp markdown link còn sót từ tài liệu crawl. Prompt đã cấm
         nhưng LLM vẫn chép lại được, nên phải chặn ở tầng code — nguyên tắc
         "không tin đầu ra của model".
      2. Đối chiếu mọi số trích dẫn [n] với số nguồn thật. Trích dẫn [7] khi chỉ
         có 5 nguồn là dấu hiệu model bịa, phải nêu rõ thay vì để người dùng tin.
      3. Báo khi câu trả lời có nội dung mà KHÔNG trích dẫn gì.

    Args:
        answer: Câu trả lời thô từ LLM.
        n_sources: Số nguồn thực tế đưa vào context.

    Returns:
        AnswerAudit — luôn có `answer` khác rỗng nếu đầu vào khác rỗng.
    """
    text = answer or ""

    cleaned = _RE_MD_LINK.sub(r"\1", text)
    n_urls = len(_RE_BARE_URL.findall(cleaned))
    cleaned = _RE_BARE_URL.sub("", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    used = sorted({int(m) for m in _RE_CITATION.findall(cleaned)})
    invalid = [n for n in used if n < 1 or n > n_sources]

    return AnswerAudit(
        answer=cleaned or text.strip(),
        citations_used=[n for n in used if n not in invalid],
        invalid_citations=invalid,
        stripped_urls=n_urls,
        has_citation=bool(used),
    )


# =============================================================================
# CÂU TRẢ LỜI MẪU CHO CÁC NHÁNH KHÔNG CẦN LLM
# =============================================================================

BLOCKED_ANSWER = (
    "Xin lỗi, tôi không thể xử lý yêu cầu này.\n\n"
    f"Tôi là {BOT_NAME} và chỉ trả lời câu hỏi dựa trên kho tài liệu về "
    f"{BOT_SCOPE}. Bạn vui lòng đặt một câu hỏi thuộc các chủ đề này."
)

EMPTY_ANSWER = (
    "Bạn chưa nhập câu hỏi. Hãy thử hỏi cụ thể, ví dụ: "
    "\"Thời hạn yêu cầu trả hàng là bao lâu?\""
)

# Dùng khi không gọi được LLM cho nhánh chitchat (hết credit / mất mạng).
CHITCHAT_FALLBACK_ANSWER = (
    f"Xin chào! Tôi là {BOT_NAME}.\n\n"
    f"Tôi tra cứu và trả lời dựa trên kho tài liệu về {BOT_SCOPE}. "
    "Mỗi câu trả lời đều kèm trích dẫn nguồn để bạn kiểm chứng.\n\n"
    "Bạn muốn hỏi gì? Ví dụ: \"Thời hạn yêu cầu hoàn tiền là bao lâu?\""
)
