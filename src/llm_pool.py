"""
LLM Provider Pool — xoay vòng nhiều nhà cung cấp/API key để chịu được rate limit.

Vì sao cần
----------
Lần chạy đánh giá RAGAS đầu tiên chết giữa chừng với lỗi thật:

    Error code: 402 - This request requires more credits, or fewer max_tokens.
    You requested up to 16384 tokens, but can only afford 4172.

Một key duy nhất là điểm chết đơn lẻ (single point of failure): hết credit, dính
429 rate limit, hoặc nhà cung cấp bảo trì là toàn bộ chatbot ngừng hoạt động —
kể cả khi đang demo trước lớp.

Thiết kế
--------
Pool giữ một danh sách "endpoint" đã sắp thứ tự ưu tiên. Mỗi endpoint gồm
(nhà cung cấp, base_url, api_key, model). Khi gọi:

    1. Duyệt các endpoint còn khoẻ theo thứ tự ưu tiên.
    2. Gọi được -> trả kết quả ngay.
    3. Lỗi -> phân loại rồi xử lý:
         429 rate limit  -> phạt nghỉ (cooldown) 60 giây, thử endpoint kế tiếp
         402 hết credit  -> phạt nghỉ 1 giờ (nạp tiền không diễn ra trong vài giây)
         401/403 sai key -> loại vĩnh viễn khỏi pool trong phiên chạy này
         404 mất model   -> loại vĩnh viễn
         5xx / timeout   -> phạt nghỉ 30 giây
    4. Hết endpoint khoẻ -> raise LLMPoolExhausted để caller chuyển sang chế độ
       extractive (Task 10 đã có sẵn nhánh này).

Vì sao xoay theo NHÀ CUNG CẤP chứ không phải nhiều tài khoản cùng một bên
------------------------------------------------------------------------
Tạo nhiều tài khoản để né hạn mức vi phạm điều khoản dịch vụ của hầu hết nhà
cung cấp. Ngược lại, hạn mức của Groq, Cerebras, Gemini, OpenRouter là ĐỘC LẬP
với nhau, nên xoay vòng qua nhiều bên vừa hợp lệ vừa cộng dồn được hạn mức.
Nếu cả nhóm muốn góp key thì mỗi người dùng key của chính mình — đó là tài khoản
thật của người thật, khác hẳn với việc một người lập hàng loạt tài khoản ảo.

Cấu hình
--------
Khai báo trong .env, bỏ trống dòng nào thì endpoint đó tự bị bỏ qua:

    GROQ_API_KEY=gsk_...
    CEREBRAS_API_KEY=csk-...
    GEMINI_API_KEY=AIza...
    GITHUB_MODELS_TOKEN=ghp_...
    MISTRAL_API_KEY=...
    OPENROUTER_API_KEY=sk-or-v1-...
    OPENAI_API_KEY=sk-proj-...

Muốn góp nhiều key của cùng một bên (ví dụ cả nhóm góp key Groq) thì ngăn bằng
dấu phẩy — pool sẽ tự tách thành nhiều endpoint riêng:

    GROQ_API_KEY=gsk_key_cua_binh,gsk_key_cua_dang,gsk_key_cua_vu
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

# =============================================================================
# ĐỊNH NGHĨA NHÀ CUNG CẤP
# =============================================================================
#
# Thứ tự trong danh sách CHÍNH LÀ thứ tự ưu tiên. Groq đứng đầu vì nhanh nhất và
# hạn mức free rộng nhất; OpenRouter xuống cuối vì tài khoản của nhóm đang cạn
# credit. Mọi nhà cung cấp ở đây đều phơi endpoint tương thích OpenAI SDK, nên
# dùng chung một client class, không cần viết adapter riêng cho từng bên.

PROVIDERS: list[dict[str, Any]] = [
    {
        "name": "groq",
        "env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    },
    {
        "name": "cerebras",
        "env": "CEREBRAS_API_KEY",
        "base_url": "https://api.cerebras.ai/v1",
        "models": ["llama-3.3-70b", "llama3.1-8b"],
    },
    {
        "name": "gemini",
        "env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-2.0-flash", "gemini-2.0-flash-lite"],
    },
    {
        "name": "github",
        "env": "GITHUB_MODELS_TOKEN",
        "base_url": "https://models.inference.ai.azure.com",
        "models": ["gpt-4o-mini", "Phi-3.5-mini-instruct"],
    },
    {
        "name": "mistral",
        "env": "MISTRAL_API_KEY",
        "base_url": "https://api.mistral.ai/v1",
        "models": ["mistral-small-latest", "open-mistral-7b"],
    },
    {
        "name": "openrouter",
        "env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "models": [
            "openai/gpt-4o-mini",
            "meta-llama/llama-3.3-70b-instruct:free",
            "deepseek/deepseek-chat-v3-0324:free",
        ],
    },
    {
        "name": "openai",
        "env": "OPENAI_API_KEY",
        "base_url": None,  # None = dùng endpoint mặc định của OpenAI SDK
        "models": ["gpt-4o-mini"],
    },
]

# Thời gian phạt nghỉ theo loại lỗi (giây).
COOLDOWN_RATE_LIMIT = 60.0      # 429 — hạn mức thường tính theo phút
COOLDOWN_NO_CREDIT = 3600.0     # 402 — nạp tiền không xảy ra trong vài giây
COOLDOWN_SERVER = 30.0          # 5xx / timeout — sự cố tạm thời
COOLDOWN_PERMANENT = float("inf")  # 401/403/404 — hỏng cấu hình, nghỉ hẳn


class LLMPoolExhausted(RuntimeError):
    """Mọi endpoint đều không dùng được. Caller nên chuyển sang chế độ extractive."""


@dataclass
class Endpoint:
    """Một tổ hợp (nhà cung cấp, key, model) có thể gọi được."""

    provider: str
    base_url: Optional[str]
    api_key: str
    model: str
    key_index: int = 0                  # Vị trí key khi một bên có nhiều key

    cooldown_until: float = 0.0
    failures: int = 0
    successes: int = 0
    _client: Any = field(default=None, repr=False)

    @property
    def label(self) -> str:
        suffix = f"#{self.key_index + 1}" if self.key_index else ""
        return f"{self.provider}{suffix}/{self.model}"

    def healthy(self, now: float) -> bool:
        return now >= self.cooldown_until

    def penalize(self, seconds: float, now: float) -> None:
        self.failures += 1
        self.cooldown_until = now + seconds

    def client(self) -> Any:
        """Tạo client OpenAI-compatible một lần rồi tái sử dụng."""
        if self._client is None:
            from openai import OpenAI

            kwargs: dict[str, Any] = {"api_key": self.api_key, "timeout": 60.0}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client


def _classify_error(exc: Exception) -> tuple[float, str]:
    """
    Suy ra hình phạt từ nội dung lỗi.

    Đọc mã trạng thái từ thuộc tính `status_code` nếu SDK có, nếu không thì dò
    trong chuỗi lỗi — các nhà cung cấp trả về định dạng lỗi không thống nhất.
    """
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()

    def has(*codes: str) -> bool:
        return any(str(status) == c or f" {c}" in text or f"code: {c}" in text
                   for c in codes)

    if has("429") or "rate limit" in text or "too many requests" in text:
        return COOLDOWN_RATE_LIMIT, "rate limit"
    if has("402") or "credit" in text or "quota" in text or "billing" in text:
        return COOLDOWN_NO_CREDIT, "hết credit"
    if has("401", "403") or "invalid api key" in text or "unauthorized" in text:
        return COOLDOWN_PERMANENT, "key không hợp lệ"
    if has("404") or "model_not_found" in text or "does not exist" in text:
        return COOLDOWN_PERMANENT, "model không tồn tại"
    if has("500", "502", "503", "504") or "timeout" in text or "connection" in text:
        return COOLDOWN_SERVER, "lỗi phía máy chủ"
    return COOLDOWN_SERVER, f"lỗi khác ({type(exc).__name__})"


class LLMPool:
    """Pool endpoint có xoay vòng, phạt nghỉ và tự phục hồi."""

    def __init__(self, endpoints: Optional[list[Endpoint]] = None) -> None:
        self.endpoints: list[Endpoint] = (
            endpoints if endpoints is not None else discover_endpoints()
        )
        self._cursor = 0

    # -- Trạng thái ---------------------------------------------------------

    def available(self) -> list[Endpoint]:
        now = time.monotonic()
        return [e for e in self.endpoints if e.healthy(now)]

    def status(self) -> list[dict[str, Any]]:
        """Ảnh chụp trạng thái pool — dùng cho trang giám sát / trace giao diện."""
        now = time.monotonic()
        return [
            {
                "endpoint": e.label,
                "healthy": e.healthy(now),
                "cooldown_left": max(0.0, e.cooldown_until - now),
                "successes": e.successes,
                "failures": e.failures,
            }
            for e in self.endpoints
        ]

    # -- Gọi ----------------------------------------------------------------

    def chat(self, messages: list[dict], **kwargs: Any) -> tuple[Any, Endpoint]:
        """
        Gọi chat completion, tự xoay sang endpoint khác khi lỗi.

        Xoay vòng bắt đầu từ vị trí con trỏ (round-robin) thay vì luôn từ đầu
        danh sách — nếu luôn bắt đầu từ endpoint số 1 thì nó gánh toàn bộ tải và
        chạm rate limit trước, trong khi các endpoint sau ngồi không.

        Args:
            messages: Danh sách message chuẩn OpenAI.
            **kwargs: model sẽ bị GHI ĐÈ theo endpoint; các tham số khác
                (temperature, top_p, max_tokens...) truyền thẳng.

        Returns:
            (response, endpoint đã dùng)

        Raises:
            LLMPoolExhausted: khi mọi endpoint đều đang bị phạt nghỉ hoặc lỗi.
        """
        kwargs.pop("model", None)
        candidates = self.available()
        if not candidates:
            raise LLMPoolExhausted(
                "Không còn endpoint nào khả dụng. "
                "Kiểm tra API key trong .env hoặc chờ hết thời gian phạt nghỉ."
            )

        # Xoay điểm bắt đầu để trải đều tải.
        start = self._cursor % len(candidates)
        ordered = candidates[start:] + candidates[:start]
        self._cursor = (self._cursor + 1) % max(1, len(candidates))

        errors: list[str] = []
        for endpoint in ordered:
            now = time.monotonic()
            if not endpoint.healthy(now):
                continue
            try:
                response = endpoint.client().chat.completions.create(
                    model=endpoint.model, messages=messages, **kwargs
                )
                endpoint.successes += 1
                return response, endpoint
            except Exception as exc:  # noqa: BLE001 — mọi lỗi đều phải xoay tiếp
                penalty, label = _classify_error(exc)
                endpoint.penalize(penalty, now)
                errors.append(f"{endpoint.label}: {label}")

        raise LLMPoolExhausted(
            "Tất cả endpoint đều lỗi trong lượt này — " + "; ".join(errors)
        )


# =============================================================================
# KHỞI TẠO
# =============================================================================

def discover_endpoints() -> list[Endpoint]:
    """
    Dựng danh sách endpoint từ biến môi trường.

    Bỏ qua nhà cung cấp không có key. Với mỗi key, chỉ lấy model ĐẦU TIÊN làm
    endpoint chính và model thứ hai làm dự phòng — không nhân bản toàn bộ tổ hợp
    key × model, vì như vậy pool sẽ toàn endpoint của cùng một nhà cung cấp và
    khi bên đó sập thì xoay vòng cũng vô ích.
    """
    endpoints: list[Endpoint] = []
    for spec in PROVIDERS:
        raw = (os.getenv(spec["env"], "") or "").strip()
        if not raw or raw.endswith("..."):     # bỏ qua giá trị placeholder
            continue

        keys = [k.strip() for k in raw.split(",") if k.strip()]
        for key_index, api_key in enumerate(keys):
            for model in spec["models"][:2]:
                endpoints.append(
                    Endpoint(
                        provider=spec["name"],
                        base_url=spec["base_url"],
                        api_key=api_key,
                        model=model,
                        key_index=key_index,
                    )
                )
    return endpoints


_pool: Optional[LLMPool] = None


def get_pool(refresh: bool = False) -> LLMPool:
    """Trả về pool dùng chung (khởi tạo một lần, lazy)."""
    global _pool
    if _pool is None or refresh:
        _pool = LLMPool()
    return _pool


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    pool = get_pool(refresh=True)
    print(f"Phát hiện {len(pool.endpoints)} endpoint:")
    for e in pool.endpoints:
        print(f"  - {e.label}")
    if not pool.endpoints:
        print("\n⚠ Chưa có API key nào trong .env. Xem hướng dẫn ở đầu file này.")
    else:
        print("\nThử gọi...")
        try:
            resp, used = pool.chat(
                [{"role": "user", "content": "Trả lời đúng một từ: OK"}],
                max_tokens=10,
            )
            print(f"  ✓ {used.label} → {resp.choices[0].message.content!r}")
        except LLMPoolExhausted as exc:
            print(f"  ✗ {exc}")
