"""
RAG Chatbot — E-commerce Support
Streamlit app kết nối RAG Retrieval (Task 9) và Generation (Task 10).

Chạy:
    streamlit run app.py
"""

import html
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Thêm project root vào sys.path để import các task từ src/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="E-commerce Support RAG",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# THEME
# =============================================================================
# Bảng màu: giấy ấm + xanh petrol sâu + đồng ánh. Chọn cặp petrol/đồng thay vì
# xanh dương/cam mặc định để giao diện không bị lẫn với template có sẵn.
# Khai báo cả hai chế độ sáng/tối vì Streamlit đổi theme theo cài đặt của máy
# người xem — demo trên máy coach mà chỉ style một chế độ là vỡ bố cục.

CSS = """
<style>
:root {
    --paper:      #FBF9F6;
    --surface:    #FFFFFF;
    --surface-2:  #F5F2EC;
    --ink:        #141C26;
    --muted:      #6B7785;
    --line:       #E6E0D8;
    --brand:      #0E4F52;
    --brand-2:    #12706F;
    --brand-soft: #E4EFEE;
    --accent:     #A85B27;
    --accent-soft:#F7EDE3;
    --ok:         #1B7A4B;
    --warn:       #9A6714;
    --radius:     14px;
    --shadow:     0 1px 2px rgba(20,28,38,.04), 0 12px 28px -14px rgba(20,28,38,.18);
    /* Font hiển thị cho tiêu đề.
       KHÔNG dùng Georgia / ui-serif / Iowan Old Style: các font này thiếu glyph
       tiếng Việt dựng sẵn (ắ, ầ, ộ, ỏ...), trình duyệt phải tự chồng dấu thanh
       lên chữ nền nên tiêu đề hiện ra kiểu "Bă´t đâ`u" — vỡ ngay ở màn hình đầu.
       Segoe UI Variable Display (Win 11) và Segoe UI phủ đủ tiếng Việt; Cambria
       là phương án serif dự phòng cũng phủ đủ. */
    --display:    "Segoe UI Variable Display", "Segoe UI", Cambria, Inter, system-ui, sans-serif;
    --sans:       "Segoe UI Variable Text", "Segoe UI", Inter, system-ui, -apple-system, sans-serif;
    --mono:       "Cascadia Code", "JetBrains Mono", Consolas, ui-monospace, monospace;
}

/* Không dùng @media (prefers-color-scheme) ở đây: .streamlit/config.toml đã khoá
   base = "light", nên nếu CSS tự đổi sang bảng màu tối theo hệ điều hành thì
   widget của Streamlit vẫn nền sáng còn nền trang thành tối — lệch nhau ngay
   trên máy coach nào để dark mode. Một bảng màu duy nhất, hiển thị nhất quán. */

.stApp { background: var(--paper); }
html, body, [class*="css"] { font-family: var(--sans); color: var(--ink); }

/* Nới khoảng thở của khối nội dung chính */
.block-container { padding-top: 2.2rem; padding-bottom: 6rem; max-width: 1180px; }

/* ---------- Header ---------- */
.hero {
    display: flex; align-items: center; gap: 18px;
    padding: 22px 26px; margin-bottom: 6px;
    background: var(--surface);
    border: 1px solid var(--line); border-radius: var(--radius);
    box-shadow: var(--shadow);
}
.hero-mark {
    flex: 0 0 auto; width: 52px; height: 52px; border-radius: 13px;
    display: grid; place-items: center; font-size: 25px;
    background: var(--brand); color: #fff;
    box-shadow: 0 6px 16px -6px var(--brand);
}
.hero-title {
    font-family: var(--display); font-size: 27px; font-weight: 600;
    letter-spacing: -.02em; line-height: 1.15; margin: 0; color: var(--ink);
}
.hero-sub { font-size: 13.5px; color: var(--muted); margin: 5px 0 0; }

/* ---------- Chip trạng thái ---------- */
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 26px; }
.chip {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 6px 12px; border-radius: 999px;
    background: var(--surface); border: 1px solid var(--line);
    font-size: 12px; color: var(--muted); white-space: nowrap;
}
.chip b { color: var(--ink); font-weight: 600; font-family: var(--mono); font-size: 11.5px; }
.chip-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--ok); }
.chip-dot.warn { background: var(--warn); }

/* ---------- Dải trace pipeline ---------- */
.trace { display: flex; align-items: stretch; flex-wrap: wrap; gap: 6px; margin: 4px 0 14px; }
.trace-step {
    display: flex; flex-direction: column; gap: 2px;
    padding: 7px 13px; border-radius: 9px;
    background: var(--surface-2); border: 1px solid var(--line);
    font-size: 11px; color: var(--muted); min-width: 84px;
}
.trace-step .k { font-weight: 600; color: var(--ink); font-size: 11.5px; letter-spacing: .02em; }
.trace-step .v { font-family: var(--mono); font-size: 11px; }
.trace-step.on { background: var(--brand-soft); border-color: var(--brand); }
.trace-step.on .k { color: var(--brand); }
.trace-step.alt { background: var(--accent-soft); border-color: var(--accent); }
.trace-step.alt .k { color: var(--accent); }
.trace-step.off { opacity: .45; }
.trace-arrow { align-self: center; color: var(--line); font-size: 13px; }

/* ---------- Thẻ nguồn ---------- */
.src {
    padding: 13px 15px; margin-bottom: 9px;
    background: var(--surface); border: 1px solid var(--line);
    border-left: 3px solid var(--brand); border-radius: 10px;
}
.src.alt { border-left-color: var(--accent); }
.src-top { display: flex; align-items: baseline; gap: 9px; margin-bottom: 7px; flex-wrap: wrap; }
.src-idx {
    font-family: var(--mono); font-size: 11px; font-weight: 700;
    color: var(--brand); background: var(--brand-soft);
    padding: 1px 7px; border-radius: 5px;
}
.src-name { font-weight: 600; font-size: 13px; color: var(--ink); word-break: break-all; }
.src-tag {
    font-size: 10.5px; text-transform: uppercase; letter-spacing: .06em;
    color: var(--muted); border: 1px solid var(--line);
    padding: 1px 7px; border-radius: 4px;
}
.src-body { font-size: 12.5px; line-height: 1.65; color: var(--muted); }

/* Thanh đo điểm — cho thấy chênh lệch giữa các nguồn bằng thị giác */
.meter { display: flex; align-items: center; gap: 9px; margin: 8px 0 9px; }
.meter-track { flex: 1; height: 4px; border-radius: 2px; background: var(--line); overflow: hidden; }
.meter-fill { height: 100%; border-radius: 2px; background: var(--brand); }
.meter-fill.alt { background: var(--accent); }
.meter-val { font-family: var(--mono); font-size: 11px; color: var(--muted); min-width: 48px; text-align: right; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
.side-label {
    font-size: 10.5px; font-weight: 700; letter-spacing: .13em;
    text-transform: uppercase; color: var(--muted);
    margin: 22px 0 9px; padding-bottom: 6px; border-bottom: 1px solid var(--line);
}
[data-testid="stSidebar"] .stButton > button {
    width: 100%; text-align: left; font-size: 12.5px; line-height: 1.45;
    padding: 9px 12px; border-radius: 9px; white-space: normal; height: auto;
    background: var(--surface-2); color: var(--ink);
    border: 1px solid var(--line); transition: all .13s ease;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--brand-soft); border-color: var(--brand); color: var(--brand);
}

/* ---------- Chat ---------- */
[data-testid="stChatMessage"] {
    background: var(--surface); border: 1px solid var(--line);
    border-radius: var(--radius); padding: 16px 18px; margin-bottom: 13px;
    box-shadow: var(--shadow);
}
[data-testid="stChatMessage"] p { font-size: 14.5px; line-height: 1.72; }
[data-testid="stChatInput"] textarea { font-size: 14px; }

/* ---------- Expander ---------- */
[data-testid="stExpander"] {
    border: 1px solid var(--line); border-radius: 11px;
    background: var(--surface-2); overflow: hidden;
}
[data-testid="stExpander"] summary { font-size: 12.5px; font-weight: 600; color: var(--ink); }

/* ---------- Vặt ---------- */
hr { border-color: var(--line); }
#MainMenu, footer { visibility: hidden; }
.empty {
    padding: 46px 30px; text-align: center;
    background: var(--surface); border: 1px dashed var(--line);
    border-radius: var(--radius);
}
.empty h3 { font-family: var(--display); font-weight: 600; font-size: 19px; margin: 0 0 7px; color: var(--ink); }
.empty p { font-size: 13px; color: var(--muted); margin: 0; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# =============================================================================
# HELPERS
# =============================================================================

def esc(value) -> str:
    """Escape để nội dung tài liệu không phá vỡ HTML của thẻ nguồn."""
    return html.escape(str(value), quote=True)


@st.cache_data(show_spinner=False, ttl=300)
def corpus_stats() -> dict:
    """
    Đếm số vector trong ChromaDB và số file corpus.

    Bọc try/except toàn bộ: app phải mở được cả khi chưa chạy Task 4, chỉ là
    thanh trạng thái sẽ báo "chưa index" thay vì crash trắng màn hình.
    """
    stats = {"vectors": 0, "docs": 0, "ready": False}
    try:
        std = PROJECT_ROOT / "data" / "standardized"
        stats["docs"] = len(list(std.rglob("*.md"))) if std.is_dir() else 0
    except OSError:
        pass
    try:
        import chromadb

        from src.task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME

        if Path(CHROMA_DIR).is_dir():
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            stats["vectors"] = client.get_collection(COLLECTION_NAME).count()
            stats["ready"] = stats["vectors"] > 0
    except Exception:  # noqa: BLE001 — thiếu DB/collection là trạng thái hợp lệ
        pass
    return stats


def embedding_label() -> str:
    try:
        from src.task4_chunking_indexing import EMBEDDING_MODEL

        return EMBEDDING_MODEL.split("/")[-1]
    except Exception:  # noqa: BLE001
        return "n/a"


def render_trace(result: dict, elapsed: float, top_k: int, reranking: bool) -> str:
    """
    Dải trace: cho thấy câu hỏi vừa rồi đi qua đúng những bước nào.

    Giá trị demo nằm ở chỗ phân biệt được HYBRID và PAGEINDEX: khi điểm cosine
    gốc tụt dưới ngưỡng, pipeline chuyển sang vectorless fallback thay vì bịa
    câu trả lời — nhìn dải này là thấy ngay, không phải mở log.
    """
    route = result.get("retrieval_source", "none")
    n_src = len(result.get("sources", []))
    is_fb = route == "pageindex"

    # Nhánh không đi qua retrieval: chỉ hiện router + kết quả, không vẽ các bước
    # semantic/BM25/RRF vì chúng KHÔNG chạy. Vẽ chúng ra sẽ nói dối về luồng.
    decision = result.get("route", "retrieve")
    if decision != "retrieve":
        label = {"chitchat": "Chitchat", "blocked": "Chặn", "empty": "Rỗng"}.get(
            decision, decision
        )
        cls = "alt" if decision == "chitchat" else "off"
        return (
            '<div class="trace">'
            f'<div class="trace-step on"><span class="k">Router</span>'
            f'<span class="v">{esc(result.get("route_reason", ""))[:34]}</span></div>'
            '<span class="trace-arrow">→</span>'
            f'<div class="trace-step {cls}"><span class="k">{label}</span>'
            f'<span class="v">không truy hồi</span></div>'
            '<span class="trace-arrow">→</span>'
            f'<div class="trace-step on"><span class="k">Trả lời</span>'
            f'<span class="v">{elapsed:.1f}s</span></div>'
            "</div>"
        )

    steps = [
        ("Router", "cần truy hồi", "on"),
        ("Semantic", f"bge-m3 · k={top_k * 2}", "on"),
        ("BM25", f"lexical · k={top_k * 2}", "on"),
        ("RRF", "k=60", "on"),
        ("Rerank", "cross-enc" if reranking else "tắt", "on" if reranking else "off"),
    ]
    parts = []
    for name, val, cls in steps:
        parts.append(
            f'<div class="trace-step {cls}"><span class="k">{esc(name)}</span>'
            f'<span class="v">{esc(val)}</span></div><span class="trace-arrow">→</span>'
        )

    route_cls = "alt" if is_fb else "on"
    route_txt = "PageIndex" if is_fb else "Hybrid"
    parts.append(
        f'<div class="trace-step {route_cls}"><span class="k">{route_txt}</span>'
        f'<span class="v">{n_src} chunk</span></div>'
    )

    model = result.get("model", "n/a")
    gen_cls = "on" if result.get("used_llm") else "alt"
    gen_txt = model.split("/")[-1] if result.get("used_llm") else "extractive"
    parts.append(
        f'<span class="trace-arrow">→</span>'
        f'<div class="trace-step {gen_cls}"><span class="k">{esc(gen_txt)}</span>'
        f'<span class="v">{elapsed:.1f}s</span></div>'
    )
    return f'<div class="trace">{"".join(parts)}</div>'


def render_sources(sources: list) -> str:
    """Thẻ nguồn kèm thanh đo điểm, chuẩn hoá theo điểm cao nhất của lượt này."""
    if not sources:
        return ""
    top = max((s.get("score") or 0) for s in sources) or 1.0
    cards = []
    for i, src in enumerate(sources, 1):
        meta = src.get("metadata") or {}
        score = src.get("score") or 0
        pct = max(3, min(100, round(score / top * 100)))
        alt = "alt" if src.get("source") == "pageindex" else ""
        body = (src.get("content") or "")[:340].replace("\n", " ").strip()
        cards.append(
            f'<div class="src {alt}">'
            f'  <div class="src-top">'
            f'    <span class="src-idx">[{i}]</span>'
            f'    <span class="src-name">{esc(meta.get("source", "unknown"))}</span>'
            f'    <span class="src-tag">{esc(meta.get("type", "?"))}</span>'
            f'  </div>'
            f'  <div class="meter">'
            f'    <div class="meter-track"><div class="meter-fill {alt}" style="width:{pct}%"></div></div>'
            f'    <span class="meter-val">{score:.4f}</span>'
            f'  </div>'
            f'  <div class="src-body">{esc(body)}…</div>'
            f'</div>'
        )
    return "".join(cards)


# =============================================================================
# SIDEBAR
# =============================================================================

SUGGESTIONS = [
    "Thời hạn yêu cầu trả hàng/hoàn tiền là bao lâu?",
    "Shopee hỗ trợ những phương thức thanh toán nào?",
    "What payment methods does the marketplace accept?",
    "Which products are sellers prohibited from listing?",
    "How do I track the shipping status of my order?",
    "Công thức nấu phở bò gồm những gì?",
]

with st.sidebar:
    st.markdown(
        '<div class="hero" style="padding:15px 16px;gap:12px;margin-bottom:4px">'
        '<div class="hero-mark" style="width:40px;height:40px;font-size:19px">🛒</div>'
        '<div><div class="hero-title" style="font-size:16px">Support RAG</div>'
        '<div class="hero-sub" style="font-size:11px">Hybrid retrieval console</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-label">Tham số truy hồi</div>', unsafe_allow_html=True)
    top_k = st.slider("Số chunk lấy về (top_k)", 3, 10, 5)
    use_rerank = st.toggle("Bật reranking", value=True)
    show_trace = st.toggle("Hiện trace pipeline", value=True)

    st.markdown('<div class="side-label">Câu hỏi mẫu</div>', unsafe_allow_html=True)
    st.caption("Câu cuối cố ý nằm ngoài phạm vi tài liệu — để thử nhánh fallback.")
    for s in SUGGESTIONS:
        if st.button(s, use_container_width=True, key=f"sug_{s[:24]}"):
            st.session_state["pending_query"] = s

    st.markdown('<div class="side-label">Kiến trúc</div>', unsafe_allow_html=True)
    st.caption(
        "Semantic (bge-m3) + BM25 → hợp nhất RRF (k=60) → rerank → "
        "PageIndex vectorless khi điểm cosine gốc < ngưỡng → LLM sinh câu trả lời có trích dẫn."
    )

    if st.session_state.get("messages"):
        st.markdown('<div class="side-label">Phiên</div>', unsafe_allow_html=True)
        if st.button("Xoá lịch sử chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    '<div class="hero">'
    '  <div class="hero-mark">🛒</div>'
    "  <div>"
    '    <h1 class="hero-title">Trợ lý Chính sách Thương mại điện tử</h1>'
    '    <p class="hero-sub">Hỏi đáp về đổi trả, thanh toán, vận chuyển, quy định người bán '
    "và bảo mật — mọi câu trả lời đều kèm trích dẫn nguồn.</p>"
    "  </div>"
    "</div>",
    unsafe_allow_html=True,
)

stats = corpus_stats()
dot = "" if stats["ready"] else " warn"
idx_txt = f'{stats["vectors"]:,} vector' if stats["ready"] else "chưa index"
st.markdown(
    f'<div class="chips">'
    f'  <span class="chip"><span class="chip-dot{dot}"></span>ChromaDB <b>{idx_txt}</b></span>'
    f'  <span class="chip">Corpus <b>{stats["docs"]} tài liệu</b></span>'
    f'  <span class="chip">Embedding <b>{esc(embedding_label())}</b></span>'
    f'  <span class="chip">Truy hồi <b>Hybrid + RRF</b></span>'
    f'  <span class="chip">Fallback <b>PageIndex</b></span>'
    f"</div>",
    unsafe_allow_html=True,
)

if not stats["ready"]:
    st.warning(
        "Chưa có vector nào trong ChromaDB. Chạy `python -m src.task4_chunking_indexing` "
        "để index corpus trước khi hỏi.",
        icon="⚠️",
    )

# =============================================================================
# LỊCH SỬ CHAT
# =============================================================================

if not st.session_state.messages:
    st.markdown(
        '<div class="empty">'
        "<h3>Bắt đầu bằng một câu hỏi</h3>"
        "<p>Chọn một câu mẫu ở thanh bên, hoặc gõ câu hỏi của bạn bên dưới. "
        "Hệ thống hỗ trợ cả tiếng Việt lẫn tiếng Anh.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("trace"):
                st.markdown(msg["trace"], unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander(f"Nguồn tham khảo — {len(msg['sources'])} chunk"):
                    st.markdown(render_sources(msg["sources"]), unsafe_allow_html=True)

# =============================================================================
# XỬ LÝ CÂU HỎI
# =============================================================================

user_input = st.chat_input("Nhập câu hỏi về chính sách / hỗ trợ khách hàng…")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy hồi tài liệu và tổng hợp câu trả lời…"):
            trace_html = ""
            warnings: list[str] = []
            try:
                from src.task10_generation import generate_with_citation

                t0 = time.perf_counter()
                response = generate_with_citation(query, top_k=top_k)
                elapsed = time.perf_counter() - t0

                answer = response.get("answer", "Chưa thể trả lời.")
                sources = response.get("sources", [])
                warnings = response.get("warnings", [])
                trace_html = render_trace(response, elapsed, top_k, use_rerank)

            except NotImplementedError:
                answer = (
                    "⚠️ **Task 10 chưa được implement.** Hãy hoàn thành "
                    "`src/task10_generation.py` để kết nối pipeline vào UI."
                )
                sources = []
            except Exception as exc:  # noqa: BLE001 — UI không được sập vì lỗi pipeline
                answer = f"❌ **Lỗi khi chạy RAG Pipeline:** {exc}"
                sources = []

        st.markdown(answer)
        if trace_html and show_trace:
            st.markdown(trace_html, unsafe_allow_html=True)
        for warning in warnings:
            st.caption(f"⚠️ {warning}")
        if sources:
            with st.expander(f"Nguồn tham khảo — {len(sources)} chunk", expanded=True):
                st.markdown(render_sources(sources), unsafe_allow_html=True)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "trace": trace_html if show_trace else "",
        }
    )
