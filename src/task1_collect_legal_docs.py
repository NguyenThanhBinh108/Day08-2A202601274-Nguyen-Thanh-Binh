"""
Task 1 — Thu thập văn bản chính sách thương mại điện tử / hỗ trợ khách hàng.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản chính sách (PDF/DOCX) từ trang chính thức của một sàn TMĐT.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, mô tả đúng nội dung.

Gợi ý nguồn (ví dụ trang công khai Shopee Vietnam — help.shopee.vn):
    - https://help.shopee.vn/portal/4/article/77251 (Chính sách trả hàng và hoàn tiền)
    - https://help.shopee.vn/portal/4/article/79198 (Phương thức thanh toán)
    - https://help.shopee.vn/portal/4/article/77244 (Chính sách bảo mật)

Gợi ý văn bản (chủ đề chính sách thương mại điện tử):
    - Chính sách đổi trả/hoàn tiền (Returns/Refund Policy)
    - Phương thức thanh toán (Payment Methods)
    - Chính sách bảo mật (Privacy Policy)
    - Quy định đăng bán sản phẩm cho người bán (Seller Listing Regulations)

Nhớ gắn metadata `customer_role` (`buyer`/`seller`/`both`) cho từng tài liệu — yêu cầu riêng
của K4 Variant (kế thừa từ Lab 07), cần thiết để viết benchmark query dùng metadata_filter.

Lưu ý: một số trang help center dùng JavaScript render nội dung (SPA) — crawl về chỉ thấy
tiêu đề mà không có nội dung thật. Đổi sang bài viết khác cùng domain thay vì cố xử lý,
và chỉ dùng nguồn công khai/được phép chia sẻ.
"""

from __future__ import annotations

import html as html_lib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # chỉ dùng cho type hint, không import thật lúc runtime
    from requests import Response

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# ---------------------------------------------------------------------------
# Hằng số cấu hình
# ---------------------------------------------------------------------------

# Test yêu cầu >= 3 file; lấy dư 1 file để còn biên an toàn nếu 1 nguồn hỏng về sau.
TARGET_FILE_COUNT = 4
MIN_REQUIRED_FILES = 3

# Trần cứng: nếu đủ 4 file rồi mà corpus vẫn thiếu từ khoá benchmark thì cho phép lấy
# thêm vài nguồn nữa, nhưng không bao giờ vượt quá con số này.
MAX_FILE_COUNT = 6

# Test cũng yêu cầu mỗi file > 1KB (file nhỏ hơn thường là trang lỗi/redirect).
MIN_FILE_SIZE = 1024

# Trang HTML crawl về mà quá ngắn nghĩa là SPA chưa render nội dung -> bỏ qua nguồn đó.
MIN_TEXT_LENGTH = 800

# User-Agent thật: nhiều CDN chặn thẳng UA mặc định "python-requests/x.y".
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 30
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 2.0
POLITE_DELAY_SECONDS = 1.0

# Font core của fpdf2 (Helvetica/Times) chỉ encode được latin-1 -> vỡ với tiếng Việt.
# Ưu tiên nạp TTF unicode có sẵn trên Windows; thứ tự = mức độ phổ biến.
UNICODE_FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("C:/Windows/Fonts/tahoma.ttf"),
    Path("C:/Windows/Fonts/times.ttf"),
    # Linux/macOS (phòng khi chấm bài trên máy khác)
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/Library/Fonts/Arial.ttf"),
)

# Corpus bắt buộc chứa các từ khoá tiếng Anh này vì benchmark query của lab dùng tiếng Anh.
REQUIRED_KEYWORDS = (
    "payment", "methods", "return", "refund",
    "seller", "listing", "order", "tracking",
)

# Nguồn chính sách chính thức của sàn TMĐT (Shopee Việt Nam — help.shopee.vn).
# customer_role ∈ {"buyer", "seller", "both"} — yêu cầu K4 Variant, dùng cho metadata_filter.
#
# VÌ SAO CHỌN SHOPEE TIẾNG VIỆT Ở TASK 1:
#   1. Đây là văn bản chính sách gốc của một sàn TMĐT thật, đúng yêu cầu đề bài (khác với
#      bài bách khoa toàn thư — loại đó chỉ mô tả *về* thương mại điện tử, không phải
#      chính sách có hiệu lực của sàn).
#   2. Task 2 đã crawl 6 bài hướng dẫn tiếng Anh của eBay. Để Task 1 lấy Shopee tiếng Việt
#      thì corpus thành SONG NGỮ, không trùng nội dung với Task 2, và câu hỏi gợi ý tiếng
#      Việt trong app.py mới có tài liệu để trả lời.
#   3. Đã kiểm chứng: help.shopee.vn trả nội dung ngay trong HTML đầu tiên (~19.600 ký tự
#      text sạch cho bài 77251), nên requests thuần là đủ, không cần playwright.
#      Docstring phía trên có cảnh báo help center thường là SPA — với Shopee thì không phải.
#
# Liệt kê dư 6 nguồn để dự phòng khi vài URL đổi hoặc chặn bot; chạy bình thường sẽ dừng
# ở TARGET_FILE_COUNT. Bốn nguồn đầu đã trải đủ 3 giá trị customer_role.
POLICY_SOURCES: list[dict[str, str]] = [
    {
        "filename": "shopee_chinh_sach_tra_hang_hoan_tien.pdf",
        "url": "https://help.shopee.vn/portal/4/article/77251",
        "title": "Shopee - Chính sách Trả hàng và Hoàn tiền",
        "customer_role": "buyer",
    },
    {
        "filename": "shopee_phuong_thuc_thanh_toan.pdf",
        "url": "https://help.shopee.vn/portal/4/article/79198",
        "title": "Shopee - Phương thức Thanh toán (Payment Methods)",
        "customer_role": "both",
    },
    {
        "filename": "shopee_chinh_sach_bao_mat.pdf",
        "url": "https://help.shopee.vn/portal/4/article/77244",
        "title": "Shopee - Chính sách Bảo mật",
        "customer_role": "both",
    },
    {
        "filename": "shopee_chinh_sach_van_chuyen.pdf",
        "url": "https://help.shopee.vn/portal/4/article/77250",
        "title": "Shopee - Chính sách Vận chuyển và Theo dõi Đơn hàng",
        "customer_role": "seller",
    },
    # --- Dự phòng ---
    {
        "filename": "ebay_user_agreement.pdf",
        "url": "https://www.ebay.com/help/policies/member-behaviour-policies/user-agreement?id=4259",
        "title": "eBay - User Agreement",
        "customer_role": "both",
    },
    {
        "filename": "ebay_seller_performance_policy.pdf",
        "url": "https://www.ebay.com/help/selling/listings/listing-tips/seller-performance-standards?id=4348",
        "title": "eBay - Seller Performance and Listing Standards",
        "customer_role": "seller",
    },
]

# ---------------------------------------------------------------------------
# Regex dùng cho html_to_text — biên dịch sẵn ở module level (rẻ, không phải việc nặng).
# Python 3 mặc định regex trên str là unicode-aware nên xử lý tiếng Việt an toàn.
# ---------------------------------------------------------------------------
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript|svg|head|nav|footer|form)\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_MAIN_RE = re.compile(r"<main\b[^>]*>(.*?)</main\s*>", re.IGNORECASE | re.DOTALL)
_ARTICLE_RE = re.compile(r"<article\b[^>]*>(.*?)</article\s*>", re.IGNORECASE | re.DOTALL)
_BODY_RE = re.compile(r"<body\b[^>]*>(.*?)</body\s*>", re.IGNORECASE | re.DOTALL)
_BLOCK_TAG_RE = re.compile(
    r"</?(p|div|br|li|tr|td|th|h[1-6]|section|article|table|ul|ol|dl|dd|dt|blockquote|pre)\b[^>]*>",
    re.IGNORECASE,
)
_ANY_TAG_RE = re.compile(r"<[^>]+>")
_CITATION_RE = re.compile(r"\[\s*(?:\d+|edit|citation needed)\s*\]", re.IGNORECASE)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INLINE_SPACE_RE = re.compile(r"[ \t\u00a0\u200b]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# Bảng thay ký tự typographic -> ASCII, dùng khi phải fallback về font latin-1.
_LATIN1_REPLACEMENTS = {
    "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u2022": "*",
    "\u00a0": " ", "\u2192": "->", "\u20ab": "VND",
}

# ---------------------------------------------------------------------------
# Nội dung dự phòng offline — CHỈ dùng khi toàn bộ nguồn mạng thất bại.
# Song ngữ Anh/Việt và cố ý chứa đủ REQUIRED_KEYWORDS để pipeline phía sau vẫn chạy được.
# ---------------------------------------------------------------------------
FALLBACK_DOCS: list[dict[str, str]] = [
    {
        "filename": "sample_returns_and_refunds_policy.pdf",
        "title": "SAMPLE - Marketplace Returns and Refunds Policy",
        "customer_role": "buyer",
        "body": """SAMPLE DOCUMENT - LAB PLACEHOLDER, NOT AN OFFICIAL POLICY.

1. Scope
This returns and refunds policy explains how a buyer may return an item bought on the
marketplace, and how the refund for that order is reviewed, approved and paid back.
The policy applies to every order placed through the marketplace checkout, regardless of
the payment methods used by the buyer.

2. Return window
A buyer may open a return request within 7 calendar days from the delivery date shown in
the order tracking timeline. For pre-order and made-to-order goods the window starts on
the day the shipment is marked delivered by the carrier.

3. Conditions for a valid return
The item must be returned with its original packaging, accessories and free gifts. The
buyer must attach evidence (photo or unboxing video) when the reason for the return is a
damaged, counterfeit or wrong item. A return request without evidence may be rejected by
the seller and closed automatically after 48 hours.

4. Refund methods and timelines
The refund is always issued to the original payment method used for the order:
- Card payment: 7 to 14 business days, depending on the issuing bank.
- E-wallet: within 24 hours after the return is approved.
- Cash on delivery: refunded to the bank account declared by the buyer, 3 to 5 business days.
Shipping fees are refunded in full when the return is caused by the seller.

5. Seller responsibilities
The seller must respond to a return request within 2 business days. If the seller does not
respond, the marketplace approves the refund on behalf of the buyer. Repeated failures
affect the seller performance score and may suspend the seller listing privileges.

6. Non-returnable items
Perishable food, personal hygiene products, digital codes already revealed, and customised
items are not eligible for return unless they arrive damaged.

--- TIẾNG VIỆT / VIETNAMESE ---

1. Phạm vi áp dụng
Chính sách trả hàng và hoàn tiền này giải thích cách người mua yêu cầu trả lại sản phẩm
đã mua trên sàn, và cách khoản hoàn tiền của đơn hàng được duyệt và chi trả.

2. Thời hạn trả hàng
Người mua có thể mở yêu cầu trả hàng trong vòng 7 ngày kể từ ngày giao hàng được ghi nhận
trên lịch sử theo dõi đơn hàng.

3. Điều kiện trả hàng hợp lệ
Sản phẩm phải còn nguyên bao bì, phụ kiện và quà tặng kèm theo. Người mua cần đính kèm
bằng chứng (ảnh hoặc video mở hộp) khi lý do trả hàng là hàng hỏng, hàng giả hoặc giao sai.

4. Phương thức và thời gian hoàn tiền
Tiền được hoàn về đúng phương thức thanh toán ban đầu: thẻ ngân hàng 7-14 ngày làm việc,
ví điện tử trong 24 giờ, thanh toán khi nhận hàng hoàn về tài khoản ngân hàng 3-5 ngày.

5. Trách nhiệm của người bán
Người bán phải phản hồi yêu cầu trả hàng trong 2 ngày làm việc. Nếu không phản hồi, sàn sẽ
duyệt hoàn tiền thay cho người mua.
""",
    },
    {
        "filename": "sample_payment_methods_and_order_tracking.pdf",
        "title": "SAMPLE - Payment Methods and Order Tracking Policy",
        "customer_role": "both",
        "body": """SAMPLE DOCUMENT - LAB PLACEHOLDER, NOT AN OFFICIAL POLICY.

1. Supported payment methods
The marketplace supports the following payment methods for every order:
- Domestic ATM card and internet banking.
- International credit or debit card (Visa, Mastercard, JCB, American Express).
- E-wallet linked to the buyer account.
- Instalment payment for orders above a configured threshold.
- Cash on delivery (COD), available only in supported districts.
A buyer can change the payment method before the order is confirmed by the seller. After
confirmation the payment method is locked and the order must be cancelled to change it.

2. Payment authorisation and capture
Card payments are authorised at checkout and captured when the seller hands the parcel to
the carrier. If the seller does not confirm the order within 48 hours the authorisation is
released automatically and no money is captured.

3. Failed payment
When a payment fails three times in a row the order is cancelled and the buyer receives a
notification. No refund step is required because the money was never captured.

4. Order tracking
Every order has a tracking code issued by the carrier. The order tracking page shows the
following states: Pending confirmation, Preparing shipment, Handed to carrier, In transit,
Out for delivery, Delivered, Delivery failed, Returning to seller.
Tracking information is refreshed every 30 minutes from the carrier API. A gap of more
than 72 hours between two tracking events lets the buyer open a support ticket.

5. Delivery failure and refund
After three failed delivery attempts the parcel returns to the seller and the refund is
issued to the original payment method within 5 business days.

--- TIẾNG VIỆT / VIETNAMESE ---

1. Các phương thức thanh toán được hỗ trợ
Sàn hỗ trợ thẻ ATM nội địa, thẻ quốc tế Visa/Mastercard/JCB, ví điện tử, trả góp và thanh
toán khi nhận hàng (COD). Người mua chỉ đổi được phương thức thanh toán trước khi người
bán xác nhận đơn hàng.

2. Uỷ quyền và ghi nợ
Giao dịch thẻ được uỷ quyền lúc thanh toán và chỉ bị trừ tiền khi người bán giao hàng cho
đơn vị vận chuyển. Nếu quá 48 giờ người bán không xác nhận, giao dịch sẽ được huỷ uỷ quyền.

3. Theo dõi đơn hàng
Mỗi đơn hàng đều có mã vận đơn. Trang theo dõi đơn hàng hiển thị các trạng thái: Chờ xác
nhận, Đang chuẩn bị hàng, Đã giao cho đơn vị vận chuyển, Đang vận chuyển, Đang giao,
Đã giao, Giao không thành công, Đang hoàn về người bán.

4. Giao hàng thất bại và hoàn tiền
Sau 3 lần giao không thành công, đơn hàng sẽ hoàn về người bán và tiền được hoàn trong 5
ngày làm việc về đúng phương thức thanh toán ban đầu.
""",
    },
    {
        "filename": "sample_seller_listing_regulations.pdf",
        "title": "SAMPLE - Seller Listing Regulations and Privacy Notice",
        "customer_role": "seller",
        "body": """SAMPLE DOCUMENT - LAB PLACEHOLDER, NOT AN OFFICIAL POLICY.

1. Who must follow these rules
Every seller who publishes a product listing on the marketplace must follow these listing
regulations. Breaking a rule can hide the listing, suspend the seller account, or block the
payment settlement of the related order.

2. Listing content requirements
A product listing must contain: an accurate title without spam keywords, at least three
real photos of the product, the correct category, the true origin and warranty period, and
a stock quantity the seller can actually deliver. The listing price must include every
mandatory fee except shipping.

3. Prohibited listings
Counterfeit goods, weapons, prescription drugs, live animals, personal data sets, and any
item banned by local law must never be listed. A prohibited listing is removed immediately
and the seller receives a penalty point.

4. Listing quality and order performance
The marketplace scores each seller on: confirmation time, cancellation rate, return and
refund rate, and tracking update quality. A seller whose cancellation rate is above 5% in a
month loses listing boost privileges for the next 30 days.

5. Privacy of buyer data
The seller receives buyer name, phone number and shipping address only to fulfil the order.
Using that personal data for marketing, selling it, or keeping it after the order is closed
violates the privacy policy and local data protection law.

--- TIẾNG VIỆT / VIETNAMESE ---

1. Đối tượng áp dụng
Mọi người bán đăng sản phẩm trên sàn đều phải tuân thủ quy định đăng bán này. Vi phạm có
thể dẫn đến ẩn tin đăng, khoá tài khoản hoặc tạm giữ tiền thanh toán của đơn hàng.

2. Yêu cầu về nội dung tin đăng
Tin đăng phải có tiêu đề đúng sản phẩm, tối thiểu 3 ảnh thật, đúng ngành hàng, ghi rõ xuất
xứ và thời gian bảo hành, và số lượng tồn kho có thật.

3. Sản phẩm bị cấm đăng bán
Hàng giả, vũ khí, thuốc kê đơn, động vật sống, dữ liệu cá nhân và các mặt hàng bị pháp
luật cấm đều không được đăng bán.

4. Chất lượng tin đăng và hiệu suất đơn hàng
Sàn chấm điểm người bán theo thời gian xác nhận, tỷ lệ huỷ đơn, tỷ lệ trả hàng hoàn tiền
và chất lượng cập nhật vận đơn.

5. Bảo mật dữ liệu người mua
Người bán chỉ được dùng tên, số điện thoại và địa chỉ của người mua để hoàn tất đơn hàng.
Dùng dữ liệu đó cho mục đích khác là vi phạm chính sách bảo mật.
""",
    },
]

# Cache module-level: giữ rỗng lúc import, chỉ được điền khi thực sự gọi hàm (lazy).
_RESPONSE_CACHE: dict[str, Any] = {}
_UNICODE_FONT_CACHE: dict[str, Optional[Path]] = {}


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def _safe_print(message: str) -> None:
    """
    In ra stdout nhưng không bao giờ crash vì console Windows dùng codepage cũ (cp1252).

    Console mặc định của Windows không encode được ký tự ✓/⚠ -> UnicodeEncodeError.
    Ở đây ta hạ cấp về ASCII thay vì để script chết giữa chừng.
    """
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="replace").decode("ascii"))


# ---------------------------------------------------------------------------
# Tải nội dung từ mạng
# ---------------------------------------------------------------------------

def _fetch(url: str) -> Optional["Response"]:
    """
    GET một URL với User-Agent thật, timeout 30s và retry tối đa MAX_RETRIES lần.

    Response được cache theo URL trong RAM để `download_file()` và bước fallback HTML
    không phải tải lại cùng một trang hai lần.

    Returns:
        Response nếu thành công (status 2xx), None nếu hỏng — KHÔNG raise ra ngoài.
    """
    if url in _RESPONSE_CACHE:
        return _RESPONSE_CACHE[url]

    try:
        import requests
    except ImportError:
        _safe_print("⚠ Chưa cài `requests` — bỏ qua bước tải từ mạng.")
        return None

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
    }

    # attempt 0 là lần thử đầu, nên tổng số lần gọi = 1 + MAX_RETRIES.
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.get(
                url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True
            )
            response.raise_for_status()
            _RESPONSE_CACHE[url] = response
            return response
        except Exception as exc:  # network, DNS, timeout, HTTP 4xx/5xx...
            remaining = MAX_RETRIES - attempt
            _safe_print(f"  ⚠ Lỗi tải ({type(exc).__name__}: {exc}) — còn {remaining} lần thử")
            if remaining > 0:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))

    _RESPONSE_CACHE[url] = None
    return None


def download_file(url: str, filename: str) -> bool:
    """
    Tải trực tiếp một file PDF về DATA_DIR/filename.

    Chỉ ghi file khi server trả đúng PDF (Content-Type `application/pdf` hoặc magic
    bytes `%PDF`). Nếu server trả HTML, hàm trả False để caller chuyển sang nhánh
    HTML -> text -> PDF.

    Args:
        url: URL cần tải.
        filename: tên file đích (tự thêm đuôi .pdf nếu thiếu).

    Returns:
        True nếu đã ghi được file PDF > MIN_FILE_SIZE, ngược lại False.
    """
    response = _fetch(url)
    if response is None:
        return False

    content_type = response.headers.get("Content-Type", "").lower()
    content = response.content
    is_pdf = "application/pdf" in content_type or content[:5].startswith(b"%PDF")
    if not is_pdf:
        return False

    filepath = DATA_DIR / (filename if filename.lower().endswith(".pdf") else f"{filename}.pdf")
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_bytes(content)
    except OSError as exc:
        _safe_print(f"  ⚠ Không ghi được {filepath.name}: {exc}")
        return False

    if filepath.stat().st_size <= MIN_FILE_SIZE:
        _safe_print(f"  ⚠ {filepath.name} chỉ {filepath.stat().st_size} bytes — bỏ file này.")
        filepath.unlink(missing_ok=True)
        return False

    _safe_print(f"  ✓ Đã tải PDF trực tiếp: {filepath.name}")
    return True


# ---------------------------------------------------------------------------
# HTML -> text
# ---------------------------------------------------------------------------

def _is_supported_script(line: str) -> bool:
    """
    Kiểm tra một dòng có nằm trong hệ chữ Latin (kể cả tiếng Việt có dấu) hay không.

    Wikipedia gắn danh sách liên kết ngôn ngữ (tiếng Hindi, Amharic, Trung, Nhật...) ngay
    trong body. Những dòng đó vừa là rác với corpus TMĐT, vừa làm fpdf2 cảnh báo thiếu
    glyph. Ta bỏ dòng nào có trên 30% chữ cái nằm ngoài khối Latin.

    Các khối được chấp nhận:
        U+0000–U+02FF  Basic Latin, Latin-1 Supplement, Latin Extended-A/B
        U+1E00–U+1EFF  Latin Extended Additional (chứa toàn bộ nguyên âm tiếng Việt)
    """
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return True  # dòng toàn số/dấu câu -> để bước lọc độ dài quyết định

    unsupported = sum(
        1 for c in letters if not (ord(c) < 0x0300 or 0x1E00 <= ord(c) <= 0x1EFF)
    )
    return unsupported / len(letters) <= 0.3


def html_to_text(html: str) -> str:
    """
    Bóc text sạch từ HTML bằng regex (không cần BeautifulSoup để giữ pipeline nhẹ).

    Các bước:
        1. Xoá comment và toàn bộ khối <script>/<style>/<nav>/<footer>... (nhiễu điều hướng).
        2. Ưu tiên vùng nội dung chính <main> → <article> → <body>.
        3. Đổi thẻ block thành xuống dòng để giữ cấu trúc đoạn.
        4. Strip mọi thẻ còn lại, unescape HTML entity, gom khoảng trắng.
        5. Bỏ dòng rác: quá ngắn, lặp lại, hoặc thuộc hệ chữ ngoài Latin/tiếng Việt.

    Regex trên `str` của Python 3 mặc định unicode-aware nên tiếng Việt không bị vỡ.
    """
    if not html:
        return ""

    text = _COMMENT_RE.sub(" ", html)
    text = _SCRIPT_STYLE_RE.sub(" ", text)

    # Thu hẹp về vùng nội dung chính nếu tìm được, tránh kéo theo menu/sidebar.
    for pattern in (_MAIN_RE, _ARTICLE_RE, _BODY_RE):
        match = pattern.search(text)
        if match and len(match.group(1)) > MIN_TEXT_LENGTH:
            text = match.group(1)
            break

    text = _BLOCK_TAG_RE.sub("\n", text)
    text = _ANY_TAG_RE.sub(" ", text)
    text = html_lib.unescape(text)
    text = _CITATION_RE.sub("", text)          # bỏ [1], [edit], [citation needed]
    text = _CONTROL_CHAR_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Gom khoảng trắng theo từng dòng, bỏ dòng rác quá ngắn (icon, dấu phân cách).
    lines: list[str] = []
    previous = None
    for raw_line in text.split("\n"):
        line = _INLINE_SPACE_RE.sub(" ", raw_line).strip()
        if len(line) < 3:
            continue
        if line == previous:  # nhiều template lặp lại y hệt một dòng
            continue
        if not _is_supported_script(line):  # danh sách liên kết ngôn ngữ của Wikipedia
            continue
        lines.append(line)
        previous = line

    return _MULTI_NEWLINE_RE.sub("\n\n", "\n".join(lines)).strip()


# ---------------------------------------------------------------------------
# text -> PDF
# ---------------------------------------------------------------------------

def _find_unicode_font() -> Optional[Path]:
    """Tìm 1 file TTF unicode dùng được cho fpdf2; kết quả được cache lại."""
    if "path" in _UNICODE_FONT_CACHE:
        return _UNICODE_FONT_CACHE["path"]

    found: Optional[Path] = None
    for candidate in UNICODE_FONT_CANDIDATES:
        try:
            if candidate.is_file():
                found = candidate
                break
        except OSError:
            continue

    _UNICODE_FONT_CACHE["path"] = found
    return found


def _sanitize_latin1(text: str) -> str:
    """
    Hạ cấp text về latin-1 khi không tìm được font unicode.

    Font core của fpdf2 (Helvetica/Times/Courier) chỉ encode được latin-1: tiếng Việt có
    dấu và dấu gạch ngang dài (—) sẽ làm fpdf2 raise. Đổi trước các ký tự typographic
    hay gặp cho dễ đọc, phần còn lại thay bằng '?' thay vì để crash.
    """
    for source, target in _LATIN1_REPLACEMENTS.items():
        text = text.replace(source, target)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _wrap_long_tokens(text: str, max_len: int = 60) -> str:
    """
    Cắt nhỏ các "từ" quá dài (URL, chuỗi hash) thành nhiều mẩu.

    fpdf2 raise `Not enough horizontal space to render a single character` nếu gặp token
    dài hơn bề rộng vùng in, nên phải chặn trước khi gọi multi_cell.
    """
    pieces: list[str] = []
    for token in text.split(" "):
        while len(token) > max_len:
            pieces.append(token[:max_len])
            token = token[max_len:]
        pieces.append(token)
    return " ".join(pieces)


def text_to_pdf(title: str, body: str, out_path: Path) -> None:
    """
    Render (title, body) thành file PDF tại out_path bằng fpdf2.

    Ưu tiên nạp TTF unicode (arial.ttf / segoeui.ttf...) để giữ được tiếng Việt có dấu;
    nếu máy không có TTF nào thì hạ cấp về Helvetica + sanitize latin-1.

    Raises:
        RuntimeError: nếu chưa cài fpdf2 hoặc PDF sinh ra nhỏ hơn ngưỡng hợp lệ.
    """
    try:
        from fpdf import FPDF
    except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
        raise RuntimeError("Chưa cài fpdf2 (`pip install fpdf2`)") from exc

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(left=18, top=18, right=18)

    font_path = _find_unicode_font()
    if font_path is not None:
        family = "LabUnicode"
        # fpdf2 >= 2.7.9: chỉ cần truyền file TTF, tham số uni=True đã bị bỏ.
        pdf.add_font(family, "", str(font_path))
        encode = str  # giữ nguyên unicode
    else:
        family = "Helvetica"
        encode = _sanitize_latin1
        _safe_print(
            "  ⚠ Không tìm thấy font TTF unicode — dùng Helvetica, "
            "ký tự tiếng Việt sẽ bị thay bằng '?'."
        )

    pdf.add_page()
    try:
        pdf.set_title(title)
    except Exception:
        pass  # metadata không quan trọng bằng nội dung

    def write_block(line_height: float, content: str) -> None:
        """
        Ghi một đoạn văn bản chiếm trọn bề rộng vùng in.

        Phải set_x() về lề trái trước mỗi lần gọi: mặc định của fpdf2 là
        new_x=XPos.RIGHT, con trỏ x nằm sát lề phải sau lần multi_cell trước, nên
        multi_cell(w=0) kế tiếp sẽ tính ra bề rộng bằng 0 và raise
        "Not enough horizontal space to render a single character".
        """
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, line_height, encode(_wrap_long_tokens(content)))

    # Tiêu đề (dùng cỡ chữ để phân cấp, không dùng style B vì chỉ đăng ký font thường)
    pdf.set_font(family, size=15)
    write_block(8, title.strip() or "Untitled")
    pdf.ln(2)

    pdf.set_font(family, size=10.5)
    for paragraph in body.replace("\r\n", "\n").split("\n"):
        line = _CONTROL_CHAR_RE.sub(" ", paragraph).rstrip()
        if not line.strip():
            pdf.ln(3)          # multi_cell với chuỗi rỗng có thể lỗi -> dùng ln()
            continue
        write_block(5.5, line)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))

    size = out_path.stat().st_size
    if size <= MIN_FILE_SIZE:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"PDF sinh ra quá nhỏ ({size} bytes) — nội dung nguồn không đủ.")


def _build_document_body(source: dict[str, str], text: str) -> str:
    """Ghép khối header truy xuất nguồn + nội dung, để PDF tự chứa metadata gốc."""
    header = (
        f"Source URL: {source['url']}\n"
        f"Customer role: {source['customer_role']}\n"
        f"Collected at: {datetime.now().isoformat(timespec='seconds')}\n"
        f"{'-' * 78}\n"
    )
    return header + text


# ---------------------------------------------------------------------------
# Fallback offline
# ---------------------------------------------------------------------------

def build_offline_fallback(count: int = 3) -> dict[str, dict[str, str]]:
    """
    Sinh `count` file PDF từ nội dung chính sách mẫu viết sẵn trong FALLBACK_DOCS.

    CHỈ dùng khi toàn bộ nguồn mạng thất bại (máy chấm không có Internet, site chặn bot...).
    Ba tài liệu mẫu đầu tiên cố ý phủ đủ REQUIRED_KEYWORDS để task 4-10 vẫn chạy được.

    Returns:
        dict map filename -> {"title", "url", "customer_role", "is_sample"} của các file đã sinh.
    """
    created: dict[str, dict[str, str]] = {}
    for doc in FALLBACK_DOCS[: max(0, count)]:
        out_path = DATA_DIR / doc["filename"]
        try:
            text_to_pdf(doc["title"], doc["body"], out_path)
        except Exception as exc:
            _safe_print(f"  ⚠ Không sinh được {doc['filename']}: {exc}")
            continue

        created[doc["filename"]] = {
            "title": doc["title"],
            "url": "offline-sample://lab08/fallback",
            "customer_role": doc["customer_role"],
            "is_sample": "true",
        }
        _safe_print(f"  ✓ Đã sinh tài liệu mẫu: {doc['filename']} ({out_path.stat().st_size} bytes)")

    if created:
        _safe_print("")
        _safe_print("!" * 78)
        _safe_print("⚠ CẢNH BÁO: đây là DỮ LIỆU MẪU do script tự sinh, KHÔNG phải văn bản thật.")
        _safe_print("⚠ Nhóm BẮT BUỘC phải thay bằng tài liệu chính sách thật (tải tay từ trang")
        _safe_print("⚠ chính thức của sàn TMĐT, đặt vào data/landing/legal/) TRƯỚC KHI NỘP BÀI.")
        _safe_print("!" * 78)
        _safe_print("")
    return created


# ---------------------------------------------------------------------------
# Điều phối
# ---------------------------------------------------------------------------

def _write_metadata(metadata: dict[str, dict[str, str]]) -> None:
    """Ghi data/landing/legal/_metadata.json (map filename -> title/url/customer_role)."""
    metadata_path = DATA_DIR / "_metadata.json"
    try:
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _safe_print(f"✓ Đã ghi metadata: {metadata_path.name} ({len(metadata)} mục)")
    except OSError as exc:
        _safe_print(f"⚠ Không ghi được _metadata.json: {exc}")


def _news_corpus_text() -> str:
    """
    Đọc phần text đã có sẵn từ Task 2 (data/landing/news/*.json).

    Từ khoá tiếng Anh trong REQUIRED_KEYWORDS là yêu cầu của TOÀN BỘ corpus, không phải
    riêng Task 1. Task 2 crawl bài help center tiếng Anh của eBay nên thường đã phủ đủ;
    nếu không tính phần đó vào, Task 1 sẽ tưởng là thiếu và kéo thêm nguồn dự phòng
    không cần thiết (tài liệu Shopee tiếng Việt vốn dĩ không chứa chữ "payment").
    """
    news_dir = DATA_DIR.parent / "news"
    if not news_dir.is_dir():
        return ""

    parts: list[str] = []
    for path in news_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        parts.append(str(data.get("title", "")))
        parts.append(str(data.get("content_markdown", "")))
    return "\n".join(parts)


def _missing_keywords(corpus_text: str) -> list[str]:
    """Trả về các từ khoá tiếng Anh bắt buộc chưa xuất hiện trong corpus đã thu."""
    lowered = (corpus_text + "\n" + _news_corpus_text()).lower()
    return [kw for kw in REQUIRED_KEYWORDS if kw not in lowered]


def _report_keyword_coverage(corpus_text: str) -> None:
    """
    Cảnh báo nếu corpus thiếu từ khoá tiếng Anh mà benchmark query của lab đang dùng.

    Không tự sửa dữ liệu — chỉ báo để nhóm chủ động bổ sung nguồn phù hợp.
    """
    missing = _missing_keywords(corpus_text)
    if missing:
        _safe_print(f"⚠ Corpus đang thiếu từ khoá tiếng Anh: {', '.join(missing)}")
        _safe_print("  → Bổ sung nguồn đúng chủ đề, hoặc chạy build_offline_fallback() để có bản mẫu.")
    else:
        _safe_print("✓ Corpus phủ đủ từ khoá tiếng Anh cần cho benchmark query.")


def collect_all() -> int:
    """
    Thu thập tài liệu chính sách vào data/landing/legal/.

    Với mỗi nguồn trong POLICY_SOURCES: thử tải PDF trực tiếp, nếu không phải PDF thì
    fetch HTML -> html_to_text() -> text_to_pdf(). Dừng khi đủ TARGET_FILE_COUNT file.
    Nguồn nào lỗi thì in cảnh báo rồi bỏ qua, không làm hỏng cả lượt chạy.

    Nếu không tải được file nào (hoặc chưa đủ MIN_REQUIRED_FILES), tự động gọi
    build_offline_fallback() và in cảnh báo rõ ràng rằng đó là dữ liệu mẫu.

    Returns:
        Số file PDF hợp lệ đã tạo trong lượt chạy này.
    """
    setup_directory()

    metadata: dict[str, dict[str, str]] = {}
    # Chỉ gom được text của nhánh HTML→PDF; PDF tải trực tiếp là binary nên không quét
    # từ khoá ở đây (task 3 sẽ convert sang markdown rồi mới kiểm tra được).
    corpus_chunks: list[str] = []
    collected = 0

    for index, source in enumerate(POLICY_SOURCES, 1):
        # Điều kiện dừng: đã đủ TARGET_FILE_COUNT file VÀ corpus đã phủ hết từ khoá
        # benchmark. Nếu còn thiếu từ khoá thì lấy thêm nguồn dự phòng, nhưng chặn
        # cứng ở MAX_FILE_COUNT để không crawl lan man.
        if collected >= TARGET_FILE_COUNT:
            if not _missing_keywords("\n".join(corpus_chunks)) or collected >= MAX_FILE_COUNT:
                break

        filename = source["filename"]
        url = source["url"]
        _safe_print(f"[{index}/{len(POLICY_SOURCES)}] {filename}")

        try:
            # Nhánh 1 — server trả thẳng PDF.
            if download_file(url, filename):
                metadata[filename] = {
                    "title": source["title"],
                    "url": url,
                    "customer_role": source["customer_role"],
                }
                collected += 1
                continue

            # Nhánh 2 — trang HTML: bóc text rồi tự render ra PDF.
            response = _fetch(url)
            if response is None:
                _safe_print("  ⚠ Bỏ qua: không tải được nội dung.")
                continue

            text = html_to_text(response.text)
            if len(text) < MIN_TEXT_LENGTH:
                # Dấu hiệu điển hình của trang SPA render bằng JavaScript.
                _safe_print(f"  ⚠ Bỏ qua: chỉ bóc được {len(text)} ký tự (nghi trang SPA).")
                continue

            out_path = DATA_DIR / filename
            text_to_pdf(source["title"], _build_document_body(source, text), out_path)

            metadata[filename] = {
                "title": source["title"],
                "url": url,
                "customer_role": source["customer_role"],
            }
            corpus_chunks.append(text)
            collected += 1
            _safe_print(f"  ✓ HTML → PDF: {filename} ({out_path.stat().st_size} bytes)")

        except Exception as exc:
            # Không để một nguồn hỏng làm dừng cả quá trình thu thập.
            _safe_print(f"  ⚠ Bỏ qua nguồn lỗi ({type(exc).__name__}): {exc}")

        finally:
            # Đặt trong finally để các nhánh `continue` ở trên vẫn phải chờ:
            # lịch sự với server và tránh bị rate-limit.
            time.sleep(POLITE_DELAY_SECONDS)

    # Không nguồn mạng nào thành công (hoặc chưa đủ số file tối thiểu) -> dùng bản mẫu.
    if collected < MIN_REQUIRED_FILES:
        if collected == 0:
            _safe_print("")
            _safe_print("⚠ TẤT CẢ nguồn mạng đều thất bại — chuyển sang dữ liệu mẫu offline.")
        else:
            _safe_print("")
            _safe_print(
                f"⚠ Chỉ thu được {collected} file (< {MIN_REQUIRED_FILES}) "
                "— bù thêm bằng dữ liệu mẫu offline."
            )
        fallback_meta = build_offline_fallback(MIN_REQUIRED_FILES - collected)
        metadata.update(fallback_meta)
        collected += len(fallback_meta)
        corpus_chunks.extend(
            doc["body"] for doc in FALLBACK_DOCS if doc["filename"] in fallback_meta
        )

    _write_metadata(metadata)
    _report_keyword_coverage("\n".join(corpus_chunks))
    return collected


if __name__ == "__main__":
    setup_directory()
    total = collect_all()

    pdf_files = sorted(
        f for f in DATA_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in {".pdf", ".docx", ".doc"}
    )
    total_bytes = sum(f.stat().st_size for f in pdf_files)

    _safe_print("")
    _safe_print("=" * 78)
    _safe_print(f"TỔNG KẾT — tạo mới {total} file trong lượt chạy này")
    _safe_print(f"Thư mục: {DATA_DIR}")
    for f in pdf_files:
        size = f.stat().st_size
        flag = "OK " if size > MIN_FILE_SIZE else "NHO"
        _safe_print(f"  [{flag}] {f.name:<52} {size:>9,} bytes")
    _safe_print(f"Tổng: {len(pdf_files)} file, {total_bytes:,} bytes")
    if len(pdf_files) < MIN_REQUIRED_FILES:
        _safe_print(f"⚠ Chưa đủ {MIN_REQUIRED_FILES} file — kiểm tra lại mạng hoặc tải tay.")
    _safe_print("=" * 78)
