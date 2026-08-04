"""
Task 2 — Crawl bài viết/hướng dẫn hỗ trợ khách hàng về thương mại điện tử.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trung tâm trợ giúp công khai của một sàn TMĐT.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: theo dõi đơn hàng, đổi phương thức thanh toán, bằng chứng hoàn tiền,
mua hàng xuyên biên giới.

Lưu ý: một số trang help center dùng JavaScript render (SPA) — nếu crawl về chỉ thấy
tiêu đề mà không có nội dung, đổi sang bài viết khác cùng domain thay vì cố xử lý.
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# Ngưỡng chất lượng: bài nào ngắn hơn ngưỡng này gần như chắc chắn là trang lỗi,
# trang redirect hoặc trang SPA chưa render kịp -> bỏ qua thay vì ghi ra file rác.
MIN_CONTENT_CHARS = 600

# Số bài muốn lưu. Đặt 6 (> mức tối thiểu 5) để có biên an toàn khi chấm điểm.
TARGET_ARTICLES = 6

# Mức tối thiểu theo yêu cầu đề bài — dùng để in cảnh báo khi crawl bị hao hụt.
MIN_ARTICLES = 5


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Danh sách URL bài viết cần crawl (eBay Help Center — HTML server-rendered, không phải SPA,
# nội dung tiếng Anh bám sát chủ đề corpus: payment methods, returns/refunds, seller listing,
# order tracking, prohibited items, privacy).
#
# 6 URL đầu là bộ chính (mỗi URL một chủ đề), các URL sau là DỰ PHÒNG cho trường hợp
# một trang bị đổi đường dẫn / redirect / chặn bot. crawl_all() dừng ngay khi đã đủ
# TARGET_ARTICLES bài hợp lệ nên phần dự phòng chỉ được dùng khi thực sự cần.
ARTICLE_URLS = [
    # --- Bộ chính: 6 chủ đề cốt lõi ---
    # Payment methods (phương thức thanh toán)
    "https://www.ebay.com/help/policies/payment-policies/accepted-payments-policy?id=4269",
    # Returns & refunds (trả hàng và hoàn tiền)
    "https://www.ebay.com/help/buying/returns-refunds/returning-item?id=4041",
    # Seller listing (đăng bán sản phẩm)
    "https://www.ebay.com/help/selling/listings/creating-managing-listings/creating-listing?id=4105",
    # Order tracking (theo dõi đơn hàng)
    "https://www.ebay.com/help/buying/shipping-delivery/tracking-your-item?id=4027",
    # Prohibited items (hàng hoá bị cấm/hạn chế)
    "https://www.ebay.com/help/policies/prohibited-restricted-items/prohibited-restricted-items-policy?id=4207",
    # Privacy (chính sách bảo mật)
    "https://www.ebay.com/help/policies/member-behaviour-policies/user-privacy-notice-privacy-policy?id=4260",

    # --- Dự phòng (chỉ crawl khi bộ chính bị hao hụt) ---
    # Paying for items (thanh toán khi mua hàng)
    "https://www.ebay.com/help/buying/paying-items/paying-items?id=4002",
    # Refund / eBay Money Back Guarantee (bằng chứng hoàn tiền)
    "https://www.ebay.com/help/buying/returns-refunds/getting-refund-ebay-money-back-guarantee?id=4210",
    # Selling practices policy (quy định dành cho người bán)
    "https://www.ebay.com/help/policies/selling-policies/selling-practices-policy?id=4346",
    # Order chưa tới nơi — tra cứu tracking và yêu cầu hoàn tiền
    "https://www.ebay.com/help/buying/returns-refunds/track-return-refund?id=4042",
    # Listing tips (mẹo đăng bán)
    "https://www.ebay.com/help/selling/listings/creating-managing-listings/listing-tips?id=4109",
    # Shipping items (đóng gói và gửi hàng)
    "https://www.ebay.com/help/selling/shipping-items/shipping-items?id=4085",
]


def _markdown_to_text(md: Any) -> str:
    """
    Chuẩn hoá kết quả markdown của crawl4ai về str.

    Từ crawl4ai 0.4 trở đi, `result.markdown` KHÔNG còn chắc chắn là str: bản 0.8.5
    trả về object `MarkdownGenerationResult`. Nếu ép str() thẳng, file JSON có thể
    chứa repr của object thay vì nội dung bài viết. Vì vậy thử lần lượt:
        raw_markdown -> fit_markdown -> str(md)
    """
    if md is None:
        return ""
    if isinstance(md, str):
        return md

    for attr in ("raw_markdown", "fit_markdown"):
        value = getattr(md, attr, None)
        if isinstance(value, str) and value.strip():
            return value

    # Fallback cuối: object lạ nhưng có __str__ hữu ích (một số bản trả markdown "giống str").
    return str(md)


def _title_from_url(url: str) -> str:
    """Sinh title tạm từ slug của URL khi trang không khai báo metadata title."""
    # Bỏ query string, lấy segment cuối của path.
    path = url.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    slug = path.rsplit("/", 1)[-1] if "/" in path else path
    # re.sub với flags=re.UNICODE (mặc định ở Python 3) để không phá ký tự tiếng Việt.
    slug = re.sub(r"[-_]+", " ", slug).strip()
    return slug.title() if slug else url


def _extract_title(result: Any, url: str) -> str:
    """Lấy title từ result.metadata (dict); không có thì đặt theo slug URL."""
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("title", "og:title", "og_title"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return _title_from_url(url)


def _unwrap_result(result: Any) -> Any:
    """
    Lấy CrawlResult đơn lẻ.

    crawl4ai 0.8.x bọc kết quả trong `CrawlResultContainer` (list-like). Container đã
    proxy attribute sang phần tử đầu, nhưng vẫn unwrap tường minh cho chắc.
    """
    if result is None:
        return None
    if hasattr(result, "markdown") or hasattr(result, "metadata"):
        return result
    try:
        return result[0]
    except (TypeError, KeyError, IndexError):
        return result


def _empty_article(url: str) -> dict:
    """Record rỗng dùng khi crawl thất bại — luôn trả dict, không bao giờ raise."""
    return {
        "url": url,
        "title": _title_from_url(url),
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": "",
    }


async def crawl_article(url: str, crawler: Optional[Any] = None) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    try:
        from crawl4ai import AsyncWebCrawler
    except ImportError as exc:  # thiếu thư viện -> degrade, không raise
        print(f"  ⚠ Chưa cài crawl4ai ({exc}). Chạy: pip install crawl4ai && playwright install chromium")
        return _empty_article(url)

    article = _empty_article(url)

    try:
        # Cho phép tái sử dụng crawler đã mở sẵn (tiết kiệm thời gian khởi động browser);
        # mặc định vẫn tự mở/đóng đúng như hướng dẫn trong docstring.
        if crawler is not None:
            raw = await crawler.arun(url=url)
        else:
            async with AsyncWebCrawler() as own_crawler:
                raw = await own_crawler.arun(url=url)
    except Exception as exc:
        # Lỗi mạng / browser binary chưa cài / timeout — báo rõ rồi trả record rỗng.
        print(f"  ⚠ Lỗi khi crawl: {type(exc).__name__}: {exc}")
        return article

    result = _unwrap_result(raw)
    if result is None:
        print("  ⚠ Crawler không trả về kết quả nào.")
        return article

    # `success` chỉ có ở CrawlResult; mặc định True để không loại nhầm object khác.
    if not getattr(result, "success", True):
        reason = getattr(result, "error_message", "") or "không rõ nguyên nhân"
        print(f"  ⚠ Crawl thất bại: {reason}")
        return article

    article["title"] = _extract_title(result, url)
    article["content_markdown"] = _markdown_to_text(getattr(result, "markdown", "")).strip()
    article["date_crawled"] = datetime.now().isoformat()
    return article


async def crawl_all() -> list:
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    if not ARTICLE_URLS:
        print("⚠ ARTICLE_URLS đang rỗng — không có gì để crawl.")
        return []

    saved_articles: list = []

    for i, url in enumerate(ARTICLE_URLS, 1):
        if len(saved_articles) >= TARGET_ARTICLES:
            print(f"→ Đã đủ {TARGET_ARTICLES} bài, bỏ qua {len(ARTICLE_URLS) - i + 1} URL dự phòng còn lại.")
            break

        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
        except Exception as exc:
            # Lưới an toàn: một URL hỏng không được làm chết cả mẻ crawl.
            print(f"  ⚠ Bỏ qua (lỗi ngoài dự kiến): {type(exc).__name__}: {exc}")
            continue

        content = article.get("content_markdown") or ""
        if len(content) < MIN_CONTENT_CHARS:
            print(
                f"  ⚠ Bỏ qua: nội dung chỉ {len(content)} ký tự (< {MIN_CONTENT_CHARS}). "
                "Trang có thể render bằng JavaScript, bị redirect hoặc chặn bot."
            )
            continue

        # Đánh số theo số bài ĐÃ LƯU để tên file liên tục article_01, article_02, ...
        filename = f"article_{len(saved_articles) + 1:02d}.json"
        filepath = DATA_DIR / filename
        try:
            filepath.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"  ⚠ Không ghi được file {filepath}: {exc}")
            continue

        saved_articles.append(article)
        print(f"  ✓ Saved: {filepath} ({len(content)} ký tự) — {article['title']}")

    # ---- Tổng kết ----
    saved = len(saved_articles)
    print("-" * 60)
    print(f"Tổng kết: đã lưu {saved} bài vào {DATA_DIR} (mục tiêu {TARGET_ARTICLES}, tối thiểu {MIN_ARTICLES}).")
    if saved < MIN_ARTICLES:
        print(f"⚠ CẢNH BÁO: chỉ có {saved}/{MIN_ARTICLES} bài — CHƯA ĐẠT yêu cầu tối thiểu!")
        print("  Cách khắc phục:")
        print("  1. Kiểm tra mạng và chạy: playwright install chromium")
        print("  2. Thay/bổ sung URL trong ARTICLE_URLS bằng bài help-center khác (ưu tiên trang HTML thuần,")
        print("     không phải SPA) — ví dụ các chủ đề: payment methods, returns & refunds, seller listing,")
        print("     order tracking, privacy policy.")
        print("  3. Mở URL bằng trình duyệt: nếu trang trắng khi tắt JavaScript thì đổi bài khác cùng domain.")
    else:
        print("✓ Đã đạt yêu cầu tối thiểu 5 bài.")

    return saved_articles


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang hướng dẫn/hỗ trợ khách hàng trên help center của sàn TMĐT")
    else:
        asyncio.run(crawl_all())
