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
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Bài hướng dẫn hỗ trợ khách hàng thật từ help.shopee.vn (trang công khai).
# customer_role gắn theo K4 Variant để lọc theo đối tượng áp dụng ở Task 4.
ARTICLE_URLS = [
    ("https://help.shopee.vn/portal/4/article/79491-Cach-tra-cuu-ma-van-don-cua-don-hang", "buyer"),
    ("https://help.shopee.vn/portal/4/article/79555-Toi-co-the-thay-doi-phuong-thuc-thanh-toan-cho-don-hang-khong", "buyer"),
    ("https://help.shopee.vn/portal/4/article/188931-Nhung-quy-dinh-chung-ve-Tra-hang-Hoan-tien-cua-Shopee", "both"),
    ("https://help.shopee.vn/portal/4/article/77268-Dieu-Khoan-Dich-Vu-Shopee-Quoc-Te", "both"),
    ("https://help.shopee.vn/portal/4/article/79308-Cach-kiem-tra-lich-su-mua-hang-tren-Shopee", "buyer"),
    ("https://help.shopee.vn/portal/4/article/79687-Lam-Sao-De-Theo-Doi-Hanh-Trinh-Don-Hang", "buyer"),
]


async def crawl_article(url: str) -> dict:
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
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        title = (result.metadata or {}).get("title") if result.metadata else None
        return {
            "url": url,
            "title": title or "Unknown",
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown or "",
        }


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, (url, customer_role) in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)
        article["customer_role"] = customer_role

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ Saved: {filepath} ({len(article['content_markdown'])} chars)")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang hướng dẫn/hỗ trợ khách hàng trên help center của sàn TMĐT")
    else:
        asyncio.run(crawl_all())
