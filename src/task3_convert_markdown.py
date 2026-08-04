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
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    converted = 0
    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            try:
                result = md.convert(str(filepath))
                output_path = output_dir / f"{filepath.stem}.md"
                output_path.write_text(result.text_content, encoding="utf-8")
                size_kb = output_path.stat().st_size / 1024
                print(f"  => OK: {output_path.name} ({size_kb:.1f} KB)")
                converted += 1
            except Exception as e:
                print(f"  => LOI: {e}")
                # Fallback: tao file .md co thong tin co ban
                output_path = output_dir / f"{filepath.stem}.md"
                fallback = f"# {filepath.stem}\n\n*Convert loi: {e}*\n"
                output_path.write_text(fallback, encoding="utf-8")

    print(f"  => Da convert {converted} file legal sang markdown")
    return converted


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                output_path = output_dir / f"{filepath.stem}.md"

                # Markdown header voi metadata day du
                header = f"# {data.get('title', 'Unknown')}\n\n"
                header += f"**Source:** {data.get('url', 'N/A')}  \n"
                header += f"**Crawled:** {data.get('date_crawled', 'N/A')}  \n"
                header += f"**Customer Role:** {data.get('customer_role', 'N/A')}  \n"
                header += f"**Category:** {data.get('category', 'N/A')}  \n\n"
                header += "---\n\n"

                content = header + data.get("content_markdown", "")
                output_path.write_text(content, encoding="utf-8")
                size_kb = output_path.stat().st_size / 1024
                print(f"  => OK: {output_path.name} ({size_kb:.1f} KB)")
                converted += 1
            except Exception as e:
                print(f"  => LOI: {e}")

    print(f"  => Da convert {converted} bai news sang markdown")
    return converted


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    legal_count = convert_legal_docs()

    print("\n--- News Articles ---")
    news_count = convert_news_articles()

    total = legal_count + news_count
    print(f"\n{'='*50}")
    print(f"HOAN THANH: {legal_count} legal + {news_count} news = {total} files")
    print(f"Output tai: {OUTPUT_DIR}")
    print(f"{'='*50}")
    return total


if __name__ == "__main__":
    convert_all()
