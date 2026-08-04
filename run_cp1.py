#!/usr/bin/env python3
"""
Script tổng hợp: xóa data_2 và chạy toàn bộ Task 1-3.

Chạy: python run_cp1.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def delete_data_2():
    """Xóa folder data_2 vì đã tích hợp vào task1/task2."""
    data_2 = ROOT / "data_2"
    if data_2.exists():
        shutil.rmtree(data_2)
        print("[OK] Da xoa folder data_2")
    else:
        print("[SKIP] Folder data_2 khong ton tai")


def run_task(script_path: str, task_name: str) -> bool:
    """Chạy một task script."""
    print(f"\n{'='*60}")
    print(f"CHAY: {task_name}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=str(ROOT),
    )
    if result.returncode == 0:
        print(f"\n[OK] {task_name} THANH CONG")
        return True
    else:
        print(f"\n[LOI] {task_name} that bai (returncode={result.returncode})")
        return False


def verify_output():
    """Kiểm tra kết quả."""
    print(f"\n{'='*60}")
    print("KIEM TRA KET QUA")
    print(f"{'='*60}")

    legal_dir = ROOT / "data" / "landing" / "legal"
    news_dir = ROOT / "data" / "landing" / "news"
    std_dir = ROOT / "data" / "standardized"

    # Legal
    if legal_dir.exists():
        pdfs = [f for f in legal_dir.iterdir() if f.suffix == ".pdf"]
        print(f"  data/landing/legal/   : {len(pdfs)} PDFs {'[OK]' if len(pdfs) >= 3 else '[FAIL]'}")
        for f in sorted(pdfs):
            print(f"    - {f.name:55s} ({f.stat().st_size/1024:.1f} KB)")
    else:
        print("  [FAIL] data/landing/legal/ khong ton tai!")

    # News
    if news_dir.exists():
        jsons = [f for f in news_dir.iterdir() if f.suffix == ".json"]
        print(f"  data/landing/news/    : {len(jsons)} JSONs {'[OK]' if len(jsons) >= 5 else '[FAIL]'}")
    else:
        print("  [FAIL] data/landing/news/ khong ton tai!")

    # Standardized
    if std_dir.exists():
        mds = list(std_dir.rglob("*.md"))
        print(f"  data/standardized/   : {len(mds)} .md files {'[OK]' if len(mds) > 0 else '[FAIL]'}")
    else:
        print("  [WARN] data/standardized/ chua ton tai (chay task3 truoc)")


if __name__ == "__main__":
    print("="*60)
    print("RUN CP1 — Task 1, 2, 3")
    print("="*60)

    # Xóa data_2
    delete_data_2()

    # Task 1: Tạo 14 PDFs
    ok1 = run_task("src/task1_collect_legal_docs.py", "Task 1: Thu thap tai lieu phap ly (14 PDFs)")

    # Task 2: Tạo 10 JSONs
    ok2 = run_task("src/task2_crawl_news.py", "Task 2: Luu bai viet huong dan (10 JSONs)")

    # Task 3: Convert sang Markdown
    ok3 = run_task("src/task3_convert_markdown.py", "Task 3: Convert sang Markdown")

    # Verify
    verify_output()

    print(f"\n{'='*60}")
    print(f"TONG KET:")
    print(f"  Task 1: {'PASS' if ok1 else 'FAIL'}")
    print(f"  Task 2: {'PASS' if ok2 else 'FAIL'}")
    print(f"  Task 3: {'PASS' if ok3 else 'FAIL'}")
    print(f"{'='*60}")
    print("\nTiep theo:")
    print("  python src/task4_chunking_indexing.py  # ~5-10 phut (download BAAI/bge-m3)")
    print("  pytest tests/test_individual.py -v")
