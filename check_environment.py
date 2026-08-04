#!/usr/bin/env python3
"""
Checkpoint 0 — Kiểm Tra Môi Trường
=====================================
Script này kiểm tra toàn bộ môi trường cần thiết cho Lab Day 8:
  - Python version
  - Các package đã cài đặt
  - Import chromadb, sentence_transformers
  - Kết nối API key (OpenRouter)
  - Cấu trúc thư mục dữ liệu

Chạy: python check_environment.py
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent


def check_python_version():
    version = sys.version_info
    print(f"  Python: {sys.version}")
    if version.major == 3 and version.minor >= 10:
        print("  ✅ Python >= 3.10 — OK")
        return True
    else:
        print("  ❌ Cần Python >= 3.10")
        return False


def check_package(package_name: str, import_name: str = None):
    """Kiểm tra một package có cài chưa."""
    if import_name is None:
        import_name = package_name
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  ✅ {package_name} == {version}")
        return True
    except ImportError:
        print(f"  ❌ {package_name} — chưa cài (pip install {package_name})")
        return False


def check_chromadb():
    """Kiểm tra chromadb import và tạo client thử."""
    try:
        import chromadb
        print(f"  ✅ chromadb == {chromadb.__version__}")
        # Thử tạo ephemeral client
        client = chromadb.EphemeralClient()
        col = client.get_or_create_collection("test_check")
        print(f"  ✅ ChromaDB EphemeralClient tạo thành công")
        return True
    except ImportError:
        print("  ❌ chromadb chưa cài")
        return False
    except Exception as e:
        print(f"  ⚠️  chromadb cài rồi nhưng lỗi: {e}")
        return False


def check_sentence_transformers():
    """Kiểm tra sentence_transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        import sentence_transformers
        print(f"  ✅ sentence_transformers == {sentence_transformers.__version__}")
        return True
    except ImportError:
        print("  ❌ sentence_transformers chưa cài")
        return False


def check_env_file():
    """Kiểm tra file .env."""
    env_path = ROOT / ".env"
    if env_path.exists():
        print(f"  ✅ .env file tồn tại: {env_path}")
        # Đọc kiểm tra key
        content = env_path.read_text()
        if "OPENROUTER_API_KEY" in content and "sk-or-" in content:
            print("  ✅ OPENROUTER_API_KEY đã được điền")
        else:
            print("  ⚠️  OPENROUTER_API_KEY chưa được điền trong .env")
        return True
    else:
        print(f"  ❌ Chưa có .env — copy từ .env.example:")
        print(f"     copy .env.example .env")
        return False


def check_api_key():
    """Kiểm tra API key thực sự hoạt động."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if api_key and api_key.startswith("sk-or-"):
            print(f"  ✅ OPENROUTER_API_KEY: {api_key[:20]}...")
            return True
        else:
            print("  ⚠️  OPENROUTER_API_KEY chưa cấu hình")
            return False
    except Exception as e:
        print(f"  ⚠️  Lỗi đọc .env: {e}")
        return False


def check_directories():
    """Kiểm tra cấu trúc thư mục."""
    dirs = [
        ROOT / "data" / "landing" / "legal",
        ROOT / "data" / "landing" / "news",
        ROOT / "data" / "standardized",
        ROOT / "src",
        ROOT / "tests",
        ROOT / "group_project",
    ]
    all_ok = True
    for d in dirs:
        if d.exists():
            print(f"  ✅ {d.relative_to(ROOT)}")
        else:
            print(f"  ❌ Thiếu thư mục: {d.relative_to(ROOT)}")
            all_ok = False
    return all_ok


def main():
    print("=" * 60)
    print("  CHECKPOINT 0 — KIỂM TRA MÔI TRƯỜNG LAB DAY 8")
    print("=" * 60)

    results = {}

    # ── Python version
    print("\n📌 1. Python Version:")
    results["python"] = check_python_version()

    # ── Core packages
    print("\n📌 2. Core Packages:")
    results["dotenv"]   = check_package("python-dotenv", "dotenv")
    results["requests"] = check_package("requests")
    results["numpy"]    = check_package("numpy")
    results["fpdf2"]    = check_package("fpdf2", "fpdf")

    # ── Task-specific packages
    print("\n📌 3. Task-Specific Packages:")
    results["chromadb"]              = check_chromadb()
    results["sentence_transformers"] = check_sentence_transformers()
    results["rank_bm25"]             = check_package("rank-bm25", "rank_bm25")
    results["langchain_splitters"]   = check_package(
        "langchain-text-splitters", "langchain_text_splitters"
    )
    results["streamlit"]             = check_package("streamlit")
    results["openai"]                = check_package("openai")
    results["markitdown"]            = check_package("markitdown")

    # ── Environment
    print("\n📌 4. Environment & API Keys:")
    results["env_file"] = check_env_file()
    results["api_key"]  = check_api_key()

    # ── Directories
    print("\n📌 5. Cấu Trúc Thư Mục:")
    results["directories"] = check_directories()

    # ── Summary
    print("\n" + "=" * 60)
    passed = sum(1 for v in results.values() if v)
    total  = len(results)
    print(f"  KẾT QUẢ: {passed}/{total} kiểm tra đạt")

    critical = ["python", "chromadb", "sentence_transformers"]
    critical_pass = all(results.get(k, False) for k in critical)

    if critical_pass:
        print("\n  ✅ CP0 PASSED — Môi trường cơ bản đã sẵn sàng!")
    else:
        print("\n  ❌ CP0 FAILED — Thiếu package quan trọng. Chạy lại:")
        print("     pip install -r requirements.txt")

    print("=" * 60)
    return critical_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
