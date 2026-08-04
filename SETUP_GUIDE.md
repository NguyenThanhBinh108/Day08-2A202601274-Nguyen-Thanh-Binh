# HƯỚNG DẪN THỰC HIỆN — CHECKPOINT 0 & 1

# Role 2: Data & Pipeline Specialist

# ====================================================

## ⚡ NHANH NHẤT: Chạy từng lệnh trong Terminal PowerShell

### CHECKPOINT 0 — Setup Môi Trường (10 phút)

```powershell
# 1. Mở PowerShell trong thư mục project
cd d:\VINAI_Team_093\LAB\Day08-2A202601274-Nguyen-Thanh-Binh

# 2. Tạo virtual environment
python -m venv .venv

# 3. Kích hoạt venv
.\.venv\Scripts\Activate.ps1

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Cài requirements (có thể mất 5-10 phút)
pip install -r requirements.txt

# 6. Cài playwright chromium (dùng cho Task 2 crawl)
playwright install chromium

# 7. Tạo file .env
copy .env.example .env
# → Sau đó mở .env và điền OPENROUTER_API_KEY=sk-or-...

# 8. Kiểm tra môi trường (CP0)
python check_environment.py
```

---

### CHECKPOINT 1 — Thu Thập Dữ Liệu (25 phút)

#### Task 1: Tải PDF Chính Sách (3+ files)

```powershell
python src/task1_collect_legal_docs.py
```

Output: 6 file PDF trong `data/landing/legal/`

#### Task 2: Crawl Bài Viết News (5+ files)

```powershell
python src/task2_crawl_news.py
```

Output: 8+ file JSON trong `data/landing/news/`

#### Task 3: Convert sang Markdown

```powershell
python src/task3_convert_markdown.py
```

Output: file .md trong `data/standardized/`

#### Kiểm tra kết quả CP1:

```powershell
dir data\landing\legal\
dir data\landing\news\
dir data\standardized\
```

---

## 📋 Checklist CP1

- [ ] ≥3 file PDF trong `data/landing/legal/`
- [ ] ≥5 file JSON trong `data/landing/news/`
- [ ] Các file .md đã có trong `data/standardized/`

---

## ⚠️ Lỗi thường gặp

| Lỗi                                    | Cách fix                                         |
| --------------------------------------- | ------------------------------------------------- |
| `MissingDependencyException` (Task 3) | `pip install "markitdown[pdf]"`                 |
| `Executable doesn't exist` (Task 2)   | `playwright install chromium`                   |
| `UnicodeEncodeError`                  | `$env:PYTHONIOENCODING="utf-8"`                 |
| `pip` không nhận diện              | Kiểm tra`.venv` đã được kích hoạt chưa |
