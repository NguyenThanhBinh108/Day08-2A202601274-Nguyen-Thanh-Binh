#!/usr/bin/env powershell
<#
.SYNOPSIS
    Script setup tự động cho Lab Day 8 — Checkpoint 0
    Role 2: Data & Pipeline Specialist

.DESCRIPTION
    Script này thực hiện toàn bộ Checkpoint 0:
    1. Tạo virtual environment .venv
    2. Cài đặt requirements.txt
    3. Cài thêm playwright chromium (cho Task 2)
    4. Copy .env.example → .env (nếu chưa có)
    5. Kiểm tra các import quan trọng

.USAGE
    Mở PowerShell trong thư mục project → chạy:
    .\setup_cp0.ps1

    Hoặc mở terminal thông thường và chạy từng lệnh theo từng bước dưới đây.
#>

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  SETUP CHECKPOINT 0 - LAB DAY 8 - ROLE 2" -ForegroundColor Cyan  
Write-Host "============================================================" -ForegroundColor Cyan

# ── Bước 1: Tạo venv
Write-Host "`n[Bước 1] Tạo môi trường ảo .venv..." -ForegroundColor Yellow
python -m venv .venv
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✅ Tạo .venv thành công" -ForegroundColor Green
} else {
    Write-Host "  ❌ Lỗi tạo .venv" -ForegroundColor Red
    exit 1
}

# ── Bước 2: Kích hoạt venv
Write-Host "`n[Bước 2] Kích hoạt .venv..." -ForegroundColor Yellow
.\.venv\Scripts\Activate.ps1
Write-Host "  ✅ Đã kích hoạt .venv" -ForegroundColor Green

# ── Bước 3: Upgrade pip
Write-Host "`n[Bước 3] Upgrade pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

# ── Bước 4: Cài requirements
Write-Host "`n[Bước 4] Cài đặt requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✅ Cài xong requirements.txt" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  Một số package có thể lỗi, tiếp tục..." -ForegroundColor Yellow
}

# ── Bước 5: Cài playwright chromium (cho Task 2 crawl)
Write-Host "`n[Bước 5] Cài Playwright Chromium (cho Task 2)..." -ForegroundColor Yellow
playwright install chromium
Write-Host "  ✅ Cài xong Playwright Chromium" -ForegroundColor Green

# ── Bước 6: Copy .env
Write-Host "`n[Bước 6] Tạo file .env..." -ForegroundColor Yellow
if (-Not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "  ✅ Đã tạo .env từ .env.example" -ForegroundColor Green
    Write-Host "  ⚠️  Nhớ điền OPENROUTER_API_KEY vào file .env!" -ForegroundColor Yellow
} else {
    Write-Host "  ✅ File .env đã tồn tại" -ForegroundColor Green
}

# ── Bước 7: Kiểm tra môi trường
Write-Host "`n[Bước 7] Kiểm tra môi trường..." -ForegroundColor Yellow
python check_environment.py

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  SETUP HOÀN TẤT! Bắt đầu Checkpoint 1:" -ForegroundColor Cyan
Write-Host "  python src\task1_collect_legal_docs.py" -ForegroundColor White
Write-Host "  python src\task2_crawl_news.py" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Cyan
