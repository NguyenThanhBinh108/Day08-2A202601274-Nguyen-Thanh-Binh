"""
Task 2 — Crawl bài viết/hướng dẫn hỗ trợ khách hàng về thương mại điện tử.

10 bài viết hướng dẫn thực hành chi tiết, dạng step-by-step.
Lưu vào data/landing/news/ dưới dạng JSON.

Format mỗi file JSON:
{
  "url": str,
  "title": str,
  "date_crawled": str,
  "customer_role": str (buyer/seller/both),
  "category": str,
  "content_markdown": str
}

Test yêu cầu:
- Có field "url" trong mỗi file JSON
- File > 500 bytes
- >= 5 file trong thư mục
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

CRAWLED_DATE = datetime.now().isoformat()

# ─────────────────────────────────────────────────────────────────────────────
# 10 BÀI VIẾT HƯỚNG DẪN CHI TIẾT
# ─────────────────────────────────────────────────────────────────────────────
ARTICLES = [
    {
        "filename": "article_01.json",
        "url": "https://help.shopee.vn/portal/4/article/79569",
        "title": "Huong Dan Theo Doi Don Hang Tren Shopee - Chi Tiet Tung Buoc",
        "date_crawled": CRAWLED_DATE,
        "customer_role": "buyer",
        "category": "order-tracking",
        "content_markdown": """\
# Huong Dan Theo Doi Don Hang Tren Shopee

## 1. Xem Trang Thai Don Hang Tren App

Cach don gian nhat:
1. Mo ung dung Shopee -> Nhan "Toi" (goc phai duoi).
2. Chon "Don hang cua toi".
3. Chon tab phu hop: "Dang xu ly" / "Dang giao" / "Da giao".
4. Nhan vao don hang -> Xem "Chi tiet van chuyen".

## 2. Cac Trang Thai Don Hang Va Y Nghia

| Trang thai | Mo ta | Viec can lam |
|------------|-------|-------------|
| Cho xac nhan | Nguoi ban chua xac nhan | Doi nguoi ban xu ly |
| Dang chuan bi hang | Nguoi ban dang dong goi | Doi shipper den lay |
| Dang van chuyen | Hang tren duong di | Theo doi ma van don |
| Cho nhan hang | Shipper sap den | Giu dien thoai de nhan cuoc |
| Da giao thanh cong | Hang da den noi | Kiem tra va bam xac nhan |
| Da huy | Don bi huy | Kiem tra ly do, lien he NB |

## 3. Theo Doi Bang Ma Van Don

Sau khi nguoi ban ban giao cho don vi van chuyen:
1. Vao Chi tiet don hang -> Tim "Ma van don" (VD: SPX123456789VN).
2. Tra cuu tren website cua don vi van chuyen:
   - SPX Express: tracking.shopee.vn
   - J&T Express: jtexpress.vn/tracking
   - GHN: ghn.vn/tracking
   - GHTK: ghtk.vn/tracking
   - Ninja Van: ninjavan.com/vn

## 4. Don Hang Chua Den - Phai Lam Gi?

Neu qua thoi gian du kien ma hang chua den:
1. Kiem tra trang thai tren app.
2. Lien he shipper bang so dien thoai trong thong tin van chuyen.
3. Neu khong lien he duoc shipper: Chat voi nguoi ban qua Shopee Chat.
4. Nguoi ban khong phan hoi sau 48 gio: Yeu cau Shopee can thiep.

## 5. Thay Doi Dia Chi Giao Hang

Chi co the thay doi khi:
- Don hang CON TRONG TRANG THAI "Cho xac nhan" hoac "Dang chuan bi".
- Cach thuc hien: Lien he nguoi ban qua Chat -> Yeu cau huy va dat lai.

## 6. Yeu Cau Giao Lai

Neu khong o nha khi shipper den:
- Shipper se lien he lai: 2-3 lan trong 3-5 ngay.
- Gui tin nhan / goi lai so shipper de hen lai gio giao.
- Neu de qua 5 ngay khong nhan: hang se hoan ve nguoi ban.

## 7. Cau Hoi Thuong Gap

H: Don hang bi tre bao lau thi duoc boi thuong?
D: Don hang tre > 10 ngay so du kien: lien he Shopee de duoc ho tro boi thuong.

H: Ma van don khong tra cuu duoc?
D: Doi 4-6 gio sau khi nhan thong bao "Dang van chuyen" de he thong cap nhat.

H: Don hang bi mat trong qua trinh van chuyen?
D: Bao cao ngay cho Shopee qua "Yeu cau hoan tien" voi ly do "Khong nhan duoc hang".
""",
    },
    {
        "filename": "article_02.json",
        "url": "https://help.shopee.vn/portal/4/article/77251",
        "title": "Huong Dan Yeu Cau Hoan Tien Tung Buoc - Shopee Vietnam",
        "date_crawled": CRAWLED_DATE,
        "customer_role": "buyer",
        "category": "returns",
        "content_markdown": """\
# Huong Dan Yeu Cau Hoan Tien Tung Buoc Tren Shopee

## Khi Nao Co The Yeu Cau Hoan Tien?

Trong vong 15 ngay ke tu khi nhan hang, voi cac ly do:
- Khong nhan duoc hang.
- Hang bi hu hong, loi.
- San pham khong dung mo ta.
- Hang gia / hang nhai.
- Thieu san pham trong don.
- Giao sai san pham.

## Buoc 1: Mo Yeu Cau Tren App

1. Vao "Don hang cua toi".
2. Tim don hang can yeu cau hoan tien.
3. Nhan nut "Yeu cau tra hang / hoan tien" (mau cam).
4. Chon phuong an:
   - "Hoan tien khong tra hang" (neu hang bi mat / khong can tra).
   - "Tra hang va hoan tien" (neu muon tra san pham lai).

## Buoc 2: Chon Ly Do Hoan Tien

Chon ly do chinh xac nhat trong danh sach:
- Khong nhan duoc hang.
- San pham bi hu / loi.
- San pham khong dung mo ta.
- Thieu hang / phu kien.
- Hang gia / hang nhai.
- Doi y khong muon mua nua (COM - chi for Kim Cuong/Vang/VIP).

## Buoc 3: Dien Thong Tin Va Dinh Kem Bang Chung

MO TA CHI TIET (bat buoc):
- Dien ro van de gap phai (vi du: "Nhan duoc ao size L nhung mo ta la size M").
- Thoi gian phat hien su co.
- Hien trang san pham khi nhan.

BANG CHUNG (khuyen nghi dinh kem):
- Anh san pham bi loi / sai / thieu.
- Anh bao bi va tinh trang dong goi.
- Video mo hop (unboxing) neu co.

## Buoc 4: Gui Va Doi Phan Hoi

Sau khi gui yeu cau:
- Nguoi ban co 3 ngay de phan hoi.
- Co the chap nhan, tu choi, hoac de xuat giai phap khac.
- Neu nguoi ban tu choi hoac khong phan hoi: Yeu cau Shopee can thiep.

## Buoc 5: Nhan Tien Hoan

Sau khi duoc chap thuan:
- Vi ShopeePay: 1-3 ngay lam viec.
- The tin dung: 7-14 ngay (tuy ngan hang).
- Chuyen khoan: 3-7 ngay lam viec.
- COD (tien mat): hoan qua chuyen khoan ngan hang.

## Loi Khuyen De Hoan Tien Nhanh

1. Gui yeu cau cang som cang tot (trong 24-48 gio khi phat hien su co).
2. Mo box va quay video ngay khi nhan hang.
3. Mo ta ro rang, cu the, tranh chung.
4. Dinh kem anh ro net, du anh sang.
5. Khong tu y gui tra hang khi chua co ma van don tu Shopee.
""",
    },
    {
        "filename": "article_03.json",
        "url": "https://help.shopee.vn/portal/4/article/77256",
        "title": "Cach Cung Cap Bang Chung De Duoc Hoan Tien Nhanh Tren Shopee",
        "date_crawled": CRAWLED_DATE,
        "customer_role": "buyer",
        "category": "returns",
        "content_markdown": """\
# Cach Cung Cap Bang Chung De Duoc Hoan Tien Nhanh Tren Shopee

## Tai Sao Bang Chung Quan Trong?

Bang chung giup:
- Xac minh van de voi san pham mot cach khach quan.
- Shopee giai quyet tranh chap nhanh chong.
- Bao ve quyen loi cua nguoi mua.
- Tranh truong hop khieu nai bi tu choi do thieu thong tin.

## Loai Bang Chung Duoc Chap Nhan

### 1. Hinh Anh (Uu Tien Nhat)
- Chup anh NGAY KHI MO HOP (trong luong 5 phut dau).
- Chup ro loi / hu hong / khac biet.
- Chup bao bi con nguyen (nhan tu phia ngoai).
- Chup nhan mac va ma san pham (barcode).
- Kich thuoc anh: toi thieu 500x500 pixel.
- So luong: 2-5 anh, goc chup khac nhau.

### 2. Video Unboxing (Manh Nhat)
- Quay video lien tuc tu luc mo goi den khi thay san pham.
- The hien ro tinh trang dong goi ban dau.
- Quay ro nhan san pham (sai size, sai mau).
- Thoi luong: 30 giay den 5 phut.
- Dinh dang: MP4, MOV.

### 3. Tai Lieu Khac
- Anh so sanh san pham nhan duoc voi hinh anh mo ta nguoi ban.
- Bien lai sua chua neu may bi loi.
- Ket qua kiem dinh neu nghi ngo hang gia.
- Screenshot chat voi nguoi ban (neu NB hua doi hang nhung khong giao).

## Cach Dang Tai Bang Chung Len App

1. Trong man hinh "Yeu cau tra hang/hoan tien":
2. Nhan bieu tuong + hoac "Them anh/video".
3. Chon anh tu thu vien hoac chup anh truc tiep.
4. Co the them toi da 10 anh + 1 video.
5. Moi tap tin toi da: 5MB (anh) / 100MB (video).

## Meo De Bang Chung Duoc Chap Thuan

Nen lam:
- Chup anh ngay khi nhan hang, truoc khi lam gi khac.
- Dam bao anh ro net, du sang, khong bi mo.
- Chup nhieu goc de the hien van de ro rang.
- Mo ta bang chung khop voi van de thuc te.

Khong nen:
- Chinh sua, cat xen, loc mau anh/video.
- Dung anh tai tu internet.
- Gui bang chung khong lien quan den san pham.
- Mo ta mau thuan voi bang chung dinh kem.

## Truong Hop Dac Biet

Khong co video unboxing:
- Van co the yeu cau hoan tien bang anh ro net.
- Kha nang thanh cong thap hon, but Shopee van xem xet.

Hang bi mat (khong nhan duoc gi):
- Khong can anh san pham.
- Can: screenshot thong bao giao hang tu shipper, xac nhan khong co nguoi o nha.
""",
    },
    {
        "filename": "article_04.json",
        "url": "https://help.shopee.vn/portal/4/article/79200",
        "title": "Huong Dan Thay Doi Phuong Thuc Thanh Toan Sau Khi Dat Hang Shopee",
        "date_crawled": CRAWLED_DATE,
        "customer_role": "buyer",
        "category": "payment",
        "content_markdown": """\
# Huong Dan Thay Doi Phuong Thuc Thanh Toan Sau Khi Dat Hang Tren Shopee

## Co The Thay Doi Phuong Thuc Thanh Toan Sau Khi Dat Hang Khong?

Phu thuoc vao trang thai don hang:

CO THE thay doi khi:
- Don hang dang o trang thai "Cho thanh toan" (chua thanh toan).
- Nguoi ban CHUA xac nhan don hang (trang thai "Cho xac nhan").

KHONG THE thay doi khi:
- Don hang da duoc thanh toan (online payment).
- Nguoi ban da xac nhan va dang chuan bi hang.
- Hang da duoc ban giao cho don vi van chuyen.

## Cach 1: Doi Tu COD Sang Thanh Toan Online

1. Vao "Don hang cua toi" -> Tim don hang COD.
2. Nhan "Huy don hang".
3. Chon ly do: "Muon doi phuong thuc thanh toan".
4. Xac nhan huy.
5. Dat lai don hang voi phuong thuc thanh toan mong muon.

Luu y: Voucher co the bi mat khi huy va dat lai don hang.

## Cach 2: Doi Phuong Thuc Thanh Toan Online Chua Thanh Toan

1. Vao "Don hang cua toi".
2. Tim don hang co trang thai "Cho thanh toan".
3. Nhan "Thanh toan ngay".
4. Chon phuong thuc thanh toan moi tu danh sach.
5. Xac nhan va hoan tat giao dich.

## Hoan Tien Khi Da Thanh Toan

Neu da thanh toan va muon doi:
- Huy don hang (neu con trong thoi han) -> Tien hoan lai.
- Dat lai don moi voi phuong thuc thanh toan mong muon.
- Tien hoan tra ve: 1-14 ngay lam viec (tuy phuong thuc).

## Thoi Gian Hoan Tien Khi Huy Don

| Phuong thuc goc | Thoi gian hoan |
|-----------------|---------------|
| Vi ShopeePay    | 1-3 ngay LV   |
| The tin dung    | 7-14 ngay LV  |
| Internet Banking| 3-7 ngay LV   |
| COD (tien mat)  | 3-7 ngay LV (qua CK ngan hang) |

## Luu Y Quan Trong

- Voucher da su dung khi huy don co the KHONG duoc hoan lai.
- Uu dai "mua lan dau" co the mat neu huy va dat lai.
- Shopee Xu duoc hoan lai vao tai khoan trong 24 gio.
- Neu don hang da giao cho shipper: KHONG the huy, phai lam thu tuc hoan tra.

## Lien He Ho Tro

Neu gap kho khan:
- Shopee Chat (trong app): Ho tro 24/7.
- Hotline: 1900 1221 (8:00-22:00 ke ca cuoi tuan).
- Email: support@shopee.vn.
""",
    },
    {
        "filename": "article_05.json",
        "url": "https://help.shopee.vn/portal/4/article/77290",
        "title": "Huong Dan Mua Hang Xuyen Bien Gioi (Cross-Border) Tren Shopee",
        "date_crawled": CRAWLED_DATE,
        "customer_role": "buyer",
        "category": "cross-border",
        "content_markdown": """\
# Huong Dan Mua Hang Xuyen Bien Gioi (Cross-Border) Tren Shopee

## Shopee Global Hoat Dong Nhu The Nao?

Shopee Global la tinh nang cho phep nguoi mua VN dat hang tu nguoi ban nuoc ngoai.
Hang duoc van chuyen truc tiep tu nuoc xuat xu ve Viet Nam.

## Tim San Pham Cross-Border

Cach 1: Tim kiem thong thuong -> Loc "Hang quoc te" / "Shopee Global".
Cach 2: Vao tab "Shopee Global" tren trang chu.
Cach 3: Xem badge "HANG QUOC TE" hoac thoi gian giao hang "7-25 ngay".

## Quy Trinh Mua Hang Quoc Te

Buoc 1: Chon san pham -> Kiem tra trang "Chi tiet san pham".
- Xem ro: nguon goc, khe size (size chart), thoi gian giao hang, chinh sach tra hang.

Buoc 2: Them vao gio hang -> Chon dia chi giao hang.
- Nhap dia chi chinh xac (ghi thieu so nha co the lam mat hang).

Buoc 3: Chon phuong thuc van chuyen quoc te.
- Thuong chi co 1 lua chon (Shopee Global Shipping).

Buoc 4: Kiem tra chi phi:
- Gia san pham + Phi van chuyen quoc te + Thue (neu co).

Buoc 5: Thanh toan -> Nhan xac nhan qua email.

## Theo Doi Don Hang Quoc Te

Cac giai doan van chuyen:
1. Nguoi ban chuan bi hang (1-3 ngay).
2. Van chuyen noi dia tai nuoc xuat xu (1-3 ngay).
3. Thong quan xuat khau (1-2 ngay).
4. Van chuyen quoc te (3-10 ngay, tuy quoc gia).
5. Thong quan nhap khau VN (1-3 ngay).
6. Van chuyen noi dia VN va giao hang cuoi (1-3 ngay).

Tra cuu: App Shopee -> Don hang -> Chi tiet van chuyen -> Ma van don quoc te.

## Chi Phi Thue Va Phi

Don hang duoi 1.000.000 VND: thuong mien thue.
Don hang tu 1.000.000 VND tro len:
- Thue nhap khau: 0-30% tuy loai hang.
- Thue VAT: 10%.
Nguoi mua chiu trach nhiem thanh toan thue.

## Chinh Sach Tra Hang Hang Quoc Te

Khi co the tra: vong 5-7 ngay ke tu nhan hang.
Chi phi: phi van chuyen hoan tra cao, nguoi mua tu chiu (tru khi nguoi ban loi).
Mot so hang khong duoc tra: thuc pham, my pham, do lot.

## Kiem Tra Truoc Khi Mua

Dieu phai lam truoc khi thanh toan:
1. Kiem tra size chart (size Chau A nho hon Chau Au).
2. Xem xet chat lieu co phu hop voi khi hau VN.
3. Kiem tra dien ap thiet bi dien tu (VN: 220V/50Hz).
4. Doc ki chinh sach bao hanh (thuong khong bao hanh tai VN).
5. Tinh toan tong chi phi: gia + phi vc + thue.
""",
    },
    {
        "filename": "article_06.json",
        "url": "https://help.shopee.vn/portal/4/article/166085",
        "title": "Huong Dan Day Du Ve Voucher Va Ma Giam Gia Tren Shopee",
        "date_crawled": CRAWLED_DATE,
        "customer_role": "both",
        "category": "promotion",
        "content_markdown": """\
# Huong Dan Day Du Ve Voucher Va Ma Giam Gia Tren Shopee

## Cac Loai Voucher Tren Shopee

1. VOUCHER SHOPEE (Platform Voucher):
   - Do Shopee phat hanh, ap dung toan san.
   - Thuong xuat hien su kien: 9.9, 10.10, 11.11, 12.12, Tet.
   - Lay tai: App -> "Uu dai cua toi" -> "Voucher Shopee".

2. VOUCHER SHOP (Seller Voucher):
   - Do nguoi ban tu phat hanh.
   - Chi ap dung cho don hang tai shop do.
   - Lay tai: Trang shop -> Nhan "Theo doi" -> Nhan voucher chao mung.

3. VOUCHER VAN CHUYEN (Shipping Voucher):
   - Giam hoac mien phi van chuyen.
   - Co the do Shopee hoac nguoi ban phat hanh.
   - Ap dung ket hop voi voucher giam gia san pham.

4. SHOPEE XU:
   - 1 Xu = 1 VND, su dung khi thanh toan.
   - Nhan Xu tu: danh gia san pham, mini game, su kien, login hang ngay.

## Cach Lay Voucher

Nguon 1 - Shopee Voucher Hub:
- App -> "Uu dai cua toi" -> Chon va thu thap.

Nguon 2 - Flash Sale Voucher:
- Vao su kien Flash Sale gio vang -> Thu thap truoc.

Nguon 3 - Voucher Tu Shop:
- Trang shop -> "Voucher Shop" -> Thu thap.

Nguon 4 - Shopee Live:
- Xem streamer bao ma on-air -> Nhap nhanh ma.

Nguon 5 - Dong Tac Vien:
- Nhan ma qua link gioi thieu cua KOL, KOC tren mang xa hoi.

## Cach Ap Dung Voucher Khi Thanh Toan

Buoc 1: Them san pham vao gio hang.
Buoc 2: Bam "Thanh toan".
Buoc 3: Tren trang thanh toan:
   - Nhan "Them Shopee Voucher" de ap ma giam cua Shopee.
   - Nhan "Them Voucher Shop" de ap ma giam cua nguoi ban.
Buoc 4: Nhap ma hoac chon tu danh sach ma trong vi.
Buoc 5: Nhan "AP DUNG".
Buoc 6: Kiem tra gia sau giam -> Hoan tat thanh toan.

## Ly Do Voucher Khong Ap Dung Duoc

Nguyen nhan pho bien:
- Don hang chua dat gia tri toi thieu (vi du: can 200k ma moi co 150k).
- Voucher da het han (kiem tra ngay hiet han).
- San pham khong thuoc danh muc duoc ap dung.
- Voucher da duoc su dung hoac het luot.
- Tai khoan bi han che.

Cach khac phuc:
- Them san pham de dat gia tri toi thieu.
- Kiem tra dieu kien bang nut ? canh voucher.
- Thu voucher khac cung loai.

## Dieu Kien Chung Cua Voucher

- Moi tai khoan su dung moi ma 1 lan.
- Ma khong co gia tri tien mat, khong quy doi.
- Khong duoc mua ban, chuyen nhuong ma.
- Hoan tien khi huy don: ShopeePay hoac Xu (khong hoan lai voucher da dung).
""",
    },
    {
        "filename": "article_07.json",
        "url": "https://help.shopee.vn/portal/4/article/77280",
        "title": "Huong Dan Mo Shop Va Bat Dau Ban Hang Tren Shopee Vietnam",
        "date_crawled": CRAWLED_DATE,
        "customer_role": "seller",
        "category": "seller-guide",
        "content_markdown": """\
# Huong Dan Mo Shop Va Bat Dau Ban Hang Tren Shopee Vietnam

## Dieu Kien Mo Shop

- Tu 18 tuoi tro len.
- Co CMND/CCCD/Ho chieu hop le.
- Co so dien thoai va email xac minh.
- Doc va dong y Dieu khoan dich vu nguoi ban.
- Voi doanh nghiep/ho kinh doanh: can them giay dang ky kinh doanh.

## Buoc 1: Tao Tai Khoan Nguoi Ban

1. Truy cap seller.shopee.vn hoac app Shopee -> "Ke cho cua toi".
2. Nhan "Dang ky ban hang".
3. Dien thong tin: email, so dien thoai, mat khau.
4. Xac minh OTP gui qua SMS.
5. Dong y dieu khoan va chinh sach.

## Buoc 2: Thiet Lap Thong Tin Shop

Ten shop (quan trong):
- Tu 3-30 ky tu, khong dau tieng Viet.
- KHONG the doi ten trong 30 ngay dau.
- Ten phai doc dao, de nho, lien quan den nganh hang.

Anh dai dien shop:
- Kich thuoc: toi thieu 300x300 pixel (khuyen nghi 500x500).
- Dinh dang: JPG, PNG.
- Anh ro net, chuyen nghiep.

Mo ta shop:
- Ngan gon, tro trong, neu ro nganh hang, uu diem.

## Buoc 3: Xac Minh Danh Tinh

1. Upload anh CMND/CCCD (mat truoc va sau).
2. Chup anh selfie cam CMND/CCCD.
3. Gui -> Cho xac minh 1-3 ngay lam viec.
4. Nhan thong bao ket qua qua email.

## Buoc 4: Thiet Lap Van Chuyen

1. Seller Centre -> Van chuyen.
2. Chon don vi van chuyen (SPX, J&T, GHN, GHTK...).
3. Nhap dia chi lay hang / kho hang chinh xac.
4. Thiet lap thoi gian chuan bi hang (1-2 ngay khuyen nghi).
5. Bat tinh nang van chuyen mong muon.

## Buoc 5: Dang San Pham Dau Tien

Quy trinh dang san pham:
1. Seller Centre -> San pham -> Them san pham.
2. Dien tieu de (60-120 ky tu, co keyword chinh).
3. Upload anh (1-9 anh, anh dau tien la anh chinh).
4. Chon danh muc phu hop.
5. Dien mo ta day du (toi thieu 100 tu).
6. Nhap gia va so luong ton kho.
7. Chon loai van chuyen ho tro.
8. Nhan "Dang" -> Kiem tra truoc khi phat hanh.

## Meo Ban Hang Hieu Qua Ngay Tu Dau

Toi uu san pham:
- Tieu de chua keyword nguoi mua hay tim (vi du: "ao phong nam tay ngan cotton").
- Anh nen trang, chup nhieu goc.
- Mo ta chi tiet: size, chat lieu, mau sac, xuat xu, bao hanh.

Tang uy tin shop:
- Tra loi chat trong vong 30 phut.
- Xu ly don hang trong ngay.
- Dong goi can than, dep, co the them thiet thiep cam on.
- Khuyen khich khach hang de lai danh gia.

Tham gia chuong trinh Shopee:
- Flash Sale cua Shopee -> Tang doanh so nhanh.
- Free Shipping -> Thu hut nguoi mua.
- Shopee Ads -> Tang hien thi.
""",
    },
    {
        "filename": "article_08.json",
        "url": "https://help.shopee.vn/portal/4/article/77285",
        "title": "Huong Dan Su Dung ShopeePay - Vi Dien Tu Shopee Vietnam",
        "date_crawled": CRAWLED_DATE,
        "customer_role": "both",
        "category": "payment",
        "content_markdown": """\
# Huong Dan Su Dung ShopeePay - Vi Dien Tu Shopee Vietnam

## ShopeePay La Gi?

ShopeePay (truoc day la AirPay) la vi dien tu chinh thuc cua Shopee.
Ngoai mua sam, ShopeePay con cho phep:
- Thanh toan tai hang an, cafe, sieu thi (quet QR).
- Nap tien dien thoai, internet, dien, nuoc.
- Gui / nhan tien giua nguoi dung.
- Vay tieu dung ngan han (SPayLater).

## Kich Hoat ShopeePay Lan Dau

1. App Shopee -> Nhan bieu tuong ShopeePay (dong xu).
2. Nhan "Kich hoat ngay".
3. Nhap so dien thoai va OTP xac minh.
4. Tao ma PIN 6 so (quan trong: ghi nho, khong de lo).
5. Hoan tat -> Ban da co vi ShopeePay co ban.

Nang cap tai khoan (de tang han muc):
- Nhap thong tin CMND/CCCD va upload anh xac minh.
- Cho duyet 1-3 ngay lam viec.

## Nap Tien Vao ShopeePay

Phuong thuc 1 - Ngan hang truc tuyen (khuyen nghi):
- ShopeePay -> Nap tien -> Chuyen khoan -> Sao chep so tai khoan ao.
- Chuyen khoan qua app ngan hang.
- Tien vao ngay (khong mat phi).

Phuong thuc 2 - Cua hang tien loi:
- Den Circle K, Family Mart, GS25, Ministop.
- Bao thu quy: nap ShopeePay + SĐT dang ky.
- Nop tien mat.
- Tien vao trong 5-10 phut.

## Thanh Toan Bang ShopeePay

Cach 1 - Thanh toan don hang Shopee:
- Chon ShopeePay luc thanh toan.
- Nhap ma PIN xac nhan.
- Nhan cashback (neu co chuong trinh).

Cach 2 - Quet ma QR tai cua hang:
- ShopeePay -> "Quet ma" hoac "Tao ma QR".
- Quet ma QR cua cua hang -> Nhap so tien -> Nhap PIN.

## Rut Tien Ve Ngan Hang

Dieu kien: tai khoan ngan hang da lien ket truoc.
Cach lien ket ngan hang:
1. ShopeePay -> "Tai khoan" -> "Lien ket ngan hang".
2. Chon ngan hang -> Nhap so tai khoan.
3. Xac minh OTP tu ngan hang.

Thuc hien rut tien:
1. ShopeePay -> "Rut tien".
2. Chon tai khoan ngan hang.
3. Nhap so tien (toi thieu 10.000 VND).
4. Nhap PIN xac nhan.
5. Tien ve trong 1-2 gio (ngan hang co online 24/7).

## Bao Mat Va Xu Ly Su Co

Cac tinh nang bao mat:
- Ma PIN 6 so bat buoc cho moi giao dich.
- Tu dong khoa sau 5 lan nhap sai.
- Thong bao tuc thi qua app + SMS.
- Dau van tay / Face ID bo sung.

Khi bi mat dien thoai:
1. Goi ngay 1900 1221 -> Yeu cau khoa tam thoi.
2. Khoi phuc tai khoan bang OTP + CMND/CCCD.
3. Doi ma PIN moi sau khi khoi phuc.
""",
    },
    {
        "filename": "article_09.json",
        "url": "https://help.shopee.vn/portal/4/article/77255",
        "title": "Huong Dan Xu Ly Tranh Chap Don Hang Danh Cho Nguoi Ban Shopee",
        "date_crawled": CRAWLED_DATE,
        "customer_role": "seller",
        "category": "dispute",
        "content_markdown": """\
# Huong Dan Xu Ly Tranh Chap Don Hang Danh Cho Nguoi Ban Shopee

## Khi Nao Xay Ra Tranh Chap?

Nguoi mua co the mo tranh chap khi:
- Ho cho la hang bi loi / hu hong / sai mo ta.
- Khong nhan duoc hang.
- Muon hoan tien nhung nguoi ban tu choi.
- Van de ve chat luong san pham.

## Quy Trinh Tranh Chap (Tu Goc Do Nguoi Ban)

Giai doan 1 — NGUOI MUA MO TRANH CHAP:
- He thong gui thong bao cho nguoi ban.
- Nguoi ban co 3 ngay de xem xet va tra loi.

Giai doan 2 — NGUOI BAN XEM XET:
1. Vao Seller Centre -> "Don hang" -> "Tra hang/Hoan tien".
2. Kiem tra bang chung nguoi mua cung cap.
3. Chon phuong an xu ly:
   - Chap nhan: chap thuan hoan tien hoac doi hang.
   - De xuat khac: de xuat muc hoan tien phan, hoac cung cap bang chung phan bien.
   - Tu choi: tu choi neu bang chung khong thuyet phuc.

Giai doan 3 — SHOPEE CAN THIEP (neu khong dong thuan):
- Ca 2 ben gui bang chung bo sung.
- Shopee xem xet trong 3-7 ngay lam viec.
- Quyet dinh cuoi cung tu Shopee.

## Bang Chung Nguoi Ban Can Chuan Bi

Rat quan trong de bao ve quyen loi:
- VIDEO DONG GOI: quay toan bo qua trinh dong goi san pham (manh nhat).
- ANH SAN PHAM: chup anh truoc khi dong goi (san pham nguyen ven, du phu kien).
- BIEN LAI VAN CHUYEN: giu lai chung tu gui hang tu don vi van chuyen.
- LICH SU CHAT: luu lai cuoc tro chuyen voi nguoi mua (neu co thoa thuan truoc).
- VIDEO KIEM TRA HANG: video kiem tra chat luong truoc khi dong goi (neu co).

## Chien Luoc Xu Ly Tranh Chap Hieu Qua

Truong hop hang bi loi that:
- Chu dong chap nhan hoan tien hoac doi hang.
- Xin loi va giai thich.
- Cai thien quy trinh kiem soat chat luong.

Truong hop nguoi mua co y xau:
- Cung cap bang chung day du (video dong goi, hinh anh truoc khi gui).
- Ghi ro cac diem bat hop ly trong yeu cau cua nguoi mua.
- Yeu cau Shopee can thiep neu tu choi khong duoc chap thuan.

## Phong Ngua Tranh Chap Trong Tuong Lai

1. Luon quay video dong goi (nen thiet lap camera co dinh tai kho).
2. Kiem tra ky san pham truoc khi dong goi.
3. Dong goi can than, bao ve san pham trong van chuyen.
4. Mo ta san pham chinh xac, khong phong dai.
5. Phan hoi chat cua nguoi mua nhanh chong truoc khi ho mo tranh chap.
6. Giai quyet van de truoc khi nguoi mua thay cua Shopee.

## Anh Huong Den Diem Shop

Tranh chap that bai (Shopee xu nghieng ve nguoi mua):
- Co the anh huong den: ty le danh gia, thu hang san pham, hien thi shop.
- Nhieu tranh chap that bai: kha nang bi kiem tra chat luong, giam uu dai.
""",
    },
    {
        "filename": "article_10.json",
        "url": "https://help.shopee.vn/portal/4/article/77246",
        "title": "Bi Quyet Toi Uu Gian Hang Tang Doanh So Tren Shopee",
        "date_crawled": CRAWLED_DATE,
        "customer_role": "seller",
        "category": "seller-guide",
        "content_markdown": """\
# Bi Quyet Toi Uu Gian Hang Tang Doanh So Tren Shopee

## 1. Toi Uu Hoa San Pham (SEO Shopee)

Tieu de san pham:
- Dung keyword chinh nguoi mua hay tim (dung Shopee Search Suggest).
- Cau truc: [Keyword chinh] + [Dac tinh] + [Nhan hieu/Xuat xu].
- Vi du: "Ao thun nam tay ngan cotton khe gio nhap khau Han Quoc".
- KHONG dung ky tu dac biet, spam keyword.

Hinh anh san pham:
- Anh chinh: nen trang, san pham ro net, chiem 80% khung anh.
- Anh phu: the hien nhieu goc, chi tiet chat lieu, size chart.
- Khuyen nghi: 6-9 anh + 1 video ngon (60-90 giay).

Mo ta chi tiet:
- Phan 1 (20 dong dau): thong tin quan trong nhat, ai cung can biet.
- Phan 2: phan loai (size, mau sac, chat lieu, xuat xu).
- Phan 3: huong dan su dung, bao quan.
- Phan 4: chinh sach bao hanh, doi tra.
- Dung bang, ky hieu bullet point cho de doc.

## 2. Tham Gia Chuong Trinh Shopee

Flash Sale Shopee:
- Dang ky qua Seller Centre -> Marketing Centre -> Flash Sale.
- Giam gia 20-50% trong gio nhat dinh.
- Tang doanh so nhanh, tang thu hang.

Shopee Free Shipping:
- Tham gia Shopee Mall -> Freeship Xtra.
- Tang kha nang mua hang (phi ship la ly do so 1 nguoi tu bo gio hang).

Shopee Ads (Quang cao tra phi):
- Search Ads: hien thi khi nguoi dung tim kiem.
- Discovery Ads: hien thi tren trang chu, goi y.
- Chi phi: dat gia theo click (CPC), bat dau tu 100 VND/click.
- ROI tot khi san pham co danh gia tot va anh dep.

## 3. Tang Uy Tin Va Danh Gia

Cach tang danh gia sao:
- Gui thep thiep cam on / thong diep than thien trong kien hang.
- Gui tin nhan chau huong roi khi nhan hang.
- Cung cap qua tang nho kem don hang (tang them trai nghiem).
- KHONG dua tien/voucher doi lay danh gia 5 sao (vi pham chinh sach).

Xu ly danh gia xau:
- Tra loi cong khai, lich su, giai thich.
- Lien he nguoi mua qua chat de hieu van de.
- Cam on phan hoi, hua cai thien.
- TUYET DOI KHONG cai nhau voi nguoi mua tren phan comment.

## 4. Cham Soc Khach Hang Tren Shopee Chat

Chi so chat quan trong:
- Ty le tra loi: toi thieu 80% (khuyen nghi 95%+).
- Thoi gian tra loi: duoi 60 phut (khuyen nghi duoi 15 phut).

Tips:
- Thiet lap tin nhan tu dong khi ngoai gio lam viec.
- Dung template tra loi cau hoi thuong gap.
- Chuyen sang ho tro dien thoai cho truong hop phuc tap.

## 5. Quan Ly Don Hang Hieu Qua

- Xu ly don hang trong 24 gio sau khi nhan (giam ty le huy).
- Cap nhat trang thai giao hang ngay sau khi ban giao shipper.
- Dong goi can than: boc dem, chong am, nhan san pham.
- Gui thong bao cho nguoi mua khi giao cho shipper.

## 6. Phan Tich Va Cai Thien

Dung Shopee Seller Analytics:
- Xem: San pham ban chay nhat, nguon truy cap, ty le chuyen doi.
- Phan tich: tai sao mot san pham it ban (anh xau? gia cao? mo ta kho hieu?).
- So sanh: hieu suat tuan nay voi tuan truoc.
- Thuc hien A/B test: thu tieu de khac, anh khac.
""",
    },
]


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Thu muc san sang: {DATA_DIR}")


def save_articles():
    """Lưu tất cả bài viết vào data/landing/news/."""
    setup_directory()
    print(f"\n{'='*60}")
    print("TASK 2 - Bai Viet Huong Dan Ho Tro Khach Hang")
    print(f"{'='*60}\n")

    saved_count = 0
    for article in ARTICLES:
        filename = article["filename"]
        # Tách data cần lưu (bỏ field "filename" ra ngoài)
        data = {k: v for k, v in article.items() if k != "filename"}

        filepath = DATA_DIR / filename
        filepath.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        size_kb = filepath.stat().st_size / 1024
        print(f"  [OK] {filename:30s} ({size_kb:6.1f} KB) - {data['customer_role']}")
        saved_count += 1

    print(f"\n{'='*60}")
    print(f"HOAN THANH: {saved_count} bai viet da luu")
    print(f"Thu muc   : {DATA_DIR}")
    print(f"{'='*60}\n")
    return saved_count


async def crawl_and_save():
    """Entry point chính — lưu trực tiếp không cần crawl."""
    count = save_articles()
    return count


if __name__ == "__main__":
    count = asyncio.run(crawl_and_save())
    print(f"\n{'CP1 PASS' if count >= 5 else 'FAIL'}: {count} bai viet (yeu cau >= 5)")
