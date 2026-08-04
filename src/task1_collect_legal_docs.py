"""
Task 1 — Thu thập văn bản chính sách thương mại điện tử / hỗ trợ khách hàng.

Nguồn dữ liệu: tận dụng data_2/k4_ecommerce/ (đã crawl thực từ help.shopee.vn)
và bổ sung 4 tài liệu mới chưa có trong data_2.

14 tài liệu PDF sẽ được tạo vào data/landing/legal/:
  1.  return-refund-policy-shopee.pdf           (buyer)
  2.  return-shipping-fee-shopee.pdf             (buyer)
  3.  payment-methods-shopee.pdf                 (buyer)
  4.  seller-listing-rules-shopee.pdf            (seller)
  5.  marketplace-operating-regulation-shopee.pdf (both)
  6.  restricted-products-policy-shopee.pdf       (seller)
  7.  delivery-process-shopee.pdf                 (buyer)
  8.  privacy-policy-shopee.pdf                   (both)
  9.  voucher-discount-policy-shopee.pdf           (both)
  10. shipping-fee-discount-program-shopee.pdf     (seller)
  11. dispute-resolution-policy-shopee.pdf         (both)   [MỚI]
  12. shopeepay-wallet-policy-shopee.pdf            (buyer)  [MỚI]
  13. seller-fee-commission-shopee.pdf              (seller) [MỚI]
  14. cross-border-shopping-shopee.pdf              (buyer)  [MỚI]

Nhớ gắn metadata `customer_role` (`buyer`/`seller`/`both`) — yêu cầu K4 Variant.
"""

import json
import time
from pathlib import Path

from fpdf import FPDF

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


# ─────────────────────────────────────────────────────────────────────────────
# 14 TÀI LIỆU CHÍNH SÁCH — nội dung chi tiết, tận dụng data_2/k4_ecommerce/
# ─────────────────────────────────────────────────────────────────────────────
LEGAL_DOCUMENTS = [

    # ── 1. Chính sách trả hàng và hoàn tiền ──────────────────────────────────
    {
        "filename": "return-refund-policy-shopee.pdf",
        "doc_id": "return-refund-policy",
        "customer_role": "buyer",
        "category": "returns",
        "source_url": "https://help.shopee.vn/portal/4/article/77251",
        "document_version": "2026-03-11",
        "title": "Chinh Sach Tra Hang va Hoan Tien - Shopee Vietnam",
        "content": """\
CHINH SACH TRA HANG VA HOAN TIEN - SHOPEE VIETNAM
Dang ngay 04/03/2026, co hieu luc tu 11/03/2026.
Doc ID: return-refund-policy | Customer Role: buyer | Category: returns

1. THOI HAN YEU CAU TRA HANG / HOAN TIEN
Nguoi mua co the gui yeu cau tra hang/hoan tien trong vong 15 ngay ke tu luc don
hang duoc cap nhat trang thai "Giao hang thanh cong". Cu the:
- Don hang do don vi van chuyen giao: 15 ngay ke tu trang thai "Giao hang thanh cong".
- Don hang do nguoi ban tu van chuyen: 15 ngay ke tu khi nguoi mua bam "Da nhan
  duoc hang", hoac 20 ngay ke tu trang thai "Lay hang thanh cong" neu nguoi mua
  chua bam xac nhan.
- Thuc pham tuoi song / dong lanh: yeu cau phai gui trong vong 24 GIO ke tu
  trang thai "Giao hang thanh cong".

2. LY DO HOP LE DE YEU CAU TRA HANG / HOAN TIEN
- Khong nhan duoc hang (mat hang, don hang chua den).
- Thieu hang (giao thieu san pham so voi don dat hang).
- Giao sai san pham (sai mau, sai size, sai model so voi mo ta).
- Hang hu hong: vo, nut, ro ri, bao bi bi hu, hang bi loi hoac khong hoat dong.
- Mo ta khong dung thuc te: hinh anh/mo ta san pham khac voi hang thuc te nhan duoc.
- Hang da qua su dung: ban giao san pham cu hoac da duoc su dung.
- Hang gia / hang nhai: san pham khong chinh hang, gia mao thuong hieu.

3. CHANGE OF MIND (COM) - DOI Y KHONG MUON MUA NUA
- Thanh vien Kim Cuong / Vang: duoc tra hang COM khong gioi han so lan.
- Thanh vien dang ky goi ShopeeVIP: tra hang COM toi da 15 lan trong moi thang
  duong lich.
- Cac thanh vien khac: khong ap dung COM (chi tra hang khi co ly do hop le).

4. QUY TRINH GUI YEU CAU TRA HANG / HOAN TIEN
Buoc 1: Dang nhap Shopee -> "Don hang cua toi" -> Chon don hang can tra.
Buoc 2: Nhan "Yeu cau tra hang/hoan tien" -> Chon ly do phu hop.
Buoc 3: Dinh kem bang chung: anh/video san pham bi loi, anh bao bi, anh nhan
        hang... (toi thieu 1 anh, khuyen nghi quay video mo hop).
Buoc 4: Mo ta chi tiet van de gap phai trong o nhan xet.
Buoc 5: Xac nhan gui yeu cau -> He thong thong bao den nguoi ban.

5. XU LY TRANH CHAP
Neu nguoi ban khong dong y / khong phan hoi trong 3 ngay, nguoi mua co the:
- Yeu cau Shopee can thiep.
- Shopee se xem xet bang chung tu ca 2 ben.
- Quyet dinh cua Shopee la quyet dinh cuoi cung.
- Thoi gian xu ly: 3-7 ngay lam viec.

6. PHI VAN CHUYEN HOAN TRA
- Nguoi ban chiu phi: loi thuoc ve nguoi ban (hang loi, sai hang, hang gia, v.v.)
  hoac giao hang that bai do loi nguoi ban.
- Khong ap dung phi hoan tra cho truong hop: tra mot phan don hang hoac nguoi
  mua tu sap xep van chuyen (se duoc hoan phi sau).

7. PHUONG THUC HOAN TIEN
Shopee chi hoan tien khi:
(a) Nguoi ban xac nhan da nhan lai hang.
(b) Nguoi mua chap nhan de nghi cua nguoi ban.
(c) Shopee quyet dinh phe duyet theo co che xu ly rieng.
Tien hoan duoc chuyen ve phuong thuc thanh toan goc cua nguoi mua:
- Vi ShopeePay: 1-3 ngay lam viec.
- The tin dung / the ghi no: 7-14 ngay lam viec (tuy ngan hang).
- Internet Banking / Chuyen khoan: 3-7 ngay lam viec.

8. LUU Y QUAN TRONG
- Hien tai Shopee KHONG ho tro yeu cau doi hang (chi ho tro tra hang/hoan tien).
- San pham ky thuat so (the game, ma nap tien), ve su kien khong ap dung tra hang.
- San pham da qua su dung ro rang (tru truong hop loi) khong duoc tra hang.
""",
    },

    # ── 2. Phí gửi hàng hoàn trả ─────────────────────────────────────────────
    {
        "filename": "return-shipping-fee-shopee.pdf",
        "doc_id": "return-shipping-fee",
        "customer_role": "buyer",
        "category": "returns",
        "source_url": "https://help.shopee.vn/portal/4/article/189477",
        "document_version": "not-stated",
        "title": "Cac Phuong Thuc Gui Hang Hoan Tra va Phi Hoan Tra - Shopee",
        "content": """\
CAC PHUONG THUC GUI HANG HOAN TRA VA PHI HOAN TRA - SHOPEE VIETNAM
Doc ID: return-shipping-fee | Customer Role: buyer | Category: returns

1. BA PHUONG THUC TRA HANG (DEU MIEN PHI VAN CHUYEN HOAN TRA)

Phuong thuc 1: Don vi van chuyen den lay hang tan noi
- He thong tu dong sap xep don vi van chuyen den lay hang tai dia chi nguoi mua.
- Nguoi mua chi can dong goi san pham can than, giao cho shipper.
- Khong phat sinh chi phi voi nguoi mua.

Phuong thuc 2: Nguoi mua tu mang hang den buu cuc chi dinh
- Chon buu cuc tren app Shopee -> Xem dia chi buu cuc gan nhat.
- Mang san pham da dong goi den buu cuc, dua ma van don cua Shopee.
- Nhan bien lai tu buu cuc de lam bang chung gui hang.
- Khong phat sinh chi phi voi nguoi mua.

Phuong thuc 3: Nguoi mua tu sap xep van chuyen (tra phi truoc, duoc hoan lai)
- Nguoi mua tu chon don vi van chuyen, tra phi truoc.
- Shopee hoan phi van chuyen hoan tra bang Shopee Xu trong vong 3-5 ngay lam viec:
  * Cung tinh/thanh pho: hoan 25.000 Shopee Xu (~25.000 VND).
  * Khac tinh/thanh pho: hoan 40.000 Shopee Xu (~40.000 VND).

2. PHI VAN CHUYEN DON HANG GOC

Hoan lai phi van chuyen don hang goc khi:
- Nguoi mua tra TOAN BO don hang -> phi van chuyen duoc hoan lai.
- Chi tra MOT PHAN don hang -> phi van chuyen KHONG duoc hoan lai.

3. DIEU KIEN DUOC HO TRO PHI HOAN TRA
Tat ca cac dieu kien sau phai duoc dap ung:
(a) Yeu cau tra hang da duoc chap nhan boi nguoi ban hoac Shopee.
(b) Co day du thong tin van don (tracking number).
(c) Xac nhan giao hang thanh cong den nguoi ban.
(d) Duoc nguoi ban hoac Shopee phe duyet hoan tien.

4. DONG GOI HANG HOA TRUOC KHI TRA
Yeu cau dong goi chuan:
- Bao quan san pham nguyen ven, tranh hu hong trong qua trinh van chuyen.
- Ghi ro thong tin: ma don hang, ten nguoi ban, dia chi nguoi ban.
- Dinh kem bo phan phu kien, qua tang (neu co) nhan duoc kem san pham.
- KHONG gui hang khi chua co ma van don tu he thong Shopee.

5. THEO DOI TRANG THAI TRA HANG
Nguoi mua co the theo doi:
- App Shopee -> "Don hang cua toi" -> Chon don hang -> "Xem chi tiet tra hang".
- Tra cuu truc tiep tren website don vi van chuyen bang ma van don.
""",
    },

    # ── 3. Phương thức thanh toán ─────────────────────────────────────────────
    {
        "filename": "payment-methods-shopee.pdf",
        "doc_id": "payment-methods",
        "customer_role": "buyer",
        "category": "payment",
        "source_url": "https://help.shopee.vn/portal/4/article/79198",
        "document_version": "not-stated",
        "title": "Phuong Thuc Thanh Toan Tren Shopee Vietnam",
        "content": """\
PHUONG THUC THANH TOAN TREN SHOPEE VIETNAM
Doc ID: payment-methods | Customer Role: buyer | Category: payment

Shopee ho tro 10 phuong thuc thanh toan chinh thuc. Khong ho tro cac phuong thuc
ngoai danh sach nay (vi du: chuyen khoan truc tiep ngoai he thong, nop tien mat
tai cua hang Shopee).

1. VI SHOPEEPAY (Vi dien tu tich hop)
- Nap tien qua: chuyen khoan ngan hang, ATM, cua hang tien loi (Circle K, GS25...).
- Giao dich tuc thi, khong can nhap lai thong tin the.
- Duoc hoan tien (cashback) va uu dai doc quyen khi thanh toan qua ShopeePay.
- Han muc: toi da 100 trieu VND/thang (tai khoan da xac minh day du).
- Muc toi thieu: 10.000 VND/giao dich.

2. THE TIN DUNG / GHI NO (Visa, Mastercard, JCB, Amex)
- Chap nhan: Visa, Mastercard, JCB, American Express.
- The quoc te phat hanh boi cac ngan hang Viet Nam.
- Giao dich toi thieu: 10.000 VND.
- Mot so ngan hang yeu cau xac thuc OTP cho giao dich truc tuyen.
- Ho tro thanh toan qua 3D Secure (Verified by Visa / Mastercard SecureCode).

3. TRA GOP QUA THE TIN DUNG
- Ap dung cho don hang tu 3 trieu VND tro len.
- Ky han: 3, 6, 9, 12 thang.
- Lai suat: 0% (cho cac chuong trinh khuyen mai co dinh) hoac theo lai suat ngan hang.
- Hop tac voi: HSBC, VPBank, Techcombank, TPBank, Sacombank, MB Bank, VIB...

4. THANH TOAN QUA MA QR (CHUYEN KHOAN NGAN HANG TRUC TUYEN)
- Quet ma QR bang app ngan hang de chuyen khoan truc tiep.
- Ho tro hon 40 ngan hang tai Viet Nam.
- Giao dich toi thieu: 10.000 VND.
- Tuc thi sau khi xac nhan.

5. THANH TOAN QUA UNG DUNG NGAN HANG
- Chuyen huong truc tiep den app ngan hang de xac nhan thanh toan.
- Khong can nhap so the hoac OTP rieng.
- Ho tro: Vietcombank, BIDV, Agribank, Techcombank, MB Bank, VPBank...

6. THE NOI DIA NAPAS (CO DANG KY INTERNET BANKING)
- The noi dia co logo NAPAS va da dang ky Internet Banking.
- Giao dich toi thieu: 10.000 VND.
- Xac nhan qua OTP gui SMS.

7. APPLE PAY (Danh cho thiet bi Apple)
- Yeu cau: iPhone, iPad, Apple Watch ho tro Apple Pay.
- Han muc: 10.000 VND den 25.000.000 VND/giao dich.
- Xac nhan bang Face ID / Touch ID / mat khau.

8. GOOGLE PAY (Danh cho thiet bi Android)
- Yeu cau: thiet bi Android ho tro NFC va Google Pay.
- Han muc: 10.000 VND den 120.000.000 VND/giao dich.
- Xac nhan bang van tay / PIN.

9. THANH TOAN KHI NHAN HANG (COD - Cash on Delivery)
- Tra tien mat khi nhan hang tu shipper.
- Ap dung cho don hang du dieu kien chap nhan COD cua nguoi ban.
- Khong phat sinh phi them (hoac phi COD rat nho, tuy shipper).
- Khong ap dung cho mot so loai hang dac biet (dien tu cao cap, hang dat truoc).
- Gia tri don hang toi da ap dung COD: 20.000.000 VND (tuy don vi van chuyen).

10. SPAYPAYLATER (MUA TRUOC TRA SAU)
- The tin dung ao, chia lam 1, 2, 3 hoac 6 ky.
- Phe duyet nhanh, khong can the vat ly.
- Lai suat: tuy ky han va chinh sach hien hanh.
- Dieu kien: tai khoan Shopee da xac minh day du, lich su giao dich tot.

GHI CHU CHUNG:
- Tien hoan tra (refund) se ve dung phuong thuc thanh toan goc.
- Shopee ap dung ma hoa SSL 256-bit va tuan thu tieu chuan PCI DSS.
- Thong tin the khong duoc luu tru tren server Shopee.
""",
    },

    # ── 4. Quy định đăng bán sản phẩm ────────────────────────────────────────
    {
        "filename": "seller-listing-rules-shopee.pdf",
        "doc_id": "seller-listing-rules",
        "customer_role": "seller",
        "category": "seller-policy",
        "source_url": "https://help.shopee.vn/portal/4/article/77246",
        "document_version": "2024-08-21",
        "title": "Quy Dinh Ve Dang Ban San Pham Tren Shopee Vietnam",
        "content": """\
QUY DINH VE DANG BAN SAN PHAM TREN SHOPEE VIETNAM
Dang ngay 14/08/2024, co hieu luc sau 7 ngay ke tu ngay dang.
Doc ID: seller-listing-rules | Customer Role: seller | Category: seller-policy

Ap dung cho tat ca nguoi ban tren san Shopee Viet Nam.

1. NOI DUNG BI CAM TRONG LISTING SAN PHAM
Nguoi ban KHONG DUOC dang san pham co cac noi dung sau:
- Noi dung phan dong, chong pha Nha nuoc, bai xich ton giao.
- Noi dung khieu dam, bao luc, dep di nhan pham.
- Vi pham quyen so huu tri tue (hang nhai, ban quyen, nhan hieu).
- San pham tu dong vat hoang da / quy hiem bi bao ve.
- Thong tin gay hieu lam cho nguoi mua ve cong dung, chat luong, xuat xu.
- Quang cao so sanh truc tiep ve gia ca / chat luong voi san pham nguoi ban khac.

2. HANH VI BI CAM KHAC
- Dang bai spam, noi dung trung lap gay nhieu lan.
- Thao tung gia (tang gia gia tao de tao khuyen mai ao).
- Danh gia ao, review gia, su dung tai khoan ao.
- Gan sai danh muc san pham.
- Ghi thong tin lien lac ca nhan (SĐT, email, Zalo...) trong mo ta hoac hinh anh.

3. YEU CAU MO TA SAN PHAM
- Tieu de: toi da 120 ky tu, phan anh dung san pham thuc te.
- Mo ta chi tiet: toi thieu 100 tu, ro rang, day du.
- Phai ghi ro: nguon goc xuat xu, thong tin bao hanh (neu co), phan loai dung
  danh muc.
- San pham co han su dung: chi duoc ban khi con it nhat 30% thoi han va toi
  thieu con 30 ngay truoc khi het han.

4. YEU CAU HINH ANH SAN PHAM
- Toi thieu 1 anh chinh, toi da 9 anh / san pham.
- Kich thuoc toi thieu: 500x500 pixel (khuyen nghi 1:1).
- Dinh dang: JPG, PNG (khong anh dong).
- Anh ro net, khong mo, khong co watermark la.
- San pham phai chiem it nhat 40% dien tich anh chinh.
- KHONG su dung anh co chua thong tin lien lac (so dien thoai, email, QR code...).

5. GIA VA KHO HANG
- Gia niem yet phai phan anh dung gia thuc ban.
- Gia toi thieu: 1.000 VND.
- Cap nhat ton kho chinh xac; tranh tinh trang "het hang gia".
- Nguoi ban co trach nhiem xu ly don hang trong thoi gian cam ket.

6. XU PHAT KHI VI PHAM
- Vi pham nhe (lan dau): canh bao, yeu cau chinh sua trong 24 gio.
- Vi pham lan 2: ha diem shop, giam hien thi san pham.
- Vi pham nghiem trong: khoa shop tam thoi (7-30 ngay).
- Vi pham cuc ky nghiem trong: xoa tai khoan vinh vien, co the bao co quan chuc nang.
""",
    },

    # ── 5. Quy chế hoạt động sàn ─────────────────────────────────────────────
    {
        "filename": "marketplace-operating-regulation-shopee.pdf",
        "doc_id": "marketplace-operating-regulation",
        "customer_role": "both",
        "category": "seller-policy",
        "source_url": "https://help.shopee.vn/portal/4/article/77245",
        "document_version": "2025-01-10",
        "title": "Quy Che Hoat Dong San Thuong Mai Dien Tu Shopee.vn",
        "content": """\
QUY CHE HOAT DONG SAN THUONG MAI DIEN TU SHOPEE.VN
Dang ngay 03/01/2025, co hieu luc sau 7 ngay ke tu ngay dang.
Doc ID: marketplace-operating-regulation | Customer Role: both | Category: seller-policy

1. TRACH NHIEM CUA NGUOI BAN
Nguoi ban phai cung cap day du va chinh xac thong tin theo quy dinh:
- Ten doanh nghiep / ho kinh doanh (neu co).
- Dia chi kinh doanh.
- Ma so thue hoac so dang ky kinh doanh.
- Thong tin lien he (so dien thoai, email).

Nguoi ban KHONG DUOC:
- Dang hang gia, hang cam, san pham vi pham quyen so huu tri tue.
- Mo ta san pham sai ve cong dung hoac quang cao so sanh gay hieu nham.
- Chia nho san pham thanh don vi nho hon neu khong cong bo ro rang.

2. QUYEN CUA SHOPEE
Shopee co quyen:
- Go san pham vi pham khoi nen tang bat ky luc nao.
- Tam khoa tai khoan nguoi ban khi co bieu hien gian lan.
- Cham dut quyen su dung dich vu ngay lap tuc voi hanh vi gian lan hoac vi pham
  phap luat.

3. QUY TRINH KHIEU NAI CUA NGUOI MUA
- Gui khieu nai qua muc "Don Mua" tren ung dung Shopee.
- Shopee dua ra quyet dinh trong vong 7 ngay lam viec (truong hop thuong thuong).
- Truong hop phuc tap: thoi gian xu ly co the dai hon.
- Nguoi tieu dung de bi ton thuong duoc uu tien xu ly theo phap luat bao ve NTD.

4. NGUOI BAN QUOC TE
- Voi nguoi ban nuoc ngoai, Shopee dong vai tro dai dien phap ly duoc chi dinh.
- Tiep nhan khieu nai tu nguoi tieu dung Viet Nam.
- Dam bao tuan thu phap luat Viet Nam ve TMDT.

5. BAO VE QUYEN LO NGUOI TIEU DUNG
Nguoi mua co quyen:
- Yeu cau thong tin day du ve san pham truoc khi mua.
- Khieu nai va duoc giai quyet theo quy trinh minh bach.
- Boi thuong thiet hai neu nguoi ban vi pham chinh sach.

6. BAN QUYEN VA SO HUU TRI TUE
- Moi noi dung dang tai len Shopee thuoc quyen so huu cua nguoi dang tai.
- Nguoi ban cap phep cho Shopee su dung noi dung cho muc dich van hanh dich vu.
- Shopee khong chiu trach nhiem ve vi pham ban quyen do nguoi ban gay ra.

7. GIAI QUYET TRANH CHAP
- Tranh chap giai quyet theo phap luat Viet Nam.
- Toa an co tham quyen: Toa an Viet Nam.
- Shopee khuyen khich thu tuc hoa giai truoc khi kien tung.
""",
    },

    # ── 6. Chính sách cấm / hạn chế sản phẩm ────────────────────────────────
    {
        "filename": "restricted-products-policy-shopee.pdf",
        "doc_id": "restricted-products-policy",
        "customer_role": "seller",
        "category": "seller-policy",
        "source_url": "https://help.shopee.vn/portal/4/article/77247",
        "document_version": "2025-05-05",
        "title": "Chinh Sach Cam / Han Che San Pham Tren Shopee Vietnam",
        "content": """\
CHINH SACH CAM / HAN CHE SAN PHAM TREN SHOPEE VIETNAM
Dang ngay 28/04/2025, co hieu luc sau 7 ngay ke tu ngay dang.
Doc ID: restricted-products-policy | Customer Role: seller | Category: seller-policy

Ap dung cho tat ca nguoi ban dang san pham tren san Shopee.

HE QUA KHI VI PHAM:
Vi pham co the dan den: xoa san pham, han che tai khoan, tam khoa hoac khoa vinh
vien tai khoan, tich thu so du tai khoan, va co the bi xu ly theo phap luat.

DANH SACH SAN PHAM BI CAM / HAN CHE:

Nhom 1 — HANG GIA / HANG NHAI / VI PHAM BAN QUYEN
- Hang nhai, hang gia, ban sao trai phep vi pham ban quyen.
- San pham gia mao nhan hieu, thuong hieu noi tieng.
- Phan mem, noi dung so sao chep trai phep.

Nhom 2 — THIET BI QUAN SU / CHINH PHU
- Trang thiet bi quan su va dong phuc lien quan.
- Tai lieu, an pham mang tinh chinh tri.
- Vu khi, dam duoc, chat no.
- Sung, vu khi va san pham co hinh dang giong vu khi.

Nhom 3 — CHAT CO KIEM SOAT / MA TUY
- Ma tuy va dung cu su dung ma tuy.
- Cac chat kich thich bi cam theo phap luat.
- Hoa chat nguy hiem va chat no.

Nhom 4 — SAN PHAM NGUOI LON / KHIEU DAM
- San pham khieu dam (tru do choi danh cho nguoi lon co phep).
- Noi dung bao luc, gay kinh di.

Nhom 5 — THUC PHAM / DUOC PHAM VI PHAM
- Thuc pham vi pham an toan ve sinh (het han, khong ro nguon goc).
- Thuoc ke don va vac-xin (tru co giay phep).
- San pham gay hai cho suc khoe, chua duoc cap phep luu hanh.

Nhom 6 — DONG VAT HOANG DA / MOI TRUONG
- San pham tu dong vat hoang da / quy hiem bi bao ve.
- Bo phan co the nguoi va hai cot.

Nhom 7 — TIEN TE / GIAY TO GIA
- Tien gia va giay to gia.
- Dich vu bat hop phap va tien te trai phep.
- Thiet bi giam sat / xam nhap trai phep.

Nhom 8 — SAN PHAM SO KHONG RO NGUON GOC
- The game, tai khoan game khong ro nguon goc.
- Key phan mem khong ro nguon goc.

Nhom 9 — SAN PHAM KHAC
- Do co va tac pham nghe thuat chua duoc cap phep.
- Vat pham ton giao / me tin lien quan den hoat dong cam.
- Thiet bi ky thuat khong dat chuan hop quy.
- San pham thuoc la.

LIEN HE BAO CAO VI PHAM:
Neu phat hien san pham vi pham, nguoi mua co the bao cao qua:
- App Shopee -> Trang san pham -> "Bao cao" -> Chon ly do.
- Trung tam tro giup: help.shopee.vn.
""",
    },

    # ── 7. Quy trình giao hàng ────────────────────────────────────────────────
    {
        "filename": "delivery-process-shopee.pdf",
        "doc_id": "delivery-process",
        "customer_role": "buyer",
        "category": "shipping",
        "source_url": "https://help.shopee.vn/portal/4/article/79569",
        "document_version": "2023-05-19",
        "title": "Quy Trinh Giao Nhan Hang Tren Shopee Vietnam",
        "content": """\
QUY TRINH GIAO NHAN HANG TREN SHOPEE VIETNAM
Dang ngay 19/05/2023. Doc ID: delivery-process | Customer Role: buyer | Category: shipping

1. QUY TRINH GIAO HANG
Buoc 1: Nguoi ban dong goi va ban giao hang cho don vi van chuyen.
Buoc 2: Don vi van chuyen nhan hang, cap nhat trang thai "Dang van chuyen".
Buoc 3: Shipper lien he nguoi mua 2-3 LAN de sap xep giao hang.
Buoc 4: Giao hang thanh cong hoac hoan hang (neu khong giao duoc).

2. TRUONG HOP KHONG GIAO DUOC HANG
Neu khong lien he duoc nguoi mua (khong nghe may / tu choi nhan):
- Don hang se HOAN VE NGUOI BAN sau khi het thoi han uu tien giao lai.
- Nguoi mua co the yeu cau giao lai trong vong khong qua 5 ngay ke tu lan lien
  he dau tien.
- Sau 5 ngay khong yeu cau giao lai -> don hang tu dong hoan ve nguoi ban.

DIA DIEM HOAN TRA PHU THUOC VAO PHUONG THUC LAY HANG BAN DAU:
- Don vi den lay tan noi -> hang hoan ve dia chi goc cua nguoi ban.
- Nguoi ban tu mang ra buu cuc -> hang hoan ve buu cuc do.

3. PHI VAN CHUYEN HOAN HANG
- Nguoi ban KHONG phai chiu phi hoan hang neu giao that bai do loi nguoi mua
  (khong nghe may, tu choi nhan, dia chi sai).

4. DONG KIEM (KIEM TRA HANG CUNG NHAN VIEN GIAO HANG)
Tu ngay 19/05/2023, nguoi mua du dieu kien co the yeu cau "dong kiem":
- Kiem tra hang cung shipper truoc khi nhan va ki bien ban.
- Voi don hang KHONG du dieu kien dong kiem:
  * Nguoi mua chi duoc kiem tra cac yeu to ben ngoai cua goi hang truoc khi
    thanh toan.
  * Mo goi hang chi duoc thuc hien SAU KHI DA THANH TOAN DAY DU.

5. KHUYEN NGHI KHI NHAN HANG
Shopee khuyen nghi nguoi mua:
- Quay / chup lai toan bo 6 mat cua kien hang truoc va sau khi mo.
- Lam bang chung neu co tranh chap ve tinh trang hang hoa.

6. CAC DON VI VAN CHUYEN TREN SHOPEE
- SPX Express (van chuyen chinh thuc cua Shopee).
- J&T Express.
- GHN (Giao hang Nhanh).
- GHTK (Giao hang Tiet Kiem).
- Ninja Van.
- ViettelPost.
- VNPost.
- Best Express.

7. THOI GIAN GIAO HANG DU KIEN
- Noi thanh: 1-2 ngay lam viec.
- Lien tinh gan: 2-4 ngay lam viec.
- Vung sau / vung xa: 4-7 ngay lam viec.
- Hang quoc te (Cross-border): 7-25 ngay (tuy quoc gia).
""",
    },

    # ── 8. Chính sách bảo mật ─────────────────────────────────────────────────
    {
        "filename": "privacy-policy-shopee.pdf",
        "doc_id": "privacy-policy",
        "customer_role": "both",
        "category": "privacy",
        "source_url": "https://help.shopee.vn/portal/4/article/77244",
        "document_version": "2026-06-11",
        "title": "Chinh Sach Bao Mat - Shopee Vietnam",
        "content": """\
CHINH SACH BAO MAT - SHOPEE VIETNAM
Dang ngay 04/06/2026, co hieu luc sau 7 ngay ke tu ngay dang.
Doc ID: privacy-policy | Customer Role: both | Category: privacy

1. CAM KET BAO VE DU LIEU CA NHAN
Shopee cam ket bao ve du lieu ca nhan nguoi dung theo quy dinh phap luat Viet Nam
ve bao ve du lieu ca nhan (PDPA - Personal Data Protection Act).

2. THU THAP DU LIEU KHI NAO
Du lieu duoc thu thap trong cac tinh huong:
- Dang ky tai khoan Shopee.
- Dien bieu mau, dat don hang, thuc hien giao dich.
- Lien he bo phan ho tro khach hang.
- Tuong tac voi thiet bi (app, website).
- Lien ket mang xa hoi (Facebook, Google...).
- Chia se vi tri dia ly (neu cap phep).
- Tham gia minigame, cuoc thi, khuyen mai.

3. LOAI DU LIEU DUOC THU THAP
Du lieu ca nhan co ban:
- Ho ten, email, ngay sinh, dia chi thanh toan / giao hang.
- So dien thoai, gioi tinh.
- Hinh anh dai dien (tuy chon).
- Giay to tuy than (CMND/CCCD, ho chieu) khi xac minh.

Du lieu ky thuat:
- Dia chi IP, loai thiet bi, trinh duyet, he dieu hanh.
- Du lieu mang, cookie, du lieu phien dang nhap.
- Du lieu vi tri GPS chinh xac (khi cap phep).
- Thong tin xem noi dung / quang cao.

Du lieu giao dich:
- Lich su mua hang, don dat hang, lich su thanh toan.
- Danh gia va nhan xet san pham.

4. MUC DICH SU DUNG DU LIEU
- Xu ly giao dich va quan ly tai khoan.
- Cham soc khach hang, giai quyet khieu nai.
- Phong chong gian lan va bao mat tai khoan.
- Tiep thi va quang cao ca nhan hoa.
- Nghien cuu va phat trien san pham.
- Tuan thu nghia vu phap ly.

5. CHIA SE VOI BEN THU BA
Du lieu co the duoc chia se voi:
- Doi tac van chuyen (de giao hang).
- Doi tac thanh toan (de xu ly giao dich).
- Co quan Nha nuoc (khi co yeu cau hop phap).
- Cong ty lien ket trong tap doan Sea Group.
- Doi tac phan tich (du lieu an danh).
Shopee KHONG BAN du lieu ca nhan cho ben thu ba.

6. BAO MAT DU LIEU
- Ma hoa SSL/TLS trong qua trinh truyen tai.
- Kiem soat truy cap nghiem ngat vao he thong.
- Giam sat va phat hien xam nhap 24/7.
- Sao luu du lieu dinh ky.

7. THOI GIAN LUU TRU
- Du lieu tai khoan: trong suot thoi gian tai khoan hoat dong.
- Du lieu giao dich: toi thieu 5 nam (theo quy dinh phap luat).
- Du lieu phien dang nhap: 30 ngay.
- Du lieu marketing: den khi nguoi dung rut lai su dong y.

8. QUYEN CUA NGUOI DUNG
- Quyen truy cap va xem du lieu ca nhan.
- Quyen chinh sua thong tin khong chinh xac.
- Quyen xoa du lieu (xoa tai khoan).
- Quyen rut lai su dong y xu ly du lieu.
- Quyen han che xu ly du lieu.
- Quyen khieu nai voi co quan bao ve du lieu.

9. COOKIE
Shopee su dung cookie:
- Cookie bat buoc: duy tri phien dang nhap, gio hang.
- Cookie phan tich: cai thien trai nghiem nguoi dung.
- Cookie quang cao: hien thi quang cao phu hop.
Nguoi dung co the tat cookie qua cai dat trinh duyet.

10. CHUYEN DU LIEU QUOC TE
Du lieu ca nhan co the duoc chuyen ra nuoc ngoai phu hop voi quy dinh phap luat
bao ve du lieu (sea Group co van phong tai Singapore, chiu su giam sat phap luat
Singapore va quoc te).

11. LIEN HE
Can bo Bao ve Du lieu (DPO): dpo.vn@shopee.com
Dia chi: Van phong Shopee tai Ha Noi va TP.HCM.
""",
    },

    # ── 9. Chính sách mã giảm giá ─────────────────────────────────────────────
    {
        "filename": "voucher-discount-policy-shopee.pdf",
        "doc_id": "voucher-discount-policy",
        "customer_role": "both",
        "category": "promotion",
        "source_url": "https://help.shopee.vn/portal/4/article/166085",
        "document_version": "2026-05-23",
        "title": "Chinh Sach Chung Ve Ma Giam Gia Shopee Vietnam",
        "content": """\
CHINH SACH CHUNG VE MA GIAM GIA SHOPEE VIETNAM
Cong bo ngay 14/11/2026 (noi dung phan dong tai tro co hieu luc tu 23/05/2026).
Doc ID: voucher-discount-policy | Customer Role: both | Category: promotion

1. PHAM VI AP DUNG
Chinh sach quan ly viec phat hanh, su dung va quyen loi cua toan bo ma uu dai
("Ma Uu Dai" / "Voucher Shopee") tren website va ung dung Shopee.vn.

Cac loai ma uu dai bao gom:
- Ma hoan Shopee Xu (cashback Xu).
- Ma giam gia lien quan chuong trinh khuyen mai.
- Ma dam bao giao hang dung hen.
- Ma mien / giam phi van chuyen.
- Ma dong phat hanh voi doi tac.
- Cac loai ma khac do Shopee quyet dinh.

2. QUYEN VA HAN CHE CUA NGUOI DUNG
- Chi chu tai khoan Shopee DA DANG KY va DANG HOAT DONG moi duoc su dung ma.
- Moi ma chi dung MOT LAN cho MOT don hang.
- Ma uu dai KHONG co gia tri tien te, KHONG the quy doi thanh tien mat.
- KHONG duoc mua ban, chuyen nhuong hay trao doi ma.
- Ma chi co gia tri cho don hang MOI dat qua nen tang Shopee.

3. CHINH SACH DONG TAI TRO (AP DUNG TU 23/05/2026)
- Shopee va nguoi ban cung chia se chi phi khuyen mai.
- Ti le chia se: Shopee 60% - Nguoi ban 40%.
- Muc toi da dong tai tro: 50.000 VND tren moi san pham.
- Nguoi ban co quyen lua chon tham gia hoac khong tham gia chuong trinh dong
  tai tro.

4. THAM QUYEN CUA SHOPEE
Shopee co quyen:
- Thay doi thoi han hieu luc ma bat ky luc nao.
- Bo sung dieu kien su dung ma.
- Tu choi ma co dau hieu gian lan.
- Huy don hang lien quan den hoat dong dang nghi.
- Tam ngung quyen truy cap tai khoan.
- Bao cao vi pham cho co quan chuc nang.

5. GIOI HAN TRACH NHIEM
Trach nhiem toi da cua Shopee doi voi khieu nai ve ma uu dai gioi han o muc
1.000.000 VND, tru khi phap luat Viet Nam co quy dinh khac.

6. HUONG DAN SU DUNG MA GIAM GIA
Buoc 1: Them san pham vao gio hang.
Buoc 2: Tien hanh thanh toan -> Trang checkout.
Buoc 3: Nhan "Them Shopee Voucher" (ma Shopee) hoac "Them Voucher Shop"
        (ma nguoi ban).
Buoc 4: Nhap ma hoac chon tu danh sach ma cua ban.
Buoc 5: Nhan "AP DUNG" -> Kiem tra gia sau giam.
Buoc 6: Hoan tat thanh toan.

7. LY DO MA KHONG AP DUNG DUOC
- Don hang chua dat gia tri toi thieu.
- Ma da het han.
- San pham khong thuoc danh muc ap dung.
- Ma da duoc su dung.
- Tai khoan bi han che.
""",
    },

    # ── 10. Chương trình ưu đãi phí vận chuyển (Seller) ──────────────────────
    {
        "filename": "shipping-fee-discount-program-shopee.pdf",
        "doc_id": "shipping-fee-discount-program",
        "customer_role": "seller",
        "category": "shipping",
        "source_url": "https://help.shopee.vn/portal/4/article/77263",
        "document_version": "2024-07-03",
        "title": "Dieu Khoan Chuong Trinh Uu Dai Phi Van Chuyen - Nguoi Ban Shopee",
        "content": """\
DIEU KHOAN VA DIEU KIEN CHUONG TRINH UU DAI PHI VAN CHUYEN
Danh cho Nguoi Ban Shopee Mall
Dang ngay 26/06/2024, co hieu luc tu 03/07/2024.
Doc ID: shipping-fee-discount-program | Customer Role: seller | Category: shipping

1. DIEU KIEN THAM GIA
- Chi ap dung cho nguoi ban thuoc SHOPEE MALL.
- Tai khoan khong bi tam khoa hoac ngung hoat dong.
- Da dap ung day du yeu cau ve chat luong dich vu cua san.
- Dang ky tham gia qua Seller Centre -> Chuong trinh van chuyen.

2. PHI DICH VU CHUONG TRINH
- Phi dich vu: 6% tren gia ban cua moi san pham.
- Muc toi da: 50.000 VND / san pham.
- Phuong thuc khau tru: tu dong kho tru tu don hang thanh cong hoac don hoan
  tra duoc duyet, truoc khi ghi co vao tai khoan nguoi ban.

3. QUYEN LOI NGUOI BAN THAM GIA
- Gian hang duoc gan nhan "FREESHIP" - thu hut nguoi mua.
- Nguoi mua duoc giam phi van chuyen khi mua hang.
- Tang kha nang hien thi tren trang chu va ket qua tim kiem.
- Tang ty le chuyen doi (conversion rate).

4. HAN CHE VA LOAI TRU
- KHONG ap dung voi nguoi ban co dau hieu lam dung khuyen mai.
- KHONG ap dung voi nguoi ban vi pham chinh sach san.
- Nguoi ban tich luy TU 6 DIEM PHAT tro len se bi tu dong tam ngung.
- Chuong trinh KHONG duoc chuyen nhuong giua cac tai khoan nguoi ban.

5. HE THONG DIEM PHAT
Vi pham chinh sach -> nhan diem phat:
- Dang san pham vi pham: 1-3 diem.
- Ti le huy don hang cao: 1-2 diem.
- Ti le phan hoi kem: 1 diem.
- Dat hang gia tao: 3 diem.
- Tu 6 diem: tam ngung chuong trinh.

6. GIOI HAN TRACH NHIEM
Shopee KHONG bao dam hoac cam ket ve:
- Viec tang so luong nguoi truy cap.
- Muc doanh so dat duoc.
Trach nhiem cua Shopee gioi han o muc phi dich vu da thanh toan.

7. DIEU CHINH CHUONG TRINH
Shopee co quyen dieu chinh dieu khoan voi thong bao truoc phu hop (toi thieu
30 ngay truoc ngay co hieu luc thay doi).

8. CACH TINH PHI VAN CHUYEN UU DAI
Vi du: San pham gia 200.000 VND, phi van chuyen goc 30.000 VND.
- Phi dich vu: 200.000 x 6% = 12.000 VND.
- Nguoi mua duoc giam: 30.000 VND phi van chuyen.
- Nguoi ban: nhan 200.000 - 12.000 = 188.000 VND (truoc cac phi khac).
""",
    },

    # ── 11. Giải quyết tranh chấp [MỚI] ─────────────────────────────────────
    {
        "filename": "dispute-resolution-policy-shopee.pdf",
        "doc_id": "dispute-resolution-policy",
        "customer_role": "both",
        "category": "dispute",
        "source_url": "https://help.shopee.vn/portal/4/article/77255",
        "document_version": "2025-01-01",
        "title": "Chinh Sach Giai Quyet Tranh Chap - Shopee Vietnam",
        "content": """\
CHINH SACH GIAI QUYET TRANH CHAP - SHOPEE VIETNAM
Doc ID: dispute-resolution-policy | Customer Role: both | Category: dispute

1. GIOI THIEU
Chinh sach nay huong dan quy trinh giai quyet tranh chap giua nguoi mua va nguoi
ban tren nen tang Shopee Vietnam, dam bao cong bang va minh bach cho ca hai ben.

2. CAC LOAI TRANH CHAP PHO BIEN
- Hang khong den (missing delivery).
- Hang bi hu hong trong qua trinh van chuyen.
- San pham khong dung mo ta (sai mau, size, chat lieu).
- Nguoi ban tu choi hoan tien.
- Nguoi mua tu choi nhan hang.
- Tranh chap ve chat luong san pham.
- Hang gia / hang nhai.

3. QUY TRINH GIAI QUYET TRANH CHAP

Giai doan 1 — Lien he truc tiep (3 ngay):
- Nguoi mua va nguoi ban tu thuong luong qua Shopee Chat.
- Nguoi ban phai phan hoi trong vong 12 GIO.
- Neu dong thuan: ket thuc tranh chap.

Giai doan 2 — Yeu cau Shopee can thiep:
- Mo yeu cau tranh chap trong don hang.
- Dinh kem bang chung: anh / video san pham, hoa don, anh bao bi.
- Mo ta chi tiet van de gap phai.

Giai doan 3 — Shopee xem xet (3-7 ngay lam viec):
- Chuyen vien Shopee xem xet bang chung tu ca hai ben.
- Co the lien he truc tiep de lay them thong tin.
- Ra quyet dinh cuoi cung.

4. YEU CAU BANG CHUNG

Nguoi mua can cung cap:
- Anh/video san pham NGAY KHI MO HOP (quan trong nhat).
- Anh bao bi va tinh trang dong goi.
- So sanh voi hinh anh mo ta cua nguoi ban.
- Anh the hien ro loi, hu hong hoac su khac biet.

Nguoi ban can cung cap:
- Bang chung dong goi va gui hang.
- Anh/video san pham truoc khi gui.
- Bien lai / chung tu van chuyen.
- Lich su chat voi nguoi mua (neu co).

5. KET QUA GIAI QUYET TRANH CHAP
Shopee co the quyet dinh:
- Hoan tien toan phan cho nguoi mua.
- Hoan tien mot phan.
- Tu choi hoan tien (neu nguoi mua sai).
- Yeu cau tra hang va hoan tien.

6. KHANG CAO
- Trong vong 3 ngay sau quyet dinh cua Shopee.
- Gui khang cao kem bang chung bo sung.
- Quyet dinh khang cao la QUYET DINH CUOI CUNG, khong the khang cao them.

7. CAM KET CUA SHOPEE
- Xet xu cong bang dua tren bang chung khach quan.
- Bao ve quyen loi ca nguoi mua va nguoi ban.
- Khong thien vi dua tren lich su giao dich.
- Minh bach trong qua trinh xu ly.

8. PHONG CHONG GIAN LAN
Neu phat hien gian lan trong yeu cau tranh chap:
- Cham dut xem xet yeu cau ngay lap tuc.
- Tai khoan vi pham bi khoa tam thoi hoac vinh vien.
- Bao cao co quan chuc nang neu can thiet.
""",
    },

    # ── 12. ShopeePay Wallet Policy [MỚI] ───────────────────────────────────
    {
        "filename": "shopeepay-wallet-policy-shopee.pdf",
        "doc_id": "shopeepay-wallet-policy",
        "customer_role": "buyer",
        "category": "payment",
        "source_url": "https://help.shopee.vn/portal/4/article/77285",
        "document_version": "2025-06-01",
        "title": "Chinh Sach Va Huong Dan ShopeePay - Vi Dien Tu Shopee",
        "content": """\
CHINH SACH VA HUONG DAN SHOPEEPAY - VI DIEN TU SHOPEE
Doc ID: shopeepay-wallet-policy | Customer Role: buyer | Category: payment

1. SHOPEEPAY LA GI?
ShopeePay (truoc day la AirPay) la vi dien tu tich hop trong app Shopee, cho phep:
- Thanh toan don hang tren Shopee va doi tac.
- Nhan va gui tien.
- Nap tien dien thoai, thanh toan hoa don dien, nuoc, internet.
- Nhan cashback va uu dai doc quyen.
- Su dung Shopee Xu (tien thuong trong he thong).

2. KICH HOAT SHOPEEPAY
Buoc 1: Mo app Shopee -> Nhan "ShopeePay" tren trang chu.
Buoc 2: Nhan "Kich hoat ngay".
Buoc 3: Xac minh so dien thoai.
Buoc 4: Tao ma PIN 6 so (bat buoc, dung cho moi giao dich).
Buoc 5 (tuy chon): Xac minh danh tinh de tang han muc.

3. NAP TIEN VAO SHOPEEPAY

Phuong thuc 1 — Qua ngan hang / chuyen khoan:
- ShopeePay -> Nap tien -> Chon Nap qua Ngan hang.
- Chuyen khoan den tai khoan ao ShopeePay duoc cung cap.
- Tien vao ngay sau khi chuyen thanh cong.

Phuong thuc 2 — Qua cua hang tien loi:
- Den Circle K, FamilyMart, Ministop, GS25, Vinmart+.
- Cung cap SĐT hoac ma QR ShopeePay tai quay.
- Nop tien mat -> Tien vao ShopeePay ngay.

4. HAN MUC SHOPEEPAY

| Loai tai khoan | Han muc nap/thang | Han muc chi/ngay |
|----------------|------------------|-----------------|
| Chua xac minh  | 20.000.000 VND   | 5.000.000 VND   |
| Xac minh co ban| 50.000.000 VND   | 20.000.000 VND  |
| Xac minh day du| 100.000.000 VND  | 50.000.000 VND  |

5. RUT TIEN TU SHOPEEPAY
Buoc 1: ShopeePay -> Rut tien.
Buoc 2: Chon tai khoan ngan hang da lien ket (phai lien ket truoc).
Buoc 3: Nhap so tien (toi thieu 10.000 VND, toi da 500.000.000 VND/lan cho ca nhan).
Buoc 4: Xac nhan bang ma PIN.
Buoc 5: Tien ve tai khoan trong 1-2 gio (ngan hang truc tuyen).

6. UU DAI KHI DUNG SHOPEEPAY
- Cashback: 5-15% cho don hang thanh toan qua ShopeePay.
- Voucher mien phi van chuyen danh rieng cho nguoi dung ShopeePay.
- Flash Sale gio vang uu tien cho nguoi dung ShopeePay.
- Tich diem Shopee Xu moi giao dich.

7. BAO MAT SHOPEEPAY
- Xac thuc 2 yeu to (2FA) cho giao dich lon.
- Ma PIN 6 so bat buoc cho moi giao dich.
- Tu dong khoa sau 5 lan nhap PIN sai.
- Thong bao tuc thi cho moi giao dich qua app va SMS.
- Bao cao giao dich bat thuong ngay lap tuc qua ho tro.

8. XU LY SU CO
Neu mat dien thoai hoac bi ro thong tin:
- Koa ngay ShopeePay: lien he 1900 1221.
- Doi ma PIN: ShopeePay -> Cai dat -> Doi ma PIN.
- Kha phuc tai khoan: Lien he ho tro va xac minh danh tinh.
""",
    },

    # ── 13. Seller Fee & Commission [MỚI] ───────────────────────────────────
    {
        "filename": "seller-fee-commission-shopee.pdf",
        "doc_id": "seller-fee-commission",
        "customer_role": "seller",
        "category": "seller-policy",
        "source_url": "https://help.shopee.vn/portal/4/article/77260",
        "document_version": "2025-01-01",
        "title": "Chinh Sach Phi Va Hoa Hong Nguoi Ban - Shopee Vietnam",
        "content": """\
CHINH SACH PHI VA HOA HONG NGUOI BAN - SHOPEE VIETNAM
Doc ID: seller-fee-commission | Customer Role: seller | Category: seller-policy

1. TONG QUAN CAU TRUC PHI
Nguoi ban tren Shopee chiu cac loai phi sau:
- Phi hoa hong (Commission Fee): % tren gia tri san pham.
- Phi thanh toan (Transaction Fee): % tren tong gia tri don hang.
- Phi dich vu van chuyen (tuy chuong trinh tham gia).

2. PHI HOA HONG THEO DANH MUC

Thoi Trang & Phu Kien:
- Quan ao, giay dep: 5% gia tri san pham.
- Tui xach, phu kien: 4% gia tri san pham.

Dien Tu & Cong Nghe:
- Dien thoai, may tinh: 2% gia tri san pham.
- Phu kien dien tu: 3% gia tri san pham.
- Do gia dung dien tu: 3% gia tri san pham.

Suc Khoe & Lam Dep:
- My pham, cham soc da: 4% gia tri san pham.
- Thuc pham chuc nang: 4% gia tri san pham.

Thuc Pham & Do Uong:
- Thuc pham kho: 3% gia tri san pham.
- Do uong: 3% gia tri san pham.

Nha Cua & Doi Song: 4% gia tri san pham.
Me & Be: 4% gia tri san pham.
The Thao & Du Lich: 3% gia tri san pham.
Sach & Van Phong Pham: 2% gia tri san pham.
O To & Xe May: 2% gia tri san pham.
Cac danh muc khac: 3-5% tuy chinh sach tung ky.

3. PHI THANH TOAN (TRANSACTION FEE)
- Muc phi: 2% tren TONG GIA TRI don hang (bao gom ca gia san pham va phi van
  chuyen neu nguoi ban ho tro phi vc).
- Ap dung cho tat ca phuong thuc thanh toan.

4. CHUONG TRINH MIEN PHI HOA HONG
Nguoi ban duoc mien phi hoa hong trong cac truong hop:
- 3 thang dau khi mo shop moi (ap dung khi du dieu kien nhat dinh).
- San pham tham gia Flash Sale do Shopee to chuc (Shopee chi tra phi).
- Nganh hang duoc Shopee ho tro phat trien (theo thoi ky).

5. CHU KY THANH TOAN CHO NGUOI BAN
- Tien duoc chuyen sau 2-5 NGAY LAM VIEC ke tu khi don hang hoan thanh.
- Rut tien toi thieu: 10.000 VND.
- Rut tien toi da / lan: 500.000.000 VND (ca nhan), 2.000.000.000 VND (DN).
- Phi rut tien: mien phi (mot so ngan hang co the ap phi rieng).

6. QUY DONG DAM BAO NGUOI BAN (SELLER GUARANTEE FUND)
- Giu lai 10% gia tri giao dich trong 30 ngay dau hoat dong.
- Muc dich: dam bao xu ly khieu nai va hoan tien.
- Giai phong tu dong sau khi du dieu kien.

7. THAY DOI PHI
- Shopee thong bao truoc it nhat 30 ngay khi dieu chinh phi.
- Thong bao qua: email, app notification, Seller Centre.
- Nguoi ban co quyen ngung kinh doanh neu khong dong y voi phi moi.

8. TRANH CHAP VE PHI
- Gui yeu cau trong vong 30 ngay ke tu ngay phat sinh.
- Qua: Seller Centre -> Ho tro -> Khieu nai ve phi.
- Thoi gian xu ly: 3-7 ngay lam viec.
""",
    },

    # ── 14. Mua hàng xuyên biên giới [MỚI] ──────────────────────────────────
    {
        "filename": "cross-border-shopping-shopee.pdf",
        "doc_id": "cross-border-shopping",
        "customer_role": "buyer",
        "category": "shipping",
        "source_url": "https://help.shopee.vn/portal/4/article/77290",
        "document_version": "2025-01-01",
        "title": "Huong Dan Mua Hang Xuyen Bien Gioi (Cross-Border) Shopee",
        "content": """\
HUONG DAN MUA HANG XUYEN BIEN GIOI (CROSS-BORDER) TREN SHOPEE
Doc ID: cross-border-shopping | Customer Role: buyer | Category: shipping

1. SHOPEE GLOBAL LA GI?
Shopee Global cho phep nguoi mua tai Viet Nam dat hang tu nguoi ban nuoc ngoai
(Singapore, Han Quoc, Dai Loan, Nhat Ban, Trung Quoc...) truc tiep tren app Shopee.

2. NHAN BIET SAN PHAM CROSS-BORDER
- Badge "HANG QUOC TE" hoac "SHOPEE GLOBAL" tren san pham.
- Thoi gian giao hang hien thi 7-25 ngay.
- Mo ta co the bang tieng nuoc ngoai kem ban dich tieng Viet.
- Gia co the tinh bang ngoai te (USD, KRW, TWD...).

3. THOI GIAN GIAO HANG DU KIEN

| Quoc gia xuat xu  | Thoi gian du kien |
|-------------------|-------------------|
| Singapore         | 5-10 ngay         |
| Han Quoc          | 7-14 ngay         |
| Dai Loan          | 7-14 ngay         |
| Nhat Ban          | 10-20 ngay        |
| Trung Quoc        | 10-25 ngay        |
| My / Chau Au      | 15-30 ngay        |

Luu y: Thoi gian co the thay doi do nghi le, thong quan hai quan, thoi tiet.

4. CHI PHI VA THUE

Phi van chuyen quoc te:
- Thuong cao hon van chuyen noi dia: 50.000 - 200.000 VND/don.
- Mot so san pham duoc mien phi van chuyen (free shipping).

Thue nhap khau:
- Don hang DUOI 1.000.000 VND: thuong mien thue nhap khau.
- Don hang TREN 1.000.000 VND: co the bi danh thue nhap khau (tuy loai hang).
- Thue VAT: 10% ap dung cho hang nhap khau.
- Nguoi mua chiu trach nhiem ve thue va phi thong quan.

5. CHINH SACH TRA HANG HANG QUOC TE
Khac biet so voi hang noi dia:
- Thoi han tra hang: 5-7 ngay (ngan hon 15 ngay cua hang noi dia).
- Chi phi van chuyen hoan tra cao hon nhieu so voi noi dia.
- Mot so mat hang dac thu khong ho tro tra hang.

6. LUU Y QUAN TRONG KHI MUA HANG QUOC TE
- Kiem tra dien ap thiet bi dien tu (Viet Nam: 220V/50Hz).
- Kiem tra quy cach (size chart cua Chau A khac Viet Nam).
- Hang quoc te thuong KHONG CO bao hanh tai Viet Nam.
- Theo doi don hang thuong xuyen vi thoi gian giao lau hon.
- Chup anh/quay video khi nhan hang de lam bang chung.

7. QUY TRINH DAT HANG QUOC TE
Buoc 1: Tim san pham voi bo loc "Hang quoc te".
Buoc 2: Chon san pham, kiem tra thoi gian giao hang du kien.
Buoc 3: Them vao gio hang -> Chon phuong thuc thanh toan.
Buoc 4: Xac nhan dia chi giao hang (bang tieng Anh hoac co phien am).
Buoc 5: Thanh toan -> Nhan xac nhan qua email/SMS.
Buoc 6: Theo doi qua App Shopee bang ma van don quoc te.

8. HOI VE THONG QUAN
Neu don hang bi giu tai hai quan:
- Shopee thong bao va huong dan cac buoc xu ly.
- Nguoi mua co the can nop thue bo sung.
- Truong hop tu choi thong quan: hoan tien trong 5-10 ngay lam viec.
""",
    },
]


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Thu muc san sang: {DATA_DIR}")


class ShopeeDocPDF(FPDF):
    """Custom PDF class với header và footer cho tài liệu Shopee."""

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(238, 77, 45)   # Shopee orange
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, "SHOPEE VIETNAM - TAI LIEU CHINH SACH", align="C", fill=True, ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 8, f"Trang {self.page_no()} | Shopee Vietnam | help.shopee.vn",
                  align="C")


def create_pdf(doc: dict) -> Path:
    """
    Tạo file PDF từ nội dung chính sách.
    Ghi nhãn customer_role và metadata rõ ràng.
    """
    pdf = ShopeeDocPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Tiêu đề
    pdf.set_font("Helvetica", "B", 14)
    title_safe = doc["title"].encode("latin-1", errors="replace").decode("latin-1")
    pdf.multi_cell(0, 8, title_safe, align="C")
    pdf.ln(3)

    # ── Metadata block
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_fill_color(245, 245, 245)
    meta_lines = [
        f"Doc ID     : {doc['doc_id']}",
        f"Role       : {doc['customer_role']}",
        f"Category   : {doc['category']}",
        f"Source URL : {doc['source_url']}",
        f"Version    : {doc.get('document_version', 'N/A')}",
        f"Retrieved  : 2026-08-04",
        f"Language   : vi",
    ]
    for line in meta_lines:
        safe = line.encode("latin-1", errors="replace").decode("latin-1")
        pdf.cell(0, 5, safe, ln=True, fill=True)
    pdf.ln(3)

    # ── Separator
    pdf.set_draw_color(238, 77, 45)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # ── Body content
    pdf.set_font("Helvetica", "", 10)
    pdf.set_line_width(0.2)
    for line in doc["content"].split("\n"):
        safe_line = line.encode("latin-1", errors="replace").decode("latin-1")
        if safe_line.strip() == "":
            pdf.ln(3)
        elif safe_line.isupper() and len(safe_line.strip()) > 5:
            # Section header
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 6, safe_line)
            pdf.set_font("Helvetica", "", 10)
        else:
            pdf.multi_cell(0, 5.5, safe_line)

    filepath = DATA_DIR / doc["filename"]
    pdf.output(str(filepath))
    return filepath


def save_metadata_index():
    """Lưu file index JSON cho tất cả tài liệu."""
    index = []
    for doc in LEGAL_DOCUMENTS:
        index.append({
            "filename": doc["filename"],
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "customer_role": doc["customer_role"],
            "category": doc["category"],
            "source_url": doc["source_url"],
            "document_version": doc.get("document_version", "N/A"),
            "language": "vi",
            "local_path": str(DATA_DIR / doc["filename"]),
        })
    index_path = DATA_DIR / "documents_index.json"
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] Index: {index_path}")


def collect_all_documents():
    """Thu thập tất cả tài liệu và lưu vào data/landing/legal/."""
    setup_directory()
    print(f"\n{'='*60}")
    print("TASK 1 - Thu Thap Van Ban Chinh Sach TMDT")
    print(f"{'='*60}\n")

    success_count = 0
    for i, doc in enumerate(LEGAL_DOCUMENTS, 1):
        print(f"[{i:02d}/{len(LEGAL_DOCUMENTS)}] {doc['filename']}")
        print(f"         Role: {doc['customer_role']} | Category: {doc['category']}")
        try:
            filepath = create_pdf(doc)
            size_kb = filepath.stat().st_size / 1024
            print(f"         => OK ({size_kb:.1f} KB)")
            success_count += 1
        except Exception as e:
            print(f"         => LOI: {e}")
        time.sleep(0.05)

    save_metadata_index()

    print(f"\n{'='*60}")
    print(f"HOAN THANH: {success_count}/{len(LEGAL_DOCUMENTS)} tai lieu PDF")
    print(f"Thu muc   : {DATA_DIR}")
    print(f"{'='*60}\n")

    print("Danh sach file:")
    for f in sorted(DATA_DIR.iterdir()):
        if f.suffix in (".pdf", ".docx", ".json"):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name:55s} ({size_kb:6.1f} KB)")

    return success_count


if __name__ == "__main__":
    count = collect_all_documents()
    print(f"\n{'CP1 PASS' if count >= 3 else 'FAIL'}: {count} tai lieu (yeu cau >= 3)")
