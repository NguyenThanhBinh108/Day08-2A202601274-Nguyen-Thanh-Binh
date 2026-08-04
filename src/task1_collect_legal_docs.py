"""
Task 1 — Thu thập văn bản chính sách thương mại điện tử / hỗ trợ khách hàng.

Nguồn: Shopee Vietnam Help Center (help.shopee.vn) — trang công khai, dùng JavaScript
render (SPA) nên không tải trực tiếp được PDF. Nội dung được trích xuất thủ công từ
trang thật rồi đóng gói lại thành PDF bằng fpdf2 (đúng như gợi ý trong docstring gốc).

Mỗi văn bản được gắn nhãn metadata `customer_role` (`buyer`/`seller`/`both`) — yêu cầu
riêng của K4 Variant, dùng ở Task 4 để lọc theo đối tượng áp dụng.
"""

import json
from pathlib import Path

from fpdf import FPDF

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
FONT_PATH = Path(r"C:\Windows\Fonts\arial.ttf")
FONT_PATH_BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


# Nội dung trích xuất thật từ help.shopee.vn (đọc trực tiếp trên trang, tháng 8/2026).
LEGAL_DOCS = [
    {
        "filename": "returns-refund-policy-shopee.pdf",
        "title": "Chính Sách Trả Hàng và Hoàn Tiền Shopee",
        "url": "https://help.shopee.vn/portal/4/article/77251",
        "customer_role": "both",
        "content": """Đối Tượng Áp Dụng
Chính sách này áp dụng cho người mua, người bán, các đơn vị vận chuyển và các bên liên quan trên nền tảng Shopee.

Điều Kiện Yêu Cầu Trả Hàng/Hoàn Tiền
Người mua có thể yêu cầu trả hàng trong các trường hợp: không nhận được Sản Phẩm, hoặc không nhận được toàn bộ các Sản Phẩm đã đặt, hoặc nhận được Sản Phẩm là hàng giả. Các trường hợp khác bao gồm sản phẩm bị lỗi, giao sai, khác biệt so với mô tả, hết hạn, hoặc trả hàng COM (khi không còn nhu cầu).

Thời Hạn
15 ngày kể từ khi đơn hàng được cập nhật giao thành công. Thực phẩm tươi sống có thời hạn 24 giờ.

Trả Hàng COM
Chỉ áp dụng cho thành viên hạng Vàng, Kim Cương hoặc người dùng gói ShopeeVIP. Người dùng ShopeeVIP được trả hàng COM tối đa 15 lần/tháng.

Chi Phí Vận Chuyển
Người bán chịu chi phí hoàn trả trong hầu hết trường hợp, ngoại trừ trả hàng một phần hoặc khi lỗi thuộc về người mua.

Hoàn Tiền
Được thực hiện qua ví ShopeePay, tài khoản ngân hàng hoặc phương thức khác sau khi người bán xác nhận nhận hàng hoặc theo quyết định của Shopee.

Bằng Chứng Trả Hàng/Hoàn Tiền
Người mua cần quay video và/hoặc chụp ảnh sản phẩm ngay khi nhận hàng và trong lúc đóng gói hàng trả để làm bằng chứng đối chiếu/tranh chấp sau này. Khi trả hàng, người mua phải đóng gói theo đúng quy định vận chuyển của Shopee và gửi trả kèm đầy đủ phụ kiện, hóa đơn thuế, giấy bảo hành (nếu có), sản phẩm phải nguyên vẹn như khi nhận.""",
    },
    {
        "filename": "payment-methods-shopee.pdf",
        "title": "Phương Thức Thanh Toán Shopee Việt Nam",
        "url": "https://help.shopee.vn/portal/4/article/79198",
        "customer_role": "buyer",
        "content": """Shopee Việt Nam hiện cung cấp 9 hình thức thanh toán cho người mua:

1. Vi ShopeePay - Vi dien tu tich hop trong ung dung Shopee, cho phep thanh toan online va tai cac cua hang chap nhan.
2. The Tin dung/Ghi no - Ho tro Visa, Mastercard, JCB, AMEX voi toi thieu 10.000 VND.
3. Tra Gop The Tin Dung - Thanh toan theo ky han, KHONG ap dung cho don hang Quoc te.
4. Thanh Toan QR - Dung dich vu ngan hang truc tuyen, toi thieu 10.000 VND.
5. Ung Dung Ngan Hang - Chuyen huong truc tiep sang app ngan hang, yeu cau lien ket ho tro.
6. The Noi Dia NAPAS - Yeu cau dang ky Internet Banking, toi thieu 10.000 VND.
7. Apple Pay - Tu 10.000 den 25.000.000 VND, chi duoc ho tro tren mot so thiet bi.
8. Google Pay - Tu 10.000 den 120.000.000 VND tren Android.
9. Thanh Toan Khi Nhan Hang (COD) - Tuy theo chinh sach tung shop.
10. SPayLater - Mua truoc tra sau voi 1, 2, 3 hoac 6 ky, chi kha dung cho Nguoi dung thoa man dieu kien.

Người dùng có thể thay đổi phương thức thanh toán trong mục "Chờ xác nhận" nếu đơn hàng chưa thanh toán 100% và không thuộc danh mục Nạp thẻ & Dịch vụ. Không thể chuyển sang thanh toán khi nhận hàng (COD) nếu trước đó đã chọn ShopeePay, thẻ nội địa NAPAS, thẻ tín dụng hoặc thẻ ghi nợ.""",
    },
    {
        "filename": "privacy-policy-shopee.pdf",
        "title": "Chính Sách Bảo Mật Shopee Việt Nam",
        "url": "https://help.shopee.vn/portal/4/article/77244",
        "customer_role": "both",
        "content": """Tổng Quan
Shopee công bố chính sách bảo mật toàn diện chi phối cách thu thập, sử dụng và bảo vệ dữ liệu cá nhân của người dùng trên nền tảng.

Phạm Vi Thu Thập Dữ Liệu
Shopee thu thập dữ liệu trong quá trình đăng ký tài khoản, giao dịch, và tương tác trên nền tảng. Thông tin bao gồm họ tên, địa chỉ email, ngày sinh, địa chỉ thanh toán và nhiều loại dữ liệu khác.

Mục Đích Sử Dụng
Dữ liệu được sử dụng để xem xét và xử lý đơn đăng ký/giao dịch của người dùng, cũng như phục vụ tiếp thị, phân tích và tuân thủ pháp luật.

Bảo Mật Dữ Liệu
Shopee thực hiện các biện pháp bảo mật khác nhau nhưng công nhận rằng không thể có sự đảm bảo an ninh tuyệt đối.

Quyền Người Dùng
Người dùng có thể rút lại sự đồng ý hoặc yêu cầu truy cập dữ liệu của mình bằng cách liên hệ dpo.vn@shopee.com.

Trẻ Em
Các Dịch Vụ này không dành cho trẻ em dưới 13 tuổi và Shopee không cố tình thu thập dữ liệu của trẻ em.

Liên Hệ
Shopee, Tầng 4-5-6, Capital Place, 29 Liễu Giai, Hà Nội.""",
    },
    {
        "filename": "seller-listing-regulations-shopee.pdf",
        "title": "Quy Định Về Đăng Bán Sản Phẩm Trên Shopee",
        "url": "https://help.shopee.vn/portal/4/article/77246",
        "customer_role": "seller",
        "content": """A. PHẠM VI VÀ ĐỐI TƯỢNG ÁP DỤNG
Đối tượng: Tất cả người bán trên Shopee. Phạm vi: quy định về việc đăng bán các sản phẩm trên nền tảng.

B. QUY ĐỊNH CHUNG
Người bán phải tuân thủ Luật Thương Mại và các quy định pháp luật về trưng bày, giới thiệu hàng hóa. Tất cả chứng từ phải được scan từ chứng từ gốc, không được làm giả, chỉnh sửa, tẩy xóa.

Nội dung nghiêm cấm đăng bán: nội dung phản động, khiêu dâm, bạo lực, thông tin rác, hàng cấm (ma túy, vũ khí), sản phẩm độc hại, con người và bộ phận cơ thể, động vật hoang dã và chế phẩm liên quan, sản phẩm vi phạm quyền sở hữu trí tuệ, sản phẩm trong Danh sách bị cấm/hạn chế của Shopee.

C. HƯỚNG DẪN ĐĂNG BÁN SẢN PHẨM
Hình ảnh sản phẩm phải rõ ràng, chi tiết tình trạng sản phẩm, tối thiểu một ảnh thật do chính người bán chụp với sản phẩm chiếm ít nhất 40% diện tích ảnh, ngôn ngữ phông nền là tiếng Việt.

Tên sản phẩm phải mô tả đúng hàng hóa bằng tiếng Việt có dấu, không dùng ký tự đặc biệt hay từ ngữ gây hiểu lầm như "Sản phẩm hot", "Miễn phí vận chuyển".

Giá sản phẩm phải tính bằng VNĐ, nghiêm cấm tăng giá gốc bất hợp lý trước khuyến mãi để phóng đại tỷ lệ giảm giá.

Quy định riêng ngành hàng có điều kiện (mỹ phẩm, thực phẩm chức năng, dược phẩm...) yêu cầu người bán cung cấp đầy đủ giấy tờ: Giấy phép kinh doanh, Chứng nhận đại lý, Phiếu công bố sản phẩm phù hợp quy định pháp luật hiện hành.

D. QUY ĐỊNH VỀ HẠN SỬ DỤNG SẢN PHẨM
Chỉ được bán hàng còn tối thiểu 30% thời hạn sử dụng và còn tối thiểu 30 ngày từ thời điểm hiện tại đối với các nhóm hàng bắt buộc có hạn sử dụng (dược phẩm, mỹ phẩm, thực phẩm...).

E. XỬ LÝ VI PHẠM
Tùy mức độ vi phạm, Shopee áp dụng các biện pháp: xóa/khóa/ẩn sản phẩm, giới hạn hoặc khóa tài khoản, yêu cầu bồi thường thiệt hại, khóa tính năng rút tiền, hoặc cung cấp thông tin cho cơ quan nhà nước có thẩm quyền.

Bản cập nhật: 14/8/2024. Có hiệu lực sau 07 ngày kể từ ngày công bố.""",
    },
]


def _make_pdf(title: str, url: str, content: str) -> FPDF:
    """Render 1 văn bản chính sách thành PDF đơn giản, hỗ trợ tiếng Việt (font Arial Unicode)."""
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("Arial", "", str(FONT_PATH))
    pdf.add_font("Arial", "B", str(FONT_PATH_BOLD))

    pdf.set_font("Arial", "B", 16)
    pdf.multi_cell(0, 10, title)
    pdf.ln(2)

    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(90, 90, 90)
    pdf.multi_cell(0, 5, f"Nguồn: {url}")
    pdf.ln(4)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 7, content)

    return pdf


def download_file(doc: dict):
    """Sinh PDF cho 1 văn bản chính sách và lưu vào DATA_DIR."""
    pdf = _make_pdf(doc["title"], doc["url"], doc["content"])
    filepath = DATA_DIR / doc["filename"]
    pdf.output(str(filepath))
    print(f"✓ Đã tạo: {filepath} ({filepath.stat().st_size} bytes)")


def write_manifest():
    """Ghi manifest metadata (url, customer_role) — Task 3/4 dùng để gắn nhãn."""
    manifest = {
        doc["filename"]: {
            "title": doc["title"],
            "url": doc["url"],
            "customer_role": doc["customer_role"],
        }
        for doc in LEGAL_DOCS
    }
    manifest_path = DATA_DIR / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Manifest: {manifest_path}")


def collect_all():
    setup_directory()
    for doc in LEGAL_DOCS:
        download_file(doc)
    write_manifest()


if __name__ == "__main__":
    collect_all()
