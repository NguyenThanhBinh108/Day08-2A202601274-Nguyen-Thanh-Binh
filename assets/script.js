const chatLog = document.querySelector("#chatLog");
const chatForm = document.querySelector("#chatForm");
const queryInput = document.querySelector("#queryInput");
const sourceList = document.querySelector("#sourceList");
const sourceCount = document.querySelector("#sourceCount");
const traceList = document.querySelector("#traceList");
const topK = document.querySelector("#topK");
const topKValue = document.querySelector("#topKValue");
const clearChat = document.querySelector("#clearChat");
const exportChat = document.querySelector("#exportChat");

const demoSources = [
  {
    title: "payment-methods-shopee.md",
    score: "0.91",
    text: "Shopee hỗ trợ ví ShopeePay, thẻ tín dụng/ghi nợ, trả góp qua thẻ, QR Code, chuyển khoản ngân hàng, NAPAS, Apple Pay, Google Pay và COD."
  },
  {
    title: "returns-refund-policy-shopee.md",
    score: "0.88",
    text: "Người mua có thể tạo yêu cầu trả hàng/hoàn tiền trong thời hạn được quy định sau khi đơn hàng giao thành công."
  },
  {
    title: "refund-evidence-guide.md",
    score: "0.83",
    text: "Bằng chứng nên gồm hình ảnh sản phẩm, tình trạng kiện hàng, mã vận đơn và mô tả rõ lý do yêu cầu hoàn tiền."
  }
];

const answerBank = [
  {
    match: ["thanh toan", "payment"],
    answer: "Shopee hỗ trợ nhiều phương thức thanh toán, gồm ví ShopeePay, thẻ tín dụng/ghi nợ, trả góp qua thẻ, QR Code, chuyển khoản qua ứng dụng ngân hàng, thẻ NAPAS, Apple Pay, Google Pay và COD. [payment-methods-shopee.md]",
    sources: [demoSources[0]]
  },
  {
    match: ["tra hang", "hoan tien", "return", "refund"],
    answer: "Để yêu cầu trả hàng hoặc hoàn tiền, người mua cần tạo yêu cầu trong thời hạn chính sách cho phép, mô tả vấn đề và gửi bằng chứng liên quan đến sản phẩm hoặc đơn hàng. [returns-refund-policy-shopee.md] [refund-evidence-guide.md]",
    sources: [demoSources[1], demoSources[2]]
  },
  {
    match: ["bang chung", "evidence"],
    answer: "Bằng chứng nên gồm hình ảnh hoặc video sản phẩm, tình trạng đóng gói, mã vận đơn, thông tin đơn hàng và mô tả rõ vấn đề cần xử lý. [refund-evidence-guide.md]",
    sources: [demoSources[2]]
  },
  {
    match: ["nguoi ban", "dang ban", "seller"],
    answer: "Người bán không được đăng bán hàng giả, hàng nhái, sản phẩm bất hợp pháp, mặt hàng vi phạm sở hữu trí tuệ hoặc sản phẩm nằm trong danh sách cấm của sàn. [product-listing-regulations-shopee.md]",
    sources: [
      {
        title: "product-listing-regulations-shopee.md",
        score: "0.86",
        text: "Quy định đăng bán yêu cầu người bán tránh hàng cấm, hàng giả/nhái, nội dung vi phạm và mặt hàng không được phép kinh doanh."
      }
    ]
  }
];

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function normalizeText(value) {
  return String(value)
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/đ/g, "d")
    .replace(/Đ/g, "D")
    .toLowerCase();
}

function addMessage(role, text) {
  const message = document.createElement("div");
  message.className = `message ${role}`;
  message.innerHTML = `
    <div class="avatar">${role === "user" ? "U" : "AI"}</div>
    <div class="bubble">
      <span class="role-label">${role === "user" ? "Bạn" : "Trợ lý RAG"}</span>
      <p>${escapeHtml(text)}</p>
    </div>
  `;
  chatLog.appendChild(message);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function pickAnswer(query) {
  const normalized = normalizeText(query);
  const found = answerBank.find((item) => item.match.some((term) => normalized.includes(term)));
  if (found) return found;

  return {
    answer: "Tôi không thể xác minh thông tin này từ nguồn hiện có. Vui lòng bổ sung dữ liệu và nối backend Task 10 để trả lời câu hỏi này bằng RAG thật.",
    sources: demoSources
  };
}

function normalizeSource(source, index) {
  const metadata = source.metadata || {};
  return {
    title: source.title || metadata.source || metadata.path || `Nguồn ${index + 1}`,
    score: Number(source.score || 0).toFixed(4),
    text: source.text || source.content || "Không có nội dung xem trước."
  };
}

function renderSources(sources) {
  const normalizedSources = sources.map(normalizeSource);
  sourceCount.textContent = normalizedSources.length;
  sourceList.innerHTML = normalizedSources.map((source) => `
    <article class="source-card">
      <div class="source-title">
        <span>${escapeHtml(source.title)}</span>
        <span>${escapeHtml(source.score)}</span>
      </div>
      <p>${escapeHtml(source.text)}</p>
    </article>
  `).join("");
}

function setTrace(active) {
  const labels = [
    "Đã nhận câu hỏi",
    `Truy xuất hybrid top-${topK.value}`,
    "Xếp hạng lại bằng RRF",
    "Dự phòng PageIndex nếu cần",
    "Sinh câu trả lời có trích dẫn",
    active ? "Đã hiển thị câu trả lời" : "Đang chờ câu hỏi"
  ];

  traceList.innerHTML = labels.map((label, index) => `
    <li class="${active || index === 0 ? "is-done" : ""}"><span></span>${label}</li>
  `).join("");
}

async function askBackend(query) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      query,
      top_k: Number(topK.value),
      use_reranking: true,
      use_pageindex_fallback: true
    })
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || data.error || "Không thể gọi backend RAG.");
  }
  return data;
}

async function handleQuery(query) {
  if (!query.trim()) return;

  addMessage("user", query);
  setTrace(true);

  try {
    const result = await askBackend(query);
    addMessage("assistant", result.answer || "Tôi không thể xác minh thông tin này từ nguồn hiện có.");
    renderSources(result.sources || []);
  } catch (error) {
    const result = pickAnswer(query);
    addMessage(
      "assistant",
      `${result.answer}\n\n[Lưu ý: Hiện đang dùng câu trả lời demo vì backend chưa chạy hoặc chưa có dữ liệu.]`
    );
    renderSources(result.sources);
  }
}

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => {
    queryInput.value = button.dataset.query;
    handleQuery(button.dataset.query);
    queryInput.value = "";
  });
});

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = queryInput.value;
  queryInput.value = "";
  handleQuery(query);
});

topK.addEventListener("input", () => {
  topKValue.textContent = topK.value;
  setTrace(false);
});

clearChat.addEventListener("click", () => {
  chatLog.innerHTML = `
    <div class="message assistant">
      <div class="avatar">AI</div>
      <div class="bubble">
        <span class="role-label">Trợ lý RAG</span>
        <p>Hội thoại đã được làm mới. Hãy nhập câu hỏi tiếp theo.</p>
      </div>
    </div>
  `;
  renderSources(demoSources);
  setTrace(false);
});

exportChat.addEventListener("click", () => {
  const messages = [...chatLog.querySelectorAll(".bubble p")].map((node) => node.textContent);
  const blob = new Blob([messages.join("\n\n---\n\n")], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "rag-chat-demo.txt";
  link.click();
  URL.revokeObjectURL(url);
});

renderSources(demoSources);
setTrace(false);
