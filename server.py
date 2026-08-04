"""
Local web server for the HTML/CSS/JS RAG chatbot.

Run:
    py server.py

Then open:
    http://localhost:5500
"""

from __future__ import annotations

import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).parent.resolve()
HOST = "localhost"
PORT = 5500

sys.path.insert(0, str(ROOT))


class RAGRequestHandler(BaseHTTPRequestHandler):
    server_version = "RAGDemoHTTP/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_json({"ok": True, "message": "Backend RAG đã sẵn sàng."})
            return

        requested_path = "index.html" if parsed.path in {"/", ""} else unquote(parsed.path.lstrip("/"))
        self.serve_static(requested_path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/chat":
            self.send_json({"error": "Không tìm thấy API."}, status=404)
            return

        try:
            payload = self.read_json_body()
            query = str(payload.get("query", "")).strip()
            top_k = int(payload.get("top_k", 5))
            use_reranking = bool(payload.get("use_reranking", True))
            use_pageindex_fallback = bool(payload.get("use_pageindex_fallback", True))

            if not query:
                self.send_json({"error": "Câu hỏi không được để trống."}, status=400)
                return

            from src.task10_generation import generate_with_citation

            result = generate_with_citation(
                query,
                top_k=top_k,
                use_reranking=use_reranking,
                use_pageindex_fallback=use_pageindex_fallback,
            )
            self.send_json({
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "retrieval_source": result.get("retrieval_source", "none"),
            })
        except Exception as exc:
            self.send_json({
                "error": "Không thể chạy pipeline RAG.",
                "detail": str(exc),
            }, status=500)

    def read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8")
        return json.loads(raw_body or "{}")

    def serve_static(self, requested_path: str) -> None:
        target = (ROOT / requested_path).resolve()
        if not self.is_safe_path(target) or not target.is_file():
            self.send_error(404, "File not found")
            return

        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        content = target.read_bytes()

        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def is_safe_path(path: Path) -> bool:
        try:
            path.relative_to(ROOT)
            return True
        except ValueError:
            return False

    def log_message(self, format: str, *args) -> None:
        print(f"[server] {self.address_string()} - {format % args}")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), RAGRequestHandler)
    print(f"Đang chạy giao diện RAG tại: http://{HOST}:{PORT}")
    print("Nhấn Ctrl+C để dừng server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
