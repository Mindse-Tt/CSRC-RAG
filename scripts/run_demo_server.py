from __future__ import annotations

import json
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from csrc_rag.retrieval.engine import RetrievalEngine  # noqa: E402


WEB_DIR = PROJECT_ROOT / "web"
ENGINE = RetrievalEngine(retrieval_mode="hybrid")


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._write_json({"status": "ok", "retrieval_mode": ENGINE.retrieval_mode})
            return
        if parsed.path == "/api/query":
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0].strip()
            if not query:
                self._write_json({"error": "missing query"}, status=HTTPStatus.BAD_REQUEST)
                return
            response = ENGINE.search(query)
            self._write_response(response)
            return
        if parsed.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/query":
            self._write_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._write_json({"error": "invalid json"}, status=HTTPStatus.BAD_REQUEST)
            return
        query = str(payload.get("query", "")).strip()
        history = payload.get("history", [])
        if not query:
            self._write_json({"error": "missing query"}, status=HTTPStatus.BAD_REQUEST)
            return
        response = ENGINE.search(query, history=history if isinstance(history, list) else None)
        self._write_response(response)

    def log_message(self, format, *args):  # noqa: A003
        return

    def _write_response(self, response) -> None:
        self._write_json(
            {
                "intent": response.intent,
                "intent_confidence": response.intent_confidence,
                "intent_method": response.intent_method,
                "intent_scores": response.intent_scores,
                "response_backend": response.response_backend,
                "response_model": response.response_model,
                "query_plan": response.query_plan,
                "answer": response.answer,
                "events": response.events,
            }
        )

    def _write_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), DemoHandler)
    print("Demo server is running at http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
