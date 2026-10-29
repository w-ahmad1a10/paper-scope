"""Web-based graphical user interface for whole-paper semantic search."""

from __future__ import annotations

import json
import socket
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .build import add_paper, build_index
from .config import DEFAULT_MODEL, INDEX_DIR
from .embed import Embedder
from .extract import extract_paper
from .index_store import IndexError
from .search import Searcher

STATIC_DIR = Path(__file__).parent / "gui_static"
HTML_FILE = STATIC_DIR / "index.html"

_searcher: Searcher | None = None
_model_name = DEFAULT_MODEL


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self) -> None:
        if not HTML_FILE.is_file():
            body = b"<html><body><h1>GUI files not found</h1></body></html>"
            self.send_response(404)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = HTML_FILE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path in ("/", "/index.html"):
            self._send_html()
            return

        if path == "/api/status":
            count = 0
            model = _model_name
            if _searcher is not None:
                count = _searcher.index.manifest.get("paper_count", 0)
                model = _searcher.index.model_name
            self._send_json({"status": "ok", "paper_count": count, "model": model})
            return

        if path == "/api/papers":
            if _searcher is None:
                self._send_json({"papers": []})
                return
            papers = []
            for paper in _searcher.index.papers:
                papers.append(
                    {
                        "key": paper.get("key", ""),
                        "title": paper.get("title", ""),
                        "year": paper.get("year"),
                        "arxiv_id": paper.get("arxiv_id"),
                        "truncated": bool(paper.get("truncated", False)),
                        "truncation_rate": paper.get("truncation_rate"),
                    }
                )
            self._send_json({"papers": papers})
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        global _searcher
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        data = parse_qs(body.decode("utf-8"))

        if path == "/api/ask":
            question = data.get("question", [""])[0].strip()
            if not question:
                self._send_json({"error": "Question is empty"}, 400)
                return
            if _searcher is None:
                self._send_json({"error": "No index loaded"}, 400)
                return
            try:
                hits = _searcher.search_all(question)
            except Exception as exc:  # noqa: BLE001 - surface search errors.
                self._send_json({"error": str(exc)}, 400)
                return

            results = []
            for rank, hit in enumerate(hits, start=1):
                p = hit.paper
                text_path = p.get("text_path", "")
                preview = ""
                if text_path:
                    try:
                        full = Path(text_path).read_text(encoding="utf-8")
                        preview = full[:1500]
                        if len(full) > 1500:
                            preview += "\n... (preview truncated)"
                    except OSError:
                        preview = "(Could not load text preview)"
                results.append(
                    {
                        "rank": rank,
                        "score": round(hit.score, 6),
                        "key": p["key"],
                        "title": p["title"],
                        "year": p.get("year"),
                        "arxiv_id": p.get("arxiv_id"),
                        "truncated": bool(p.get("truncated", False)),
                        "truncation_rate": p.get("truncation_rate"),
                        "preview": preview,
                    }
                )
            self._send_json({"results": results})
            return

        if path == "/api/ask-document":
            pdf_path = data.get("path", [""])[0].strip()
            if not pdf_path:
                self._send_json({"error": "PDF path is required"}, 400)
                return
            if _searcher is None:
                self._send_json({"error": "No index loaded"}, 400)
                return
            try:
                pdf_path_obj = Path(pdf_path)
                if not pdf_path_obj.exists():
                    self._send_json({"error": f"PDF not found: {pdf_path}"}, 400)
                    return

                extracted = extract_paper(pdf_path_obj)
                embedder = Embedder(_model_name)
                query_vector = embedder.encode_one(extracted.text)

                hits = _searcher.index.all_scores(query_vector)

                results = []
                for rank, hit in enumerate(hits, start=1):
                    p = hit.paper
                    text_path = p.get("text_path", "")
                    preview = ""
                    if text_path:
                        try:
                            full = Path(text_path).read_text(encoding="utf-8")
                            preview = full[:1500]
                            if len(full) > 1500:
                                preview += "\n... (preview truncated)"
                        except OSError:
                            preview = "(Could not load text preview)"
                    results.append(
                        {
                            "rank": rank,
                            "score": round(hit.score, 6),
                            "key": p["key"],
                            "title": p["title"],
                            "year": p.get("year"),
                            "arxiv_id": p.get("arxiv_id"),
                            "truncated": bool(p.get("truncated", False)),
                            "truncation_rate": p.get("truncation_rate"),
                            "preview": preview,
                        }
                    )
                self._send_json({"results": results})
                return
            except Exception as exc:  # noqa: BLE001 - surface search errors.
                self._send_json({"error": str(exc)}, 400)
                return

        if path == "/api/add":
            pdf_path = data.get("path", [""])[0].strip()
            name = data.get("name", [""])[0].strip()
            if not pdf_path or not name:
                self._send_json({"error": "Path and name are required"}, 400)
                return

            try:
                manifest = add_paper(
                    title=name,
                    source_path=Path(pdf_path),
                    model_name=_model_name,
                    index_dir=INDEX_DIR,
                )
                _searcher = Searcher()
                self._send_json(
                    {
                        "ok": True,
                        "paper_count": manifest["paper_count"],
                        "message": f"Collection now has {manifest['paper_count']} papers.",
                    }
                )
            except Exception as exc:  # noqa: BLE001 - surface add errors.
                self._send_json({"error": str(exc)}, 400)
            return

        if path == "/api/rebuild":
            try:
                build_index(model_name=_model_name, index_dir=INDEX_DIR)
                _searcher = Searcher()
                count = _searcher.index.manifest.get("paper_count", 0)
                self._send_json(
                    {
                        "ok": True,
                        "paper_count": count,
                        "message": "Index rebuilt.",
                    }
                )
            except Exception as exc:  # noqa: BLE001 - surface rebuild errors.
                self._send_json({"error": str(exc)}, 400)
            return

        self.send_response(404)
        self.end_headers()


def _load_index() -> None:
    global _searcher, _model_name
    try:
        _searcher = Searcher()
        _model_name = _searcher.index.model_name
    except IndexError:
        _searcher = None
        _model_name = DEFAULT_MODEL


def launch_gui() -> None:
    _load_index()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"

    def serve() -> None:
        server.serve_forever()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    webbrowser.open(url)

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
