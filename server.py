"""Local HTTP API and static HTML frontend for Lookalike.

Run: .venv/Scripts/python.exe server.py
"""
from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from io import BytesIO
import json
import secrets
import threading
from time import monotonic, perf_counter
from urllib.parse import urlsplit

from PIL import Image, ImageOps
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.formparsers import MultiPartParser
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
import uvicorn

from isr.config import ARTIFACTS, DATA, INDEXES, METHODS, MODEL_PATH, ROOT, label
from isr.dataset import fingerprint, image_path, load_records
from isr.evaluation import evaluate
from isr.features import Encoder, MAX_UPLOAD_BYTES, open_rgb
from isr.retrieval import SearchIndex, build_index, refine_query

# Valid uploads stay in RAM; the request guard caps the entire multipart body.
MultiPartParser.spool_max_size = MAX_UPLOAD_BYTES + 1024 * 1024


class LocalEngine:
    def __init__(self):
        self.records = load_records()
        self.by_id = {row.sha256: row for row in self.records}
        self.gallery = [row for row in self.records if row.split == "gallery"]
        self.queries = [row for row in self.records if row.split == "queries"]
        self.dataset_id = fingerprint(self.records)
        self.indexes = {}
        self.encoders = {}
        self.sessions = OrderedDict()
        self.lock = threading.RLock()
        self.maintenance = threading.Lock()

    def card(self, row):
        return {"id": row.sha256, "category": row.category, "label": label(row.category),
                "split": row.split, "filename": image_path(row).name,
                "image_url": f"/api/images/{row.sha256}", "thumbnail_url": f"/api/images/{row.sha256}?thumbnail=1"}

    def resources(self, method):
        if method not in METHODS:
            raise ValueError("Choose CLIP semantic search or the color baseline.")
        metadata = INDEXES / method / "metadata.json"
        if not metadata.is_file():
            raise FileNotFoundError(f"The {method} index is missing. Build it from the Collection panel.")
        key = (metadata.stat().st_mtime_ns, MODEL_PATH.stat().st_mtime_ns if method == "clip" and MODEL_PATH.exists() else 0)
        with self.lock:
            if method not in self.indexes or self.indexes[method][0] != key:
                self.indexes[method] = (key, SearchIndex(method, self.records))
                self.encoders[method] = Encoder(method)
            return self.indexes[method][1], self.encoders[method]

    def overview(self):
        categories = sorted({row.category for row in self.gallery}, key=label)
        return {"gallery_count": len(self.gallery), "query_count": len(self.queries),
                "categories": [{"id": c, "label": label(c),
                                "gallery_count": sum(r.category == c for r in self.gallery)} for c in categories],
                "examples": [self.card(row) for row in self.queries],
                "showcase": [self.card(next(row for row in self.gallery if row.category == c))
                             for c in ["cup", "laptop", "butterfly", "watch"] if c in categories],
                "methods": [{"id": m, "label": name, "ready": (INDEXES / m / "metadata.json").is_file()
                             and (m != "clip" or MODEL_PATH.is_file())} for m, name in METHODS.items()]}

    def search(self, image, method, k, category=None):
        index, encoder = self.resources(method)
        start = perf_counter()
        vector = encoder.encode([image])[0]
        encoded = perf_counter()
        results = index.search(vector, k)
        end = perf_counter()
        token = secrets.token_urlsafe(24)
        with self.lock:
            now = monotonic()
            for key in list(self.sessions):
                if now - self.sessions[key]["created"] > 1800:
                    self.sessions.pop(key)
            while len(self.sessions) >= 128:
                self.sessions.popitem(last=False)
            self.sessions[token] = {"vector": vector, "method": method, "k": k, "category": category,
                                    "ids": {r["index_id"] for r in results}, "created": now}
        return self.result_payload(results, method, token, (encoded - start) * 1000,
                                   (end - encoded) * 1000, category)

    def result_payload(self, results, method, token, encode_ms, search_ms, category, refined=False):
        cards = []
        for result in results:
            cards.append({**self.card(self.by_id[result["sha256"]]), "rank": result["rank"],
                          "score": result["score"], "index_id": result["index_id"]})
        return {"method": method, "method_label": METHODS[method], "token": token,
                "results": cards, "encode_ms": encode_ms, "search_ms": search_ms,
                "total_ms": encode_ms + search_ms, "refined": refined,
                "precision": sum(r["category"] == category for r in cards) / len(cards) if category else None}

    def feedback(self, token, selected, reset=False):
        with self.lock:
            session = self.sessions.get(token)
            if not session or monotonic() - session["created"] > 1800:
                raise ValueError("This search has expired. Search your image again.")
            if not isinstance(selected, list) or any(type(idx) is not int for idx in selected):
                raise ValueError("Select valid result images.")
            if not reset and (not selected or not set(selected).issubset(session["ids"])):
                raise ValueError("Select at least one image from the current results.")
            index, _ = self.resources(session["method"])
            start = perf_counter()
            vector = session["vector"] if reset else refine_query(session["vector"], index.vectors[selected])
            results = index.search(vector, session["k"])
            elapsed = (perf_counter() - start) * 1000
            session["ids"] = {row["index_id"] for row in results}
            return self.result_payload(results, session["method"], token, 0, elapsed,
                                       session["category"], refined=not reset)

    def report(self):
        path = ARTIFACTS / "evaluation.json"
        if not path.exists():
            return {"available": False, "reason": "Run the evaluation to measure both search methods."}
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["dataset_fingerprint"] != self.dataset_id:
            return {"available": False, "reason": "The dataset changed. Rebuild the indexes and run evaluation again."}
        for summary in report["summaries"]:
            meta = INDEXES / summary["method"] / "metadata.json"
            if not meta.exists() or json.loads(meta.read_text())["built_at"] != summary["index_built_at"]:
                return {"available": False, "reason": "The indexes changed. Run evaluation to refresh the report."}
        return {"available": True, **report}


@lru_cache(maxsize=1)
def engine():
    return LocalEngine()


@lru_cache(maxsize=600)
def image_thumbnail(image_id):
    row = engine().by_id.get(image_id)
    if row is None:
        raise HTTPException(404, "Image not found.")
    img = ImageOps.contain(open_rgb(image_path(row)), (560, 440), method=Image.Resampling.LANCZOS)
    out = BytesIO()
    img.save(out, "JPEG", quality=88)
    return out.getvalue()


async def home(request):
    return FileResponse(ROOT / "frontend/index.html", headers={"Cache-Control": "no-cache"})


async def overview(request):
    return JSONResponse(await run_in_threadpool(lambda: engine().overview()))


async def dataset_image(request):
    image_id = request.path_params["image_id"]
    local = await run_in_threadpool(engine)
    row = local.by_id.get(image_id)
    if not row:
        raise HTTPException(404, "Image not found.")
    headers = {"Cache-Control": "public, max-age=86400"}
    if request.query_params.get("thumbnail") == "1":
        return Response(await run_in_threadpool(image_thumbnail, image_id), media_type="image/jpeg", headers=headers)
    return FileResponse(image_path(row), headers=headers)


def integer(value, name, minimum, maximum):
    try:
        number = int(value)
    except (ValueError, TypeError):
        raise ValueError(f"{name} must be a whole number.") from None
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return number


async def search(request):
    local = await run_in_threadpool(engine)
    async with request.form(max_files=1, max_fields=4, max_part_size=MAX_UPLOAD_BYTES) as form:
        method = str(form.get("method", "clip"))
        if method not in {"clip", "histogram", "both"}:
            raise ValueError("Unknown search method.")
        k = integer(form.get("k", 10), "Number of matches", 1, 20)
        query_id = form.get("query_id")
        uploaded = form.get("image")
        category = None
        if uploaded and query_id:
            raise ValueError("Choose one query image.")
        if isinstance(uploaded, UploadFile):
            content = await uploaded.read(MAX_UPLOAD_BYTES + 1)
            image = await run_in_threadpool(open_rgb, content)
        elif isinstance(query_id, str) and query_id in local.by_id:
            row = local.by_id[query_id]
            if row.split != "queries":
                raise ValueError("Choose a held-out example query.")
            image = await run_in_threadpool(open_rgb, image_path(row))
            category = row.category
        else:
            raise ValueError("Upload an image or choose an example first.")
        methods = ["histogram", "clip"] if method == "both" else [method]
        output = []
        for choice in methods:
            output.append(await run_in_threadpool(local.search, image, choice, k, category))
        return JSONResponse({"searches": output})


async def feedback(request):
    data = await request.json()
    if not isinstance(data, dict) or not isinstance(data.get("token"), str):
        raise ValueError("A valid search is required for feedback.")
    reset = data.get("reset", False)
    if type(reset) is not bool:
        raise ValueError("Invalid reset value.")
    result = await run_in_threadpool(lambda: engine().feedback(data["token"], data.get("selected", []), reset))
    return JSONResponse(result)


async def collection(request):
    local = await run_in_threadpool(engine)
    split = request.query_params.get("split", "gallery")
    if split not in {"gallery", "queries"}:
        raise ValueError("Unknown image split.")
    category = request.query_params.get("category", "all")
    if category != "all" and category not in {r.category for r in local.records}:
        raise ValueError("Unknown category.")
    page = integer(request.query_params.get("page", 1), "Page", 1, 100)
    rows = [r for r in local.records if r.split == split and (category == "all" or r.category == category)]
    pages = max(1, (len(rows) + 11) // 12)
    if page > pages:
        raise ValueError("This page does not exist.")
    return JSONResponse({"images": [local.card(r) for r in rows[(page - 1) * 12:page * 12]],
                         "page": page, "pages": pages, "total": len(rows)})


async def report(request):
    return JSONResponse(await run_in_threadpool(lambda: engine().report()))


async def run_evaluation(request):
    local = await run_in_threadpool(engine)
    if not local.maintenance.acquire(blocking=False):
        raise HTTPException(409, "Another evaluation or index build is running. Try again shortly.")
    try:
        await run_in_threadpool(evaluate)
        return JSONResponse(local.report())
    finally:
        local.maintenance.release()


async def rebuild(request):
    method = request.path_params["method"]
    if method not in METHODS:
        raise ValueError("Unknown search method.")
    local = await run_in_threadpool(engine)
    if not local.maintenance.acquire(blocking=False):
        raise HTTPException(409, "Another evaluation or index build is running. Try again shortly.")
    try:
        metadata = await run_in_threadpool(build_index, method)
        with local.lock:
            local.indexes.pop(method, None)
            local.sessions.clear()
        return JSONResponse({"message": f"{METHODS[method]} index is ready.", "metadata": metadata})
    finally:
        local.maintenance.release()


async def download(request):
    files = {"summary": "evaluation-summary.csv", "queries": "evaluation-queries.csv", "categories": "evaluation-categories.csv"}
    name = files.get(request.path_params["name"])
    if name is None or not (ARTIFACTS / name).exists():
        raise HTTPException(404, "Run evaluation before downloading this report.")
    if not (await run_in_threadpool(lambda: engine().report()))["available"]:
        raise HTTPException(409, "Run evaluation to refresh the report before downloading it.")
    return FileResponse(ARTIFACTS / name, filename=name, media_type="text/csv")


async def error_response(request, error):
    if isinstance(error, HTTPException):
        return JSONResponse({"error": error.detail}, status_code=error.status_code)
    if isinstance(error, FileNotFoundError):
        return JSONResponse({"error": str(error)}, status_code=409)
    if isinstance(error, ValueError):
        return JSONResponse({"error": str(error)}, status_code=400)
    return JSONResponse({"error": "The request could not be completed. Check the local server log and try again."}, status_code=500)


class LocalRequestGuard:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        if scope["method"] in {"POST", "PUT", "DELETE"}:
            origin = headers.get(b"origin", b"").decode()
            if origin and urlsplit(origin).netloc != headers.get(b"host", b"").decode():
                return await JSONResponse({"error": "Use this app from its local address."}, status_code=403)(scope, receive, send)
        maximum = MAX_UPLOAD_BYTES + 1024 * 1024
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            length = maximum + 1
        if length > maximum:
            return await JSONResponse({"error": "Upload an image smaller than 10 MB."}, status_code=413)(scope, receive, send)
        received = 0
        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > maximum:
                raise HTTPException(413, "Upload an image smaller than 10 MB.")
            return message
        async def secure_send(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).extend([
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"content-security-policy", b"default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")])
            await send(message)
        await self.app(scope, limited_receive, secure_send)


app = Starlette(routes=[Route("/", home), Route("/api/overview", overview),
    Route("/api/images/{image_id}", dataset_image), Route("/api/search", search, methods=["POST"]),
    Route("/api/feedback", feedback, methods=["POST"]), Route("/api/collection", collection),
    Route("/api/evaluation", report), Route("/api/evaluation", run_evaluation, methods=["POST"]),
    Route("/api/indexes/{method}/build", rebuild, methods=["POST"]),
    Route("/api/download/{name}", download), Mount("/assets", StaticFiles(directory=ROOT / "frontend"), name="assets")],
    middleware=[Middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"]), Middleware(LocalRequestGuard)],
    exception_handlers={HTTPException: error_response, ValueError: error_response,
                        FileNotFoundError: error_response, Exception: error_response})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
