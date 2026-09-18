"""Exercise the real ASGI routes without a network connection or a new client dependency."""
import asyncio
import json

from isr.config import DATA
from server import app, engine


def request(path, method="GET", body=b"", content_type=None, headers=None):
    sent = []
    delivered = False
    path, _, query = path.partition("?")
    raw_headers = [(b"host", b"localhost:8000"), (b"content-length", str(len(body)).encode())]
    if content_type:
        raw_headers.append((b"content-type", content_type.encode()))
    raw_headers.extend(headers or [])
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
             "http_version": "1.1", "method": method, "scheme": "http", "path": path,
             "raw_path": path.encode(), "query_string": query.encode(), "root_path": "",
             "headers": raw_headers, "client": ("127.0.0.1", 12345), "server": ("localhost", 8000)}
    async def receive():
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}
    async def send(message):
        sent.append(message)
    asyncio.run(app(scope, receive, send))
    status = next(message["status"] for message in sent if message["type"] == "http.response.start")
    data = b"".join(message.get("body", b"") for message in sent)
    return status, data


def multipart(fields, image=None):
    boundary = "lookalike-test-boundary"
    chunks = []
    for key, value in fields.items():
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    if image is not None:
        chunks.extend([f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="test.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode(), image, b"\r\n"])
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def test_frontend_and_overview():
    status, html = request("/")
    assert status == 200
    assert b"/assets/styles.css" in html and b"/assets/app.js" in html
    status, data = request("/api/overview")
    overview = json.loads(data)
    assert status == 200
    assert overview["gallery_count"] == 400 and overview["query_count"] == 100
    assert len(overview["examples"]) == 100


def test_image_search_feedback_and_reset():
    example = next(row for row in engine().queries if row.category == "cup")
    body, kind = multipart({"query_id": example.sha256, "method": "clip", "k": 5})
    status, data = request("/api/search", "POST", body, kind)
    result = json.loads(data)["searches"][0]
    assert status == 200
    assert len(result["results"]) == 5
    assert result["results"][0]["category"] == "cup"
    assert all(row["split"] == "gallery" for row in result["results"])
    for reset in [False, True]:
        payload = {"token": result["token"], "selected": [result["results"][0]["index_id"]], "reset": reset}
        status, data = request("/api/feedback", "POST", json.dumps(payload).encode(), "application/json")
        refined = json.loads(data)
        assert status == 200 and refined["refined"] is not reset


def test_upload_compare_and_invalid_image():
    example = next(row for row in engine().queries if row.category == "laptop")
    body, kind = multipart({"method": "both", "k": 5}, (DATA / example.path).read_bytes())
    status, data = request("/api/search", "POST", body, kind)
    results = json.loads(data)["searches"]
    assert status == 200 and {row["method"] for row in results} == {"clip", "histogram"}
    assert all(row["precision"] is None for row in results)
    body, kind = multipart({"method": "clip"}, b"not an image")
    status, data = request("/api/search", "POST", body, kind)
    assert status == 400 and "could not be read" in json.loads(data)["error"]


def test_unknown_method_gallery_query_and_bad_feedback_rejected():
    for fields in [{"query_id": engine().queries[0].sha256, "method": "unknown"},
                   {"query_id": engine().gallery[0].sha256, "method": "clip"},
                   {"query_id": engine().queries[0].sha256, "k": "-1"}]:
        body, kind = multipart(fields)
        status, _ = request("/api/search", "POST", body, kind)
        assert status == 400
    status, _ = request("/api/feedback", "POST", b'{"token":"invalid","selected":[-1]}', "application/json")
    assert status == 400


def test_collection_and_evaluation():
    status, data = request("/api/collection?category=cup&split=queries&page=1")
    collection = json.loads(data)
    assert status == 200 and collection["total"] == 10
    assert all(row["category"] == "cup" and row["split"] == "queries" for row in collection["images"])
    status, data = request("/api/evaluation")
    assert status == 200 and json.loads(data)["available"]
    status, data = request("/api/download/summary")
    assert status == 200 and b"precision_at_10" in data


def test_image_access_and_request_guards():
    status, data = request(f"/api/images/{engine().gallery[0].sha256}?thumbnail=1")
    assert status == 200 and data.startswith(b"\xff\xd8")
    status, _ = request("/api/images/not-a-dataset-id")
    assert status == 404
    status, _ = request("/api/search", "POST", headers=[(b"origin", b"https://example.com")])
    assert status == 403
    status, _ = request("/api/search", "POST", headers=[(b"content-length", b"999999999")])
    assert status == 413
