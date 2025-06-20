import os
import importlib
from pathlib import Path

import pytest
faiss = pytest.importorskip("faiss")

# Set environment variables for temporary paths in each test


@pytest.fixture(name="client")
def flask_client(tmp_path, monkeypatch):
    os.environ["INDEX_FILE"] = str(tmp_path / "index.faiss")
    os.environ["METADATA_FILE"] = str(tmp_path / "meta.pkl")
    os.environ["CHUNKS_FILE"] = str(tmp_path / "chunks.pkl")
    os.environ["UPLOAD_FOLDER"] = str(tmp_path / "uploads")
    # Import/reload after setting env vars
    import web_app
    importlib.reload(web_app)

    def fake_create(**_kwargs):
        class Choice:
            def __init__(self):
                self.message = type("M", (), {"content": "ok"})()
        return type("R", (), {"choices": [Choice()]})()

    monkeypatch.setattr(web_app.client.chat.completions, "create", fake_create)
    return web_app.app.test_client()


def test_home_page(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_upload_updates_index(client):
    pdf_path = Path("pdfs") / "Sai Teja Resume.pdf"
    with pdf_path.open("rb") as f:
        resp = client.post("/upload", data={"file": (f, "resume.pdf")}, content_type="multipart/form-data")
    assert resp.status_code == 200
    idx = faiss.read_index(os.environ["INDEX_FILE"])
    assert idx.ntotal > 0


def test_chat_endpoint(client):
    pdf_path = Path("pdfs") / "Sai Teja Resume.pdf"
    with pdf_path.open("rb") as f:
        client.post("/upload", data={"file": (f, "resume.pdf")}, content_type="multipart/form-data")
    resp = client.post("/chat", json={"question": "test"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "answer" in data
