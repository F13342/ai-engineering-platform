import asyncio
from pathlib import Path

import httpx
import pytest
from docx import Document

from ai_platform.api.app import create_app
from ai_platform.document_generation import DocumentGenerationService


@pytest.fixture(autouse=True)
def configure_api_key(monkeypatch):
    monkeypatch.setenv("AI_PLATFORM_API_KEY", "test-secret")


def make_request(app, method: str, path: str, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_generate_arabic_word_document_returns_download(
    tmp_path: Path,
) -> None:
    service = DocumentGenerationService(output_dir=tmp_path)
    app = create_app(document_generation_service=service)

    response = make_request(
        app,
        "POST",
        "/v1/documents/generate",
        headers={"X-API-Key": "test-secret"},
        json={
            "title": "تقرير اختبار",
            "content": "# مقدمة\nهذا مستند عربي تجريبي.\n- النقطة الأولى",
            "output_format": "docx",
            "language": "ar",
        },
    )

    assert response.status_code == 200
    assert "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in (
        response.headers["content-type"]
    )
    assert response.content.startswith(b"PK")
    assert len(response.content) > 100

    saved_files = list(tmp_path.glob("*.docx"))
    assert len(saved_files) == 1

    document = Document(saved_files[0])
    extracted_text = "\n".join(p.text for p in document.paragraphs)
    assert "تقرير اختبار" in extracted_text
    assert "هذا مستند عربي تجريبي." in extracted_text


def test_document_generation_rejects_request_without_api_key(
    tmp_path: Path,
) -> None:
    app = create_app(
        document_generation_service=DocumentGenerationService(
            output_dir=tmp_path
        )
    )

    response = make_request(
        app,
        "POST",
        "/v1/documents/generate",
        json={
            "title": "Test",
            "content": "Some content",
            "output_format": "docx",
            "language": "en",
        },
    )

    assert response.status_code == 401
    assert list(tmp_path.iterdir()) == []


def test_document_generation_rejects_unsupported_format(
    tmp_path: Path,
) -> None:
    app = create_app(
        document_generation_service=DocumentGenerationService(
            output_dir=tmp_path
        )
    )

    response = make_request(
        app,
        "POST",
        "/v1/documents/generate",
        headers={"X-API-Key": "test-secret"},
        json={
            "title": "Test",
            "content": "Some content",
            "output_format": "txt",
            "language": "en",
        },
    )

    assert response.status_code == 422
    assert list(tmp_path.iterdir()) == []
