import asyncio

import httpx

from ai_platform.api.app import create_app


def make_request(app, method: str, path: str, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_chat_page_returns_html() -> None:
    app = create_app()

    response = make_request(app, "GET", "/chat")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert response.headers["cache-control"] == "no-store"

    html = response.text
    assert 'id="apiKey"' in html
    assert 'id="messages"' in html
    assert 'id="prompt"' in html
    assert 'X-API-Key' in html
    assert '/v1/language-model/generate' in html
    assert "localStorage" not in html
    assert "sessionStorage" not in html


def test_chat_page_is_not_listed_as_an_api_operation() -> None:
    app = create_app()
    schema = app.openapi()

    assert "/chat" not in schema["paths"]
    assert "/v1/language-model/generate" in schema["paths"]
