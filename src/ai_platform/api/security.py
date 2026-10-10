"""API key authentication for protected administrative endpoints."""

import hmac
import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader


api_key_header = APIKeyHeader(
    name="X-API-Key",
    scheme_name="APIKeyAuth",
    auto_error=False,
)


def require_api_key(
    api_key: str | None = Security(api_key_header),
) -> None:
    """Reject administrative requests without a valid configured API key."""
    configured_key = os.environ.get("AI_PLATFORM_API_KEY")

    if not configured_key:
        raise HTTPException(
            status_code=503,
            detail="Administrative API authentication is not configured.",
        )

    if api_key is None or not hmac.compare_digest(api_key, configured_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
        )
