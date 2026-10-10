"""Application entry point for the AI Engineering Platform."""

import uvicorn


class Application:
    """Configure and launch the AI Platform API server."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535.")

        self.host = host
        self.port = port

    def run(self) -> None:
        """Start the API server."""
        uvicorn.run(
            "ai_platform.api.app:app",
            host=self.host,
            port=self.port,
            reload=False,
        )
