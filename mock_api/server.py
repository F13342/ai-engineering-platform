import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


DATA_FILE = Path(__file__).parent.parent / "data" / "raw" / "api_students.json"


class StudentAPIHandler(BaseHTTPRequestHandler):
    """Serve student data through a local REST API."""

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path != "/students":
            self.send_error(404, "Endpoint not found")
            return

        try:
            data = json.loads(
                DATA_FILE.read_text(encoding="utf-8")
            )
        except (FileNotFoundError, json.JSONDecodeError):
            self.send_error(500, "Unable to load student data")
            return

        response = json.dumps(data).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()

        self.wfile.write(response)

    def log_message(self, format: str, *args: object) -> None:
        """Suppress default HTTP server logging."""
        return


def main() -> None:
    """Start the local student REST API."""
    server = HTTPServer(("localhost", 8000), StudentAPIHandler)

    print("Mock Student API running at http://localhost:8000/students")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nMock Student API stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()