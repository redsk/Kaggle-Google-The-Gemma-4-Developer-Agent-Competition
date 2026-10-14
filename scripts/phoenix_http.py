"""HTTP-only entry point for pinned Phoenix 20.21.4.

Upstream's gRPC listener binds [::] independently of PHOENIX_HOST. We export
OTLP/HTTP, so disable that unused listener in this process. Installed Phoenix
files remain untouched. The normal Phoenix CLI handles SQLite, HTTP and the UI.
"""
from unittest.mock import patch
from importlib.metadata import version

if version("arize-phoenix") != "20.21.4":
    raise SystemExit("Review the HTTP-only launcher before changing the pinned Phoenix version.")

from phoenix.server.grpc_server import GrpcServer
from phoenix.server.main import main


class HttpOnlyGrpcServer(GrpcServer):
    async def __aenter__(self):
        return None


if __name__ == "__main__":
    with patch("phoenix.server.app.GrpcServer", HttpOnlyGrpcServer):
        main()
