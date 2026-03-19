"""CLI entry point for the Segmentum Personality Analysis API server.

Usage::

    python -m segmentum.api_cli --port 8000
    python -m segmentum.api_cli --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Segmentum Personality Analyzer API server",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is required to run the API server.\n"
            "Install with: pip install segmentum[api]",
            file=sys.stderr,
        )
        sys.exit(1)

    uvicorn.run(
        "segmentum.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
