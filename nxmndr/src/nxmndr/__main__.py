# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""nxmndr command-line interface.

Provides simple discovery and dispatch to common entry points.

Examples:
  python -m nxmndr --help
  python -m nxmndr server --port 8080 --azure-proxy
  python -m nxmndr grpc --grpc-port 50051 --grpc-only
"""

from __future__ import annotations

import argparse
import sys

from nxmndr import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nxmndr", description="Multi-format vision inference toolkit")
    p.add_argument("--version", action="version", version=f"nxmndr {__version__}")
    sub = p.add_subparsers(dest="command")

    # Unified server (aiohttp + optional azure proxy)
    sp_server = sub.add_parser(
        "server", help="Run unified HTTP server (inference + optional Azure proxy)"
    )
    sp_server.add_argument("--port", type=int, default=8080)
    sp_server.add_argument(
        "--host", default="127.0.0.1", help="bind address (default: 127.0.0.1, this machine only)"
    )
    sp_server.add_argument(
        "--azure-proxy", action="store_true", help="Enable Azure OpenAI proxy endpoints"
    )

    # Legacy gRPC server
    sp_grpc = sub.add_parser("grpc", help="Run legacy gRPC inference server only")
    sp_grpc.add_argument("--grpc-port", type=int, default=50051)
    sp_grpc.add_argument(
        "--host", default="127.0.0.1", help="bind address (default: 127.0.0.1, this machine only)"
    )

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "server":
        from nxmndr.server import run_unified_server
        import asyncio

        asyncio.run(
            run_unified_server(host=args.host, port=args.port, include_azure_proxy=args.azure_proxy)
        )
    elif args.command == "grpc":
        from nxmndr.server import serve

        serve(port=args.grpc_port, host=args.host)
    else:
        build_parser().print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
