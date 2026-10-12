# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Server package.

The server entry points live in ``nxmndr.server.server`` (torch, grpc, aiohttp) and
are imported only when one of them is requested, so importing a submodule such as
``nxmndr.server.managers`` (for example in a spawned PyTorch RPC worker) does not
load the whole server stack.
"""

_LAZY_SERVER_NAMES = ("serve", "run_unified_server", "create_server", "stop_server")

__all__ = list(_LAZY_SERVER_NAMES)


def __getattr__(name):
    if name in _LAZY_SERVER_NAMES:
        from . import server as _server

        return getattr(_server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
