# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Importing nxmndr.server or one of its submodules does not import server.py.

An eager package __init__ made every importer of
nxmndr.server.managers (such as the spawned PyTorch RPC worker) load the whole
server stack.
"""

from __future__ import annotations

import json
import subprocess
import sys

PROBE = """
import json, sys
import nxmndr.server
after_package = "nxmndr.server.server" in sys.modules
import nxmndr.server.managers
after_managers = "nxmndr.server.server" in sys.modules
aiohttp_after_managers = "aiohttp" in sys.modules
from nxmndr.server import create_server, run_unified_server, serve, stop_server
import nxmndr.server.server as module
resolved = all(f is getattr(module, f.__name__)
               for f in (create_server, run_unified_server, serve, stop_server))
try:
    nxmndr.server.no_such_name
    missing = "no error"
except AttributeError:
    missing = "AttributeError"
print(json.dumps(dict(after_package=after_package, after_managers=after_managers,
                      aiohttp_after_managers=aiohttp_after_managers, resolved=resolved,
                      missing=missing, all=sorted(nxmndr.server.__all__))))
"""


def test_server_module_loads_only_when_an_entry_point_is_requested():
    out = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=300, check=True
    )
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result == {
        "after_package": False,
        "after_managers": False,
        "aiohttp_after_managers": False,
        "resolved": True,
        "missing": "AttributeError",
        "all": ["create_server", "run_unified_server", "serve", "stop_server"],
    }
