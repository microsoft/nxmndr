# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""A PyTorch RPC worker target that proves a spawned child ran it.

``marker_worker`` has the signature of ``managers._run_rpc_worker``. It writes the
file named by ``NXMNDR_TEST_RPC_MARKER`` and ends the child there, before the
readiness handshake, so no torch RPC rendezvous is attempted. It lives in an
importable module because a ``spawn`` child unpickles its target by reference.
"""

from __future__ import annotations

import os
from pathlib import Path

MARKER_ENV = "NXMNDR_TEST_RPC_MARKER"


def marker_worker(master_addr, master_port, ready_port, startup_timeout_s, device="cpu"):
    Path(os.environ[MARKER_ENV]).write_text(
        f"pid={os.getpid()} ready_port={ready_port}", encoding="utf-8"
    )
    raise SystemExit(0)
