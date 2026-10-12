# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Model doubles for a separate worker process.

Plugin integration tests run a real nxmndr gRPC server in the isolated worker
environment and drive it from QGIS over OpenSession / StreamPredict / CloseSession.
This module puts deterministic doubles below the production load path of such a
process, so the server, the shared dispatcher, the SAM handler, the YOLO adapter,
the registry loaders, ``build_model_record`` and the ModelManager cache and leases
all run unchanged:

- Hugging Face repo :data:`SAM_DOUBLE_REPO` loads the SAM 3 double of
  ``tst.support.stream_doubles`` (``FakeHFSamModel``: config ``model_type`` ``sam3``,
  text prompts ``field`` / ``nothing`` / anything else, and an unprompted mask stack).
  Every other repo still goes to the real Hugging Face loader.
- ``nxmndr.models.sam.load_sam_variant`` returns the tracker double for geometry
  prompts (box area plus 5x5 squares at positive points, minus squares at negative
  points) and counts loads and disposals.
- ``ultralytics`` is ``tst.support.ultralytics_double`` (``YOLO(path)`` reads a
  checkpoint written by ``ultralytics_double.write_checkpoint``).

Installation is explicit test bootstrap, never a production fallback: call
:func:`install` in the process, or start the process through this module. With the
backend's ``src`` directory and the directory that contains ``tst`` on
``PYTHONPATH``, and the worker's own Python:

``python -m tst.support.worker_doubles [--stats FILE] serve [--cache-dir DIR]``
    Installs the doubles, starts ``nxmndr.server.server.create_server`` on
    ``127.0.0.1:0`` and prints one JSON line ``{"endpoint": "127.0.0.1:<port>",
    "pid": <pid>}`` to stdout; logs go to stderr. It stops (``stop_server``) when
    stdin closes or receives a line ``stop``.

``python -m tst.support.worker_doubles [--stats FILE] run MODULE [ARGS...]``
    Installs the doubles (their logs go to stderr), then runs ``MODULE`` as
    ``__main__`` with ``ARGS`` (for example the plugin's ``nxmndr_worker``) with the
    original stdout, so a module that speaks a protocol on stdout keeps it clean.

``--stats FILE`` writes counters as JSON when the process exits: YOLO checkpoints
opened, the distinct YOLO ``conf``/``iou``/``max_det`` predict arguments, SAM doubles built, SAM variant loads and disposals per (variant, device),
unprompted SAM calls. Tests use them to prove that models load once per job.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import runpy
import sys
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

SAM_DOUBLE_REPO = "nxmndr-test/sam3-double"


@dataclass
class Installed:
    """Handles to the installed doubles (counters are read by :meth:`stats`)."""

    sam_log: List[Dict[str, object]] = field(default_factory=list)
    sam_models: List[object] = field(default_factory=list)
    variants: Optional[object] = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def stats(self) -> Dict[str, object]:
        from tst.support.ultralytics_double import FakeYOLO

        variants = self.variants
        with self.lock:
            sam_models = list(self.sam_models)
        options = []
        with FakeYOLO.lock:
            for entry in FakeYOLO.predict_options:
                if entry not in options:
                    options.append(dict(entry))
        return {
            "yolo_checkpoints_opened": list(FakeYOLO.constructed),
            "yolo_predict_options": options,  # distinct conf/iou/max_det arguments, first-seen order
            "sam_models_built": len(sam_models),
            "sam_unprompted_calls": sum(m.unprompted_calls for m in sam_models),
            "sam_variant_loads": {f"{v}@{d}": n for (v, d), n in sorted(variants.calls.items())},
            "sam_variant_disposals": {
                f"{v}@{d}": n for (v, d), n in sorted(variants.disposals.items())
            },
        }


_installed: Optional[Installed] = None


def install() -> Installed:
    """Install every double into this process (idempotent) and return the handles."""

    global _installed
    if _installed is not None:
        return _installed

    from nxmndr.inference import inference as _runtime  # noqa: F401  (registers the loaders)
    from nxmndr.models import HuggingFaceModelSpec, get_registry
    from nxmndr.models import sam as sam_support
    from tst.support import ultralytics_double
    from tst.support.stream_doubles import CountingVariantLoader, FakeHFSamModel

    installed = Installed()
    installed.variants = CountingVariantLoader(installed.sam_log)

    registry = get_registry()
    real_hf_loader = registry.get_loader(HuggingFaceModelSpec(repo_id=SAM_DOUBLE_REPO))

    def load_huggingface(spec, provider, session):
        if spec.repo_id != SAM_DOUBLE_REPO:
            return real_hf_loader(spec, provider, session)
        model = FakeHFSamModel(installed.sam_log, repo_path=f"double://{SAM_DOUBLE_REPO}")
        with installed.lock:
            installed.sam_models.append(model)
        return model

    registry.register_spec("huggingface", HuggingFaceModelSpec, load_huggingface)
    sam_support.load_sam_variant = installed.variants
    sys.modules["ultralytics"] = ultralytics_double.make_module()
    _installed = installed
    return installed


def _write_stats(path: str) -> None:
    if _installed is None:
        return
    tmp = f"{path}.part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_installed.stats(), fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _stdout_to_stderr() -> int:
    """Point file descriptor 1 at stderr (nxmndr's loggers write to stdout) and return
    a duplicate of the original stdout."""

    sys.stdout.flush()
    saved = os.dup(1)
    os.dup2(2, 1)
    return saved


def _serve(cache_dir: Optional[str], announce_fd: int) -> int:
    from nxmndr.server import server

    srv, port, service = server.create_server("127.0.0.1", 0, model_cache_dir=cache_dir, max_cores=1)
    line = json.dumps({"endpoint": f"127.0.0.1:{port}", "pid": os.getpid()}) + "\n"
    os.write(announce_fd, line.encode("utf-8"))
    os.close(announce_fd)
    try:
        for line in sys.stdin:
            if line.strip() == "stop":
                break
    finally:
        server.stop_server(srv, service)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tst.support.worker_doubles",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--stats", help="write the doubles' counters to this JSON file at exit")
    sub = parser.add_subparsers(dest="mode", required=True)
    serve = sub.add_parser("serve", help="run a loopback nxmndr gRPC server with the doubles")
    serve.add_argument("--cache-dir", default=None, help="model artifact directory (create_server)")
    run = sub.add_parser("run", help="run MODULE as __main__ with the doubles installed")
    run.add_argument("module")
    run.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    stdout_fd = _stdout_to_stderr()  # installing logs; stdout stays clean for a protocol
    install()
    if args.stats:
        atexit.register(_write_stats, args.stats)
    if args.mode == "serve":
        return _serve(args.cache_dir, stdout_fd)
    sys.stdout.flush()
    os.dup2(stdout_fd, 1)  # the module gets the real stdout back
    os.close(stdout_fd)
    sys.argv = [args.module, *args.args]
    runpy.run_module(args.module, run_name="__main__", alter_sys=True)
    return 0


__all__ = ["SAM_DOUBLE_REPO", "Installed", "install", "main"]


if __name__ == "__main__":
    sys.exit(main())
