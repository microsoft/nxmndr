# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Persistent server-side registry for model usage metadata."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..logging import get_logger

logger = get_logger(__name__)


CACHE_DIR_ENV = "NXMNDR_CACHE_DIR"


def default_cache_root() -> Path:
    """``$NXMNDR_CACHE_DIR`` if set, else ``~/.cache/nxmndr``. Read on each call.

    Holds the registry file (``model_registry.json``) and the default model
    artifact directory (``models/``). Tests point it at a temporary directory.
    """

    configured = os.environ.get(CACHE_DIR_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "nxmndr"


@dataclass
class RegistryEntry:
    entry_id: str
    model_id: str
    display_name: str
    source: str
    task: str
    format: str
    last_used_epoch_ms: int
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "model_id": self.model_id,
            "display_name": self.display_name,
            "source": self.source,
            "task": self.task,
            "format": self.format,
            "last_used_epoch_ms": self.last_used_epoch_ms,
            "metadata": dict(self.metadata or {}),
        }

    @staticmethod
    def from_dict(payload: Dict[str, object]) -> "RegistryEntry":
        return RegistryEntry(
            entry_id=str(payload.get("entry_id") or uuid.uuid4().hex),
            model_id=str(payload.get("model_id") or ""),
            display_name=str(payload.get("display_name") or ""),
            source=str(payload.get("source") or ""),
            task=str(payload.get("task") or ""),
            format=str(payload.get("format") or ""),
            last_used_epoch_ms=int(payload.get("last_used_epoch_ms") or 0),
            metadata={
                str(k): str(v) for k, v in (payload.get("metadata") or {}).items()
            },
        )


class ModelRegistryStore:
    """Manage persistent registry of model usage on the provider."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        artifact_root: Path | None = None,
    ) -> None:
        self._path = Path(path) if path else default_cache_root() / "model_registry.json"
        self._artifact_root = (
            Path(artifact_root) if artifact_root else default_cache_root() / "models"
        )
        self._lock = threading.Lock()
        self._entries: Dict[str, RegistryEntry] = {}
        self._by_model: Dict[str, str] = {}
        self._load()

    # Public API --------------------------------------------------
    def record_usage(
        self,
        *,
        model_id: str,
        display_name: str,
        source: str,
        task: str,
        model_format: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> RegistryEntry:
        """Insert or update a registry entry for a model."""

        now_ms = int(time.time() * 1000)
        metadata = metadata or {}
        entry = RegistryEntry(
            entry_id=self._by_model.get(model_id) or uuid.uuid4().hex,
            model_id=model_id,
            display_name=display_name,
            source=source,
            task=task,
            format=model_format,
            last_used_epoch_ms=now_ms,
            metadata={str(k): str(v) for k, v in metadata.items() if v is not None},
        )

        with self._lock:
            self._entries[entry.entry_id] = entry
            self._by_model[model_id] = entry.entry_id
            self._save_locked()
        return entry

    def list_entries(self, provider_id: Optional[str] = None) -> List[RegistryEntry]:
        # Provider ID currently unused but reserved for future multi-provider support
        with self._lock:
            entries = list(self._entries.values())
        entries.sort(key=lambda item: item.last_used_epoch_ms, reverse=True)
        return entries

    def evict(self, entry_id: str) -> bool:
        with self._lock:
            entry = self._entries.pop(entry_id, None)
            if not entry:
                return False
            if entry.model_id in self._by_model:
                del self._by_model[entry.model_id]
            self._save_locked()
            return True

    def find_by_artifact_sha(self, sha256_hex: str) -> Optional[RegistryEntry]:
        if not sha256_hex:
            return None
        with self._lock:
            for entry in self._entries.values():
                if entry.metadata.get("artifact_sha256") == sha256_hex:
                    return entry
        return None

    def artifact_root(self) -> Path:
        self._artifact_root.mkdir(parents=True, exist_ok=True)
        return self._artifact_root

    def resolve_artifact_path(self, sha256_hex: str, ext: str = "") -> Path:
        ext = f".{ext.lstrip('.')}" if ext else ""
        return self.artifact_root() / f"{sha256_hex}{ext}"

    # Internal helpers --------------------------------------------
    def _ensure_dir(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        try:
            if not self._path.exists():
                return
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(data, Iterable):
                logger.warning("Registry file malformed: expected list")
                return
            for item in data:
                if not isinstance(item, dict):
                    continue
                entry = RegistryEntry.from_dict(item)
                if not entry.model_id:
                    continue
                self._entries[entry.entry_id] = entry
                self._by_model[entry.model_id] = entry.entry_id
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load model registry: %s", exc)

    def _save_locked(self) -> None:
        self._ensure_dir()
        payload = [entry.to_dict() for entry in self._entries.values()]
        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(self._path)


__all__ = ["ModelRegistryStore", "RegistryEntry"]
