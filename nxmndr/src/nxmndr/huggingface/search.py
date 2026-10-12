# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import json
import logging
from pprint import pformat
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Iterable, Any
from dataclasses import dataclass, field

import requests

from nxmndr.constants import get_extension_rank

# huggingface_hub and nxmndr.models (torch) are imported only by the functions that
# use them, so this module imports on a host without the ML runtime (it ships in
# the nxmndr-wire distribution).
if TYPE_CHECKING:
    from nxmndr.models.models import HuggingFaceModelSpec

"""Utility helpers for searching HuggingFace Hub models.

This module provides a resilient search wrapper around the public HuggingFace
models REST API as well as a convenience helper for checking if a model is
likely usable for inference (i.e. it exposes a pipeline tag).
"""

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 10


@dataclass
class HuggingFaceSearchResponse:
    """Container for HuggingFace search responses with metadata."""

    models: List[Dict[str, Any]] = field(default_factory=list)
    total: int = 0
    headers: Dict[str, str] = field(default_factory=dict)
    has_more: bool = False
    error: Optional[str] = None


def hf_model_supports_inference(model_id: str) -> bool:
    """Return True if the model exposes a pipeline tag (usable for inference).

    A missing tag does not always mean unusable (custom architectures may still
    work) but this serves as a quick heuristic.
    """
    from huggingface_hub import HfApi

    try:
        api = HfApi()
        info = api.model_info(model_id)
        return bool(getattr(info, "pipeline_tag", None))
    except Exception as e:  # Network / API errors
        logger.warning("Failed to fetch model info for %s: %s", model_id, e)
        return False


def search_huggingface_models(
    query: str,
    limit: int = 10,
    offset: int = 0,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    auth_token: Optional[str] = None,
    full: bool = True,
) -> HuggingFaceSearchResponse:
    """Search HuggingFace models by query string.

    Args:
            query: Search term.
            limit: Max number of results.
            offset: Result offset for pagination.
            timeout: HTTP timeout in seconds.
            auth_token: Optional Hugging Face access token for private repos.
            full: Whether to request full metadata payloads.

    Returns:
        HuggingFaceSearchResponse containing models and metadata.
    """
    base_url = "https://huggingface.co/api/models"
    requested_limit = limit
    params = {"search": query, "limit": requested_limit + 1, "offset": offset}
    if full:
        params["full"] = "true"

    headers = {"Accept": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    try:
        resp = requests.get(base_url, params=params, timeout=timeout, headers=headers)
        resp.raise_for_status()
        models = resp.json()
        if not isinstance(models, list):
            logger.warning("Unexpected response type from HF API (expected list)")
            return HuggingFaceSearchResponse(
                models=[],
                total=0,
                headers=dict(resp.headers),
                error="Unexpected response type from HuggingFace API.",
            )

        has_more = False
        if len(models) > requested_limit:
            has_more = True
            models = models[:requested_limit]

        total_count = 0
        if "x-total-count" in resp.headers:
            try:
                total_count = int(resp.headers["x-total-count"])
                if total_count > 0 and offset + requested_limit < total_count:
                    has_more = True
                elif total_count and offset + requested_limit >= total_count:
                    has_more = False
            except (ValueError, TypeError):
                logger.debug(
                    "Unable to parse x-total-count header: %s",
                    resp.headers["x-total-count"],
                )

        return HuggingFaceSearchResponse(
            models=models,
            total=total_count,
            headers=dict(resp.headers),
            has_more=has_more,
        )
    except requests.Timeout:
        logger.error("HuggingFace search timed out (query=%s, timeout=%ss)", query, timeout)
        return HuggingFaceSearchResponse(
            models=[],
            total=0,
            error="HuggingFace search timed out.",
        )
    except requests.RequestException as e:
        logger.error("HuggingFace search failed: %s", e)
        return HuggingFaceSearchResponse(
            models=[],
            total=0,
            error=str(e),
        )
    except ValueError as e:
        logger.error("Failed to decode HuggingFace response JSON: %s", e)
        return HuggingFaceSearchResponse(
            models=[],
            total=0,
            error="Failed to decode HuggingFace response JSON.",
        )


def format_search_results(models: List[Dict]) -> str:
    """Return a pretty formatted string of model entries for logging / printing."""
    return pformat(models, indent=2, width=100, compact=False)


def log_search_results(models: List[Dict]) -> None:
    """Log HuggingFace model search results (info level)."""
    if not models:
        logger.info("No models returned from search")
        return
    logger.info("HuggingFace search returned %d models", len(models))
    for m in models:
        # Log a minimal subset to avoid very large blobs
        subset = {k: m.get(k) for k in ("modelId", "pipeline_tag", "tags", "downloads") if k in m}
        logger.debug("Model: %s", json.dumps(subset, ensure_ascii=False))


# ----------------------------- Conversion Utilities -----------------------------

_PREFERRED_WEIGHT_FILENAMES = [
    # Highest priority explicit names
    "model.safetensors",
    "diffusion_pytorch_model.safetensors",
    "pytorch_model.safetensors",
]

# Use centralized extension preference order from constants module
# Note: EXTENSION_PREFERENCE_ORDER imported at top of file


def _select_primary_weight_file(siblings: Sequence[Dict[str, object]]) -> Optional[str]:
    """Heuristically choose the best weight file from HF 'siblings'.

    Args:
            siblings: Sequence of sibling dicts each expected to contain 'rfilename'.

    Returns:
            Selected filename or None if no plausible weight file is found.
    """
    if not siblings:
        return None

    # Normalize list of filenames
    filenames: List[str] = [
        s.get("rfilename") for s in siblings if isinstance(s, dict) and s.get("rfilename")
    ]
    # 1. Explicit preferred names
    for preferred in _PREFERRED_WEIGHT_FILENAMES:
        if preferred in filenames:
            return preferred
    # 2. Any file matching ordered extensions (keep earliest best candidate)
    best: Optional[str] = None
    best_rank = 1_000
    for fn in filenames:
        import os

        _, ext = os.path.splitext(fn)
        rank = get_extension_rank(ext)
        if rank < best_rank:
            best_rank = rank
            best = fn
    return best


## NOTE: Legacy direct dict->spec conversion moved below after HFModelSearchResult class.


@dataclass
class HFModelSearchResult:
    """Structured representation of a single HuggingFace model search result.

    Provides convenient attribute access and conversion to a
    HuggingFaceModelSpec via the .to_spec() method.
    """

    repo_id: str
    pipeline_tag: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    downloads: Optional[int] = None
    siblings: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Optional["HFModelSearchResult"]:
        if not isinstance(d, dict):
            return None
        repo_id = d.get("modelId") or d.get("id")
        if not repo_id:
            return None
        return cls(
            repo_id=repo_id,
            pipeline_tag=d.get("pipeline_tag"),
            tags=d.get("tags") or [],
            downloads=d.get("downloads"),
            siblings=d.get("siblings") or [],
            raw=d,
        )

    def fetch_files(self, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> List[str]:
        """Fetch (refresh) the list of files for this model from HuggingFace Hub.

        Returns a list of repository filenames and updates self.siblings in-place.

        Args:
                timeout: HTTP timeout in seconds.

        Returns:
                List of filenames present in the repository (may be empty on error).
        """
        api_url = f"https://huggingface.co/api/models/{self.repo_id}"
        params = {"full": "true"}
        try:
            resp = requests.get(api_url, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            siblings = data.get("siblings") or []
            if isinstance(siblings, list):
                self.siblings = siblings  # refresh in-place
                self.raw = data
                filenames = [
                    s.get("rfilename")
                    for s in siblings
                    if isinstance(s, dict) and s.get("rfilename")
                ]
                return filenames
        except requests.Timeout:
            logger.warning("Timeout while fetching files for %s", self.repo_id)
        except requests.RequestException as e:
            logger.warning("Failed to fetch files for %s: %s", self.repo_id, e)
        except ValueError as e:
            logger.warning("Invalid JSON while fetching files for %s: %s", self.repo_id, e)
        return []

    # Backwards compatible adapter
    def to_spec(self, require_weights: bool = True) -> Optional[HuggingFaceModelSpec]:
        from nxmndr.models.models import HuggingFaceModelSpec

        filename = _select_primary_weight_file(self.siblings)
        if not filename and require_weights:
            return None
        if not filename:  # fallback default if relaxed
            filename = "pytorch_model.bin"
        return HuggingFaceModelSpec(repo_id=self.repo_id, filename=filename)


def parse_hf_search_results(
    results: Iterable[Dict[str, object]],
) -> List[HFModelSearchResult]:
    """Parse raw list of search result dictionaries into HFModelSearchResult objects."""
    parsed: List[HFModelSearchResult] = []
    for r in results:
        obj = HFModelSearchResult.from_dict(r)  # type: ignore[arg-type]
        if obj:
            parsed.append(obj)
    return parsed


__all__ = [
    "hf_model_supports_inference",
    "search_huggingface_models",
    "format_search_results",
    "log_search_results",
    "HFModelSearchResult",
    "parse_hf_search_results",
    "HuggingFaceSearchResponse",
]


if __name__ == "__main__":  # Manual quick test
    # Configured here, not at import: a host process that imports this module keeps
    # its own logging configuration.
    logging.basicConfig(level=logging.INFO)
    query = "landsat"
    response = search_huggingface_models(query, limit=5)
    log_search_results(response.models)
    print(format_search_results(response.models))
