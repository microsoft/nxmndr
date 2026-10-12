# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Deterministic model doubles for streaming and SAM tests (no weights, no network).

Doubles sit BELOW the production code paths: records enter through the real
ModelManager ``loader`` hook, predictions run through the shared dispatcher, and SAM
prompts run through ``nxmndr.models.sam`` with only the transformers model and
processor replaced.
"""

from __future__ import annotations

import threading
from collections import Counter, defaultdict
from types import SimpleNamespace
from typing import Callable, Dict, List

import numpy as np
import torch

from nxmndr.server import managers


# ------------------------------------------------------------------ plain models


class SegLogitsModel:
    """Two-class segmentation logits; class 1 wins where the first band is > 0.5.

    With ``return_embeddings`` it returns ``{"output", "embeddings"}`` like the
    Hugging Face wrapper, embeddings = per-band mean over the chip.
    """

    def __init__(self):
        self.calls: List[Dict[str, object]] = []

    def predict(self, arr, return_embeddings=False):
        arr = np.asarray(arr, dtype=np.float32)
        self.calls.append({"shape": arr.shape, "return_embeddings": return_embeddings})
        band = arr[..., 0] if arr.ndim == 3 else arr.reshape(arr.shape[-2:])
        h, w = band.shape
        logits = np.zeros((1, 2, h, w), dtype=np.float32)
        logits[0, 1] = band - 0.5
        if return_embeddings:
            emb = arr.reshape(-1, arr.shape[-1]).mean(axis=0).astype(np.float32)[None, :]
            return {"output": logits, "embeddings": emb}
        return logits


class EchoModel:
    """Returns the input unchanged (lets tests check the bytes the server decoded)."""

    def __init__(self):
        self.seen: List[np.ndarray] = []

    def predict(self, arr, return_embeddings=False):
        self.seen.append(np.array(arr, copy=True))
        return np.array(arr, copy=True)


class GatedModel:
    """Blocks inside predict until ``release`` is set; ``entered`` fires on entry."""

    def __init__(self, inner=None):
        self.inner = inner or EchoModel()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def predict(self, arr, return_embeddings=False):
        self.calls += 1
        self.entered.set()
        if not self.release.wait(timeout=30):
            raise TimeoutError("GatedModel was never released")
        return self.inner.predict(arr, return_embeddings=return_embeddings)


# ------------------------------------------------------------------ SAM doubles


class _Batch(dict):
    """dict of tensors with ``.to(device)`` like a transformers BatchFeature."""

    def to(self, device):
        return self


class FakeSamProcessor:
    """Records every call; mimics the shapes of the SAM 3 processors (no rescaling)."""

    def __init__(self, log: List[Dict[str, object]], kind: str):
        self.log = log
        self.kind = kind

    def __call__(self, images=None, text=None, return_tensors=None, **prompt):
        arr = np.asarray(images)
        h, w = arr.shape[0], arr.shape[1]
        entry = {"kind": self.kind, "text": text, "image_shape": arr.shape}
        entry.update({k: v for k, v in prompt.items()})
        self.log.append(entry)
        batch = _Batch(pixel_values=torch.zeros(1, 3, 4, 4), original_sizes=torch.tensor([[h, w]]))
        if text is not None:
            batch["text"] = text
        for key, value in prompt.items():
            batch[key] = value
        return batch

    # tracker post-processing: threshold logits, keep (objects, masks, H, W)
    def post_process_masks(self, masks, original_sizes, mask_threshold=0.0, **kwargs):
        self.log.append({"kind": self.kind, "post_process_mask_threshold": mask_threshold})
        return [masks[0] > mask_threshold]

    # text post-processing: score filter, then binarize
    def post_process_instance_segmentation(self, outputs, threshold, mask_threshold, target_sizes):
        self.log.append(
            {"kind": self.kind, "score_threshold": threshold, "mask_threshold": mask_threshold}
        )
        keep = outputs["scores"] > threshold
        return [
            {
                "masks": outputs["probs"][keep] > mask_threshold,
                "boxes": torch.zeros((int(keep.sum()), 4)),
                "scores": outputs["scores"][keep],
            }
        ]


class FakeSamTrackerModel(torch.nn.Module):
    """One object: box area, plus 5x5 squares at positive points, minus squares at negatives."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="sam3_tracker")
        self.calls = 0

    def forward(self, original_sizes=None, input_points=None, input_labels=None,
                input_boxes=None, multimask_output=True, **kwargs):
        self.calls += 1
        h, w = [int(v) for v in original_sizes[0]]
        mask = np.zeros((h, w), dtype=bool)
        if input_boxes is not None:
            x0, y0, x1, y1 = [int(round(v)) for v in input_boxes[0][0]]
            mask[y0:y1, x0:x1] = True
        if input_points is not None:
            for (x, y), lab in zip(input_points[0][0], input_labels[0][0]):
                x, y = int(round(x)), int(round(y))
                mask[max(0, y - 2): y + 3, max(0, x - 2): x + 3] = bool(lab)
        logits = torch.from_numpy(np.where(mask, 4.0, -4.0).astype(np.float32))
        return SimpleNamespace(pred_masks=logits[None, None, None])  # (1, obj, 1, H, W)


class FakeSam3Model(torch.nn.Module):
    """Text prompts: ``field`` gives two parcels plus a whole-chip group mask;
    ``nothing`` gives only low-score candidates; anything else one centered square."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="sam3")
        self.calls = 0

    def forward(self, original_sizes=None, text=None, **kwargs):
        self.calls += 1
        h, w = [int(v) for v in original_sizes[0]]
        probs = []
        scores = []
        if text == "field":
            left = np.zeros((h, w), np.float32)
            left[:, : w // 2] = 0.9
            right = np.zeros((h, w), np.float32)
            right[:, w // 2:] = 0.9
            whole = np.full((h, w), 0.9, np.float32)
            probs, scores = [left, right, whole], [0.8, 0.7, 0.95]
        elif text == "nothing":
            probs, scores = [np.full((h, w), 0.9, np.float32)], [0.01]
        else:
            sq = np.zeros((h, w), np.float32)
            sq[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 0.9
            probs, scores = [sq], [0.6]
        return {"probs": torch.from_numpy(np.stack(probs)), "scores": torch.tensor(scores)}


class FakeHFSamModel:
    """Stands in for nxmndr's HuggingFaceModel wrapping a SAM 3 checkpoint."""

    def __init__(self, log, repo_path="/nonexistent/sam3-snapshot"):
        self.model = FakeSam3Model()
        self.processor = FakeSamProcessor(log, "text")
        self.repo_path = repo_path
        self.unprompted_calls = 0

    def predict(self, arr, return_embeddings=False):
        """Unprompted mode: a fixed (3, H, W) uint8 mask stack, like the SAM 3 path."""
        self.unprompted_calls += 1
        arr = np.asarray(arr)
        h, w = arr.shape[0], arr.shape[1]
        masks = np.zeros((3, h, w), dtype=np.uint8)
        masks[0, : h // 2] = 1
        masks[1, h // 2:] = 1
        masks[2, :, : w // 3] = 1
        return masks


class NamedLikeSamModel(SegLogitsModel):
    """A non-SAM model whose name, path and ID all contain 'sam'.

    SAM routing must come from the loaded model's capability, never from its name.
    """

    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(config=SimpleNamespace(model_type="unet"))
        self.repo_path = "/data/samples/sam-lookalike"


# ------------------------------------------------------------------ loader


class CountingLoader:
    """ModelManager ``loader`` that builds records from registered doubles by source.

    ``factories[source]() -> (model_obj, backend)``; counts loads and disposals.
    """

    def __init__(self, factories: Dict[str, Callable[[], tuple]]):
        self.factories = dict(factories)
        self.loads: Counter = Counter()
        self.records: Dict[str, List[object]] = defaultdict(list)
        self.lock = threading.Lock()

    def __call__(self, model_spec, key, metadata, device_plan):
        # The spec carries the source string verbatim; the key may normalize it.
        source = (
            getattr(model_spec, "model_path", None)
            or getattr(model_spec, "repo_id", None)
            or key.source
        )
        if source not in self.factories:
            raise RuntimeError(f"no double registered for {source!r}")
        model_obj, backend = self.factories[source]()
        record = managers.ModelRecord(
            model=model_obj,
            backend=backend,
            metadata=dict(metadata or {}),
            spec=model_spec,
            device_models={},
        )
        with self.lock:
            self.loads[source] += 1
            self.records[source].append(record)
        return record


class CountingVariantLoader:
    """Replacement for ``nxmndr.models.sam.load_sam_variant`` that counts loads and
    disposals (``close`` or ``dispose``) of the resources it creates."""

    def __init__(self, log):
        self.log = log
        self.calls: Counter = Counter()
        self.disposals: Counter = Counter()
        self.tokens: List[object] = []  # the token argument of every call
        self.lock = threading.Lock()

    def __call__(self, variant, capability, device, token=None):
        from nxmndr.models import sam as sam_support

        key = (variant, str(device))
        with self.lock:
            self.calls[key] += 1
            self.tokens.append(token)
        loader = self

        class _Counted(sam_support.SamVariantResource):
            def close(self):
                with loader.lock:
                    loader.disposals[key] += 1
                super().close()

            dispose = close

        if variant == sam_support.SAM_VARIANT_GEOMETRY:
            return _Counted(FakeSamTrackerModel(), FakeSamProcessor(self.log, "tracker"))
        return _Counted(FakeSam3Model(), FakeSamProcessor(self.log, "text"))
