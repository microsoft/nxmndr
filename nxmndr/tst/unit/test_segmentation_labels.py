# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Segmentation label maps never wrap or truncate.

``prepare_segmentation_mask`` (and its confidence variant) cast predictions to ``uint16``
labels. A value outside ``[0, 65535]`` or a fractional value is refused instead of being
cast (a model that returned its input unchanged produced 65536, which a plain cast
turned into label 0); the dispatcher then returns the prediction unchanged as a
``raw`` result with a warning.
"""

from __future__ import annotations

import numpy as np
import pytest

from nxmndr.inference.image_utils import prepare_segmentation_mask, prepare_segmentation_mask_with_confidence
from nxmndr.server.dispatch import shape_result

SEGMENTATION = {"task_type": "segmentation"}


@pytest.mark.parametrize("prediction", [
    np.array([[0.0, 65536.0]], dtype=np.float32),  # identity output; a uint16 cast wraps it to 0
    np.array([[0, 65536]], dtype=np.int64),
    np.array([[-1, 2]], dtype=np.int32),
    np.array([[0.0, 2.5]], dtype=np.float32),
    np.array([[0.0, np.nan]], dtype=np.float32),
    np.array([[[0, 70000]]], dtype=np.int64),  # (1, H, W) integer mask
    np.array([[[[0.0, 1.5], [2.0, 3.0]]]], dtype=np.float32),  # (N, 1, H, W) single channel
])
def test_values_a_uint16_cast_would_change_are_refused(prediction):
    with pytest.raises(ValueError, match="uint16 range|fractional or non-finite"):
        prepare_segmentation_mask(prediction)
    with pytest.raises(ValueError, match="uint16 range|fractional or non-finite"):
        prepare_segmentation_mask_with_confidence(prediction)


@pytest.mark.parametrize("prediction", [
    np.array([[0.0, 3.0, 65535.0]], dtype=np.float32),
    np.array([[0, 3, 65535]], dtype=np.int64),
    np.array([[[0, 3, 65535]]], dtype=np.int32),
    np.array([[[[0.0, 3.0, 65535.0], [0.0, 3.0, 65535.0]]]], dtype=np.float64),  # (N, 1, H, W)
])
def test_whole_labels_in_range_are_kept_exactly(prediction):
    expected = np.asarray(prediction).reshape(-1, 3).astype(np.int64).tolist()
    mask = prepare_segmentation_mask(prediction)
    assert mask.dtype == np.uint16 and mask.tolist() == expected
    mask, confidence = prepare_segmentation_mask_with_confidence(prediction)
    assert mask.tolist() == expected and (confidence == 1.0).all()


def test_logits_still_become_argmax_labels():
    logits = np.zeros((3, 2, 2), np.float32)
    logits[2, 0, 1] = 5.0
    assert prepare_segmentation_mask(logits).tolist() == [[0, 2], [0, 0]]


def test_the_dispatcher_keeps_an_out_of_range_identity_output_as_raw_values():
    identity = np.array([[0.0, 1.0], [65535.0, 65536.0]], dtype=np.float32)
    result = shape_result(identity, options=SEGMENTATION, model_metadata={})
    assert result.metadata["result_type"] == "raw"
    assert result.metadata["task_warning"].startswith("segmentation_fallback:")
    assert result.dtype == "float32" and result.array.tolist() == identity.tolist()  # 65536 stays 65536

    in_range = shape_result(identity[:1], options=SEGMENTATION, model_metadata={})
    assert in_range.metadata["result_type"] == "segmentation_mask" and in_range.dtype == "uint16"
    assert in_range.array.tolist() == [[0, 1]]
