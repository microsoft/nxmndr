# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""The Ultralytics YOLO adapter on its own: chip conversion, mask stack, loading.

``ultralytics.YOLO`` is the deterministic double of ``tst.support.ultralytics_double``;
everything below it is production code.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from nxmndr.models import ultralytics_yolo as yolo_adapter
from tst.support.ultralytics_double import FakeYOLO, make_module, write_checkpoint


@pytest.fixture
def fake_ultralytics(monkeypatch):
    FakeYOLO.reset()
    monkeypatch.setitem(sys.modules, "ultralytics", make_module())
    return FakeYOLO


def _load(tmp_path, name="yolo.pt", **settings):
    path = write_checkpoint(tmp_path / name, **settings)
    spec = yolo_adapter.UltralyticsModelSpec(model_path=str(path))
    return yolo_adapter.load_ultralytics_yolo(spec, None, None)


def _blocks(h=16, w=16, dtype=np.uint8):
    """Band 0 holds instances 1 (top-left 8x8), 2 (top-right 8x8), 3 (bottom 4 rows)."""
    chip = np.zeros((h, w, 3), dtype=dtype)
    chip[:8, :8, 0] = 1
    chip[:8, 8:, 0] = 2
    chip[12:, :, 0] = 3
    chip[:, :, 1] = 9  # other bands never define instances
    return chip


def _expected(chip):
    band = chip[:, :, 0]
    return np.stack([(band == v) for v in (1, 2, 3)]).astype(np.uint8)


# The detection settings every predict call gets when the request names none: the paper's
# DelineateAnything run (conf 0.1) and Ultralytics' iou and max_det defaults.
DEFAULT_DETECTION = {"conf": 0.1, "iou": 0.7, "max_det": 300}


# ------------------------------------------------------------------ chip conversion


def test_uint8_rgb_chip_is_passed_as_a_writable_copy():
    chip = _blocks()
    chip.setflags(write=False)
    out = yolo_adapter.to_yolo_chip(chip)
    assert out.dtype == np.uint8 and out.shape == (16, 16, 3) and out.flags.writeable
    np.testing.assert_array_equal(out, chip)


def test_band_rules_repeat_one_band_drop_extra_bands_and_refuse_two():
    one = np.arange(16, dtype=np.uint8).reshape(4, 4, 1)
    np.testing.assert_array_equal(yolo_adapter.to_yolo_chip(one), np.repeat(one, 3, axis=2))
    np.testing.assert_array_equal(yolo_adapter.to_yolo_chip(one[:, :, 0]), np.repeat(one, 3, axis=2))
    four = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
    np.testing.assert_array_equal(yolo_adapter.to_yolo_chip(four), four[:, :, :3])
    with pytest.raises(yolo_adapter.InstanceInputError):
        yolo_adapter.to_yolo_chip(np.zeros((4, 4, 2), np.uint8))
    with pytest.raises(yolo_adapter.InstanceInputError):
        yolo_adapter.to_yolo_chip(np.zeros((1, 4, 4, 3), np.uint8))


def test_non_uint8_data_is_scaled_clipped_and_nan_is_zero():
    unit = np.full((2, 2, 3), 0.5, dtype=np.float32)
    assert yolo_adapter.to_yolo_chip(unit).tolist() == np.full((2, 2, 3), 127, np.uint8).tolist()
    wide = np.array([[[0, 300, 7]]], dtype=np.uint16)
    assert yolo_adapter.to_yolo_chip(wide).tolist() == [[[0, 255, 7]]]
    nan = np.array([[[np.nan, 2.0, -3.0]]], dtype=np.float32)
    assert yolo_adapter.to_yolo_chip(nan).tolist() == [[[0, 2, 0]]]


# ------------------------------------------------------------------ mask stack


def test_instances_come_back_at_chip_size_in_model_order(tmp_path, fake_ultralytics):
    model = _load(tmp_path)
    chip = _blocks()
    masks = model.predict(chip)
    assert masks.dtype == np.uint8 and masks.flags.c_contiguous
    np.testing.assert_array_equal(masks, _expected(chip))
    assert fake_ultralytics.constructed == [str(tmp_path / "yolo.pt")]
    call = model.yolo.calls[0]
    assert call["kwargs"] == {"verbose": False, "retina_masks": True, **DEFAULT_DETECTION}
    assert call["shape"] == (16, 16, 3) and call["dtype"] == "uint8"


def test_group_mask_is_dropped_unless_disabled(tmp_path, fake_ultralytics):
    model = _load(tmp_path, group_mask=True)
    chip = _blocks()
    np.testing.assert_array_equal(model.predict(chip), _expected(chip))
    kept = model.predict(chip, drop_group_masks=False)
    assert kept.shape == (4, 16, 16)
    np.testing.assert_array_equal(kept[0], (chip[:, :, 0] > 0).astype(np.uint8))


@pytest.mark.parametrize("empty", ["none", "zeros"])
def test_nothing_found_is_a_valid_empty_stack(tmp_path, fake_ultralytics, empty):
    model = _load(tmp_path, empty=empty, mask_divisor=2)
    masks = model.predict(np.zeros((16, 12, 3), np.uint8))
    assert masks.shape == (0, 16, 12) and masks.dtype == np.uint8


@pytest.mark.parametrize("mask_dtype", ["uint8", "float32"])
def test_smaller_masks_are_resized_to_the_chip(tmp_path, fake_ultralytics, mask_dtype):
    model = _load(tmp_path, mask_divisor=2, mask_dtype=mask_dtype)
    chip = _blocks()
    assert model.yolo.predict(chip)[0].masks.data.shape == (3, 8, 8)
    np.testing.assert_array_equal(model.predict(chip), _expected(chip))


def test_an_instance_lost_in_the_resize_is_dropped(tmp_path, fake_ultralytics):
    model = _load(tmp_path, mask_divisor=4)
    chip = np.zeros((16, 16, 3), np.uint8)
    chip[:8, :8, 0] = 1  # survives sampling at every 4th pixel
    chip[1:3, 1:3, 0] = 2  # between the sampled pixels: gone at mask resolution
    masks = model.predict(chip)
    assert masks.shape == (1, 16, 16)
    block = np.zeros((16, 16), np.uint8)
    block[:8, :8] = 1  # instance 1 at mask resolution, back at chip size
    np.testing.assert_array_equal(masks[0], block)


def test_letterboxed_masks_are_refused_not_stretched():
    results = [type("R", (), {"masks": type("M", (), {"data": np.ones((1, 16, 16), np.uint8)})()})()]
    with pytest.raises(ValueError, match="aspect ratio"):
        yolo_adapter.instance_stack(results, 16, 8)
    with pytest.raises(ValueError, match=r"\(N, H, W\)"):
        yolo_adapter.instance_stack(
            [type("R", (), {"masks": type("M", (), {"data": np.ones((16, 16))})()})()], 16, 16
        )


def test_a_build_without_retina_masks_is_called_again_without_it(tmp_path, fake_ultralytics):
    model = _load(tmp_path, retina_masks_supported=False)
    chip = _blocks()
    np.testing.assert_array_equal(model.predict(chip), _expected(chip))
    assert [c["kwargs"] for c in model.yolo.calls] == [{"verbose": False, **DEFAULT_DETECTION}]


def test_detection_settings_reach_predict(tmp_path, fake_ultralytics):
    model = _load(tmp_path)
    model.predict(_blocks(), conf=0.25, iou=0.5, max_det=7)
    assert model.yolo.calls[0]["kwargs"] == {"verbose": False, "retina_masks": True, "conf": 0.25, "iou": 0.5,
                                             "max_det": 7}
    assert (yolo_adapter.DEFAULT_CONF, yolo_adapter.DEFAULT_IOU, yolo_adapter.DEFAULT_MAX_DET) == (0.1, 0.7, 300)
    assert fake_ultralytics.predict_options == [DEFAULT_DETECTION | {"conf": 0.25, "iou": 0.5, "max_det": 7}]


def test_predict_refuses_embeddings(tmp_path, fake_ultralytics):
    with pytest.raises(ValueError, match="no embeddings"):
        _load(tmp_path).predict(_blocks(), return_embeddings=True)


# ------------------------------------------------------------------ loading


def test_a_detection_only_yolo_model_is_refused(tmp_path, fake_ultralytics):
    with pytest.raises(ValueError, match="'detect' model"):
        _load(tmp_path, task="detect")


def test_missing_ultralytics_is_reported(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    assert yolo_adapter.ultralytics_available() is False
    with pytest.raises(ImportError, match="needs the ultralytics package"):
        _load(tmp_path)


def test_availability_does_not_import_ultralytics(monkeypatch):
    monkeypatch.delitem(sys.modules, "ultralytics", raising=False)
    yolo_adapter.ultralytics_available()
    assert "ultralytics" not in sys.modules
