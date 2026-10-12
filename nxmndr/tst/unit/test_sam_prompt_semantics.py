# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SAM prompt decoding, capability resolution and tracker prompt semantics.

The tracker nesting is checked against the real transformers Sam3TrackerProcessor
(constructed offline with a default image processor, no weights), with a spy in
place of the model: one image, one object, labels kept, the box sent as a box.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nxmndr.models import sam


# ------------------------------------------------------------------ decoding


def test_negative_labels_and_box_are_preserved():
    prompt = sam.parse_sam_prompt(
        {
            "sam_input_points": "[[12.5, 40.0], [100.0, 7.25]]",
            "sam_input_labels": "[1, 0]",
            "sam_input_bbox": "[0, 0, 256, 200]",
            "sam_text_prompt": "  road ",
        }
    )
    assert prompt.points == ((12.5, 40.0), (100.0, 7.25))
    assert prompt.labels == (1, 0)
    assert prompt.box == (0.0, 0.0, 256.0, 200.0)
    assert prompt.text == "road"
    assert prompt.has_geometry and prompt.has_prompt
    assert (prompt.conf_threshold, prompt.mask_threshold) == (0.5, 0.5)


@pytest.mark.parametrize(
    "options",
    [
        {"sam_input_points": "[[1, 2]]"},
        {"sam_input_labels": "[1]"},
        {"sam_input_points": "[[1, 2], [3, 4]]", "sam_input_labels": "[1]"},
        {"sam_input_points": "[[1, 2]]", "sam_input_labels": "[2]"},
        {"sam_input_points": "[[1, 2]]", "sam_input_labels": "[true]"},
        {"sam_input_points": "[[[[1, 2]]]]", "sam_input_labels": "[1]"},
        {"sam_input_points": "[[1, NaN]]", "sam_input_labels": "[1]"},
        {"sam_input_points": "[[1, 2", "sam_input_labels": "[1]"},
        {"sam_input_bbox": "[1, 2, 3"},
        {"sam_input_bbox": "[30, 40, 10, 20]"},
        {"sam_input_bbox": "[1, 2, 3]"},
        {"sam_conf_threshold": "1.5"},
        {"sam_mask_threshold": "-0.1"},
        {"sam_conf_threshold": "high"},
    ],
)
def test_malformed_prompt_options_raise(options):
    with pytest.raises(sam.SamPromptError):
        sam.parse_sam_prompt(options)


def test_no_prompt_is_not_an_error():
    prompt = sam.parse_sam_prompt({"task_type": "segmentation"})
    assert not prompt.has_prompt


# ------------------------------------------------------------------ capability


def _wrapped(model_type, repo_path="/snapshots/abc"):
    return SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(model_type=model_type)), repo_path=repo_path
    )


def test_capability_comes_from_the_loaded_architecture_not_the_name():
    cap = sam.resolve_sam_capability(_wrapped("sam3"), SimpleNamespace(repo_id="org/x", revision="r1"))
    assert (cap.family, cap.record_variant, cap.pretrained_path, cap.revision) == (
        "sam3", "sam3", "/snapshots/abc", "r1"
    )
    tracker = sam.resolve_sam_capability(_wrapped("sam3_tracker"))
    assert tracker.record_variant == "sam3_tracker"
    # names, paths and IDs containing "sam" do not make a model SAM
    lookalike = _wrapped("unet", repo_path="/data/samples/sam3")
    assert sam.resolve_sam_capability(lookalike, SimpleNamespace(repo_id="facebook/sam3")) is None
    assert sam.resolve_sam_capability(object()) is None


def test_capability_falls_back_to_the_records_own_repo_never_a_hardcoded_one():
    cap = sam.resolve_sam_capability(_wrapped("sam3", repo_path=None), SimpleNamespace(repo_id="me/my-sam"))
    assert cap.pretrained_path == "me/my-sam"


# ------------------------------------------------------------------ input


def test_image_conversion_rejects_non_uint8_instead_of_truncating():
    assert sam.sam_image_from_array(np.zeros((4, 5, 3), np.uint8)).size == (5, 4)
    assert sam.sam_image_from_array(np.zeros((1, 4, 5, 3), np.uint8)).size == (5, 4)
    assert sam.sam_image_from_array(np.zeros((4, 5, 1), np.uint8)).mode == "L"
    with pytest.raises(sam.SamInputError):
        sam.sam_image_from_array(np.full((4, 5, 3), 300, np.uint16))
    with pytest.raises(sam.SamInputError):
        sam.sam_image_from_array(np.zeros((4, 5, 3), np.float32))
    with pytest.raises(sam.SamInputError):
        sam.sam_image_from_array(np.zeros((3, 4, 5), np.uint8)[None, None])


# ------------------------------------------------------------------ tracker


@pytest.fixture(scope="module")
def tracker_processor():
    from transformers import Sam2ImageProcessorFast, Sam3TrackerProcessor

    return Sam3TrackerProcessor(image_processor=Sam2ImageProcessorFast())


class _SpyTracker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = {}

    def forward(self, **inputs):
        self.inputs = inputs
        masks = torch.full((1, 1, 1, 256, 256), -4.0)
        masks[..., 64:128, 64:128] = 4.0
        return SimpleNamespace(pred_masks=masks)


def test_points_labels_and_box_form_one_object_prompt(tracker_processor):
    spy = _SpyTracker()
    image = sam.sam_image_from_array(np.zeros((64, 48, 3), np.uint8))
    out = sam.run_sam3_tracker_inference(
        spy, tracker_processor, image,
        points=((10.0, 20.0), (30.0, 5.0)), labels=(1, 0), box=(1.0, 2.0, 40.0, 50.0), device="cpu",
    )
    assert tuple(spy.inputs["pixel_values"].shape[:1]) == (1,)  # one image, not one per point
    assert tuple(spy.inputs["input_points"].shape) == (1, 1, 2, 2)  # image, object, point, xy
    assert spy.inputs["input_labels"].tolist() == [[[1, 0]]]  # negative label kept
    assert tuple(spy.inputs["input_boxes"].shape) == (1, 1, 4)  # a real box prompt
    assert spy.inputs["multimask_output"] is False
    assert out["masks"].shape == (1, 64, 48) and out["masks"].dtype == np.uint8
    assert out["masks"].any()


def test_box_alone_is_not_expanded_into_corner_points(tracker_processor):
    spy = _SpyTracker()
    image = sam.sam_image_from_array(np.zeros((64, 48, 3), np.uint8))
    sam.run_sam3_tracker_inference(spy, tracker_processor, image, box=(1.0, 2.0, 40.0, 50.0), device="cpu")
    assert "input_points" not in spy.inputs and "input_labels" not in spy.inputs
    assert tuple(spy.inputs["input_boxes"].shape) == (1, 1, 4)


def test_mask_threshold_is_applied_to_tracker_logits(tracker_processor):
    image = sam.sam_image_from_array(np.zeros((64, 48, 3), np.uint8))
    full = sam.run_sam3_tracker_inference(
        _SpyTracker(), tracker_processor, image, box=(1.0, 2.0, 40.0, 50.0), device="cpu",
        mask_threshold=0.5,
    )
    none = sam.run_sam3_tracker_inference(
        _SpyTracker(), tracker_processor, image, box=(1.0, 2.0, 40.0, 50.0), device="cpu",
        mask_threshold=1.0,
    )
    assert full["masks"].any() and not none["masks"].any()


# ------------------------------------------------------------------ text


class _TextProcessor:
    def __call__(self, images=None, text=None, return_tensors=None):
        class _B(dict):
            def to(self, device):
                return self

        return _B(original_sizes=torch.tensor([[8, 6]]))

    def post_process_instance_segmentation(self, outputs, threshold, mask_threshold, target_sizes):
        return [outputs]


def test_empty_text_result_keeps_the_chip_size():
    model = lambda **kw: {"masks": torch.zeros((0, 8, 6), dtype=torch.bool),  # noqa: E731
                          "boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,))}
    out = sam.run_sam3_text_inference(model, _TextProcessor(), np.zeros((8, 6, 3), np.uint8), "x", device="cpu")
    assert out["masks"].shape == (0, 8, 6) and out["masks"].dtype == np.uint8


def test_text_results_drop_group_masks():
    left = torch.zeros((8, 6), dtype=torch.bool)
    left[:, :3] = True
    right = ~left
    whole = torch.ones((8, 6), dtype=torch.bool)
    model = lambda **kw: {"masks": torch.stack([left, right, whole]),  # noqa: E731
                          "boxes": torch.zeros((3, 4)), "scores": torch.tensor([0.8, 0.7, 0.9])}
    out = sam.run_sam3_text_inference(model, _TextProcessor(), np.zeros((8, 6, 3), np.uint8), "x", device="cpu")
    assert out["masks"].shape == (2, 8, 6)
    assert out["masks"].dtype == np.uint8


def test_handler_prefers_geometry_over_text():
    calls = []

    def get_variant(name):
        calls.append(name)
        raise RuntimeError("stop after routing")

    prompt = sam.parse_sam_prompt({"sam_text_prompt": "field", "sam_input_bbox": json.dumps([0, 0, 2, 2])})
    with pytest.raises(RuntimeError):
        sam.handle_sam_inference(np.zeros((4, 4, 3), np.uint8), prompt, get_variant)
    assert calls == [sam.SAM_VARIANT_GEOMETRY]
