# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""The PyTorch model catalog and the catalog output rule, in-process.

- ``list_model_classes``: torchvision's classification and segmentation names, the
  served torchgeo names when torchgeo imports, none of torchgeo when it does not
  (the import is blocked with ``sys.modules``); every listed name of either family is
  served or excluded with a reason.
- ``resolve``/``build``: the errors for unknown, unserved and unavailable names and
  for bad metadata; constructor arguments from metadata or from the weights file;
  a weights file that does not match fails with the class, its arguments and the
  first differing entry, in one line.
- ``input_size``/``list_input_sizes``: the served names that take one chip size, each
  of which runs at that size and fails at 512 (the plugin's default chip), while every
  other served name runs at 224 and at 512 (models built on the ``meta`` device where
  their builder allows).
- ``dispatch.run_prediction`` on a catalog record: class vector, label map and
  embedding families, the shapes that fit none, and a chip of another size than a
  fixed-size model takes.

Weights are random and saved by each test; nothing is downloaded. The gRPC service
and the RPC worker run in tst/integration/test_model_catalog_service.py.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torchvision

from nxmndr.inference import inference_pb2
from nxmndr.models import catalog
from nxmndr.server import dispatch
from nxmndr.server.model_cache import _PYTORCH_CONSTRUCTOR_KEYS, cache_key_from_spec
from nxmndr.tensor_bundle import unpack_tensor_bundle

try:
    import torchgeo.models as torchgeo_models
except ImportError:
    torchgeo_models = None

needs_torchgeo = pytest.mark.skipif(
    torchgeo_models is None, reason="torchgeo is not installed (the nxmndr[geo] extra)"
)


def _clear_caches():
    catalog._family_models.cache_clear()
    catalog._family_names.cache_clear()
    catalog.list_model_classes.cache_clear()
    catalog.input_size.cache_clear()
    catalog.list_input_sizes.cache_clear()


@pytest.fixture
def fresh_catalog():
    _clear_caches()
    yield
    _clear_caches()


def _save(model, path):
    torch.save(model.state_dict(), str(path))
    return str(path)


# ------------------------------------------------------------------ the served names


def test_served_names_are_the_families_image_models(fresh_catalog):
    names = catalog.list_model_classes()
    assert all(catalog.is_catalog_name(name) for name in names)
    assert list(names) == sorted(names, key=lambda n: tuple(n.split(":", 1)))  # family, name

    tv = torchvision.models
    classification = tv.list_models(module=tv)
    segmentation = tv.list_models(module=tv.segmentation)
    served_tv = [n.split(":", 1)[1] for n in names if n.startswith("torchvision:")]
    assert served_tv == sorted(classification + segmentation)
    assert "resnet18" in served_tv and "fcn_resnet50" in served_tv
    for name in set(tv.list_models()) - set(served_tv):
        module = tv.get_model_builder(name).__module__.rsplit(".", 1)[0]
        assert module in catalog.TORCHVISION_EXCLUDED, name
        with pytest.raises(catalog.CatalogError, match="is not served"):
            catalog.resolve(f"torchvision:{name}")

    served_tg = [n.split(":", 1)[1] for n in names if n.startswith("torchgeo:")]
    if torchgeo_models is None:
        assert served_tg == []
    else:
        assert served_tg == sorted(catalog.TORCHGEO_SERVED)
        listed = set(torchgeo_models.list_models())
        # Every torchgeo name is served or excluded with a reason, so a new torchgeo
        # release cannot add an unproven name.
        assert listed == set(catalog.TORCHGEO_SERVED) | set(catalog.TORCHGEO_EXCLUDED)
        assert not set(catalog.TORCHGEO_SERVED) & set(catalog.TORCHGEO_EXCLUDED)


def test_without_torchgeo_the_catalog_lists_torchvision_only(fresh_catalog, monkeypatch):
    monkeypatch.setitem(sys.modules, "torchgeo", None)  # import torchgeo raises ImportError
    monkeypatch.setitem(sys.modules, "torchgeo.models", None)
    names = catalog.list_model_classes()
    assert names and all(name.startswith("torchvision:") for name in names)
    with pytest.raises(catalog.CatalogUnavailableError, match=r"nxmndr\[geo\]"):
        catalog.resolve("torchgeo:unet")


@needs_torchgeo
def test_with_torchgeo_the_catalog_lists_both_families(fresh_catalog):
    names = catalog.list_model_classes()
    assert "torchgeo:unet" in names and "torchvision:resnet18" in names
    assert names.index("torchgeo:unet") < names.index("torchvision:resnet18")


@pytest.mark.parametrize(
    "model_class, message",
    [
        ("resnet18", "is not registered in this server"),
        ("foo:bar", "families: torchvision, torchgeo"),
        ("torchvision:", "Capabilities entry 'pytorch_model_classes'"),
        ("torchvision:no_such_model", "is not in torchvision.models.list_models()"),
        ("torchvision:raft_large", "is not served: takes two images"),
        ("torchvision:fasterrcnn_resnet50_fpn", "is not served: returns per-image detection"),
        ("torchvision:quantized_resnet18", "is not served: int8 inference"),
    ],
)
def test_unknown_and_unserved_names_are_refused_with_where_the_catalog_comes_from(
    model_class, message
):
    with pytest.raises(catalog.CatalogError, match=message.replace("(", r"\(").replace(")", r"\)")):
        catalog.resolve(model_class)


def test_constructor_arguments_are_integers_of_at_least_one():
    assert catalog.parse_arguments({"num_classes": "5", "in_channels": " 4 ", "x": "y"}) == {
        "num_classes": 5,
        "in_channels": 4,
    }
    assert catalog.parse_arguments({"num_classes": ""}) == {}
    assert catalog.parse_arguments(None) == {}
    for bad in ("0", "-2", "five", "2.5"):
        with pytest.raises(catalog.CatalogError, match="must be an integer >= 1"):
            catalog.parse_arguments({"num_classes": bad})
    assert tuple(catalog.CONSTRUCTOR_KEYS) == _PYTORCH_CONSTRUCTOR_KEYS


# ------------------------------------------------------------------------- build


def test_build_takes_the_class_count_from_the_weights_file_or_the_metadata(tmp_path):
    torch.manual_seed(0)
    reference = torchvision.models.resnet18(weights=None, num_classes=5).eval()
    path = _save(reference, tmp_path / "r18.pt")
    chip = torch.rand(32, 32, 3)
    with torch.no_grad():
        expected = reference(chip.permute(2, 0, 1)[None])
    for metadata in (None, {"num_classes": "5"}, {"num_classes": "5", "in_channels": "3"}):
        model = catalog.build("torchvision:resnet18", path, metadata)
        assert model.catalog_info() == {
            "model_family": "torchvision",
            "model_class": "torchvision:resnet18",
            "num_classes": "5",
            "in_channels": "3",
            "input_size": "",
        }
        with torch.no_grad():
            torch.testing.assert_close(model(chip), expected)
    for chip in (torch.rand(32, 32, 4), torch.rand(1, 32, 32, 3)):
        with pytest.raises(ValueError, match=r"takes one \[H, W, 3\] chip; got shape"):
            model(chip)


def test_torchvision_models_take_three_bands_only(tmp_path):
    assert catalog.parse_arguments({"in_channels": "3"}, "torchvision:resnet18") == {
        "in_channels": 3
    }
    # Refused before any build (the server checks at LoadModel: INVALID_ARGUMENT).
    with pytest.raises(catalog.CatalogError, match="takes 3-band chips .* in_channels=4"):
        catalog.parse_arguments({"in_channels": "4"}, "torchvision:resnet18")
    path = _save(torchvision.models.resnet18(weights=None, num_classes=5), tmp_path / "r18.pt")
    with pytest.raises(catalog.CatalogError, match="takes 3-band chips .* in_channels=4"):
        catalog.build("torchvision:resnet18", path, {"in_channels": "4"})


def test_build_reads_the_aux_head_of_torchvision_segmentation(tmp_path):
    torch.manual_seed(0)
    reference = torchvision.models.get_model(
        "deeplabv3_mobilenet_v3_large",
        weights=None,
        weights_backbone=None,
        num_classes=4,
        aux_loss=True,
    ).eval()
    path = _save(reference, tmp_path / "dl.pt")
    model = catalog.build("torchvision:deeplabv3_mobilenet_v3_large", path)
    assert model.catalog_info()["num_classes"] == "4"
    chip = torch.rand(64, 64, 3)
    with torch.no_grad():
        torch.testing.assert_close(model(chip), reference(chip.permute(2, 0, 1)[None])["out"])


def test_a_weights_file_that_does_not_match_fails_with_the_first_differing_entry(tmp_path):
    model = torchvision.models.resnet18(weights=None, num_classes=5)
    path = _save(model, tmp_path / "r18.pt")
    with pytest.raises(catalog.CatalogError) as wrong_classes:
        catalog.build("torchvision:resnet18", path, {"num_classes": "7"})
    assert str(wrong_classes.value) == (
        f"weights file {path} does not match torchvision:resnet18(num_classes=7) (strict load): "
        "shape mismatch fc.weight file [5, 512] model [7, 512] "
        "(first of 2 differing entries: 2 with another shape)"
    )

    with pytest.raises(catalog.CatalogError) as other_architecture:
        catalog.build("torchvision:resnet34", path)
    resnet34 = torchvision.models.resnet34(weights=None, num_classes=5).state_dict()
    missing = [key for key in resnet34 if key not in model.state_dict()]
    assert missing[0] == "layer1.2.conv1.weight" and len(missing) > 1
    assert all(key in resnet34 for key in model.state_dict())  # nothing unexpected
    assert str(other_architecture.value) == (
        f"weights file {path} does not match torchvision:resnet34(num_classes=5) (strict load): "
        f"missing layer1.2.conv1.weight (first of {len(missing)} differing entries: "
        f"{len(missing)} missing)"
    )

    blob = tmp_path / "model.pt"
    torch.save({"state_dict": {"w": torch.zeros(1)}, "epoch": 3}, str(blob))
    with pytest.raises(
        catalog.CatalogError, match="is not a state dict .* holds \\['state_dict', 'epoch'\\]"
    ):
        catalog.build("torchvision:resnet18", str(blob))
    with pytest.raises(catalog.CatalogError, match="not found"):
        catalog.build("torchvision:resnet18", str(tmp_path / "absent.pt"))

    # A whole pickled model: torch's refusal runs to many lines of advice; the error
    # keeps its reason, in one line.
    pickled = tmp_path / "pickled.pt"
    torch.save(model, str(pickled))
    with pytest.raises(catalog.CatalogError) as whole_model:
        catalog.build("torchvision:resnet18", str(pickled))
    assert str(whole_model.value) == (
        f"torchvision:resnet18: weights file {pickled} is not a PyTorch state dict "
        "(UnpicklingError: Unsupported global: GLOBAL torchvision.models.resnet.ResNet was not "
        "an allowed global by default)"
    )


def test_catalog_error_messages_are_one_line():
    assert str(catalog.CatalogError("a\n\tb  c\n")) == "a b c"


@needs_torchgeo
def test_torchgeo_unet_takes_classes_and_bands_from_the_weights_file(tmp_path):
    torch.manual_seed(0)
    reference = torchgeo_models.get_model("unet", weights=None, classes=3, in_channels=4).eval()
    path = _save(reference, tmp_path / "unet.pt")
    model = catalog.build("torchgeo:unet", path)
    assert model.catalog_info() == {
        "model_family": "torchgeo",
        "model_class": "torchgeo:unet",
        "num_classes": "3",
        "in_channels": "4",
        "input_size": "",
    }
    chip = torch.rand(64, 64, 4)
    with torch.no_grad():
        torch.testing.assert_close(model(chip), reference(chip.permute(2, 0, 1)[None]))
    with pytest.raises(catalog.CatalogError, match="has no class head"):
        catalog.build("torchgeo:tilenet", path, {"num_classes": "3"})


# ------------------------------------------------------------- fixed input sizes

# The served names that take one square chip size, with that size: torchvision's
# VisionTransformer (image_size) and MaxVit (INPUT_SIZES_NOT_ON_THE_MODEL), torchgeo's
# timm ViTs, DINOv2 ViTs and ScaleMAE (strict patch_embed.img_size) and EarthLoc
# (image_size). test_every_served_name_runs_at_its_input_size_only proves the list.
FIXED_INPUT_SIZES = {
    "torchgeo:earthloc": 320,
    "torchgeo:scalemae_large_patch16": 224,
    "torchgeo:vit_base_patch14_dinov2": 518,
    "torchgeo:vit_base_patch16_224": 224,
    "torchgeo:vit_huge_patch14_224": 224,
    "torchgeo:vit_large_patch16_224": 224,
    "torchgeo:vit_small_patch14_dinov2": 518,
    "torchgeo:vit_small_patch16_224": 224,
    "torchvision:maxvit_t": 224,
    "torchvision:vit_b_16": 224,
    "torchvision:vit_b_32": 224,
    "torchvision:vit_h_14": 224,
    "torchvision:vit_l_16": 224,
    "torchvision:vit_l_32": 224,
}
OTHER_SIZE = 512  # the plugin's default chip size; no fixed size above is 512


def test_the_fixed_input_sizes_are_listed_in_catalog_order(fresh_catalog):
    names = catalog.list_model_classes()
    expected = [(name, FIXED_INPUT_SIZES[name]) for name in names if name in FIXED_INPUT_SIZES]
    assert list(catalog.list_input_sizes()) == expected
    assert len(expected) == (14 if torchgeo_models is not None else 6)
    assert catalog.input_size("torchvision:resnet18") is None
    with pytest.raises(catalog.CatalogError, match="is not served"):
        catalog.input_size("torchvision:raft_large")


def test_one_failing_input_size_probe_leaves_only_that_name_out(fresh_catalog, monkeypatch):
    real = catalog._probe_model
    failed = []

    def probe(family, name, kwargs):
        if (family, name) == ("torchvision", "vit_b_16"):
            failed.append(name)
            raise RuntimeError("probe failed")
        return real(family, name, kwargs)

    monkeypatch.setattr(catalog, "_probe_model", probe)
    names = catalog.list_model_classes()
    expected = [
        (name, FIXED_INPUT_SIZES[name])
        for name in names
        if name in FIXED_INPUT_SIZES and name != "torchvision:vit_b_16"
    ]
    assert list(catalog.list_input_sizes()) == expected
    assert list(catalog.list_input_sizes()) == expected
    assert failed == ["vit_b_16"]  # the result is cached: the failing probe runs once


def _runs(model, bands, size, device):
    try:
        with torch.no_grad():
            model(torch.zeros(1, bands, size, size, device=device))
    except Exception:  # the model refuses the size: an assertion, a shape error
        return False
    return True


@pytest.mark.parametrize("model_class", catalog.list_model_classes())
def test_every_served_name_runs_at_its_input_size_only(model_class):
    family, name = catalog.resolve(model_class)
    kwargs = catalog._builder_kwargs(family, name, {})
    bands = 3
    if family == catalog.FAMILY_TORCHGEO and catalog.TORCHGEO_SERVED[name].channel_arg:
        kwargs[catalog.TORCHGEO_SERVED[name].channel_arg] = bands  # tilenet's default is 4
    try:
        with torch.device("meta"):
            model = catalog._construct(family, name, kwargs).eval()
        device = "meta"
    except (NotImplementedError, RuntimeError):  # torchvision RegNet builds on the CPU
        model = catalog._construct(family, name, kwargs).eval()
        device = "cpu"
    size = catalog.input_size(model_class)
    assert size == FIXED_INPUT_SIZES.get(model_class)
    if size is None:
        assert _runs(model, bands, 224, device) and _runs(model, bands, OTHER_SIZE, device)
    else:
        assert _runs(model, bands, size, device) and not _runs(model, bands, OTHER_SIZE, device)


def test_a_fixed_size_model_refuses_a_chip_of_another_size():
    calls = []

    def run(chip, metadata):
        record = SimpleNamespace(backend="pytorch", metadata=metadata, model=None, spec=None)
        return dispatch.run_prediction(
            SimpleNamespace(record=record),
            image=np.zeros(chip, dtype=np.float32),
            options={},
            device_id="cpu:0",
            torch_device="cpu",
            infer=lambda array, return_embeddings: calls.append(array.shape)
            or np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        )

    metadata = {
        "task": 0,
        "model_family": "torchvision",
        "model_class": "torchvision:vit_b_32",
        "num_classes": "3",
        "in_channels": "3",
        "input_size": "224",
    }
    for chip in ((512, 512, 3), (224, 256, 3)):
        with pytest.raises(dispatch.MalformedPayloadError) as refused:
            run(chip, metadata)
        assert str(refused.value) == (
            f"torchvision:vit_b_32 takes only 224 x 224 chips; this chip is {chip[0]} x {chip[1]}"
        )
        assert refused.value.code == dispatch.ERROR_MALFORMED_PAYLOAD
    assert calls == []  # refused before the model ran
    assert _raw(run((224, 224, 3), metadata)).shape == (224, 224)
    assert _raw(run((512, 512, 3), {**metadata, "input_size": ""})).shape == (512, 512)
    assert calls == [(224, 224, 3), (512, 512, 3)]


def test_the_cache_key_includes_the_catalog_arguments():
    def spec(fmt, **metadata):
        msg = inference_pb2.ModelSpec(
            format=fmt, source="/w.pt", model_class="torchvision:resnet18"
        )
        for key, value in metadata.items():
            msg.metadata.add(key=key, value=value)
        return msg

    base = cache_key_from_spec(spec(inference_pb2.PYTORCH))
    five = cache_key_from_spec(spec(inference_pb2.PYTORCH, num_classes="5"))
    assert five != base
    assert cache_key_from_spec(spec(inference_pb2.PYTORCH, num_classes="05")) == five
    assert cache_key_from_spec(spec(inference_pb2.PYTORCH, num_classes="5", note="x")) == five
    assert cache_key_from_spec(spec(inference_pb2.PYTORCH, in_channels="4")) not in (base, five)
    onnx = cache_key_from_spec(spec(inference_pb2.ONNX))
    assert cache_key_from_spec(spec(inference_pb2.ONNX, num_classes="5")) == onnx


# -------------------------------------------------------------- catalog output rule


def _run(output, *, chip=(4, 5, 3), num_classes="3", options=None, family="torchvision"):
    metadata = {"task": 0, "model_class": "torchvision:x"}
    if family:
        metadata.update(model_family=family, num_classes=num_classes, in_channels="3")
    record = SimpleNamespace(backend="pytorch", metadata=metadata, model=None, spec=None)
    lease = SimpleNamespace(record=record)
    return dispatch.run_prediction(
        lease,
        image=np.zeros(chip, dtype=np.float32),
        options=options or {},
        device_id="cpu:0",
        torch_device="cpu",
        infer=lambda array, return_embeddings: output,
    )


def _raw(result):
    assert result.metadata["payload_format"] == "raw"
    return np.frombuffer(result.output, dtype=result.dtype).reshape(result.shape)


def test_a_class_vector_becomes_a_constant_label_tile():
    logits = np.array([[0.1, 2.0, -1.0]], dtype=np.float32)
    result = _run(logits)
    assert result.metadata["result_type"] == "segmentation_mask"
    assert result.metadata["output_family"] == dispatch.OUTPUT_CLASS_VECTOR
    tile = _raw(result)
    assert tile.dtype == np.uint16 and tile.shape == (4, 5) and (tile == 1).all()

    with_confidence = _run(logits, options={"return_confidence": "true", "return_embeddings": "1"})
    bundle = unpack_tensor_bundle(with_confidence.output)
    assert with_confidence.metadata["has_confidence"] == "true"
    assert (bundle["mask"] == 1).all() and bundle["mask"].shape == (4, 5)
    softmax = np.exp(logits[0] - logits[0].max())
    softmax /= softmax.sum()
    np.testing.assert_allclose(bundle["confidence"], np.full((4, 5), softmax[1]), rtol=1e-6)
    np.testing.assert_array_equal(bundle["embeddings"], logits)


def test_per_pixel_logits_keep_the_label_map_path():
    logits = np.random.default_rng(0).normal(size=(1, 3, 4, 5)).astype(np.float32)
    result = _run(logits, options={"task_type": "classification"})
    assert result.metadata["output_family"] == dispatch.OUTPUT_LABEL_MAP
    assert result.metadata["result_type"] == "segmentation_mask"
    np.testing.assert_array_equal(_raw(result), np.argmax(logits[0], axis=0).astype(np.uint16))


@pytest.mark.parametrize(
    "shape, num_classes, options",
    [
        ((1, 8), "", {}),  # no class head
        ((1, 8, 2, 2), "", {}),  # a feature map
        ((1, 8), "0", {}),  # built without a head (timm num_classes=0)
        ((1, 3), "3", {"task_type": "embedding"}),  # the embedding task
    ],
)
def test_features_are_embeddings(shape, num_classes, options):
    features = np.random.default_rng(1).normal(size=shape).astype(np.float32)
    result = _run(features, num_classes=num_classes, options=options)
    assert result.metadata["result_type"] == "embeddings"
    assert result.metadata["output_family"] == dispatch.OUTPUT_EMBEDDINGS
    np.testing.assert_array_equal(unpack_tensor_bundle(result.output)["embeddings"], features)


@pytest.mark.parametrize(
    "shape, num_classes, why",
    [
        ((1, 3, 2, 2), "3", "per-pixel logits must be chip-sized"),
        ((1, 196, 8), "", "the families are"),  # a token sequence
        ((2, 3), "3", "the families are"),  # batch 2
        ((8,), "3", "the families are"),
        ((1, 4), "3", "built with num_classes=3"),
        ((1, 1), "1", "one class logit has no argmax"),
    ],
)
def test_outputs_that_fit_no_family_are_rejected_with_their_shape(shape, num_classes, why):
    with pytest.raises(dispatch.UnsupportedOutputError) as rejected:
        _run(np.zeros(shape, dtype=np.float32), num_classes=num_classes)
    assert f"shape {list(shape)}" in str(rejected.value) and why in str(rejected.value)
    assert rejected.value.code == dispatch.ERROR_INFERENCE_FAILED


def test_models_outside_the_catalog_keep_the_raw_result():
    logits = np.array([[0.1, 2.0, -1.0]], dtype=np.float32)
    result = _run(logits, family="")
    assert result.metadata["result_type"] == "raw" and "output_family" not in result.metadata
    np.testing.assert_array_equal(_raw(result), logits)
