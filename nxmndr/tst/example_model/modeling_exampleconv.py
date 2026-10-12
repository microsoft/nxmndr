# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Unified example model (test-only).

Contains:
  * ExampleModel (simple conv classifier)
  * HuggingFace wrapper (ExampleConvConfig, ExampleConvModel)
  * Export utilities for PyTorch & ONNX (export, round_trip)

Kept exclusively under tst/ to avoid leaking experimental code into the main package.
Referenced by config.json via trust_remote_code auto_map.
"""

from pathlib import Path

import onnxruntime as ort
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, ImageProcessingMixin
import numpy as np

cwd = Path(__file__).parent
ONNX_PATH = cwd / "example_model.onnx"
PYTORCH_PATH = cwd / "example_model.pth"
HF_WEIGHTS_PATH = cwd / "pytorch_model.bin"


class ExampleImageProcessor(ImageProcessingMixin):
    """Identity image processor for ExampleConvModel.

    This processor performs no transformations, passing through input as-is.
    """

    model_input_names = ["pixel_values"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, images, return_tensors=None, **kwargs):
        """Process images with identity transform.

        Args:
            images: Input images (numpy array, torch tensor, or PIL images)
            return_tensors: "pt" for PyTorch tensors, "np" for numpy, None for list

        Returns:
            Dictionary with "pixel_values" key containing processed images
        """
        # Convert to numpy if needed
        if torch.is_tensor(images):
            images_np = images.cpu().numpy()
        elif isinstance(images, np.ndarray):
            images_np = images
        else:
            # Assume PIL or similar - convert to numpy
            images_np = np.array(images)

        # Ensure 4D shape (batch, channels, height, width)
        if images_np.ndim == 3:
            images_np = images_np[np.newaxis, ...]

        # Convert to float32 and ensure writable copy
        images_np = np.array(images_np, dtype=np.float32, copy=True)

        # Return based on requested format
        if return_tensors == "pt":
            pixel_values = torch.from_numpy(images_np)
        elif return_tensors == "np":
            pixel_values = images_np
        else:
            pixel_values = images_np.tolist()

        return {"pixel_values": pixel_values}


class ExampleConvConfig(PretrainedConfig):
    model_type = "exampleconv"

    def __init__(
        self,
        num_channels: int = 3,
        image_size: int = 32,
        hidden_channels: int = 16,
        num_labels: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.image_size = image_size
        self.hidden_channels = hidden_channels
        self.num_labels = num_labels


class ExampleModel(nn.Module):
    def __init__(
        self,
        num_channels: int = 3,
        hidden_channels: int = 16,
        image_size: int = 32,
        num_labels: int = 10,
    ):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_channels * image_size * image_size, num_labels)
        self._image_size = image_size

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ExampleConvModel(PreTrainedModel):
    config_class = ExampleConvConfig
    base_model_prefix = "exampleconv"

    def __init__(self, config: ExampleConvConfig):
        super().__init__(config)
        self.backbone = ExampleModel(
            num_channels=config.num_channels,
            hidden_channels=config.hidden_channels,
            image_size=config.image_size,
            num_labels=config.num_labels,
        )
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.LongTensor | None = None,
        return_dict: bool = True,
    ):
        logits = self.backbone(pixel_values)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        if not return_dict:
            return (logits,) if loss is None else (loss, logits)
        return {"loss": loss, "logits": logits}


def export(target_dir: str | Path | None = None, seed: int | None = None):
    """Export artifacts into target_dir (defaults to tst/example_model).

    If the default PyTorch artifact already exists in tst/example_model, reuse it
    to avoid recreating test fixtures.

    ``seed``: when given, ``torch.manual_seed(seed)`` runs before the model is
    constructed, so the weights (and the export input) are the same on every run
    and every machine. tst/conftest.py pins it.
    """
    base_dir = Path(target_dir) if target_dir is not None else cwd
    base_dir.mkdir(parents=True, exist_ok=True)

    pth_path = base_dir / "example_model.pth"
    onnx_path = base_dir / "example_model.onnx"
    hf_weights_path = base_dir / "pytorch_model.bin"

    if target_dir is None and PYTORCH_PATH.exists():
        return

    if seed is not None:
        torch.manual_seed(seed)
    model = ExampleModel().eval()
    dummy_input = torch.randn(1, 3, 32, 32)

    torch.save(model.state_dict(), pth_path)
    torch.save(model.state_dict(), hf_weights_path)
    torch.onnx.export(model, dummy_input, str(onnx_path), opset_version=11, export_params=True)

    # Newer PyTorch may write external data files (.onnx.data); consolidate
    # into a single self-contained ONNX file so the server can copy it standalone.
    # onnx is available as a transitive dep of onnxscript (used by torch.onnx).
    data_path = Path(str(onnx_path) + ".data")
    if data_path.exists():
        import onnx
        onnx_model = onnx.load(str(onnx_path), load_external_data=True)
        onnx.save_model(onnx_model, str(onnx_path), save_as_external_data=False)
        data_path.unlink(missing_ok=True)


def round_trip():
    """Validate ONNX output numerically matches PyTorch forward pass."""
    model = ExampleModel().eval()
    model.load_state_dict(torch.load(PYTORCH_PATH, weights_only=True))
    dummy_input = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        original_output = model(dummy_input)
    ort_session = ort.InferenceSession(str(ONNX_PATH))
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    onnx_output = torch.tensor(ort_outs[0])
    assert torch.allclose(original_output, onnx_output, atol=1e-6), "Outputs are not close!"


__all__ = [
    "ExampleModel",
    "ExampleConvConfig",
    "ExampleConvModel",
    "ExampleImageProcessor",
    "export",
    "round_trip",
    "ONNX_PATH",
    "PYTORCH_PATH",
    "HF_WEIGHTS_PATH",
]

if __name__ == "__main__":
    export()
