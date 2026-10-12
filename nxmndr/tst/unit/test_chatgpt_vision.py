# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Tests for ChatGPT Vision inference integration.
"""

import os
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from nxmndr.exceptions import PredictionError, ValidationError
from nxmndr.gpt import (
    AzureChatGptVisionProvider,
    ChatGptVisionModel,
    ChatGptVisionModelSpec,
    OpenAIChatGptVisionProvider,
    build_prompt,
    get_task_prompt,
)
from nxmndr.inference import InferenceSession
from nxmndr.tasks import Task


class TestChatGptVisionModelSpec:
    """Test ChatGPT Vision model specification."""

    def test_default_values(self):
        """Test default model spec values."""
        spec = ChatGptVisionModelSpec()

        assert spec.model_name == "gpt-4o"
        assert spec.system_prompt is None
        assert spec.max_tokens == 1000
        assert spec.temperature == 0.0
        assert spec.name is None

    def test_custom_values(self):
        """Test custom model spec values."""
        spec = ChatGptVisionModelSpec(
            model_name="gpt-4-vision-preview",
            system_prompt="Test prompt",
            max_tokens=500,
            temperature=0.5,
            name="test-model",
        )

        assert spec.model_name == "gpt-4-vision-preview"
        assert spec.system_prompt == "Test prompt"
        assert spec.max_tokens == 500
        assert spec.temperature == 0.5
        assert spec.name == "test-model"

    def test_validation_success(self):
        """Test successful validation."""
        spec = ChatGptVisionModelSpec()
        spec.validate()  # Should not raise

    def test_validation_negative_tokens(self):
        """Test validation with negative max_tokens."""
        spec = ChatGptVisionModelSpec(max_tokens=-1)

        with pytest.raises(ValidationError, match="max_tokens must be positive"):
            spec.validate()

    def test_validation_invalid_temperature(self):
        """Test validation with invalid temperature."""
        spec = ChatGptVisionModelSpec(temperature=3.0)

        with pytest.raises(ValidationError, match="temperature must be between"):
            spec.validate()

    def test_from_json(self):
        """Test JSON deserialization."""
        json_str = '{"model_name": "gpt-4o", "max_tokens": 800, "name": "test"}'
        spec = ChatGptVisionModelSpec.from_json(json_str)

        assert spec.model_name == "gpt-4o"
        assert spec.max_tokens == 800
        assert spec.name == "test"


class TestChatGptVisionModel:
    """Test ChatGPT Vision model wrapper."""

    def setup_method(self):
        """Set up test fixtures."""
        self.spec = ChatGptVisionModelSpec()
        self.provider = Mock()
        self.model = ChatGptVisionModel(self.spec, self.provider)

    def test_preprocess_numpy_array(self):
        """Test preprocessing numpy array."""
        # Create test image (HWC format)
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        result = self.model.preprocess(image_array)

        assert "image_b64" in result
        assert "original_size" in result
        assert "format" in result
        assert result["format"] == "jpeg"
        assert isinstance(result["image_b64"], str)

    def test_preprocess_grayscale_array(self):
        """Test preprocessing grayscale numpy array."""
        # Create grayscale image
        gray_array = np.random.randint(0, 255, (224, 224), dtype=np.uint8)

        result = self.model.preprocess(gray_array)

        assert "image_b64" in result
        # Should convert grayscale to RGB
        assert result["original_size"] == (224, 224)

    def test_preprocess_chw_format(self):
        """Test preprocessing CHW format array."""
        # Create image in CHW format (channels first)
        chw_array = np.random.rand(3, 224, 224)

        result = self.model.preprocess(chw_array)

        assert "image_b64" in result
        assert result["original_size"] == (224, 224)

    def test_preprocess_pil_image(self):
        """Test preprocessing PIL Image."""
        pil_image = Image.new("RGB", (100, 100), color="red")

        result = self.model.preprocess(pil_image)

        assert "image_b64" in result
        assert result["original_size"] == (100, 100)

    def test_preprocess_large_image_resize(self):
        """Test that large images get resized."""
        # Create large image that should be resized
        large_image = np.random.randint(0, 255, (3000, 3000, 3), dtype=np.uint8)

        result = self.model.preprocess(large_image)

        # Should be resized to fit within limits
        size = result["original_size"]
        assert max(size) <= 2048

    def test_preprocess_invalid_input(self):
        """Test preprocessing with invalid input."""
        with pytest.raises(
            PredictionError, match="Failed to preprocess image: Unsupported input type"
        ):
            self.model.preprocess([1, 2, 3])  # Invalid type

    def test_predict_delegates_to_provider(self):
        """Test that predict delegates to provider."""
        test_data = np.zeros((10, 10, 3))
        expected_result = "test result"
        self.provider.predict.return_value = expected_result

        result = self.model.predict(test_data)

        assert result == expected_result
        self.provider.predict.assert_called_once_with(test_data)

    def test_predict_no_provider(self):
        """Test predict without provider."""
        model = ChatGptVisionModel(self.spec, None)

        with pytest.raises(PredictionError, match="No inference provider"):
            model.predict(np.zeros((10, 10, 3)))

    def test_postprocess_dict_response(self):
        """Test postprocessing ChatGPT API response."""
        api_response = {"choices": [{"message": {"content": "Test response content"}}]}

        result = self.model.postprocess(api_response)

        assert result == "Test response content"

    def test_postprocess_classification_task(self):
        """Test postprocessing for classification task."""
        text_response = """
        1. Dog
        2. Cat  
        3. Bird
        """

        result = self.model.postprocess(text_response, Task.IMAGE_CLASSIFICATION)

        assert isinstance(result, dict)
        assert "classes" in result
        assert "Dog" in result["classes"]
        assert "Cat" in result["classes"]


class TestOpenAIChatGptVisionProvider:
    """Test OpenAI-specific ChatGPT Vision inference provider."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        provider = OpenAIChatGptVisionProvider(api_key="test-key")

        assert provider.api_key == "test-key"
        assert provider.timeout == 30
        assert provider._client is None

    def test_init_from_environment(self):
        """Test initialization from environment variable."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            provider = OpenAIChatGptVisionProvider()

            assert provider.api_key == "env-key"

    def test_get_provider_name(self):
        """Test provider name."""
        provider = OpenAIChatGptVisionProvider(api_key="test-key")
        assert provider._get_provider_name() == "OpenAI"

    def test_get_model_identifier(self):
        """Test model identifier uses model name."""
        spec = ChatGptVisionModelSpec(model_name="gpt-4o")
        provider = OpenAIChatGptVisionProvider(api_key="test-key")
        provider.model_spec = spec

        assert provider._get_model_identifier() == "gpt-4o"

    @patch("openai.OpenAI")
    def test_get_client_initialization(self, mock_openai):
        """Test OpenAI client initialization."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_client.models.list.return_value = []  # Mock validation

        provider = OpenAIChatGptVisionProvider(api_key="test-key")
        client = provider._get_client()

        assert client == mock_client
        mock_openai.assert_called_once_with(api_key="test-key", timeout=30)
        mock_client.models.list.assert_called_once()

    @patch("openai.OpenAI")
    def test_get_client_passes_organization_and_project(self, mock_openai):
        """An explicit project reaches the OpenAI client beside the organization."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_client.models.list.return_value = []

        provider = OpenAIChatGptVisionProvider(api_key="test-key", organization="org-1", project="proj-1")
        provider._get_client()

        mock_openai.assert_called_once_with(
            api_key="test-key", timeout=30, organization="org-1", project="proj-1"
        )


class TestAzureChatGptVisionProvider:
    """Test Azure OpenAI-specific ChatGPT Vision inference provider."""

    def test_init_with_endpoint(self):
        """Test initialization with explicit endpoint."""
        provider = AzureChatGptVisionProvider(
            endpoint_url="https://test.openai.azure.com/", deployment_name="gpt-4-vision"
        )

        assert provider.endpoint_url == "https://test.openai.azure.com/"
        assert provider.deployment_name == "gpt-4-vision"
        assert provider.timeout == 30
        assert provider._client is None

    def test_init_from_environment(self):
        """Test initialization from environment variables."""
        with patch.dict(
            os.environ,
            {"ENDPOINT_URL": "https://env.openai.azure.com/", "DEPLOYMENT_NAME": "env-deployment"},
        ):
            provider = AzureChatGptVisionProvider()

            assert provider.endpoint_url == "https://env.openai.azure.com/"
            assert provider.deployment_name == "env-deployment"

    def test_init_no_endpoint(self):
        """Test initialization without endpoint raises error."""
        with pytest.raises(Exception, match="Azure endpoint URL is required"):
            AzureChatGptVisionProvider()

    def test_get_provider_name(self):
        """Test provider name."""
        provider = AzureChatGptVisionProvider(endpoint_url="https://test.openai.azure.com/")
        assert provider._get_provider_name() == "Azure OpenAI"

    def test_get_model_identifier(self):
        """Test model identifier uses deployment name."""
        provider = AzureChatGptVisionProvider(
            endpoint_url="https://test.openai.azure.com/", deployment_name="my-deployment"
        )

        assert provider._get_model_identifier() == "my-deployment"

    @patch("openai.AzureOpenAI")
    @patch("azure.identity.get_bearer_token_provider")
    @patch("azure.identity.DefaultAzureCredential")
    def test_get_client_initialization(
        self, mock_credential, mock_token_provider, mock_azure_openai
    ):
        """Test Azure OpenAI client initialization."""
        mock_client = Mock()
        mock_azure_openai.return_value = mock_client
        mock_client.models.list.return_value = []  # Mock validation
        mock_token_provider.return_value = Mock()
        mock_credential.return_value = Mock()

        provider = AzureChatGptVisionProvider(
            endpoint_url="https://test.openai.azure.com/", deployment_name="test-deployment"
        )
        client = provider._get_client()

        assert client == mock_client
        mock_azure_openai.assert_called_once()
        call_args = mock_azure_openai.call_args[1]
        assert call_args["azure_endpoint"] == "https://test.openai.azure.com/"
        assert call_args["api_version"] == "2025-01-01-preview"
        assert call_args["timeout"] == 30
        mock_client.models.list.assert_called_once()


class TestPromptFunctions:
    """Test prompt building functions."""

    def test_get_task_prompt_classification(self):
        """Test getting classification task prompt."""
        prompt = get_task_prompt(Task.IMAGE_CLASSIFICATION)

        assert "identify" in prompt.lower()
        assert "objects" in prompt.lower()

    def test_get_task_prompt_detection(self):
        """Test getting detection task prompt."""
        prompt = get_task_prompt(Task.OBJECT_DETECTION)

        assert "objects" in prompt.lower()
        assert "locate" in prompt.lower() or "location" in prompt.lower()

    def test_build_prompt_custom(self):
        """Test building prompt with custom text."""
        custom = "Custom analysis prompt"
        result = build_prompt(custom_prompt=custom)

        assert custom in result

    def test_build_prompt_task_based(self):
        """Test building prompt based on task."""
        result = build_prompt(task=Task.IMAGE_CLASSIFICATION)

        assert "identify" in result.lower()

    def test_build_prompt_with_system(self):
        """Test building prompt with system instructions."""
        result = build_prompt(
            task=Task.IMAGE_CLASSIFICATION, system_prompt="Be precise and detailed"
        )

        assert "System instructions" in result
        assert "Be precise and detailed" in result

    def test_build_prompt_precedence(self):
        """Test that custom prompt takes precedence over task."""
        custom = "Custom prompt"
        result = build_prompt(task=Task.IMAGE_CLASSIFICATION, custom_prompt=custom)

        assert custom in result
        # Should not contain task-specific text since custom takes precedence
        task_prompt = get_task_prompt(Task.IMAGE_CLASSIFICATION)
        assert task_prompt not in result


class TestIntegration:
    """Integration tests with InferenceSession."""

    def test_inference_session_integration(self):
        """Test full integration with InferenceSession using OpenAI provider."""
        spec = ChatGptVisionModelSpec()
        provider = OpenAIChatGptVisionProvider(api_key="test-key")

        # Mock the provider's predict method
        provider.predict = Mock(return_value="Mock response")

        session = InferenceSession(spec, provider)

        # Mock the provider initialization to avoid real API calls
        with patch.object(provider, "_get_client"):
            with patch.object(provider, "load_spec"):
                test_image = np.zeros((100, 100, 3), dtype=np.uint8)

                result = session.run(test_image, prompt="Test prompt")

                assert result == "Mock response"
                provider.predict.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
