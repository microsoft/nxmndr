# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
ChatGPT Vision inference provider with comprehensive API credential management.
"""

import os
import time
from typing import Optional, Union, Dict, Any

from nxmndr.tasks import Task
from ..exceptions import InferenceError, ValidationError
from ..logging import get_logger
from ..inference.inference import InferenceProvider
from .models import ChatGptVisionModelSpec, Gpt4VisionModelSpec, Gpt5VisionModelSpec
from .prompts import build_prompt

logger = get_logger(__name__)


class ChatGptVisionError(InferenceError):
    """Specific error for ChatGPT Vision API issues."""

    pass


class ChatGptVisionInferenceProvider(InferenceProvider):
    """
    Base class for ChatGPT Vision inference providers.

    Handles common functionality like rate limiting, error handling, and API calls.
    """

    def __init__(self, timeout: int = 30):
        """
        Initialize base ChatGPT Vision inference provider.

        Args:
            timeout: Request timeout in seconds
        """
        super().__init__()

        self.timeout = timeout

        # Client will be initialized lazily by subclasses
        self._client = None
        self.model_spec: Optional[ChatGptVisionModelSpec] = None

        # Rate limiting and retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0  # Base delay between retries

        # Register the model with the registry
        self._register_model()

    def _register_model(self):
        """Register ChatGPT Vision model with the model registry."""
        try:
            from ..models import get_registry
            from .models import ChatGptVisionModel, ChatGptVisionModelSpec

            def load_chatgpt_vision(spec, provider, session):
                """Load ChatGPT Vision model."""
                if not isinstance(provider, ChatGptVisionInferenceProvider):
                    raise ValueError(
                        "ChatGptVisionModelSpec requires ChatGptVisionInferenceProvider"
                    )
                return ChatGptVisionModel(spec, provider)

            registry = get_registry()
            registry.register_spec("chatgpt_vision", ChatGptVisionModelSpec, load_chatgpt_vision)
            logger.debug("Registered ChatGPT Vision model with registry")
        except Exception as e:
            logger.warning(f"Failed to register ChatGPT Vision model: {e}")

    def _get_client(self):
        """Get or initialize the OpenAI client. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _get_client")

    def _get_model_identifier(self) -> str:
        """Get the model identifier to use in API calls. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _get_model_identifier")

    def _get_provider_name(self) -> str:
        """Get the provider name for logging. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _get_provider_name")

    def _validate_client(self):
        """Validate client credentials by making a test request."""
        try:
            # Make a minimal request to validate credentials
            self._client.models.list()
        except Exception as e:
            error_msg = str(e).lower()
            if (
                "authentication" in error_msg
                or "api key" in error_msg
                or "unauthorized" in error_msg
            ):
                raise ChatGptVisionError(
                    f"{self._get_provider_name()} authentication failed - check credentials"
                )
            elif "organization" in error_msg:
                raise ChatGptVisionError("Invalid OpenAI organization ID")
            elif "endpoint" in error_msg:
                raise ChatGptVisionError("Invalid endpoint URL")
            else:
                # Log but don't fail - might be temporary network issue
                logger.warning(f"Client validation warning: {e}")

    def load_spec(
        self, model_spec: Union[ChatGptVisionModelSpec, Gpt4VisionModelSpec, Gpt5VisionModelSpec]
    ):
        """Load model specification and validate configuration."""
        if not isinstance(
            model_spec, (ChatGptVisionModelSpec, Gpt4VisionModelSpec, Gpt5VisionModelSpec)
        ):
            raise ValidationError(
                f"Expected ChatGptVisionModelSpec, Gpt4VisionModelSpec, or Gpt5VisionModelSpec, got {type(model_spec)}"
            )

        # Validate the spec
        model_spec.validate()

        self.model_spec = model_spec

        # Initialize client to validate credentials early unless skipped for tests
        if not self._should_skip_client_validation():
            self._get_client()

        logger.info(
            f"Loaded ChatGPT Vision model spec: {model_spec.model_name} on {self._get_provider_name()}"
        )

    def _get_token_params(self) -> Dict[str, Any]:
        """Get appropriate token parameters based on model type."""
        from .models import Gpt5VisionModelSpec

        if isinstance(self.model_spec, Gpt5VisionModelSpec):
            # GPT-5 uses max_completion_tokens
            return {"max_completion_tokens": self.model_spec.max_completion_tokens}
        else:
            # GPT-4 and earlier use max_tokens
            return {"max_tokens": self.model_spec.max_tokens}

    def predict(self, input_data, task: Optional[Task] = None, prompt: Optional[str] = None) -> str:
        """
        Perform inference using ChatGPT Vision API.

        Args:
            input_data: Image data (numpy array, PIL Image, or file path)
            task: Optional task type for prompt selection
            prompt: Optional custom prompt (overrides task-based prompt)

        Returns:
            String response from ChatGPT Vision API
        """
        if not self.model_spec:
            raise ChatGptVisionError("Model spec not loaded. Call load_spec() first.")

        try:
            # Import the model class here to avoid circular imports
            from .models import ChatGptVisionModel

            # Create model instance for preprocessing
            model = ChatGptVisionModel(self.model_spec, self)

            # Preprocess image
            processed_data = model.preprocess(input_data, task)
            image_b64 = processed_data["image_b64"]

            # Build prompt
            final_prompt = build_prompt(
                task=task, custom_prompt=prompt, system_prompt=self.model_spec.system_prompt
            )

            # Make API call with retry logic
            response = self._call_api_with_retry(image_b64, final_prompt)

            # Postprocess response
            return model.postprocess(response, task)

        except Exception as e:
            if isinstance(e, ChatGptVisionError):
                raise
            else:
                raise ChatGptVisionError(f"Prediction failed: {str(e)}")

    def _call_api_with_retry(self, image_b64: str, prompt: str) -> Dict[str, Any]:
        """Make API call with retry logic for rate limiting and transient errors."""
        client = self._get_client()

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(
                    f"Making {self._get_provider_name()} Vision API call (attempt {attempt + 1})"
                )

                response = client.chat.completions.create(
                    model=self._get_model_identifier(),
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                                },
                            ],
                        }
                    ],
                    **self._get_token_params(),
                    temperature=self.model_spec.temperature,
                )

                logger.debug("ChatGPT Vision API call successful")
                return response.model_dump()

            except Exception as e:
                error_msg = str(e).lower()

                # Handle rate limiting
                if "rate limit" in error_msg or "quota" in error_msg:
                    if attempt < self.max_retries:
                        delay = self.retry_delay * (2**attempt)  # Exponential backoff
                        logger.warning(
                            f"Rate limited, retrying in {delay}s (attempt {attempt + 1})"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        raise ChatGptVisionError("Rate limit exceeded, max retries reached")

                # Handle authentication errors (don't retry)
                elif "authentication" in error_msg or "unauthorized" in error_msg:
                    raise ChatGptVisionError(
                        f"{self._get_provider_name()} authentication failed - check credentials"
                    )

                # Handle model/deployment not found (don't retry)
                elif (
                    "model" in error_msg
                    and "not found" in error_msg
                    or "deployment" in error_msg
                    and "not found" in error_msg
                ):
                    raise ChatGptVisionError(
                        f"{self._get_provider_name()} model/deployment {self._get_model_identifier()} not available"
                    )

                # Handle other transient errors
                elif attempt < self.max_retries and any(
                    keyword in error_msg
                    for keyword in ["timeout", "connection", "network", "temporary", "server error"]
                ):
                    delay = self.retry_delay * (attempt + 1)
                    logger.warning(f"Transient error, retrying in {delay}s: {e}")
                    time.sleep(delay)
                    continue

                # Re-raise non-retryable errors
                else:
                    raise ChatGptVisionError(f"API call failed: {str(e)}")

        raise ChatGptVisionError("Max retries exceeded")

    # --- Internal helpers ---
    def _should_skip_client_validation(self) -> bool:
        """Return True if credential validation should be skipped (e.g., tests).

        Skips when API key resembles a test key or explicit env override is set.
        """
        if os.getenv("NXMNDR_SKIP_OPENAI_VALIDATION"):
            return True
        api_key = getattr(self, "api_key", None)
        if not api_key:
            return False
        lowered = api_key.lower()
        if lowered in {"test-key", "dummy", "fake", "sk-test"} or lowered.startswith("test-"):
            return True
        return False

    def has_gpu(self) -> bool:
        """ChatGPT Vision API runs on OpenAI's infrastructure."""
        return True  # OpenAI handles GPU acceleration

    def close(self):
        """Clean up resources."""
        if self._client:
            # OpenAI client doesn't need explicit cleanup
            self._client = None
        logger.debug("ChatGptVisionInferenceProvider closed")


class OpenAIChatGptVisionProvider(ChatGptVisionInferenceProvider):
    """
    ChatGPT Vision inference provider for OpenAI API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 30,
        project: Optional[str] = None,
    ):
        """
        Initialize OpenAI ChatGPT Vision inference provider.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var
            organization: OpenAI organization ID. If None, reads from OPENAI_ORG_ID env var
            base_url: Custom API base URL. If None, uses OpenAI default
            timeout: Request timeout in seconds
            project: OpenAI project ID, sent as the client's ``project``. If None, the
                openai client reads OPENAI_PROJECT_ID itself
        """
        super().__init__(timeout=timeout)

        # API credential management
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.organization = organization or os.getenv("OPENAI_ORG_ID")
        self.project = project
        self.base_url = base_url

        logger.info("Initialized OpenAI ChatGPT Vision provider")

    def _get_client(self):
        """Lazy initialization of OpenAI client with credential validation."""
        if self._client is None:
            try:
                from openai import OpenAI

                if not self.api_key:
                    raise ChatGptVisionError(
                        "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                        "or provide api_key parameter."
                    )

                client_kwargs = {"api_key": self.api_key, "timeout": self.timeout}

                if self.organization:
                    client_kwargs["organization"] = self.organization
                if self.project:
                    client_kwargs["project"] = self.project
                if self.base_url:
                    client_kwargs["base_url"] = self.base_url

                self._client = OpenAI(**client_kwargs)

                # Test client validity with a minimal request
                self._validate_client()

                logger.info("OpenAI client initialized successfully")

            except ImportError:
                raise ChatGptVisionError("OpenAI package not installed. Run: pip install openai")
            except Exception as e:
                raise ChatGptVisionError(f"Failed to initialize OpenAI client: {str(e)}")

        return self._client

    def _get_model_identifier(self) -> str:
        """Get the OpenAI model name."""
        return self.model_spec.model_name

    def _get_provider_name(self) -> str:
        """Get the provider name for logging."""
        return "OpenAI"


class AzureChatGptVisionProvider(ChatGptVisionInferenceProvider):
    """
    ChatGPT Vision inference provider for Azure OpenAI with Entra ID authentication.
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        deployment_name: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Initialize Azure ChatGPT Vision inference provider.

        Args:
            endpoint_url: Azure OpenAI endpoint URL. If None, reads from ENDPOINT_URL env var
            deployment_name: Azure OpenAI deployment name. If None, reads from DEPLOYMENT_NAME env var
            timeout: Request timeout in seconds
        """
        super().__init__(timeout=timeout)

        # Azure OpenAI configuration
        self.endpoint_url = endpoint_url or os.getenv("ENDPOINT_URL")
        self.deployment_name = deployment_name or os.getenv("DEPLOYMENT_NAME", "gpt-5")

        if not self.endpoint_url:
            raise ChatGptVisionError(
                "Azure endpoint URL is required. Set ENDPOINT_URL environment variable "
                "or provide endpoint_url parameter."
            )

        logger.info(f"Initialized Azure ChatGPT Vision provider with endpoint: {self.endpoint_url}")

    def _get_client(self):
        """Lazy initialization of Azure OpenAI client with Entra ID authentication."""
        if self._client is None:
            try:
                from openai import AzureOpenAI
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider

                # Initialize Azure credential provider
                token_provider = get_bearer_token_provider(
                    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
                )

                self._client = AzureOpenAI(
                    azure_endpoint=self.endpoint_url,
                    azure_ad_token_provider=token_provider,
                    api_version="2025-01-01-preview",
                    timeout=self.timeout,
                )

                # Test client validity with a minimal request
                self._validate_client()

                logger.info(
                    f"Azure OpenAI client initialized successfully for endpoint: {self.endpoint_url}"
                )

            except ImportError:
                raise ChatGptVisionError(
                    "Required packages not installed. Run: pip install openai azure-identity"
                )
            except Exception as e:
                raise ChatGptVisionError(f"Failed to initialize Azure OpenAI client: {str(e)}")

        return self._client

    def _get_model_identifier(self) -> str:
        """Get the Azure deployment name."""
        return self.deployment_name

    def _get_provider_name(self) -> str:
        """Get the provider name for logging."""
        return "Azure OpenAI"
