# ChatGPT Vision Integration

## Overview

The ChatGPT Vision integration provides a parallel inference path that sends images to OpenAI's ChatGPT Vision API for analysis, completely bypassing traditional ML model execution. This enables sophisticated image understanding using state-of-the-art vision-language models without requiring local compute resources.

## Features

- **API Credential Management**: All OpenAI API credentials are securely managed within the inference provider
- **Task-Based Prompts**: Automatic prompt selection based on `Task` enum values
- **Custom Prompts**: Support for custom analysis prompts
- **Image Format Flexibility**: Handles numpy arrays, PIL Images, and file paths
- **Automatic Preprocessing**: Image resizing, format conversion, and base64 encoding
- **Error Handling**: Comprehensive error handling with retry logic for rate limiting
- **Response Processing**: Structured output parsing for different task types

## Quick Start

### 1. Install Dependencies

```bash
pip install openai Pillow
```

### 2. Set API Key

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 3. Basic Usage

```python
import numpy as np
from nxmndr.inference import InferenceSession
from nxmndr.gpt import ChatGptVisionModelSpec, AzureChatGptVisionProvider
from nxmndr.tasks import Task

# Create model specification
spec = ChatGptVisionModelSpec(
    model_name="gpt-4o",
    max_tokens=1000,
    temperature=0.0
)

# Create Azure provider and session (recommended for enterprise)
provider = AzureChatGptVisionProvider(
    endpoint_url="https://your-resource.openai.azure.com/",
    deployment_name="gpt-4-vision-preview"
)
session = InferenceSession(spec, provider)

# Analyze image with custom prompt
image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
result = session.run(image, prompt="Describe what you see in this image.")
print(result)

# Task-based analysis
classification_result = session.run(image, task=Task.IMAGE_CLASSIFICATION)
detection_result = session.run(image, task=Task.OBJECT_DETECTION)
```

## Configuration

### ChatGptVisionModelSpec Parameters

```python
@dataclass
class ChatGptVisionModelSpec(ModelSpec):
    model_name: str = "gpt-4o"                    # OpenAI model to use
    system_prompt: Optional[str] = None           # System-level instructions
    max_tokens: int = 1000                        # Maximum response length
    temperature: float = 0.0                      # Response creativity (0.0-2.0)
    name: Optional[str] = None                    # Optional model identifier
```

### Provider Configuration

```python
# For Azure OpenAI (recommended)
provider = AzureChatGptVisionProvider(
    endpoint_url="https://your-resource.openai.azure.com/",
    deployment_name="gpt-4-vision-preview",
    timeout=60  # Request timeout in seconds
)

# For OpenAI API (alternative)
from nxmndr.gpt import OpenAIChatGptVisionProvider
provider = OpenAIChatGptVisionProvider(
    api_key="your-openai-api-key",
    organization="your-org-id",  # Optional
    project="your-project-id",   # Optional; if omitted the openai client reads OPENAI_PROJECT_ID
    timeout=60
)
```

## Supported Input Formats

The system automatically handles various image input formats:

### Numpy Arrays
```python
# HWC format (Height, Width, Channels)
image_hwc = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

# CHW format (Channels, Height, Width) - automatically converted
image_chw = np.random.rand(3, 224, 224)

# Grayscale - automatically converted to RGB
image_gray = np.random.randint(0, 255, (224, 224), dtype=np.uint8)

# Normalized arrays (0-1 range) - automatically scaled to 0-255
image_normalized = np.random.rand(224, 224, 3)
```

### PIL Images
```python
from PIL import Image

# Load from file
image = Image.open("photo.jpg")

# Create programmatically  
image = Image.new('RGB', (100, 100), color='red')
```

### File Paths
```python
# Direct file path (string or Path object)
result = session.run("path/to/image.jpg", task=Task.IMAGE_CLASSIFICATION)
```

## Task-Specific Analysis

The system provides optimized prompts for different analysis tasks:

### Image Classification
```python
result = session.run(image, task=Task.IMAGE_CLASSIFICATION)
# Uses prompt optimized for identifying and categorizing objects
```

### Object Detection  
```python
result = session.run(image, task=Task.OBJECT_DETECTION)
# Uses prompt optimized for finding and locating objects
```

### Segmentation Analysis
```python
result = session.run(image, task=Task.SEGMENTATION)
# Uses prompt optimized for describing image regions and segments
```

### Custom Analysis
```python
# Medical image analysis
medical_prompt = """
Analyze this image as if it were a medical scan. Identify any areas of 
concern, anomalies, or notable features. Be thorough but note this is 
for educational purposes only.
"""
result = session.run(image, prompt=medical_prompt)

# Architectural analysis
arch_prompt = """
Examine this image from an architectural perspective. Identify structural 
elements, design styles, materials, and spatial composition.
"""
result = session.run(image, prompt=arch_prompt)
```

## Advanced Usage

### System Prompts
```python
spec = ChatGptVisionModelSpec(
    system_prompt="You are an expert art historian. Analyze images with focus on artistic techniques, historical context, and cultural significance."
)
```

### Custom Prompt Functions
```python
from inference.gpt.prompts import create_custom_classification_prompt

# Create classification prompt for specific categories
categories = ["dog", "cat", "bird", "car", "bicycle"]
prompt = create_custom_classification_prompt(categories)
result = session.run(image, prompt=prompt)
```

### Response Processing
```python
# For classification tasks, get structured output
result = session.run(image, task=Task.IMAGE_CLASSIFICATION)

if isinstance(result, dict) and 'classes' in result:
    print("Detected classes:", result['classes'])
    print("Full response:", result['raw_text'])
else:
    print("Text response:", result)
```

## Error Handling

The system includes comprehensive error handling:

### Azure Configuration Issues
```python
try:
    provider = AzureChatGptVisionProvider()
    session = InferenceSession(spec, provider)
    result = session.run(image)
except ChatGptVisionError as e:
    if "endpoint" in str(e):
        print("Check your ENDPOINT_URL environment variable")
    elif "rate limit" in str(e):
        print("Rate limit exceeded, try again later")
```

### Input Validation
```python
try:
    result = session.run(invalid_input)
except PredictionError as e:
    print(f"Input processing failed: {e}")
```

## Rate Limiting and Retries

The provider automatically handles rate limiting with exponential backoff:

- **Max Retries**: 3 attempts by default
- **Exponential Backoff**: 1s, 2s, 4s delays
- **Rate Limit Detection**: Automatic detection of rate limit errors
- **Graceful Degradation**: Clear error messages when limits exceeded

## Performance Considerations

### Image Size Optimization
- Images larger than 2048px are automatically resized
- JPEG compression used for optimal upload size
- Base64 encoding handled automatically

### Token Management
```python
# Adjust max_tokens based on expected response length
spec = ChatGptVisionModelSpec(
    max_tokens=500,    # For short descriptions
    # max_tokens=2000, # For detailed analysis
)
```

### Caching Strategy
Consider implementing response caching for repeated analyses:

```python
# Example: Cache responses by image hash
import hashlib

def get_image_hash(image_array):
    return hashlib.md5(image_array.tobytes()).hexdigest()

# Check cache before making API call
image_hash = get_image_hash(image)
if image_hash in response_cache:
    return response_cache[image_hash]
```

## Integration with Existing Pipeline

The ChatGPT Vision provider seamlessly integrates with the existing inference architecture:

### Unified Interface
```python
# Same interface as other inference providers
session = InferenceSession(spec, provider)  # Works for any provider type
result = session.run(input_data, task=task)  # Consistent API
```

### Provider Switching
```python
# Easy switching between local and Azure ChatGPT inference
if use_azure_chatgpt:
    provider = AzureChatGptVisionProvider()
    spec = ChatGptVisionModelSpec()
else:
    provider = LocalInferenceProvider()
    spec = PytorchModelSpec(...)

session = InferenceSession(spec, provider)
```

### Registry Integration
```python
# Registered with unified model registry
from nxmndr.models import get_registry

registry = get_registry()
spec_class = registry.get_spec_class("chatgpt_vision")
# Returns ChatGptVisionModelSpec
```

## Troubleshooting

### Common Issues

1. **"OpenAI API key is required"**
   - Set `OPENAI_API_KEY` environment variable
   - Or pass `api_key` parameter to provider

2. **"Rate limit exceeded"**
   - Wait for rate limit to reset
   - Upgrade OpenAI plan for higher limits
   - Implement request batching/queuing

3. **"Model not found"**
   - Check model name (use "gpt-4o", "gpt-4-vision-preview", etc.)
   - Verify API access to vision models

4. **"Image too large"**
   - Images are auto-resized, but check input format
   - Ensure image data is valid numpy array or PIL Image

### Debug Mode
```python
import logging
logging.getLogger('inference.gpt').setLevel(logging.DEBUG)

# Enable detailed logging for debugging
```

## Examples

See `example_chatgpt_vision.py` for comprehensive usage examples including:
- Basic image description
- Task-based analysis
- Custom domain-specific prompts  
- Error handling scenarios
- Real image file processing

## Testing

Run the test suite:
```bash
pytest tst/test_chatgpt_vision.py -v
```

The tests include:
- Model specification validation
- Image preprocessing with various formats
- Provider initialization and configuration
- Prompt building and task mapping
- Error handling scenarios
- Integration testing with mocked API responses