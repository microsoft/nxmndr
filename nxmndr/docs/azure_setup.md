# Azure OpenAI ChatGPT Vision Setup Guide

This guide helps you set up the ChatGPT Vision integration with Azure OpenAI using Entra ID authentication.

## Prerequisites

1. **Azure Subscription** with access to Azure OpenAI services
2. **Azure OpenAI Resource** with GPT-4 Vision model deployed
3. **Azure CLI** installed and configured
4. **Python Dependencies** installed: `pip install openai azure-identity`

## Step 1: Create Azure OpenAI Resource

1. Go to Azure Portal → Create Resource → Azure OpenAI
2. Choose subscription, resource group, region, and name
3. Select pricing tier (Standard recommended)
4. Review and create the resource

## Step 2: Deploy GPT-4 Vision Model

1. Navigate to your Azure OpenAI resource
2. Go to "Model deployments" → "Create new deployment"
3. Select model: `gpt-4` with vision capabilities
4. Choose deployment name (e.g., `gpt-4-vision-preview`)
5. Set capacity and deploy

## Step 3: Configure Authentication

### Option A: Azure CLI (Recommended for Development)
```bash
az login
```

### Option B: Service Principal (Recommended for Production)
```bash
export AZURE_CLIENT_ID="your-client-id"
export AZURE_CLIENT_SECRET="your-client-secret" 
export AZURE_TENANT_ID="your-tenant-id"
```

### Option C: Managed Identity (For Azure Resources)
No additional configuration needed - automatically available in Azure VMs, App Service, etc.

## Step 4: Set Environment Variables

```bash
export ENDPOINT_URL="https://your-resource-name.openai.azure.com/"
export DEPLOYMENT_NAME="gpt-4-vision-preview"
```

Replace:
- `your-resource-name` with your Azure OpenAI resource name
- `gpt-4-vision-preview` with your actual deployment name

## Step 5: Test the Setup

Run the example script to verify everything works:

```bash
python examples/example_chatgpt_vision.py
```

Or use the Azure provider directly:

```python
from nxmndr.gpt import AzureChatGptVisionProvider, ChatGptVisionModelSpec
from nxmndr.inference import InferenceSession

# Create provider
provider = AzureChatGptVisionProvider(
    endpoint_url="https://your-resource.openai.azure.com/",
    deployment_name="gpt-4-vision-preview"
)

# Create model spec
spec = ChatGptVisionModelSpec(model_name="gpt-4o")

# Create session and run inference
session = InferenceSession(spec, provider)
result = session.run("path/to/image.jpg", prompt="Describe this image")
print(result)
```

## Troubleshooting

### Authentication Issues
- Ensure you're logged in: `az account show`
- Check credentials: `az account get-access-token --resource https://cognitiveservices.azure.com/`
- For service principal: verify client ID, secret, and tenant ID

### Endpoint Issues
- Verify endpoint URL format: `https://your-resource.openai.azure.com/`
- Check resource exists: `az cognitiveservices account show --name your-resource --resource-group your-rg`
- Ensure deployment exists and is active

### Permission Issues
- Verify role assignment: "Cognitive Services OpenAI User" or "Cognitive Services OpenAI Contributor"
- Check Azure RBAC: `az role assignment list --assignee your-user-id`

### Model Availability
- Confirm GPT-4 Vision is available in your region
- Check deployment status in Azure Portal
- Verify model deployment name matches DEPLOYMENT_NAME environment variable

## Security Best Practices

1. **Use Managed Identity** when possible (Azure resources)
2. **Rotate credentials** regularly for service principals  
3. **Store secrets securely** using Azure Key Vault
4. **Apply least privilege** - only grant necessary permissions
5. **Monitor usage** through Azure Monitor and logs

## Environment-Specific Configuration

### Development
- Use Azure CLI authentication (`az login`)
- Set environment variables in `.env` file
- Test with small images first

### Staging/Production
- Use Managed Identity or Service Principal
- Configure secrets in Azure Key Vault
- Set up monitoring and alerting
- Use dedicated Azure OpenAI resource per environment

## Cost Optimization

1. **Monitor token usage** - GPT-4 Vision is more expensive
2. **Optimize image sizes** - resize images before sending
3. **Use appropriate models** - consider GPT-4o vs GPT-4-vision-preview
4. **Set quotas** to prevent unexpected costs
5. **Implement caching** for repeated analysis

## Alternative: OpenAI Platform (non-Azure) via the proxy

The `--azure-proxy` server can also front models on the OpenAI platform
(`https://api.openai.com`) — useful when an Azure subscription has no quota for a
model. Any endpoint whose URL host is `api.openai.com` is authenticated with
`OPENAI_API_KEY` instead of Azure credentials, forwarded to `/v1/images/edits` or
`/v1/chat/completions` (no `api-version`), and the deployment name is sent as the
`model` field, replacing any `model` the client sends. The plugin-facing routes are
unchanged, so the QGIS plugin only needs
to be in proxy mode pointing at this server. `<PREFIX>_PROVIDER=openai` (or
`azure_openai`) sets the identity explicitly for any host, for example a loopback test
stub; without it the host decides as above. Routes and request shapes are defined in
`nxmndr/server/proxy_routes.py`.

```bash
export OPENAI_API_KEY="sk-..."             # from platform.openai.com/api-keys
export VISION_DEPLOYMENT_NAME="gpt-image-1" # any Image API model, e.g. gpt-image-2
export VISION_ENDPOINT_URL="https://api.openai.com"
export VISION_TYPE="vision"
nxmndr-server --azure-proxy --http-port 8080
```

The proxy listens on 127.0.0.1 unless started with `--host`; it has no authentication
and spends these credentials for every caller.

No `az login` is needed for a proxy that only fronts OpenAI-platform endpoints; the
Azure credential is created lazily on the first Azure-bound request. GPT Image
models on the OpenAI platform may require completing API Organization Verification
in the OpenAI developer console before first use.

## Additional Resources

- [Azure OpenAI Documentation](https://docs.microsoft.com/azure/cognitive-services/openai/)
- [Azure Identity Documentation](https://docs.microsoft.com/python/api/azure-identity/)
- [OpenAI Python SDK](https://github.com/openai/openai-python)