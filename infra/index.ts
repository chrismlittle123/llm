import { defineConfig, createContainer, createSecret } from "@palindrom-ai/infra";

const config = defineConfig({
  cloud: "gcp",
  region: "europe-west2",
  project: "christopher-little-dev",
  environment: process.env.PULUMI_STACK || "dev",
});

const openaiKey = createSecret("openai-api-key", { config });
const anthropicKey = createSecret("anthropic-api-key", { config });
const langfusePublicKey = createSecret("langfuse-public-key", { config });
const langfuseSecretKey = createSecret("langfuse-secret-key", { config });

const llmGateway = createContainer("llm-gateway", {
  config,
  image: `europe-west2-docker.pkg.dev/christopher-little-dev/llm-gateway/llm-gateway:${process.env.IMAGE_TAG || "latest"}`,
  port: 8000,
  size: "medium",
  minInstances: 1,
  maxInstances: 5,
  environment: {
    SERVICE_NAME: "llm-gateway",
    SERVICE_ENVIRONMENT: "production",
    DEFAULT_MODEL: "gpt-4o",
    DEFAULT_TIMEOUT: "60",
    DEFAULT_MAX_RETRIES: "3",
  },
  link: [openaiKey, anthropicKey, langfusePublicKey, langfuseSecretKey],
});

export const gatewayUrl = llmGateway.url;
