# ============================================================================
# DOCKERFILE - ARM64 for AgentCore Runtime
# ============================================================================
# File: Dockerfile

FROM --platform=linux/arm64 ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-cache

# Copy application code
COPY agent.py ./

# Expose AgentCore Runtime port
EXPOSE 8080

# Run with ADOT auto-instrumentation for observability
CMD ["opentelemetry-instrument", "uv", "run", "uvicorn", "agent:api", "--host", "0.0.0.0", "--port", "8080"]
