FROM python:3.12-slim

# Install uv from Astral's official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Bytecode compilation for faster container startup
ENV UV_COMPILE_BYTECODE=1

# Install dependencies (pyproject.toml + uv.lock for reproducibility)
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
RUN uv sync --no-dev --frozen

EXPOSE 9000

ENTRYPOINT ["uv", "run", "deepseek-cursor-proxy"]
CMD ["--host", "0.0.0.0", "--port", "9000"]
