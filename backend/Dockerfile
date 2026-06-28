# --- Stage 1: Builder (Using UV to install dependencies) ---
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

# Setting up work environment
WORKDIR /app

# Copy only dependency management files (to leverage Docker cache)
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment (.venv)
# --frozen: Ensures installing exactly what's in the lock file
# --no-cache: Saves space in the image
RUN uv sync --frozen --no-cache

# --- Stage 2: Runtime (The final and lean image) ---
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install basic system tools (optional, sometimes needed for certain PDF/DB libraries)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the Builder
COPY --from=builder /app/.venv /app/.venv

# Copy the application source code
COPY app ./app

# Update PATH to use the virtual environment as default
ENV PATH="/app/.venv/bin:$PATH"

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Default command (will be overridden by docker-compose, but good to have)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]