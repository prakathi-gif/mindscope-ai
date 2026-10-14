# MindScope AI — Production Multi-Stage Dockerfile
# Serves both the FastAPI REST inference engine (Port 8000) and the Streamlit dashboard (Port 8501).

FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT_STREAMLIT=8501 \
    PORT_FASTAPI=8000

# Install essential system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications first for Docker layer caching
COPY requirements.txt .

# Install dependencies (CPU-optimized PyTorch by default for lightweight container size)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application code and pre-trained artifacts
COPY . .

# Expose Streamlit and FastAPI ports
EXPOSE 8501 8000

# Health check using FastAPI health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Make start script executable
RUN chmod +x entrypoint.sh

# Default command: launch both services
CMD ["./entrypoint.sh"]

