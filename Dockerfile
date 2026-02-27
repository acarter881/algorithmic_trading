FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (needed by thefuzz[speedup] and cryptography)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

# Copy application source
COPY src/ src/
COPY config/ config/

# Re-install in editable mode so the entry point resolves
RUN pip install --no-cache-dir -e .

# Create directories for runtime data
RUN mkdir -p /data /logs

# Default environment: paper trading
ENV AUTOTRADER__KALSHI__ENVIRONMENT=demo \
    AUTOTRADER__DATABASE__URL=sqlite:////data/autotrader_paper.db \
    AUTOTRADER__LOGGING__LOG_DIR=/logs

VOLUME ["/data", "/logs"]

ENTRYPOINT ["autotrader"]
CMD ["run", "--config-dir", "config", "--environment", "demo"]
