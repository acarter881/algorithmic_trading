FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (needed by thefuzz[speedup] and cryptography)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy project metadata and source
COPY pyproject.toml README.md ./
COPY src/ src/
COPY config/ config/

# Install the package
RUN pip install --no-cache-dir .

# Create directories for runtime data
RUN mkdir -p /data /logs

# Default environment: paper trading
ENV AUTOTRADER__KALSHI__ENVIRONMENT=demo \
    AUTOTRADER__DATABASE__URL=sqlite:////data/autotrader_paper.db \
    AUTOTRADER__LOGGING__LOG_DIR=/logs

VOLUME ["/data", "/logs"]

ENTRYPOINT ["autotrader"]
CMD ["run", "--config-dir", "config"]
