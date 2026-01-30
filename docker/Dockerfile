# Multi-stage build for vector cache optimizer
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -u 1001 appuser

# Copy requirements first for better caching
COPY requirements.txt .

# Build stage - install all dependencies including build tools
FROM base as builder

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Production stage - minimal runtime image
FROM base as production

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser alembic/ ./alembic/
COPY --chown=appuser:appuser alembic.ini .

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose application port
EXPOSE 8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Development stage for local development
FROM base as development

RUN pip install --upgrade pip setuptools wheel
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# Install development tools
RUN pip install \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    mypy \
    ruff

# Copy application code
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Testing stage
FROM development as testing

# Run tests during build
USER root
RUN chown -R appuser:appuser /app
USER appuser

# Run linting and tests
RUN python -m ruff check src/
RUN python -m mypy src/
RUN python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Production-ready stage with optimizations
FROM python:3.11-slim as optimized

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set production environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO

WORKDIR /app

# Create non-root user with specific UID for security
RUN groupadd -r appuser && useradd -r -g appuser -u 1001 appuser

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-deps -r requirements.txt

# Copy application files with proper permissions
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/

# Create required directories
RUN mkdir -p /app/logs /app/data /app/models && \
    chown -R appuser:appuser /app && \
    chmod 755 /app/src

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check with custom endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health/ready || exit 1

# Use exec form for better signal handling
ENTRYPOINT ["python", "-m"]
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--access-log"]