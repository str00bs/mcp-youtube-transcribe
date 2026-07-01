# YouTube Transcription MCP Server
# Multi-stage build for production deployment

# Build stage
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set up application
WORKDIR /app
COPY . .

# Create data directories
RUN mkdir -p /app/data/downloads /app/data/temp

# Set environment variables
ENV DOWNLOAD_DIR=/app/data/downloads
ENV TEMP_DIR=/app/data/temp
ENV WHISPER_MODEL=small
ENV LOG_LEVEL=INFO
ENV CLEANUP_TEMP_FILES=true
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.server import mcp; print('OK')" || exit 1

# Default command (stdio mode for MCP)
CMD ["python", "main.py", "--stdio"]

# Alternative: HTTP mode
# CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8080"]
