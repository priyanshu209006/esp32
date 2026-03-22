FROM python:3.11-slim

# Install ffmpeg for pydub MP3→PCM conversion
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (layer-cached; only re-runs when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py .

# Run as non-root for security
RUN adduser --disabled-password --gecos "" appuser
USER appuser

# Render injects PORT — we honour it with ${PORT:-8000} fallback
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
