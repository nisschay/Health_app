FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOME=/home/user

# libpq5 for PostgreSQL; tesseract and poppler back the scanned-PDF fallback,
# which silently degraded without them because only the Python wrappers shipped.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    tesseract-ocr \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run as uid 1000 and only $HOME is writable.
RUN useradd -m -u 1000 user
WORKDIR $HOME/app

COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

USER user
EXPOSE 7860

HEALTHCHECK --interval=60s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-7860}/health" || exit 1

# Shell form so $PORT is expanded: Spaces set 7860, other hosts inject their own.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
