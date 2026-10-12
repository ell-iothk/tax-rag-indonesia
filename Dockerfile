# API saja. Qdrant dan Ollama jalan sebagai layanan terpisah.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# README.md ikut karena pyproject.toml merujuknya lewat readme = "README.md"
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install -e . && pip install "uvicorn[standard]" fastapi httpx

COPY api/ api/
COPY scripts/ scripts/
COPY data/raw/ data/raw/

EXPOSE 8000
CMD ["python", "scripts/mulai_api.py"]
