FROM python:3.13-slim AS base

# Install system deps for pdfplumber, Docling, and ChromaDB
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source code and config
COPY src/ src/
COPY rubric/ rubric/
COPY .env.example .env.example
COPY DOMAIN_NOTES.md README.md ./

# Create .refinery output directories
RUN mkdir -p .refinery/profiles .refinery/pageindex .refinery/chroma_db

ENTRYPOINT ["uv", "run", "python", "-m", "src.main"]
CMD ["--help"]
