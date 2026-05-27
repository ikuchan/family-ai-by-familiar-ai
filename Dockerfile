FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libpq-dev \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY migration/ ./migration/

RUN uv sync --no-group dev --no-group gui

CMD ["uv", "run", "familiar", "--no-tui"]
