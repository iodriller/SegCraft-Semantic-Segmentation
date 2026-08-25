FROM ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 AS uv

FROM python:3.11-slim-bookworm@sha256:0bee7276f83efd4a1ee05bbbf4281d95ed28e079220a9457f25a93e3f1e3c31b AS builder
COPY --from=uv /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE.md ./
COPY src/ ./src/
COPY configs/ ./configs/
RUN uv sync --frozen --extra web --no-editable

FROM python:3.11-slim-bookworm@sha256:0bee7276f83efd4a1ee05bbbf4281d95ed28e079220a9457f25a93e3f1e3c31b AS runtime
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 \
    SEGCRAFT_HOST=0.0.0.0 SEGCRAFT_PORT=8000 SEGCRAFT_OPEN_BROWSER=0
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && useradd --create-home --uid 10001 segcraft \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder --chown=segcraft:segcraft /app /app
RUN mkdir -p /app/outputs && chown segcraft:segcraft /app/outputs
USER segcraft
EXPOSE 8000
VOLUME ["/app/outputs"]
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=12 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]
CMD ["segcraft-web"]
