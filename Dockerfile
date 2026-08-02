# Redress: frontend build, then a Python runtime that serves both.
#
# One image rather than two services. The API and the SPA are served from the
# same origin in production, which is what the Vite dev proxy imitates — so
# the request path exercised in development is the one that ships, and CORS
# never becomes load-bearing.
#
# Ollama is deliberately NOT in this image. It needs the GPU, it holds several
# gigabytes of weights, and it has its own release cadence; baking it in would
# produce a 20 GB image that has to be rebuilt to change a model. See
# docker-compose.yml.

# ---------- stage 1: build the SPA ----------
FROM node:24-alpine AS frontend

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci || npm install

COPY frontend/ ./
RUN npm run build

# ---------- stage 2: runtime ----------
FROM python:3.13-slim

# Never buffer stdout: container logs should appear as they happen, not when
# the process exits.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    REDRESS_FRONTEND_DIST=/app/static

WORKDIR /app

COPY backend/pyproject.toml ./
# Installed from the manifest rather than pinned here so the container and a
# local checkout resolve the same dependency set.
RUN pip install --no-cache-dir \
    "pydantic>=2.7" \
    "fastapi>=0.111" \
    "uvicorn[standard]>=0.30" \
    "python-multipart>=0.0.9" \
    "pypdf>=5.0" \
    "pyyaml>=6.0"

COPY backend/app ./app
COPY --from=frontend /build/dist ./static

# Runs as a non-root user: this process accepts arbitrary uploaded files from
# the network and parses them with a PDF library. That is exactly the surface
# where a parser bug turns into code execution, so it should not be root's.
RUN useradd --create-home --uid 10001 redress && chown -R redress /app
USER redress

EXPOSE 8000

# Ollama lives on the host or in a sibling container; the default assumes the
# compose network. One worker, matching the single-audit concurrency the job
# runner enforces — a second worker would only queue behind the same GPU.
ENV OLLAMA_HOST=http://ollama:11434 \
    REDRESS_MODEL=qwen3.5:9b

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
