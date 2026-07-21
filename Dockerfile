# Copyright 2026 Query Farm LLC - https://query.farm
#
# Single image serving BOTH transports of the vgi-recommender worker:
#   docker run ... IMG            -> HTTP server on $PORT (default 8000; /health, VGI RPC)
#   docker run -i ... IMG stdio   -> stdio worker DuckDB spawns on-host
# See docker-entrypoint.sh. The worker is keyless and stateless: it buffers the
# interaction relation per execution and fits an implicit-feedback ALS model in
# process — no persisted state, so no /data volume.
# syntax=docker/dockerfile:1
FROM python:3.13-slim

ARG VERSION=0.0.0
ARG GIT_COMMIT=unknown
ARG SOURCE_URL=https://github.com/Query-farm/vgi-recommender

LABEL org.opencontainers.image.title="vgi-recommender" \
      org.opencontainers.image.description="Collaborative-filtering recommendations (implicit-feedback ALS) for DuckDB via VGI (stdio + HTTP)" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.licenses="MIT" \
      farm.query.vgi.transports='["http","stdio"]'

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    # Single-threaded BLAS keeps ALS deterministic (pinned random_state); this
    # mirrors what vgi_recommender.recommender sets before numpy imports.
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

WORKDIR /app

# curl backs the HEALTHCHECK and the CI /health smoke. libgomp1 is the OpenMP
# runtime `implicit` (and scipy) link against at import time — absent from -slim,
# so without it `import vgi_recommender.worker` fails with `libgomp.so.1: cannot
# open shared object file`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install the worker + HTTP-serving extra from the source tree.
COPY pyproject.toml README.md LICENSE ./
COPY vgi_recommender ./vgi_recommender
RUN pip install '.[serve]'

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=8s \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
