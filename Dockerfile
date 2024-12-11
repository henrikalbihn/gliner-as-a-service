FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV PYTHONUNBUFFERED=1

WORKDIR /workspace/

USER root

# Install ffmpeg, libsm6, libxext6, gcc, curl
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 gcc curl -y

# Install uv
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=ghcr.io/astral-sh/uv:0.4.15 /uv /bin/uv

# Place executables in the environment at the front of the path
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#using-the-environment
ENV PATH="/workspace/.venv/bin:$PATH"

# Compile bytecode
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#compiling-bytecode
ENV UV_COMPILE_BYTECODE=1

# uv Cache
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#caching
ENV UV_LINK_MODE=copy

# Install dependencies
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#intermediate-layers
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

ENV PYTHONPATH=/workspace

# COPY ./scripts /workspace/scripts

# Copy project files
COPY ./pyproject.toml ./uv.lock /workspace/
COPY ./app /workspace/app
# COPY ./appV1.py /workspace/appV1.py
COPY ./scripts /workspace/scripts
RUN chmod +x /workspace/scripts/* && \
    chown 1001:1001 /workspace/scripts/*

WORKDIR /workspace

RUN mkdir -p /workspace/models && \
    chown -R 1001:1001 /workspace/models

# Sync the project
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#intermediate-layers
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# Give user 1001 write access to cache
RUN mkdir -p /.cache && \
    chown -R 1001:1001 /.cache

EXPOSE 8000

USER 1001
