# syntax=docker/dockerfile:1.7

FROM python:3.11-slim-bookworm

ARG INSTALL_CODEX=1
ARG INSTALL_MOLSCRIBE=1
ARG DOWNLOAD_MODELS=1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/chemx-venv \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    XDG_CACHE_HOME=/opt/chemx-cache \
    CHEMX_MOLSCRIBE_MODEL=/opt/chemx-models/molscribe/swin_base_char_aux_1m680k.pth \
    CHEMX_MARKER_PAGE_CHUNK_SIZE=1 \
    CHEMX_OCR_COMMAND="tesseract {image} stdout -l eng" \
    CHEMX_MOLSCRIBE_COMMAND="/opt/molscribe-venv/bin/python /workspace/scripts/molscribe_predict.py --model /opt/chemx-models/molscribe/swin_base_char_aux_1m680k.pth {image}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        fonts-dejavu \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        nodejs \
        npm \
        tesseract-ocr \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /workspace

COPY pyproject.toml uv.lock README.md ./
COPY docs ./docs
COPY scripts ./scripts
COPY src ./src
COPY tests ./tests

RUN mkdir -p /workspace/runs /opt/chemx-cache /opt/uv-cache /opt/chemx-models/molscribe

RUN uv sync --frozen --extra dev --extra ui --extra gold

RUN if [ "$INSTALL_CODEX" = "1" ]; then npm install -g @openai/codex; fi

RUN if [ "$INSTALL_MOLSCRIBE" = "1" ]; then \
        uv python install 3.10 \
        && uv venv --python 3.10 /opt/molscribe-venv \
        && /opt/molscribe-venv/bin/python -m pip install --upgrade pip \
        && /opt/molscribe-venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==1.13.1 torchvision==0.14.1 \
        && /opt/molscribe-venv/bin/pip install molscribe; \
    else \
        python -m venv /opt/molscribe-venv; \
    fi

RUN if [ "$DOWNLOAD_MODELS" = "1" ]; then \
        uv run --no-sync python scripts/download_datalab_cache.py --include-weights; \
    fi

RUN if [ "$INSTALL_MOLSCRIBE" = "1" ] && [ "$DOWNLOAD_MODELS" = "1" ]; then \
        uv run --no-sync python scripts/download_molscribe_model.py --output "$CHEMX_MOLSCRIBE_MODEL"; \
    fi

ENTRYPOINT ["uv", "run", "--no-sync", "chemx"]
CMD ["--help"]
