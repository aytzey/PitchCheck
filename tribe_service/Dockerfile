FROM python:3.11-slim AS rust-core-builder

ENV PIP_NO_CACHE_DIR=1 \
    PATH=/root/.cargo/bin:$PATH

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  curl \
  pkg-config \
  && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --profile minimal --default-toolchain stable

RUN pip install --upgrade pip setuptools wheel && \
  pip install "maturin>=1.7,<2"

COPY tribe_service/rust_core ./rust_core

RUN maturin build \
  --manifest-path rust_core/Cargo.toml \
  --release \
  --features extension-module \
  -o /wheels

FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRIBE_CACHE_DIR=/models

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
  ffmpeg \
  git \
  build-essential \
  libsndfile1 \
  && rm -rf /var/lib/apt/lists/*

COPY tribe_service/requirements.txt ./requirements.txt

RUN pip install --upgrade pip setuptools wheel

RUN pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1 \
  torchvision==0.22.1

RUN pip install uv==0.11.2

RUN ln -sf /usr/local/bin/uv /usr/bin/uv && \
  ln -sf /usr/local/bin/uvx /usr/bin/uvx

RUN pip install -r requirements.txt

COPY --from=rust-core-builder /wheels /tmp/pitchcheck-wheels

RUN pip install /tmp/pitchcheck-wheels/*.whl && \
  rm -rf /tmp/pitchcheck-wheels

RUN pip install --upgrade \
  git+https://github.com/huggingface/transformers.git@9914a3641f7aaaabb0bcdfcd73a54a1cfa70c3e7

RUN python -m spacy download en_core_web_lg

RUN pip install --no-deps git+https://github.com/facebookresearch/tribev2.git@main

COPY tribe_service /app/tribe_service

RUN python /app/tribe_service/patch_tribev2_whisperx.py

RUN mkdir -p /models

EXPOSE 8090

CMD ["uvicorn", "tribe_service.app:app", "--host", "0.0.0.0", "--port", "8090"]
