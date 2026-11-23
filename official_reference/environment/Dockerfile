FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /uvx /usr/local/bin/
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates build-essential pkg-config \
    libegl1 libgl1 libglib2.0-0 libglfw3 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*
COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
ENV UV_CACHE_DIR=/workspace/cache/uv \
    UV_LINK_MODE=copy \
    MPLCONFIGDIR=/workspace/cache/matplotlib \
    TRITON_CACHE_DIR=/workspace/cache/triton \
    WANDB_DATA_DIR=/workspace/cache/wandb \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    MAX_JOBS=2
WORKDIR /workspace
CMD ["bash"]
