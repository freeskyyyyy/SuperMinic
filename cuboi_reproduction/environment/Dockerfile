# Reuse the cached research OS/graphics/uv foundation, never its project .venv.
FROM supermimic-research:bootstrap

ARG http_proxy
ARG https_proxy
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.12 \
    && ln -sf "$(uv python find 3.12)" /usr/local/bin/python \
    && ln -sf "$(uv python find 3.12)" /usr/local/bin/python3
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

ENV UV_PYTHON=3.12 \
    UV_CACHE_DIR=/workspace/cache/uv \
    UV_LINK_MODE=copy \
    HOME=/workspace/cache/home \
    MPLCONFIGDIR=/workspace/cache/matplotlib \
    TRITON_CACHE_DIR=/workspace/cache/triton \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    MAX_JOBS=2
WORKDIR /workspace/src/SuperMimic-Reproduced
CMD ["sleep", "infinity"]
