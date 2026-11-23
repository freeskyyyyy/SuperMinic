#!/usr/bin/env bash
set -eu
cd /workspace/src/SuperMimic-Reproduced
sha256sum uv.lock > /workspace/environment/original_lock.sha256
env -u PYTHONPATH uv sync --locked --group dev
env -u PYTHONPATH uv pip freeze --python .venv/bin/python > /workspace/environment/installed-packages.txt
.venv/bin/python --version
.venv/bin/python -c 'import torch, mujoco, warp; print("torch", torch.__version__, "CUDA build", torch.version.cuda, "mujoco", mujoco.__version__, "warp", warp.__version__)'
