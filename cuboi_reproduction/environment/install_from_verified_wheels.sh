#!/usr/bin/env bash
set -eu
cd /workspace/src/SuperMimic-Reproduced
env -u PYTHONPATH uv pip install --python .venv/bin/python \
  --no-deps --no-index --find-links /workspace/cache/wheelhouse \
  --require-hashes -r /workspace/environment/remaining-wheels.txt
env -u PYTHONPATH uv sync --locked --offline --group dev --inexact \
  --no-install-package nvidia-cublas --no-install-package nvidia-cudnn-cu13 \
  --no-install-package nvidia-cufft --no-install-package nvidia-cusolver \
  --no-install-package nvidia-cusparselt-cu13 --no-install-package nvidia-nccl-cu13 \
  --no-install-package triton
.venv/bin/python /workspace/scripts/patch_cusparselt_tag.py
env -u PYTHONPATH uv pip check --python .venv/bin/python
env -u PYTHONPATH uv pip freeze --python .venv/bin/python > /workspace/environment/installed-packages.txt
.venv/bin/python --version
.venv/bin/python -c 'import torch, mujoco, warp; print("torch", torch.__version__, "CUDA build", torch.version.cuda, "mujoco", mujoco.__version__, "warp", warp.__version__)'
