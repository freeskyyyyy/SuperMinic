"""Exercise actual CUDA, Warp and NVIDIA EGL operations in the new environment."""

import json
import os
import platform
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np
import torch
import warp as wp
from OpenGL import GL

from supermimic.data.motion_convert import load_g1_model


@wp.kernel
def square(source: wp.array(dtype=wp.float32), result: wp.array(dtype=wp.float32)):
    index = wp.tid()
    result[index] = source[index] * source[index]


root = Path(__file__).resolve().parents[1]
assert torch.cuda.is_available(), "CUDA is unavailable"
matrix = torch.randn(256, 256, device="cuda", requires_grad=True)
loss = (matrix @ matrix).square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(loss) and torch.isfinite(matrix.grad).all()

wp.init()
source = wp.array(np.arange(64, dtype=np.float32), device="cuda:0")
result = wp.zeros(64, dtype=wp.float32, device="cuda:0")
wp.launch(square, dim=64, inputs=[source, result], device="cuda:0")
np.testing.assert_array_equal(result.numpy(), np.arange(64, dtype=np.float32) ** 2)

model = load_g1_model()
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, height=120, width=160)
renderer.update_scene(data)
pixels = renderer.render()
vendor = GL.glGetString(GL.GL_VENDOR).decode()
render_device = GL.glGetString(GL.GL_RENDERER).decode()
assert pixels.shape == (120, 160, 3)
assert "NVIDIA" in vendor, f"unexpected software EGL renderer: {vendor} / {render_device}"
renderer.close()

report = {
    "python": platform.python_version(), "architecture": platform.machine(),
    "torch": torch.__version__, "cuda_build": torch.version.cuda,
    "cuda_device": torch.cuda.get_device_name(0),
    "compute_capability": torch.cuda.get_device_capability(0),
    "cuda_forward_backward": True, "warp": wp.__version__, "warp_kernel": True,
    "mujoco": mujoco.__version__, "g1_nq": model.nq, "g1_nv": model.nv,
    "egl_vendor": vendor, "egl_renderer": render_device, "render_shape": list(pixels.shape),
}
(root / "reports/P1_compute_render.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
