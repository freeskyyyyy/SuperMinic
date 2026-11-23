"""Run small, real GPU computations before enabling simulation or training."""
import json
from pathlib import Path

import torch
import warp as wp


@wp.kernel
def twice(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    y[i] = 2.0 * x[i]


def main():
    assert torch.cuda.is_available(), "PyTorch CUDA unavailable"
    x = torch.randn(128, 128, device="cuda", requires_grad=True)
    loss = (x @ x.T).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss).item()
    assert torch.isfinite(x.grad).all().item()
    assert x.grad.abs().sum().item() > 0
    wp.init()
    a = wp.ones(16, dtype=float, device="cuda:0")
    b = wp.zeros_like(a)
    wp.launch(twice, dim=16, inputs=[a, b], device="cuda:0")
    wp.synchronize()
    assert (b.numpy() == 2.0).all()
    report = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "warp": wp.__version__,
        "torch_forward_backward": "passed",
        "warp_kernel": "passed",
    }
    Path("/workspace/reports/gpu-check.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
