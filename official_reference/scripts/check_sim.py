"""Validate one real motion and 64 G1 physics environments without training."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from mjlab.scripts.csv_to_npz import MotionLoader

from supermimic.scripts.train_utils import create_env
from supermimic.tasks.env_cfgs import gencoord_env_cfg


def main():
    base = Path("/workspace")
    motion = MotionLoader(
        str(base / "data/run1_subject5.csv"), 30.0, 50.0, device="cpu"
    )
    arrays = motion.arrays()
    assert all(
        np.isfinite(v).all() for k, v in arrays.items() if k != "joint_names"
    )
    motion_path = base / "data/run1_subject5_50hz.npz"
    np.savez(motion_path, **arrays)
    cfg = gencoord_env_cfg()
    cfg.commands["motion"].motion_file = str(motion_path)
    env = create_env(cfg, 64, "cuda:0")
    try:
        env.reset()
        action = torch.zeros((64, env.num_actions), device="cuda:0")
        start = time.perf_counter()
        resets = 1
        for step in range(1000):
            _, reward, _, _ = env.step(action)
            if step % 100 == 0:
                assert torch.isfinite(reward).all().item()
                assert torch.isfinite(env.unwrapped.sim.data.qpos).all().item()
                assert torch.isfinite(env.unwrapped.sim.data.qvel).all().item()
                env.reset()
                resets += 1
        torch.cuda.synchronize()
        report = {
            "num_envs": 64,
            "control_steps": 1000,
            "explicit_resets": resets,
            "elapsed_seconds_including_kernel_warmup": time.perf_counter() - start,
            "num_actions": env.num_actions,
            "converted_frames": int(arrays["joint_pos"].shape[0]),
            "control_dt": env.unwrapped.step_dt,
            "physics_step_reset_finite_check": "passed",
            "policy": "zero action; not a trained policy",
            "visual_review": "pending",
        }
        (base / "reports/g1-physics-check.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
