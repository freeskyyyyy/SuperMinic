"""Run 31 original-runner iterations to verify real updates, not convergence."""
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from supermimic.runners.supermimic_runner import SuperMimicRunner
from supermimic.scripts.train import TrainConfig
from supermimic.scripts.train_utils import create_env, setup_device

base = Path("/workspace")
torch.manual_seed(42)
np.random.seed(42)
task = "SuperMimic-GenCoord-G1"
cfg = TrainConfig.from_task(task)
cfg.env.commands["motion"].motion_file = str(base / "data/run1_subject5_50hz.npz")
cfg.env.commands["motion"].sampling_mode = "external"
cfg.runner.run_name = "engineering-smoke64-h10-eight-transitions"
cfg.runner.actor_obs = "proprio"
cfg.runner.history_length = 10
cfg.runner.wm.horizon = 9  # Nine recorded states provide eight scored transitions.
cfg.runner.wm.batch_size = 64
cfg.runner.wm.num_batches = 1
cfg.runner.policy.batch_size = 64
cfg.runner.max_iterations = 31
cfg.runner.log_interval = 5
cfg.runner.save_interval = 30
(base / "reports/training-smoke-config.json").write_text(
    json.dumps(asdict(cfg.runner), indent=2) + "\n", encoding="utf-8"
)
env = create_env(cfg.env, 64, setup_device("cuda:0"))
runner = SuperMimicRunner(env, cfg.runner, task, device="cuda:0")
models = {
    "actor": runner.policy_learner.actor,
    "world_model": runner._dynamics,
    "critic": runner.policy_learner.critic,
}
before = {name: [p.detach().cpu().clone() for p in m.parameters()]
          for name, m in models.items()}
started = time.perf_counter()
runner.run()
changed = {}
for name, model in models.items():
    parameters = list(model.parameters())
    assert all(torch.isfinite(p).all().item() for p in parameters), name
    changed[name] = any(
        not torch.equal(old, new.detach().cpu())
        for old, new in zip(before[name], parameters, strict=True)
    )
    assert changed[name], f"{name} parameters did not update"
report = {
    "iterations": 31,
    "num_envs": 64,
    "elapsed_seconds_including_compilation": time.perf_counter() - started,
    "parameter_updates": changed,
    "finite_parameters": True,
    "scope": "engineering smoke test, not tracking convergence",
}
(base / "reports/training-smoke-check.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
