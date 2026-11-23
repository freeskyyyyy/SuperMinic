"""Exercise stop, checkpoint and resume on the real G1 environment."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import supermimic.runners.supermimic_runner as runner_module
from supermimic.runners.supermimic_runner import SuperMimicRunner
from supermimic.scripts.train import TrainConfig
from supermimic.scripts.train_utils import create_env, setup_device

base = Path("/workspace")
output = base / "runs" / sys.argv[1]
phase = sys.argv[2]
output.mkdir(parents=True, exist_ok=True)
stop_file = output / "stop_request"
cfg = TrainConfig.from_task("SuperMimic-GenCoord-G1")
cfg.env.commands["motion"].motion_file = str(base / "data/run1_subject5_50hz.npz")
cfg.env.commands["motion"].sampling_mode = "external"
cfg.runner.actor_obs = "proprio"
cfg.runner.history_length = 10
cfg.runner.wm.horizon = 9
cfg.runner.wm.batch_size = 16
cfg.runner.wm.num_batches = 1
cfg.runner.policy.batch_size = 16
cfg.runner.max_iterations = 20
cfg.runner.log_interval = 5
cfg.runner.save_interval = 19
cfg.runner.checkpoint_training = True
cfg.runner.stop_file = str(stop_file) if phase == "save" else None
cfg.runner.resume_path = str(output / "phase-save/training_state.pt") if phase == "resume" else None
torch.manual_seed(42)
np.random.seed(42)
runner_module.export_actor_as_onnx = lambda **kwargs: []
runner_module.log_model_artifact = lambda files: None


class SmokeRunner(SuperMimicRunner):
  def _create_log_dir(self):
    path = output / f"phase-{phase}"
    path.mkdir(parents=True, exist_ok=True)
    return path

  def step(self):
    result = super().step()
    if phase == "save" and self.replay_buffer.num_transitions == 16:
      stop_file.touch()
    return result


env = create_env(cfg.env, 16, setup_device("cuda:0"))
runner = SmokeRunner(env, cfg.runner, "SuperMimic-GenCoord-G1", device="cuda:0")
runner.run()
checkpoint = output / f"phase-{phase}/training_state.pt"
state = torch.load(checkpoint, map_location="cpu", weights_only=False)
assert state["next_step"] == (16 if phase == "save" else 20)
assert state["world_optimizers"][0]["state"]
assert state["actor_optimizers"][0]["state"]
assert state["critic_optimizers"][0]["state"]
if phase == "resume":
  previous = torch.load(output / "phase-save/training_state.pt", map_location="cpu", weights_only=False)
  assert any(not torch.equal(state["actor"][k], previous["actor"][k]) for k in state["actor"])
  assert state["replay"]["num_transitions"] == 20
  assert state["replay"]["storage"]["truncated"][:, 15].all()
print(json.dumps({"phase": phase, "next_step": state["next_step"], "optimizer_states": "present", "checkpoint": str(checkpoint)}))
