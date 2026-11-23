"""Training checkpoint with an explicit reset of simulation episodes on resume."""

import os
import random
from dataclasses import asdict

import numpy as np
import torch


def _training_config(cfg):
  config = asdict(cfg)
  for key in ("max_iterations", "checkpoint_training", "upload_checkpoint_artifacts", "eval_checkpoint_interval", "resume_path", "stop_file", "log_interval", "save_interval", "run_name", "wandb_project"):
    config.pop(key)
  if not config["table_a5"]:
    config.pop("table_a5")  # Match the config of checkpoints saved before this option.
  return config


def save_training_checkpoint(path, runner, next_step, ratio):
  policy = runner.policy_learner
  sampler = runner._sampler
  state = {
    "task_id": runner.task_id,
    "config": _training_config(runner.cfg),
    "num_envs": runner.replay_buffer.num_envs,
    "next_step": next_step,
    "ratio_prev": ratio._prev,
    "world_model": runner._wm.bundle(),
    "actor": policy.actor.state_dict(),
    "critic": policy.critic.state_dict() if hasattr(policy, "critic") else None,
    "world_optimizers": [x.state_dict() for x in runner.world_model_learner.optimizers],
    "actor_optimizers": [x.state_dict() for x in policy.actor_optimizers],
    "critic_optimizers": [x.state_dict() for x in getattr(policy, "critic_optimizers", [])],
    "replay": runner.replay_buffer.state_dict(),
    "sampler": None if sampler is None else {
      "value": sampler.value.clone(),
      "count": sampler.count.clone(),
      "effort": sampler.effort.clone(),
    },
    "torch_rng": torch.get_rng_state(),
    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    "numpy_rng": np.random.get_state(),
    "python_rng": random.getstate(),
  }
  temporary = path.with_suffix(path.suffix + ".tmp")
  torch.save(state, temporary)
  os.replace(temporary, path)


def restore_training_checkpoint(path, runner, ratio):
  state = torch.load(path, map_location="cpu", weights_only=False)
  if state["task_id"] != runner.task_id:
    raise ValueError("checkpoint task does not match current task")
  if state["config"] != _training_config(runner.cfg) or state["num_envs"] != runner.replay_buffer.num_envs:
    raise ValueError("checkpoint training configuration does not match")
  policy = runner.policy_learner
  runner._wm.load_bundle(state["world_model"])
  policy.actor.load_state_dict(state["actor"])
  if state["critic"] is not None:
    policy.critic.load_state_dict(state["critic"])
  for optimizers, saved in (
    (runner.world_model_learner.optimizers, state["world_optimizers"]),
    (policy.actor_optimizers, state["actor_optimizers"]),
    (getattr(policy, "critic_optimizers", []), state["critic_optimizers"]),
  ):
    for optimizer, optimizer_state in zip(optimizers, saved, strict=True):
      rates = [group["lr"] for group in optimizer.param_groups]
      optimizer.load_state_dict(optimizer_state)
      # Preserve the original device tensors when restoring a CPU-loaded checkpoint.
      for group, rate in zip(optimizer.param_groups, rates, strict=True):
        if torch.is_tensor(rate):
          rate.copy_(group["lr"])
          group["lr"] = rate
  runner.replay_buffer.load_state_dict(state["replay"])
  runner.replay_buffer.mark_resume_boundary()
  if state["sampler"] is not None:
    sampler = runner._sampler
    sampler.value.copy_(state["sampler"]["value"])
    sampler.count.copy_(state["sampler"]["count"])
    sampler.effort.copy_(state["sampler"]["effort"])
    runner.motion_cmd.set_sampling_distribution(sampler._distribution())
  ratio._prev = state["ratio_prev"]
  torch.set_rng_state(state["torch_rng"])
  if state["cuda_rng"] is not None:
    torch.cuda.set_rng_state_all(state["cuda_rng"])
  np.random.set_state(state["numpy_rng"])
  random.setstate(state["python_rng"])
  return state["next_step"]
